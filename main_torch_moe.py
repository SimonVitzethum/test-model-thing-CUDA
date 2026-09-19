"""PyTorch-Port von test-model-thing (MLX-Original: main.py) + MoE.

- Byte-Embedding (256 -> dim), recurrente States mit gelerntem Decay.
- P0.1: echtes Truncated-BPTT-Fenster über alle Gewichte (statt 1-Step-
  Dummy-Grad-Hack). States werden nur an Fenstergrenzen detached.
- Entropie-adaptives Sampling nur für Generierung, nie als Trainingsziel.
- MoE-Feedforward mit Top-k Routing, getrennte total/aktiv-Zählung.

Datei ist bewusst standalone (nur torch nötig).
"""
import argparse
import glob
import itertools
import os
import random
import sys
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embed = nn.Embedding(256, dim)

    def forward(self, x: torch.Tensor):
        return self.embed(x)


class Decoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.decode = nn.Linear(dim, 256)
        self.stop = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor):
        return self.decode(x), torch.sigmoid(self.stop(x))


class MoELayer(nn.Module):
    """Ein recurrenter Layer; Feedforward-Anteil als MoE (E=1 == dense)."""

    def __init__(self, dim: int, num_experts: int = 1, top_k: int = 1):
        super().__init__()
        assert num_experts >= 1
        assert 1 <= top_k <= num_experts
        self.dim = dim
        self.num_experts = num_experts
        self.top_k = top_k if num_experts > 1 else 1

        # trainierbar (wie MLX: self.decay)
        self.decay = nn.Parameter(torch.zeros(dim))

        # Buffer: recurrenter Carry (kein Gradient über Fenstergrenzen).
        # decaytrace/embedtrace sind deprecated (nur für alte Checkpoints
        # lesbar gehalten) und werden nicht mehr für Grad-Surgery benutzt.
        self.register_buffer("states", torch.zeros(dim))
        self.register_buffer("decaytrace", torch.zeros(dim))
        self.register_buffer("embedtrace", torch.zeros(256, dim))
        self.register_buffer("usage", torch.zeros(num_experts))

        self.norm = nn.LayerNorm(dim)
        self.experts = nn.ModuleList(
            [nn.Linear(dim, dim, bias=False) for _ in range(num_experts)]
        )
        self.router = nn.Linear(dim, num_experts) if num_experts > 1 else None
        self.silu = nn.SiLU()

    def forward_moe(self, h_norm: torch.Tensor):
        """Batched MoE: h_norm (..., dim) -> y (..., dim).

        E=1 ist dense. Sonst Top-k über Softmax-Gewichte; Experten werden
        gestapelt und per gather kombiniert (E klein, Korrektheit > Kernel).
        """
        if self.num_experts == 1:
            return self.silu(self.experts[0](h_norm)), None, None, None, None
        logits = self.router(h_norm)  # (..., E)
        probs = torch.softmax(logits, dim=-1)
        _top_v, top_idx = torch.topk(logits, self.top_k, dim=-1)  # (..., k)
        w = torch.gather(probs, -1, top_idx)
        w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        outs = torch.stack([self.silu(e(h_norm)) for e in self.experts],
                           dim=-2)  # (..., E, dim)
        idx = top_idx.unsqueeze(-1).expand(*top_idx.shape, h_norm.shape[-1])
        sel = torch.gather(outs, -2, idx)  # (..., k, dim)
        y = (sel * w.unsqueeze(-1)).sum(dim=-2)
        return y, probs, top_idx, w, logits

    def forward(self, enc: torch.Tensor, x: torch.Tensor, prev: torch.Tensor):
        """Ein Zeitschritt. prev ist der Carry (mit Grad innerhalb Fenster).
        Alle Tensoren (dim,) single-stream oder (B, dim) batched."""
        decay = torch.sigmoid(self.decay)  # (dim,) broadcastet über Batch
        state = decay * prev + enc
        h_norm = self.norm(state)
        if self.num_experts == 1:
            y = self.silu(self.experts[0](h_norm))
            aux = torch.zeros((), device=x.device)
            zloss = torch.zeros((), device=x.device)
            info = None
        else:
            y, probs, top_idx, w, logits = self.forward_moe(h_norm)
            # Uniformitäts-Aux pro Position: E*sum(p^2)-1 in [0, E-1].
            # Single-Token: Skalar; batched: (...,) und Caller mittelt.
            aux = self.num_experts * (probs * probs).sum(dim=-1) - 1.0
            zloss = torch.logsumexp(logits, dim=-1).pow(2)
            info = (probs.detach(), top_idx.detach())
        out = x + y
        return out, state, decay, aux, zloss, info


class Model(nn.Module):
    def __init__(self, dim: int, layers: int, temp: float = 0.75,
                 num_experts: int = 1, top_k: int = 1,
                 aux_coef: float = 0.01, zloss_coef: float = 0.001,
                 bptt: int = 64, ema_tau: float = 0.99,
                 latent_w: float = 1.0, ce_w: float = 1.0,
                 var_w: float = 1.0, stop_w: float = 1.0):
        super().__init__()
        self.dim = dim
        self.layercount = layers
        self.temp = temp
        self.num_experts = num_experts
        self.top_k = top_k if num_experts > 1 else 1
        self.aux_coef = aux_coef
        self.zloss_coef = zloss_coef
        self.bptt = max(1, bptt)
        self.ema_tau = ema_tau
        self.latent_w = latent_w
        self.ce_w = ce_w
        self.var_w = var_w
        self.stop_w = stop_w

        self.encoder = Encoder(dim)
        # P0.2: entkoppelter Ziel-Encoder (EMA), kein Gradient.
        self.target_encoder = Encoder(dim)
        with torch.no_grad():
            self.target_encoder.load_state_dict(self.encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.decoder = Decoder(dim)
        self.layers = nn.ModuleList(
            [MoELayer(dim, num_experts, self.top_k) for _ in range(layers)]
        )

    # -- Sampling / Utils (MLX-nah) --
    @torch.no_grad()
    def sample(self, output: torch.Tensor) -> int:
        probs = torch.softmax(output, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum() / torch.log(torch.tensor(256.0))
        temp = max(0.1, self.temp * (1.0 - self.temp * float(entropy)))
        return torch.distributions.Categorical(logits=output / temp).sample().item()

    @torch.no_grad()
    def reset(self):
        for layer in self.layers:
            layer.states.zero_()
            layer.decaytrace.zero_()
            layer.embedtrace.zero_()
            layer.usage.zero_()

    def device(self):
        return next(self.parameters()).device

    # -- Forward eines Steps (frozen / Generierung, 1 Byte) --
    @torch.no_grad()
    def step_frozen(self, c: torch.Tensor):
        enc = self.encoder(c)
        x = enc
        for layer in self.layers:
            x, state, decay, aux, zloss, info = layer(enc, x, layer.states)
            layer.states.copy_(state)
            if info is not None:
                for e in info[1].tolist():
                    layer.usage[e] += 1
        return self.decoder(x)

    def step(self, c: torch.Tensor, dummies=None, frozen: bool = False):
        # dummies-Arg nur noch aus Kompatibilität (wird ignoriert).
        if frozen:
            return (None, None, None, None, None, None), self.step_frozen(c)
        enc = self.encoder(c)
        x = enc
        states, decays, auxs, zlosses, infos = [], [], [], [], []
        for layer in self.layers:
            x, state, decay, aux, zloss, info = layer(enc, x, layer.states)
            states.append(state)
            decays.append(decay)
            auxs.append(aux)
            zlosses.append(zloss)
            infos.append(info)
        output, stop = self.decoder(x)
        return (x, states, decays, auxs, zlosses, infos), (output, stop)

    def loss_terms(self, x, output, stop, nextb, end, auxs, zlosses):
        loss = self.var_w * torch.clamp(1.0 - torch.sqrt(x.var() + 1e-4), min=0.0)
        if nextb is not None:
            n = torch.tensor(nextb, device=x.device, dtype=torch.long)
            with torch.no_grad():
                tgt = self.target_encoder(n)
            if self.latent_w > 0:
                loss = loss + self.latent_w * torch.mean((x - tgt) ** 2)
            if self.ce_w > 0:
                loss = loss + self.ce_w * (
                    -output[n] + torch.logsumexp(output, dim=-1))
            if self.stop_w > 0:
                target = torch.tensor([1.0 if end else 0.0], device=x.device)
                loss = loss + self.stop_w * torch.mean((stop.view(-1) - target) ** 2)
            if self.num_experts > 1 and len(auxs) > 0:
                loss = loss + self.aux_coef * torch.stack(auxs).mean()
                loss = loss + self.zloss_coef * torch.stack(zlosses).mean()
        return loss

    @torch.no_grad()
    def update_target_ema(self):
        for t, o in zip(self.target_encoder.parameters(),
                        self.encoder.parameters()):
            t.mul_(self.ema_tau).add_(o.detach(), alpha=1.0 - self.ema_tau)

    @torch.no_grad()
    def frozen_call(self, currb: int):
        dev = self.device()
        c = torch.tensor(currb, device=dev, dtype=torch.long)
        (_, states, decays, auxs, zlosses, infos), (output, stop) = self.step(c, frozen=True)
        return self.sample(output), float(stop.view(-1)[0])

    def train_window(self, curr_list: list, next_list: list, end_list: list,
                     optimizer) -> tuple:
        """TBPTT über ein Fenster: voller Gradient für ALLE Gewichte
        (Encoder, Decoder, Experten, Router, Norm, Decay) innerhalb des
        Fensters; Carry wird nur an der Fenstergrenze detached."""
        dev = self.device()
        assert len(curr_list) == len(next_list) == len(end_list) and len(curr_list) > 0
        T = len(curr_list)
        carries = [l.states.detach().clone() for l in self.layers]
        losses = []
        last_out, last_stop = None, None
        for t in range(T):
            c = torch.tensor(curr_list[t], device=dev, dtype=torch.long)
            enc = self.encoder(c)
            x = enc
            auxs, zlosses = [], []
            new_carries = []
            for i, layer in enumerate(self.layers):
                x, state, decay, aux, zloss, _info = layer(enc, x, carries[i])
                new_carries.append(state)
                auxs.append(aux)
                zlosses.append(zloss)
            carries = new_carries
            output, stop = self.decoder(x)
            last_out, last_stop = output, stop
            losses.append(self.loss_terms(x, output, stop, next_list[t],
                                          end_list[t], auxs, zlosses))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            self.update_target_ema()
            for layer, final in zip(self.layers, carries):
                layer.states.copy_(final.detach())
            b = self.sample(last_out.detach())
            s = float(last_stop.detach().view(-1)[0])
        return b, s, float(loss.detach())

    def train_step(self, currb: int, nextb, end: bool, optimizer) -> tuple:
        # Kompatibilitäts-Wrapper: Fenster der Länge 1.
        b, s, _ = self.train_window([currb], [nextb], [end], optimizer)
        return b, s

    def train_batch(self, curr: torch.Tensor, nxt: torch.Tensor,
                    end: torch.Tensor, carries, optimizer,
                    frozen: bool = False):
        """Vektorisiertes TBPTT über (B, T) Bytes. Ein Backward pro Batch.

        curr/nxt: (B, T) long; end: (B, T) bool. carries: Liste mit je
        (B, dim)-Carry pro Layer (Start des Fensters). Gibt
        (loss, ce_mean, neue_carries) zurück; frozen=True schiebt nur
        Carries ohne Gewichts-Update.
        """
        B, T = curr.shape
        dev = curr.device
        if carries is None:
            carries = [torch.zeros(B, self.dim, device=dev)
                       for _ in range(self.layercount)]
        if frozen:
            with torch.no_grad():
                for t in range(T):
                    enc = self.encoder(curr[:, t])  # (B, dim)
                    x = enc
                    new_carries = []
                    for i, layer in enumerate(self.layers):
                        x, state, _d, _a, _z, _in = layer(enc, x, carries[i])
                        new_carries.append(state)
                    carries = new_carries
                    self.decoder(x)
            return 0.0, 0.0, [c.detach() for c in carries]

        losses, ce_parts = [], []
        for t in range(T):
            enc = self.encoder(curr[:, t])  # (B, dim)
            x = enc
            auxs, zlosses = [], []
            new_carries = []
            for i, layer in enumerate(self.layers):
                x, state, _d, aux, zloss, _in = layer(enc, x, carries[i])
                new_carries.append(state)
                auxs.append(aux)
                zlosses.append(zloss)
            carries = new_carries
            output, stop = self.decoder(x)  # (B,256), (B,1)
            step_loss = self.var_w * torch.clamp(
                1.0 - torch.sqrt(x.var() + 1e-4), min=0.0)
            with torch.no_grad():
                tgt = self.target_encoder(nxt[:, t])  # (B, dim)
            if self.latent_w > 0:
                step_loss = step_loss + self.latent_w * ((x - tgt) ** 2).mean()
            ce_tok = (-output.gather(1, nxt[:, t : t + 1]).squeeze(1)
                      + torch.logsumexp(output, dim=-1))  # (B,)
            ce_parts.append(ce_tok.detach().mean())
            if self.ce_w > 0:
                step_loss = step_loss + self.ce_w * ce_tok.mean()
            if self.stop_w > 0:
                target = end[:, t].float().unsqueeze(1)
                step_loss = step_loss + self.stop_w * ((stop - target) ** 2).mean()
            if self.num_experts > 1:
                step_loss = step_loss + self.aux_coef * torch.stack(
                    [a.mean() for a in auxs]).mean()
                step_loss = step_loss + self.zloss_coef * torch.stack(
                    [z.mean() for z in zlosses]).mean()
            losses.append(step_loss)
        loss = torch.stack(losses).mean()
        ce_mean = torch.stack(ce_parts).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            self.update_target_ema()
        return (float(loss.detach()), float(ce_mean.detach()),
                [c.detach() for c in carries])

    def __call__(self, currb: int, nextb, end: bool, frozen: bool, optimizer=None):
        if frozen:
            return self.frozen_call(currb)
        assert optimizer is not None, "train_step braucht optimizer"
        return self.train_step(currb, nextb, end, optimizer)

    # -- Param-Zählung: total vs. aktiv --
    def count_total(self) -> int:
        e, k, d, L = self.num_experts, self.top_k, self.dim, self.layercount
        per_layer = e * d * d + 3 * d + (d * e + e if e > 1 else 0)
        return 256 * d + L * per_layer + 256 * d + 256 + d + 1

    def count_active(self) -> int:
        e, k, d, L = self.num_experts, self.top_k, self.dim, self.layercount
        per_layer = k * d * d + 3 * d + (d * e + e if e > 1 else 0)
        return 256 * d + L * per_layer + 256 * d + 256 + d + 1

    def count(self) -> int:
        return self.count_total()

    # -- Save/Load: nur Gewichte, nie recurrentes Memory (P0.4) --
    METABUFFERS = ("states", "usage", "decaytrace", "embedtrace")

    def weights_state_dict(self):
        return {k: v for k, v in self.state_dict().items()
                if not any(k.endswith("." + b) for b in self.METABUFFERS)}

    def save(self, path: str):
        tmp = os.path.join(os.path.dirname(path) or ".",
                           "temporary-" + os.path.basename(path))
        torch.save(
            {"model": self.weights_state_dict(),
             "meta": {"dim": self.dim, "layers": self.layercount,
                      "experts": self.num_experts, "top_k": self.top_k}},
            tmp,
        )
        os.replace(tmp, path)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        data = torch.load(path, map_location="cpu", weights_only=True)
        try:
            self.load_state_dict(data["model"], strict=False)
        except Exception:
            pass


class Runtime:
    def __init__(self, path: str, threshold: float, dim: int, layers: int,
                 temp: float, lr: float, num_experts: int = 1, top_k: int = 1,
                 aux_coef: float = 0.01, zloss_coef: float = 0.001,
                 bptt: int = 64, ema_tau: float = 0.99,
                 latent_w: float = 1.0, ce_w: float = 1.0,
                 var_w: float = 1.0, stop_w: float = 1.0):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        self.model = Model(dim, layers, temp, num_experts, top_k,
                           aux_coef, zloss_coef, bptt, ema_tau,
                           latent_w, ce_w, var_w, stop_w)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.path = path
        self.threshold = threshold
        self.step = 0

    def save_periodic(self):
        self.step += 1
        if self.step % 500 == 0:
            self.model.save(self.path)

    def call(self, c: int, n, end: bool, save: bool, frozen: bool):
        out = self.model(c, n, end, frozen,
                         None if frozen else self.optimizer)
        if save:
            self.save_periodic()
        return out

    def write(self, b: int):
        sys.stdout.buffer.write(bytes([b]))
        sys.stdout.flush()

    def chat(self, save: bool, frozen: bool):
        train_w = (not frozen)
        while True:
            text = input(f"\n[{self.now()}]\nUser >> ")
            data = (text + "\n").encode("utf-8")
            pairs = [(c, n) for c, n in itertools.pairwise(data)]
            if train_w and len(pairs) > 0:
                # Prompt per TBPTT-Fenster mittrainieren (mit Zielen).
                for s in range(0, len(pairs), self.model.bptt):
                    ch = pairs[s:s + self.model.bptt]
                    cc = [c for c, _ in ch]
                    nn_ = [n for _, n in ch]
                    ee = [i == len(pairs) - 1 for i in
                          range(s, min(s + self.model.bptt, len(pairs)))]
                    self.model.train_window(cc, nn_, ee, self.optimizer)
                    if save:
                        self.save_periodic()
            else:
                for i, (c, n) in enumerate(pairs):
                    self.model.frozen_call(c)
            print(f"\n[{self.now()}]\nModel >> ", end="", flush=True)
            b = data[-1]
            # Generierung ohne Ziel: Gewichte immer frozen (nur Carry läuft).
            for _ in range(512):
                b, stop = self.model.frozen_call(b)
                self.write(b)
                if stop > self.threshold:
                    print()
                    break
            else:
                print()

    def train(self, save: bool, frozen: bool, dataset: str, batch: int = 8,
                max_carry: int = 2048):
        files = glob.glob(dataset, recursive=True)
        if not files:
            raise FileNotFoundError(f"Glob {dataset!r} fand nichts.")
        random.shuffle(files)
        T = self.model.bptt
        dev = self.device
        logged = 0
        while True:
            for file in files:
                with open(file, "rb") as f:
                    raw = f.read(1 << 20)  # 1 MiB pro File und Epoche
                if len(raw) < batch * 16 + 1:
                    continue
                ids = list(raw)
                n = (len(ids) - 1) // batch
                streams = torch.tensor(
                    [ids[i * n: i * n + n] for i in range(batch)],
                    dtype=torch.long, device=dev)  # (B, n)
                carries = None
                carried = 0  # Tokens seit Reset (P0.4: Drift begrenzen)
                for s in range(0, n - 1, T):
                    e = min(s + T, n - 1)
                    if e - s < 8:
                        continue
                    if carried >= max_carry:
                        carries = None  # Reset an lies: frischer Carry
                        carried = 0
                    curr = streams[:, s:e]
                    nxt = streams[:, s + 1: e + 1]
                    end = (nxt == 10)  # \n als EOS-Markierung (P1.6)
                    loss, ce, carries = self.model.train_batch(
                        curr, nxt, end, carries, self.optimizer,
                        frozen=frozen)
                    carried += (e - s)
                    if not frozen:
                        logged += 1
                        if logged % 20 == 0:
                            print(f"\n[batch {logged}] loss {loss:.4f} "
                                  f"CE {ce:.4f} BPC {ce / 0.6931:.4f} "
                                  f"({batch}x{e-s} tok)",
                                  flush=True)
                    if save:
                        self.save_periodic()

    def now(self):
        return datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

    def __call__(self, mode: str, dataset: str, save: bool, frozen: bool,
                 batch: int = 8, max_carry: int = 2048):
        self.model.load(self.path)
        self.model.reset()  # P0.4: Inference startet immer mit Null-Memory
        print(f"device: {self.device}, total: {self.model.count_total():,}, "
              f"aktiv: {self.model.count_active():,} "
              f"(E={self.model.num_experts}, k={self.model.top_k})")
        try:
            if mode == "train":
                self.train(save, frozen, dataset, batch=batch,
                           max_carry=max_carry)
            elif mode == "chat":
                self.chat(save, frozen)
        finally:
            if save:
                # kompletter Checkpoint: Gewichte + Optimizer, ohne Memory.
                tmp = os.path.join(os.path.dirname(self.path) or ".",
                                   "temporary-" + os.path.basename(self.path))
                torch.save({"model": self.model.weights_state_dict(),
                            "optim": self.optimizer.state_dict()}, tmp)
                os.replace(tmp, self.path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test-model-thing torch+moe")
    parser.add_argument("path")
    parser.add_argument("mode", choices=["train", "chat"])
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.set_defaults(save=True)
    parser.add_argument("--dataset", default="wikipedia_clean/**/wiki_*")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--experts", type=int, default=1)
    parser.add_argument("--topk", type=int, default=1)
    parser.add_argument("--temp", type=float, default=0.75)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--bptt", type=int, default=64)
    parser.add_argument("--ema-tau", type=float, default=0.99)
    parser.add_argument("--latent-w", type=float, default=1.0)
    parser.add_argument("--ce-w", type=float, default=1.0)
    parser.add_argument("--var-w", type=float, default=1.0)
    parser.add_argument("--stop-w", type=float, default=1.0)
    parser.add_argument("--no-latent", dest="latent_w", action="store_const",
                        const=0.0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-carry", type=int, default=2048)
    args = parser.parse_args()

    rt = Runtime(path=args.path, threshold=args.threshold, dim=args.dim,
                 layers=args.layers, temp=args.temp, lr=args.lr,
                 num_experts=args.experts, top_k=args.topk, bptt=args.bptt,
                 ema_tau=args.ema_tau, latent_w=args.latent_w,
                 ce_w=args.ce_w, var_w=args.var_w, stop_w=args.stop_w)
    rt(args.mode, args.dataset, args.save, args.frozen, batch=args.batch,
       max_carry=args.max_carry)
