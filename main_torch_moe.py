"""PyTorch-Port von test-model-thing (MLX-Original: main.py) + MoE.

Behält die Original-Semantik bei:
- Byte-Embedding (256 -> dim), recurrente States mit gelerntem Decay,
  RTU-ähnliche Eligibility-Traces (embedtrace/decaytrace),
  Latent-MSE + CE + Variance-Hinge + Stop-MSE, Test-Time-Training,
  entropie-adaptives Sampling.
- Neu: Feedforward pro Layer als Mixture-of-Experts mit Top-k Routing,
  getrennte Zählung totaler vs. aktiver Parameter.

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

        # Buffer (kein Gradient, werden manuell fortgeschrieben)
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
        """Gibt (y, probs, top_idx, weights) zurück."""
        if self.num_experts == 1:
            return self.silu(self.experts[0](h_norm)), None, None, None
        logits = self.router(h_norm)  # (E,)
        probs = torch.softmax(logits, dim=-1)
        top_vals, top_idx = torch.topk(logits, self.top_k)
        w = probs[top_idx]
        w = w / (w.sum() + 1e-9)
        y = torch.zeros_like(h_norm)
        for j, e_idx in enumerate(top_idx.tolist()):
            y = y + w[j] * self.silu(self.experts[e_idx](h_norm))
        return y, probs, top_idx, w, logits

    def forward(self, enc: torch.Tensor, x: torch.Tensor, dummy: torch.Tensor):
        decay = torch.sigmoid(self.decay)  # (dim,)
        prev = self.states.detach()  # kein BPTT durch die Zeit (wie MLX stop_gradient)
        state = decay * prev + enc + dummy
        h_norm = self.norm(state)
        if self.num_experts == 1:
            y = self.silu(self.experts[0](h_norm))
            aux = torch.zeros((), device=x.device)
            zloss = torch.zeros((), device=x.device)
            info = None
        else:
            y, probs, top_idx, w, logits = self.forward_moe(h_norm)
            # Single-Token Uniformitäts-Aux: E*sum(p^2)-1 in [0, E-1], 0=uniform.
            aux = self.num_experts * (probs * probs).sum() - 1.0
            zloss = torch.logsumexp(logits, dim=-1).pow(2)
            info = (probs.detach(), top_idx.detach())
        out = x + y
        return out, state, decay, aux, zloss, info


class Model(nn.Module):
    def __init__(self, dim: int, layers: int, temp: float = 0.75,
                 num_experts: int = 1, top_k: int = 1,
                 aux_coef: float = 0.01, zloss_coef: float = 0.001):
        super().__init__()
        self.dim = dim
        self.layercount = layers
        self.temp = temp
        self.num_experts = num_experts
        self.top_k = top_k if num_experts > 1 else 1
        self.aux_coef = aux_coef
        self.zloss_coef = zloss_coef

        self.encoder = Encoder(dim)
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

    # -- Forward eines Steps --
    def step(self, c: torch.Tensor, dummies=None, frozen: bool = False):
        dev = c.device if isinstance(c, torch.Tensor) else self.device()
        if dummies is None:
            dummies = [torch.zeros(self.dim, device=dev) for _ in range(self.layercount)]
        enc = self.encoder(c)
        x = enc
        states, decays, auxs, zlosses, infos = [], [], [], [], []
        for i, layer in enumerate(self.layers):
            x, state, decay, aux, zloss, info = layer(enc, x, dummies[i])
            if frozen:
                with torch.no_grad():
                    layer.states.copy_(state.detach())
                    if info is not None:
                        for e in info[1].tolist():
                            layer.usage[e] += 1
            states.append(state)
            decays.append(decay)
            auxs.append(aux)
            zlosses.append(zloss)
            infos.append(info)
        output, stop = self.decoder(x)
        return (x, states, decays, auxs, zlosses, infos), (output, stop)

    def loss_terms(self, x, output, stop, nextb, end, auxs, zlosses):
        loss = torch.clamp(1.0 - torch.sqrt(x.var() + 1e-4), min=0.0)
        if nextb is not None:
            n = torch.tensor(nextb, device=x.device, dtype=torch.long)
            with torch.no_grad():
                tgt = self.encoder(n)
            loss = loss + torch.mean((x - tgt) ** 2)
            loss = loss - output[n] + torch.logsumexp(output, dim=-1)
            target = torch.tensor([1.0 if end else 0.0], device=x.device)
            loss = loss + torch.mean((stop.view(-1) - target) ** 2)
            if self.num_experts > 1 and len(auxs) > 0:
                loss = loss + self.aux_coef * torch.stack(auxs).mean()
                loss = loss + self.zloss_coef * torch.stack(zlosses).mean()
        return loss

    @torch.no_grad()
    def frozen_call(self, currb: int):
        dev = self.device()
        c = torch.tensor(currb, device=dev, dtype=torch.long)
        (_, states, decays, auxs, zlosses, infos), (output, stop) = self.step(c, frozen=True)
        return self.sample(output), float(stop.view(-1)[0])

    def train_step(self, currb: int, nextb, end: bool, optimizer) -> tuple:
        """Ein Online-Step mit RTRL-ähnlicher Trace-Korrektur (MLX-Original treu)."""
        dev = self.device()
        c = torch.tensor(currb, device=dev, dtype=torch.long)

        # alte Buffer-Snapshots (prev) für Trace-Formeln
        prev_states = [l.states.clone() for l in self.layers]
        prev_embed = [l.embedtrace.clone() for l in self.layers]
        prev_decaytr = [l.decaytrace.clone() for l in self.layers]

        dummies = [torch.zeros(self.dim, device=dev, requires_grad=True)
                   for _ in range(self.layercount)]

        (x, states, decays, auxs, zlosses, infos), (output, stop) = self.step(
            c, dummies, frozen=False
        )
        loss = self.loss_terms(x, output, stop, nextb, end, auxs, zlosses)

        params = list(self.parameters())
        grads = torch.autograd.grad(loss, params + dummies, allow_unused=True)
        g_params, g_dummies = grads[:len(params)], grads[len(params):]

        grad_map = {id(p): (p, g) for p, g in zip(params, g_params)}
        enc_w = self.encoder.embed.weight
        _, enc_g = grad_map[id(enc_w)]
        if enc_g is None:
            enc_g = torch.zeros_like(enc_w)
        else:
            enc_g = enc_g.clone()

        # Trace-Korrektur pro Layer (faithful zum MLX-Original)
        oh = torch.zeros(256, self.dim, device=dev)
        oh[currb, :] = 1.0
        for i, layer in enumerate(self.layers):
            dlds = g_dummies[i]
            if dlds is None:
                dlds = torch.zeros(self.dim, device=dev)
            dec = decays[i].detach()  # (dim,)
            # Encoder: akkumuliere über Layer (Original macht += über 16 Layer)
            enc_g = enc_g + dlds.unsqueeze(0) * (prev_embed[i] * dec.unsqueeze(0))
            # Decay: überschreibe mit Trace-Version
            new_decaytr = dec * prev_decaytr[i] + dec * (1.0 - dec) * prev_states[i]
            p_obj, _ = grad_map[id(layer.decay)]
            grad_map[id(layer.decay)] = (p_obj, (dlds * new_decaytr).detach())
            # Buffer-Updates
            with torch.no_grad():
                layer.states.copy_(states[i].detach())
                layer.decaytrace.copy_(new_decaytr.detach())
                layer.embedtrace.copy_(
                    (prev_embed[i] * dec.unsqueeze(0) + oh).detach()
                )
                if infos[i] is not None:
                    for e in infos[i][1].tolist():
                        layer.usage[e] += 1

        grad_map[id(enc_w)] = (enc_w, enc_g.detach())

        optimizer.zero_grad(set_to_none=True)
        for _, (p, g) in grad_map.items():
            if g is not None:
                p.grad = g
        optimizer.step()

        with torch.no_grad():
            b = self.sample(output.detach())
            s = float(stop.detach().view(-1)[0])
        return b, s

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

    # -- Save/Load (state_dict inkl. Buffer) --
    def save(self, path: str):
        tmp = os.path.join(os.path.dirname(path) or ".",
                           "temporary-" + os.path.basename(path))
        torch.save(
            {"model": self.state_dict(),
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
                 aux_coef: float = 0.01, zloss_coef: float = 0.001):
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(dev)
        self.model = Model(dim, layers, temp, num_experts, top_k, aux_coef, zloss_coef)
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
        while True:
            text = input(f"\n[{self.now()}]\nUser >> ")
            data = (text + "\n").encode("utf-8")
            for i, (c, n) in enumerate(itertools.pairwise(data)):
                self.call(c, n, i == len(data) - 2, save, frozen)
            print(f"\n[{self.now()}]\nModel >> ", end="", flush=True)
            b = data[-1]
            while True:
                b, stop = self.call(b, None, False, save, frozen)
                self.write(b)
                if stop > self.threshold:
                    print()
                    break

    def train(self, save: bool, frozen: bool, dataset: str):
        files = glob.glob(dataset, recursive=True)
        if not files:
            raise FileNotFoundError(f"Glob {dataset!r} fand nichts.")
        random.shuffle(files)
        while True:
            for file in files:
                with open(file, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        data = line.encode("utf-8")
                        if len(data) < 2:
                            continue
                        for i, (c, n) in enumerate(itertools.pairwise(data)):
                            b, _ = self.call(c, n, i == len(data) - 2, save, frozen)
                            self.write(b)

    def now(self):
        return datetime.now().strftime("%d/%m/%Y, %H:%M:%S")

    def __call__(self, mode: str, dataset: str, save: bool, frozen: bool):
        # Optimizer-State laden falls vorhanden (Datei speichert nur Modell;
        # Optimizer startet frisch — bewusst, sonst TTT-Drift über Runs)
        self.model.load(self.path)
        print(f"device: {self.device}, total: {self.model.count_total():,}, "
              f"aktiv: {self.model.count_active():,} "
              f"(E={self.model.num_experts}, k={self.model.top_k})")
        try:
            if mode == "train":
                self.train(save, frozen, dataset)
            elif mode == "chat":
                self.chat(save, frozen)
        finally:
            if save:
                # kompletter Checkpoint inkl. Optimizer
                tmp = os.path.join(os.path.dirname(self.path) or ".",
                                   "temporary-" + os.path.basename(self.path))
                torch.save({"model": self.model.state_dict(),
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
    args = parser.parse_args()

    rt = Runtime(path=args.path, threshold=args.threshold, dim=args.dim,
                 layers=args.layers, temp=args.temp, lr=args.lr,
                 num_experts=args.experts, top_k=args.topk)
    rt(args.mode, args.dataset, args.save, args.frozen)
