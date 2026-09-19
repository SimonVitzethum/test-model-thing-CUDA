"""CoLA-Probe für den PyTorch+MoE-Port (Gegenstück zu benchmark.py)."""
import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from main_torch_moe import Model


class Classification(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, 2)

    def forward(self, x: torch.Tensor):
        return self.proj(x)


def cola(filepath: str):
    data = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 4:
                    data.append((parts[3].encode("utf-8"), int(parts[1])))
    except FileNotFoundError:
        pass
    return data


def mcc(tp, tn, fp, fn):
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return ((tp * tn - fp * fn) / den if den != 0 else 0.0) * 100


@torch.no_grad()
def rollout(model: Model, b_s: bytes):
    model.reset()
    dev = next(model.parameters()).device
    for b in b_s:
        c = torch.tensor(b, device=dev, dtype=torch.long)
        model.step(c, frozen=True)
    return model.layers[-1].states.clone()


def benchmark(model, data: list, train: bool, head, optimizer):
    tp = tn = fp = fn = 0
    head.train(train)
    for i, (b_s, label) in enumerate(data):
        if len(b_s) == 0:
            continue
        state = rollout(model, b_s)
        target = torch.tensor([label], device=state.device)
        if train:
            optimizer.zero_grad(set_to_none=True)
            choice = head(state)
            loss = F.cross_entropy(choice[None, :], target)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                choice = head(state)
        predicted = int(torch.argmax(choice).item())
        if predicted == 1 and label == 1:
            tp += 1
        elif predicted == 0 and label == 0:
            tn += 1
        elif predicted == 1 and label == 0:
            fp += 1
        elif predicted == 0 and label == 1:
            fn += 1
        if i > 0 and i % 500 == 0:
            print(f"[{i}/{len(data)-1}] {'train' if train else 'held'}: "
                  f"T+ {tp}, T- {tn}, F+ {fp}, F- {fn} ({mcc(tp,tn,fp,fn):.4f})")
    print(f"[{len(data)-1}/{len(data)-1}] {'train' if train else 'held'}: "
          f"T+ {tp}, T- {tn}, F+ {fp}, F- {fn} ({mcc(tp,tn,fp,fn):.4f})")


def run(path: str, epochs: int, split: float, data: str,
        dim: int, layers: int, experts: int, topk: int):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint fehlt: {path!r}.")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model(dim=dim, layers=layers, num_experts=experts, top_k=topk)
    model.to(dev)
    model.load(path)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    print(f"total: {model.count_total():,}, aktiv: {model.count_active():,}")

    head = Classification(model.dim).to(dev)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3)

    rows = cola(data)
    if len(rows) < 2:
        raise FileNotFoundError("CoLA fehlt/ungültig: https://nyu-mll.github.io/CoLA/.")
    s = int(len(rows) * min(max(split, 0.0), 1.0))
    train, held = rows[:s], rows[s:]
    print("Benchmark startet.")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        benchmark(model, train, True, head, optimizer)
        benchmark(model, held, False, head, optimizer)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("path")
    p.add_argument("epochs", type=int)
    p.add_argument("split", type=float)
    p.add_argument("--data", default="CoLA/original/raw/in_domain_train.tsv")
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--layers", type=int, default=16)
    p.add_argument("--experts", type=int, default=1)
    p.add_argument("--topk", type=int, default=1)
    a = p.parse_args()
    run(a.path, a.epochs, a.split, a.data, a.dim, a.layers, a.experts, a.topk)
