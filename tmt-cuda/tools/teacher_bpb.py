"""Bits per byte of a candidate teacher on the same held-out set the model
is measured on, so the two numbers can be compared directly.

Token-level cross entropy is summed in bits and divided by the number of
*bytes* those tokens cover, which is what makes it a BPB rather than a
perplexity."""
import sys, math, time, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"
path = sys.argv[2] if len(sys.argv) > 2 else "/home/simon/tmt-data/enwik8/held.bin"
nbytes = int(sys.argv[3]) if len(sys.argv) > 3 else 2_000_000
win = 1024

raw = open(path, "rb").read(nbytes)
text = raw.decode("utf-8", errors="replace")
print(f"{name}: {len(raw)} bytes, {len(text)} chars", flush=True)

tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to("cuda").eval()

ids = tok(text, return_tensors=None, add_special_tokens=False)["input_ids"]
print(f"{len(ids)} tokens, {len(raw)/len(ids):.2f} bytes per token", flush=True)

nll_bits, counted_tokens = 0.0, 0
t0 = time.time()
with torch.no_grad():
    for s in range(0, len(ids) - 1, win):
        chunk = ids[s:s + win + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk], device="cuda")
        logits = model(x).logits[0, :-1]
        tgt = x[0, 1:]
        # The vocabulary is large enough that one float copy of the whole
        # logit block does not fit, so the log-softmax runs in row slices.
        total = 0.0
        for a in range(0, logits.shape[0], 128):
            part = logits[a:a + 128].float()
            lp = torch.log_softmax(part, -1).gather(1, tgt[a:a + 128, None]).squeeze(1)
            total += float(-lp.sum())
            del part, lp
        nll_bits += total / math.log(2)
        del logits
        counted_tokens += tgt.numel()
        if s % (win * 20) == 0:
            print(f"  {s}/{len(ids)}  running BPB {nll_bits / (counted_tokens * len(raw) / len(ids)):.4f}", flush=True)

bytes_counted = counted_tokens * len(raw) / len(ids)
print(f"\ntokens scored: {counted_tokens}")
print(f"BPB: {nll_bits / bytes_counted:.4f}")
print(f"(bits per token: {nll_bits / counted_tokens:.3f})")
dt = time.time() - t0
rate = counted_tokens / dt
print(f"scoring: {dt:.1f} s, {rate:.0f} tokens/s, {rate * len(raw) / len(ids) / 1e6:.2f} MB/s")
for label, mb in (("enwik8 100 MB", 100), ("enwik9 1 GB", 1000)):
    print(f"  {label}: {mb * 1e6 / (rate * len(raw) / len(ids)) / 3600:.1f} h")
