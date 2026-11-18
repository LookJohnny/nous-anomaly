"""Read activations out of any HuggingFace causal LM. Needs torch and transformers.

For each record the chat prompt is built with the model's own template and the target text is teacher-forced.
Layers are chosen by relative depth, so the same settings work on any model size. Saved per record and layer:
  pre    state at the last prompt position (before the reply starts)         site "reply" only
  mean   mean state over the target tokens (the reply, or the last user message for site "user")
  early  mean over the first 16 target tokens                                site "reply" only

  python -m nous_anomaly.extract --model Qwen/Qwen3-4B-Instruct-2507 --records data/records.jsonl --out feats/qwen3-4b
  ... --shard 0/4   run on 4 GPUs in parallel; finished chunks are skipped on restart.
"""
import argparse, glob, json, os, time
import numpy as np

DEPTHS = (0.25, 0.4, 0.5, 0.6, 0.75)
POOLS = ("pre", "mean", "early")
EARLY = 16
MAX_TARGET_TOKENS = 256


def layers_for(n_layers, depths=DEPTHS):
    """Relative depth -> index into hidden_states (0 is the embedding output, n_layers the last block)."""
    return sorted({int(min(n_layers, max(1, round(d * n_layers)))) for d in depths})


def chat_text(tok, messages, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return "".join(f"<{m['role']}>\n{m['content']}\n" for m in msgs) + "<assistant>\n"


def positions(tok, record):
    """-> (input ids, index of the last prompt token or None, [start, end) of the target tokens)."""
    if record["site"] == "reply":
        prompt = tok(chat_text(tok, record["messages"], record.get("system")), add_special_tokens=False)["input_ids"]
        target = tok(record["target"], add_special_tokens=False)["input_ids"][:MAX_TARGET_TOKENS]
        if not prompt or not target:
            raise ValueError("empty prompt or target")
        return prompt + target, len(prompt) - 1, (len(prompt), len(prompt) + len(target))
    # Site "user": no system prompt, so a persona cannot leak into a reading of the user's message.
    text = chat_text(tok, record["messages"])
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    c0 = text.rindex(record["target"]); c1 = c0 + len(record["target"])
    idx = [i for i, (s, e) in enumerate(enc["offset_mapping"]) if s >= c0 and e <= c1 and e > s]
    if not idx:
        raise ValueError("target span not found")
    return enc["input_ids"], None, (idx[0], idx[-1] + 1)


def features(model, tok, record, layers, device):
    import torch
    ids, last_prompt, (a, b) = positions(tok, record)
    with torch.no_grad():
        hs = model(torch.tensor([ids], device=device), output_hidden_states=True, use_cache=False).hidden_states
    out = np.full((len(POOLS), len(layers), hs[0].shape[-1]), np.nan, dtype=np.float32)
    for j, L in enumerate(layers):
        h = hs[L][0].float()
        if last_prompt is not None:
            out[0, j] = h[last_prompt].cpu().numpy()
            out[2, j] = h[a:min(b, a + EARLY)].mean(0).cpu().numpy()
        out[1, j] = h[a:b].mean(0).cpu().numpy()
    return out


def load_model(name, device=None, dtype=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    dtype = dtype or (torch.bfloat16 if device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:                                    # transformers before 4.56 names the argument torch_dtype
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    model = model.to(device).eval()
    return tok, model, device


def run(model, tok, device, records, out_dir, *, depths=DEPTHS, shard=(0, 1), chunk=256, log=print):
    os.makedirs(out_dir, exist_ok=True)
    layers = layers_for(model.config.num_hidden_layers, depths)
    meta = dict(model=getattr(model.config, "_name_or_path", ""), n_layers=model.config.num_hidden_layers,
                layers=layers, depths=list(depths), pools=list(POOLS), early=EARLY)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    mine = [r for i, r in enumerate(records) if i % shard[1] == shard[0]]
    t0 = time.time(); failed = []
    for c in range(0, len(mine), chunk):
        path = os.path.join(out_dir, f"feats-{shard[0]}of{shard[1]}-{c // chunk:05d}.npz")
        if os.path.exists(path):
            continue
        ids, arrays = [], []
        for r in mine[c:c + chunk]:
            try:
                arrays.append(features(model, tok, r, layers, device)); ids.append(r["id"])
            except ValueError:
                failed.append(r["id"])
        if ids:
            tmp = path + ".tmp.npz"
            np.savez(tmp, ids=np.array(ids), x=np.stack(arrays).astype(np.float16))
            os.replace(tmp, path)
        log(f"{min(c + chunk, len(mine))}/{len(mine)} records, {time.time() - t0:.0f}s, failed {len(failed)}")
    return dict(meta, records=len(mine), failed=failed)


class Features:
    """All chunks of one model's features, addressed by record id."""

    def __init__(self, out_dir):
        with open(os.path.join(out_dir, "meta.json")) as f:
            self.meta = json.load(f)
        ids, xs = [], []
        for path in sorted(glob.glob(os.path.join(out_dir, "feats-*.npz"))):
            if path.endswith(".tmp.npz"):
                continue
            z = np.load(path)
            ids.extend(z["ids"].tolist()); xs.append(z["x"])
        self.x = np.concatenate(xs) if xs else np.zeros((0, len(POOLS), len(self.meta["layers"]), 0), np.float16)
        self.index = {i: n for n, i in enumerate(ids)}
        self.layers = self.meta["layers"]

    def __contains__(self, record_id):
        return record_id in self.index

    def get(self, ids, pool, layer):
        """-> float64 [len(ids), d] for one pooling and one layer (a hidden_states index from meta['layers'])."""
        rows = [self.index[i] for i in ids]
        return self.x[rows, POOLS.index(pool), self.layers.index(layer)].astype(np.float64)


def main(argv=None):
    from .data import load
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1", help="i/n: process every n-th record starting at i")
    ap.add_argument("--depths", type=float, nargs="*", default=list(DEPTHS))
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    i, n = map(int, args.shard.split("/"))
    tok, model, device = load_model(args.model, args.device)
    result = run(model, tok, device, load(args.records), args.out, depths=tuple(args.depths), shard=(i, n))
    print(json.dumps({k: v for k, v in result.items() if k != "failed"} | {"failed": len(result["failed"])}))


if __name__ == "__main__":
    main()
