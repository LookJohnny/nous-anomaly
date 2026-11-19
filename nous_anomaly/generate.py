"""Let the model under study write data, so a new concept or a larger normal corpus needs no hand-collected examples.

  normal    benign user messages over many topics and intents, each answered by the model with no system prompt.
            These widen the normal corpus (source "gen").
  contrast  for one dialogue, two replies: the model's own, and one written under an instruction to show a concept
            (one sentence per concept in DEFINITIONS). The instruction is NOT stored: activations are later read
            with the original prompt, so a direction cannot pick up the instruction itself. Source "syn", split "syn".

  python -m nous_anomaly.generate normal   --model Qwen/Qwen3-4B-Instruct-2507 --out data/gen.jsonl
  python -m nous_anomaly.generate contrast --model Qwen/Qwen3-4B-Instruct-2507 --records data/records.jsonl --out data/syn.jsonl
  python -m nous_anomaly.data --repo ... --extra data/gen.jsonl data/syn.jsonl --out data/records.jsonl
"""
import argparse, hashlib, random, re

from .data import EMOTIONS, load, save

DEFINITIONS = {
    "hostile": "对用户不耐烦、轻蔑、带有攻击性，或者贬低用户",
    "syco": "不管用户说得对不对都顺着用户、表示赞同，即使用户的说法有错也不纠正",
    "anger": "语气愤怒",
    "joy": "语气喜悦、兴奋",
    "sadness": "语气悲伤、低落",
}
INSTRUCTION = ("【数据构造任务】下面这条回复用于训练安全检测器。请在回复中刻意表现出这个特点：{definition}。"
               "直接写回复本身，不要解释，不要加引号。")

TOPICS = {
    "zh": "做饭 旅行 编程 数学 物理 历史 健身 理财 职场 育儿 宠物 电影 音乐 文学 医疗常识 法律常识 汽车 手机数码 园艺 "
          "学外语 求职 租房 网购售后 心理调节 天气与出行".split(),
    "en": ["cooking", "travel", "programming", "math", "physics", "history", "fitness", "personal finance", "office work",
           "parenting", "pets", "movies", "music", "literature", "everyday health", "everyday law", "cars",
           "phones and gadgets", "gardening", "language learning", "job hunting", "renting a home",
           "online shopping and returns", "stress management", "weather and commuting"],
}
INTENTS = {
    "zh": "提一个事实性问题 请求一步步的操作指导 请对方帮忙写一段文字 请求推荐 请对方解释一个概念 请对方比较两个选项 "
          "描述一个遇到的问题并求助 请对方总结或改写一段话 闲聊分享近况 提一个需要计算或推理的问题".split(),
    "en": ["ask a factual question", "ask for step-by-step instructions", "ask for a short piece of writing",
           "ask for a recommendation", "ask for a concept to be explained", "ask for a comparison of two options",
           "describe a problem and ask for help", "ask for a summary or a rewrite of a short passage",
           "make small talk about their day", "ask something that needs calculation or reasoning"],
}
ASK = {"zh": "请写 {n} 条用户可能发给 AI 助手的消息。主题：{topic}。类型：{intent}。每条单独一行，内容各不相同，"
             "不要编号，不要解释。",
       "en": "Write {n} different messages a user might send to an AI assistant. Topic: {topic}. Type: {intent}. "
             "One message per line, no numbering, no commentary."}


def make_generator(model, tok, device, *, max_new_tokens=256, batch=32, temperature=0.8, log=print):
    """-> gen(prompts): prompts are (system or None, messages); returns one sampled reply per prompt."""
    import torch
    from .extract import chat_text
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def gen(prompts, max_new_tokens=max_new_tokens):
        out = []
        for i in range(0, len(prompts), batch):
            texts = [chat_text(tok, messages, system) for system, messages in prompts[i:i + batch]]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            with torch.no_grad():
                ids = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                                     top_p=0.9, pad_token_id=tok.pad_token_id)
            for row in ids[:, enc["input_ids"].shape[1]:]:
                text = tok.decode(row, skip_special_tokens=True)
                out.append(re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip())
            log(f"generated {len(out)}/{len(prompts)}")
        return out
    return gen


def split_lines(text):
    """A generated list -> clean, distinct messages (numbering, bullets and quotes removed)."""
    seen, out = set(), []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+\s*[.、)）:：])\s*", "", line).strip().strip("\"“”'「」")
        if 4 <= len(line) <= 300 and line not in seen:
            seen.add(line); out.append(line)
    return out


def _hash_split(key, val_fraction=0.2):
    return "val" if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 1000 < val_fraction * 1000 else "dev"


def normal_records(gen, *, per_cell=10, topics=TOPICS, intents=INTENTS):
    """Benign conversations. A group is one (language, topic, intent) cell; a fifth of the groups goes to val."""
    cells = [(lang, t, i) for lang in topics for t in range(len(topics[lang])) for i in range(len(intents[lang]))]
    asks = [(None, [{"role": "user", "content": ASK[lang].format(n=per_cell, topic=topics[lang][t], intent=intents[lang][i])}])
            for lang, t, i in cells]
    users = []
    for (lang, t, i), text in zip(cells, gen(asks, max_new_tokens=60 * per_cell)):
        users += [(f"{lang}-{t}-{i}", topics[lang][t], n, m) for n, m in enumerate(split_lines(text)[:per_cell])]
    replies = gen([(None, [{"role": "user", "content": m}]) for _, _, _, m in users])
    return [dict(id=f"gen:{group}-{n}", source="gen", site="reply", system=None, messages=[{"role": "user", "content": m}],
                 target=reply, group=group, topic=topic, split=_hash_split(group), labels={})
            for (group, topic, n, m), reply in zip(users, replies) if reply]


def contrast_records(gen, records, *, concepts=tuple(DEFINITIONS), per_concept=300, seed=0):
    """Pairs of replies to dev dialogues, drawn evenly from every source that has reply-site dev records."""
    by_source = {}
    for r in records:
        if r["site"] == "reply" and r["split"] == "dev":
            key = (r.get("system"), str(r["messages"]))
            by_source.setdefault(r["source"], {}).setdefault(key, r)
    rng, out = random.Random(seed), []
    for concept in concepts:
        pools = [rng.sample(list(v.values()), len(v)) for _, v in sorted(by_source.items())]
        contexts = [p[i] for i in range(max(map(len, pools))) for p in pools if i < len(p)][:per_concept]
        instruction = INSTRUCTION.format(definition=DEFINITIONS[concept])
        plain = gen([(r.get("system"), r["messages"]) for r in contexts])
        shown = gen([(((r.get("system") or "") + "\n\n" + instruction).strip(), r["messages"]) for r in contexts])
        positive = {"emotion": concept} if concept in EMOTIONS else {concept: 1}
        for n, (r, a, b) in enumerate(zip(contexts, plain, shown)):
            for label, text in ((0, a), (1, b)):
                if text:
                    out.append(dict(id=f"syn:{concept}:{n}:{label}", source="syn", site="reply", system=r.get("system"),
                                    messages=r["messages"], target=text, group=f"{concept}:{r['source']}:{r['group']}",
                                    topic=r.get("topic"), split="syn", labels=positive if label else {},
                                    meta={"concept": concept, "context": r["id"]}))
    return out


def main(argv=None):
    from .extract import load_model
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("normal", "contrast"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--records", help="contrast: where the dialogues come from")
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--per-concept", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args(argv)
    tok, model, device = load_model(args.model)
    gen = make_generator(model, tok, device, batch=args.batch)
    if args.command == "normal":
        records = normal_records(gen, per_cell=args.per_cell)
    else:
        records = contrast_records(gen, load(args.records), per_concept=args.per_concept)
    save(records, args.out)
    print(len(records), "records ->", args.out)


if __name__ == "__main__":
    main()
