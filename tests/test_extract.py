import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from nous_anomaly import extract

TEXTS = ["你好，我的包裹丢了。", "非常抱歉，我马上帮您查。", "算了，问了也是等通知。", "滚开，别烦我。", "你是客服助手。"]
TEMPLATE = ("{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}\n{% endfor %}"
            "{% if add_generation_prompt %}<assistant>{% endif %}")


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    """A randomly initialised 8-layer Llama and a character tokenizer, saved to disk; nothing is downloaded."""
    from tokenizers import Regex, Tokenizer, models, pre_tokenizers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    chars = sorted(set("".join(TEXTS) + "<>systemuserassistant\n"))
    core = Tokenizer(models.WordLevel({c: i for i, c in enumerate(["[UNK]"] + chars)}, unk_token="[UNK]"))
    core.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), "isolated")
    tok = PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]")
    tok.chat_template = TEMPLATE
    torch.manual_seed(0)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=len(chars) + 1, hidden_size=32, intermediate_size=64,
                                         num_hidden_layers=8, num_attention_heads=4, num_key_value_heads=4)).eval()
    path = tmp_path_factory.mktemp("tiny")
    model.save_pretrained(path); tok.save_pretrained(path)
    return str(path)


def _reply(i=0, target=TEXTS[1]):
    return dict(id=f"r{i}", site="reply", system=TEXTS[4], messages=[{"role": "user", "content": TEXTS[0]}], target=target)


def _user(i=0):
    return dict(id=f"u{i}", site="user", system=TEXTS[4], target=TEXTS[2],
                messages=[{"role": "user", "content": TEXTS[0]}, {"role": "assistant", "content": TEXTS[1]},
                          {"role": "user", "content": TEXTS[2]}])


def test_layers_follow_relative_depth():
    assert extract.layers_for(36) == [9, 14, 18, 22, 27]
    assert extract.layers_for(8) == [2, 3, 4, 5, 6]
    assert extract.layers_for(2) == [1, 2]                      # never the embedding layer, never past the last block


def test_positions_point_at_the_prompt_end_and_the_target(tiny):
    tok, model, device = extract.load_model(tiny, device="cpu")
    ids, last, (a, b) = extract.positions(tok, _reply())
    assert tok.decode(ids[:last + 1]).replace(" ", "").endswith("<assistant>") and a == last + 1
    assert tok.decode(ids[a:b]).replace(" ", "") == TEXTS[1]
    assert TEXTS[4] in tok.decode(ids).replace(" ", "")          # the reply site keeps the system prompt
    ids, last, (a, b) = extract.positions(tok, _user())
    assert last is None and tok.decode(ids[a:b]).replace(" ", "") == TEXTS[2]
    assert TEXTS[4] not in tok.decode(ids).replace(" ", "")      # the user site drops it


def test_features_match_a_manual_forward_pass(tiny):
    tok, model, device = extract.load_model(tiny, device="cpu")
    layers = extract.layers_for(model.config.num_hidden_layers)
    f = extract.features(model, tok, _reply(), layers, device)
    ids, last, (a, b) = extract.positions(tok, _reply())
    with torch.no_grad():
        hs = model(torch.tensor([ids]), output_hidden_states=True).hidden_states
    assert len(hs) == 9 and f.shape == (3, 5, 32)
    for j, L in enumerate(layers):
        assert np.allclose(f[0, j], hs[L][0, last].numpy(), atol=1e-5)
        assert np.allclose(f[1, j], hs[L][0, a:b].mean(0).numpy(), atol=1e-5)
        assert np.allclose(f[2, j], hs[L][0, a:a + extract.EARLY].mean(0).numpy(), atol=1e-5)
    # Causal model: the pre-reply state cannot depend on the reply that follows it.
    other = extract.features(model, tok, _reply(target=TEXTS[3]), layers, device)
    assert np.allclose(f[0], other[0], atol=1e-5) and not np.allclose(f[1], other[1], atol=1e-3)
    u = extract.features(model, tok, _user(), layers, device)
    assert np.isnan(u[0]).all() and np.isnan(u[2]).all() and np.isfinite(u[1]).all()


def test_run_shards_resume_and_read_back(tiny, tmp_path):
    tok, model, device = extract.load_model(tiny, device="cpu")
    records = [_reply(i, TEXTS[1 + i % 3]) for i in range(7)] + [_user(i) for i in range(3)]
    records.append(dict(_reply(99), target=""))                  # unusable record: reported, not fatal
    out = str(tmp_path / "feats")
    r0 = extract.run(model, tok, device, records, out, shard=(0, 2), chunk=2, log=lambda *_: None)
    r1 = extract.run(model, tok, device, records, out, shard=(1, 2), chunk=2, log=lambda *_: None)
    assert r0["failed"] + r1["failed"] == ["r99"]
    feats = extract.Features(out)
    assert set(feats.index) == {r["id"] for r in records} - {"r99"} and feats.layers == [2, 3, 4, 5, 6]
    direct = extract.features(model, tok, records[3], feats.layers, device)
    assert np.allclose(feats.get(["r3"], "mean", 4)[0], direct[1, 2], atol=2e-2, rtol=2e-2)   # stored as float16
    assert np.isnan(feats.get(["u0"], "pre", 4)).all()
    # A second run finds every chunk on disk and does no model work.
    class Boom:
        config = model.config
        def __call__(self, *a, **k): raise AssertionError("recomputed a finished chunk")
    extract.run(Boom(), tok, device, records[:10], out, shard=(0, 2), chunk=2, log=lambda *_: None)


def test_tokens_are_saved_in_the_basis_and_average_to_the_mean_pool(tiny, tmp_path):
    tok, model, device = extract.load_model(tiny, device="cpu")
    rng = np.random.default_rng(0)
    np.savez(tmp_path / "basis.npz", layer=4, mean=rng.normal(size=32), scale=rng.uniform(0.5, 2, size=32),
             components=np.linalg.qr(rng.normal(size=(32, 32)))[0][:6])
    basis = extract.load_basis(tmp_path / "basis.npz", device)
    records = [_reply(0), _reply(1, TEXTS[3]), _user(0)]
    out = str(tmp_path / "feats")
    extract.run(model, tok, device, records, out, chunk=2, log=lambda *_: None, basis=basis)
    feats = extract.Features(out)
    assert feats.meta["token_layer"] == 4
    z = np.load(tmp_path / "basis.npz")
    for r, t in zip(records, feats.tokens([r["id"] for r in records])):
        a, b = extract.positions(tok, r)[2]
        assert t.shape == (b - a, 6)
        mean_pool = feats.get([r["id"]], "mean", 4)[0]
        assert np.allclose(t.mean(0), ((mean_pool - z["mean"]) / z["scale"]) @ z["components"].T, atol=5e-2)
