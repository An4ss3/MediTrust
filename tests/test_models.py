"""Tests for src/models.py (CPU, random weights: no download needed)."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("timm")

from src.models import (BACKBONES, build_model, count_parameters, export_onnx,
                        attention_modules, manual_attention, set_fused_attention)

torch.manual_seed(0)


@pytest.fixture(scope="module", params=list(BACKBONES))
def model(request):
    return build_model(request.param, pretrained=False).eval()


def test_forward_shape(model):
    with torch.no_grad():
        out = model(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 9) and torch.isfinite(out).all()


def test_gradients_reach_input_and_head(model):
    model.zero_grad()
    x = torch.rand(2, 3, 64, 64, requires_grad=True)
    model(x).logsumexp(1).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    head = model.backbone.get_classifier()
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_preprocess_resize_and_normalise(model):
    x = torch.full((1, 3, 64, 64), 0.5)
    y = model.preprocess(x)
    assert y.shape == (1, 3, 224, 224)
    expected = (0.5 - model.mean) / model.std
    assert torch.allclose(y, expected.expand_as(y), atol=1e-6)


def test_parameter_counts():
    r18 = count_parameters(build_model("R18", pretrained=False))
    deit = count_parameters(build_model("DEIT", pretrained=False))
    assert 11.0e6 < r18 < 11.5e6   # 11.7 M with the 1000-class head, ~11.2 M with 9 classes
    assert 5.4e6 < deit < 5.6e6    # 5.7 M with the 1000-class head


@pytest.mark.parametrize("fused", [True, False])
def test_manual_attention_matches_timm(fused):
    m = build_model("DEIT", pretrained=False).eval()
    set_fused_attention(m, fused)
    blocks = attention_modules(m)
    assert len(blocks) == 12 and blocks[0].num_heads == 3
    tokens = torch.randn(2, 197, 192)
    for attn in (blocks[0], blocks[-1]):
        with torch.no_grad():
            ref = attn(tokens)
            out, w = manual_attention(attn, tokens)
        assert w.shape == (2, 3, 197, 197)
        assert torch.allclose(w.sum(-1), torch.ones(2, 3, 197), atol=1e-5)
        assert (out - ref).abs().max() < 1e-5


@pytest.mark.parametrize("name", list(BACKBONES))
def test_onnx_export_parity(name, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnx")
    m = build_model(name, pretrained=False).eval()
    path = str(tmp_path / f"{name}.onnx")
    export_onnx(m, path)
    x = torch.rand(4, 3, 64, 64)  # batch 4 checks the dynamic batch axis
    with torch.no_grad():
        ref = m(x).numpy()
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"image": x.numpy()})[0]
    assert out.shape == (4, 9)
    assert abs(out - ref).max() < 1e-3
    assert (out.argmax(1) == ref.argmax(1)).all()
