"""Tests for the parameter-efficient fine-tuning helpers in ssl_extractor.

``_apply_adapters``/``_apply_ssf`` traverse a real HuggingFace ``ViTModel``'s
module tree by attribute name (``.layers``, ``.attention``, ``.mlp.fc1/fc2``,
...). That tree isn't part of any public API contract and had zero test
coverage before this file -- an attribute-path mistake here fails silently
or crashes deep inside a checkpoint load, not at the call site. These tests
exist specifically to catch that class of regression: load a real model,
apply each transform, and confirm the output shape is unchanged and (since
both are zero/identity-initialized) numerically matches the untouched
baseline exactly.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

import torch
from transformers import ViTModel

from ic_core.ssl_extractor import _apply_adapters, _apply_ssf

MODEL_NAME = "WinKawaks/vit-tiny-patch16-224"


@pytest.fixture(scope="module")
def dummy_input():
    return torch.zeros(1, 3, 224, 224)


@pytest.fixture(scope="module")
def baseline_output(dummy_input):
    model = ViTModel.from_pretrained(MODEL_NAME, ignore_mismatched_sizes=True).eval()
    with torch.no_grad():
        return model(pixel_values=dummy_input).last_hidden_state


def test_apply_adapters_preserves_shape_and_matches_baseline_at_zero_init(
    dummy_input, baseline_output
):
    model = ViTModel.from_pretrained(MODEL_NAME, ignore_mismatched_sizes=True)
    model = _apply_adapters(model, bottleneck=16).eval()

    with torch.no_grad():
        out = model(pixel_values=dummy_input).last_hidden_state

    assert out.shape == baseline_output.shape
    # Adapter up-projections are zero-initialized, so the adapter's residual
    # contribution is exactly zero right after construction.
    assert torch.allclose(out, baseline_output)
    assert len(model.adapters) == 2 * len(model.layers)


def test_apply_ssf_preserves_shape_and_matches_baseline_at_identity_init(
    dummy_input, baseline_output
):
    model = ViTModel.from_pretrained(MODEL_NAME, ignore_mismatched_sizes=True)
    model = _apply_ssf(model).eval()

    with torch.no_grad():
        out = model(pixel_values=dummy_input).last_hidden_state

    assert out.shape == baseline_output.shape
    # SSF scale=1/shift=0 at init is a no-op, so wrapping shouldn't change
    # a single output value yet.
    assert torch.allclose(out, baseline_output)
    for layer in model.layers:
        assert hasattr(layer.attention.q_proj, "scale")
        assert hasattr(layer.mlp.fc1, "scale")
