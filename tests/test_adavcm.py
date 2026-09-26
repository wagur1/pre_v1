"""Unit tests for AdaVCM modules."""
import torch
import pytest
import numpy as np

from src.models import (
    SpatioTemporalSalience,
    BoundaryAwareFilter,
    TemporalBackgroundRegularizer,
    AdaptivePolicyNet,
    AdaVCM,
)
from src.metrics import bd_rate


def test_salience_and_motion():
    sal = SpatioTemporalSalience(dilate_radius=3)
    x = torch.rand(2, 3, 4, 32, 32)
    # Give random boxes
    boxes = [[[4, 4, 12, 12]], [[8, 8, 20, 20]]]
    w_map = sal(x, boxes=boxes)
    assert w_map.shape == (2, 1, 4, 32, 32)
    assert (w_map >= 0.0).all() and (w_map <= 1.0).all()
    # Check that object centers have high weight
    assert w_map[0, 0, 0, 8, 8] > 0.8


def test_boundary_aware_filter_preserves_foreground():
    filt = BoundaryAwareFilter(default_sigma=3.0)
    x = torch.rand(1, 3, 2, 32, 32)
    # Mask with foreground in center
    w_map = torch.zeros(1, 1, 2, 32, 32)
    w_map[:, :, :, 10:20, 10:20] = 1.0

    out = filt(x, w_map, sigma=3.5)
    assert out.shape == x.shape
    # Core foreground pixels must be 100% bit-exact identical
    fg_diff = torch.abs(out[:, :, :, 12:18, 12:18] - x[:, :, :, 12:18, 12:18])
    assert fg_diff.max().item() < 1e-6
    # Background must be smoothed (lower variance)
    bg_orig_var = torch.var(x[:, :, :, :8, :8])
    bg_out_var = torch.var(out[:, :, :, :8, :8])
    assert bg_out_var < bg_orig_var


def test_temporal_regularizer():
    tbr = TemporalBackgroundRegularizer(default_alpha=0.8)
    x = torch.rand(1, 3, 5, 32, 32)
    w_map = torch.zeros(1, 1, 5, 32, 32)  # All background
    out = tbr(x, w_map)
    assert out.shape == x.shape
    # Background temporal difference should decrease
    orig_diff = torch.abs(x[:, :, 1:] - x[:, :, :-1]).mean()
    out_diff = torch.abs(out[:, :, 1:] - out[:, :, :-1]).mean()
    assert out_diff < orig_diff


def test_policy_net():
    policy = AdaptivePolicyNet()
    x = torch.rand(2, 3, 4, 32, 32)
    w_map = torch.rand(2, 1, 4, 32, 32)
    sigma, alpha, scale = policy(x, w_map, qp=35.0)
    assert sigma.shape == (2,)
    assert alpha.shape == (2,)
    assert (sigma >= 1.0).all() and (sigma <= 5.0).all()
    assert (alpha >= 0.0).all() and (alpha <= 0.95).all()


def test_end_to_end_adavcm():
    model = AdaVCM(learnable_policy=True)
    x = torch.rand(1, 3, 4, 32, 32)
    res = model(x, qp=32.0)
    assert "preprocessed" in res
    assert "weight_map" in res
    assert res["preprocessed"].shape == (1, 3, 4, 32, 32)


def test_bd_rate_calculation():
    ra = [100.0, 200.0, 400.0, 800.0]
    ma = [0.40, 0.50, 0.60, 0.70]
    # Test method has 20% lower rate at same accuracy
    rt = [80.0, 160.0, 320.0, 640.0]
    mt = [0.40, 0.50, 0.60, 0.70]
    bd = bd_rate(ra, ma, rt, mt)
    assert bd < -15.0 and bd > -25.0


def test_policy_net_gradient_flow():
    """Verify that gradients flow differentiably through AdaVCM into PolicyNet parameters."""
    model = AdaVCM(learnable_policy=True)
    x = torch.rand(2, 3, 3, 32, 32, requires_grad=True)
    res = model(x, qp=32.0)
    loss = res["preprocessed"].sum()
    loss.backward()

    # Verify that PolicyNet parameters received gradients
    has_grad = False
    for name, p in model.policy_net.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "PolicyNet failed to receive gradients from preprocessed output!"
