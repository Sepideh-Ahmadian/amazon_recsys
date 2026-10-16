import math

import pytest
import torch

from amazon_recsys.training.losses import bce_loss, bpr_loss, get_loss


def test_bpr_hand_computed():
    # -log sigmoid(2 - 1) = log(1 + e^-1)
    out = bpr_loss(torch.tensor([2.0]), torch.tensor([1.0]))
    assert out.item() == pytest.approx(math.log(1 + math.exp(-1)))


def test_bpr_equal_scores_is_log2():
    assert bpr_loss(torch.zeros(4), torch.zeros(4)).item() == pytest.approx(math.log(2))


def test_bpr_decreases_when_positive_scores_rise():
    neg = torch.tensor([0.5, 0.5])
    low = bpr_loss(torch.tensor([0.0, 0.0]), neg)
    high = bpr_loss(torch.tensor([1.0, 3.0]), neg)
    assert high < low


def test_bpr_accepts_multiple_negatives():
    pos = torch.tensor([1.0, 2.0])
    neg = torch.tensor([[0.0, 0.5], [1.0, 1.5]])
    manual = -torch.nn.functional.logsigmoid(pos.unsqueeze(1) - neg).mean()
    assert bpr_loss(pos, neg).item() == pytest.approx(manual.item())


def test_bce_hand_computed():
    # pos logit 0 -> log 2; neg logit 0 -> log 2; mean of both = log 2
    assert bce_loss(torch.zeros(3), torch.zeros(3)).item() == pytest.approx(math.log(2))
    # pos=2 -> log(1+e^-2); neg=-1 -> log(1+e^-1); averaged
    expected = 0.5 * (math.log(1 + math.exp(-2)) + math.log(1 + math.exp(-1)))
    assert bce_loss(torch.tensor([2.0]), torch.tensor([-1.0])).item() == pytest.approx(expected)


def test_bce_decreases_with_better_separation():
    worse = bce_loss(torch.tensor([0.0]), torch.tensor([0.0]))
    better = bce_loss(torch.tensor([3.0]), torch.tensor([-3.0]))
    assert better < worse


def test_registry_lookup():
    assert get_loss("bpr") is bpr_loss
    with pytest.raises(KeyError, match="Available"):
        get_loss("nope")
