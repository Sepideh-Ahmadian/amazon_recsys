import random

import numpy as np
import torch

from amazon_recsys.utils.seed import set_seed


def _draw():
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_same_seed_same_draws():
    set_seed(7)
    a = _draw()
    set_seed(7)
    assert a == _draw()


def test_different_seed_different_draws():
    set_seed(1)
    a = _draw()
    set_seed(2)
    assert a != _draw()
