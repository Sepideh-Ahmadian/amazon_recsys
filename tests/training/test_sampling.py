import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from amazon_recsys.training.sampling import UniformNegativeSampler, epoch_batches

N_USERS, N_ITEMS = 6, 10


def _train():
    # user u interacted with items u, u+1, u+2 (mod N_ITEMS)
    users = np.repeat(np.arange(N_USERS), 3)
    items = np.array([(u + k) % N_ITEMS for u in range(N_USERS) for k in range(3)])
    return users, items


def _sampler(seed=0):
    users, items = _train()
    return UniformNegativeSampler(users, items, N_ITEMS, seed=seed)


def test_shape_and_dtype():
    out = _sampler().sample(np.array([0, 1, 2, 3]), n_neg=5)
    assert tuple(out.shape) == (4, 5)
    assert out.dtype == np.dtype("int64") or str(out.dtype) == "torch.int64"


@given(st.lists(st.integers(0, N_USERS - 1), min_size=1, max_size=40), st.integers(1, 4))
@settings(max_examples=40, deadline=None)
def test_negatives_never_train_positives(user_list, n_neg):
    users, items = _train()
    positives = set(zip(users.tolist(), items.tolist(), strict=True))
    neg = _sampler(seed=3).sample(np.array(user_list), n_neg)
    for u, row in zip(user_list, neg.tolist(), strict=True):
        assert all((u, j) not in positives for j in row)
        assert all(0 <= j < N_ITEMS for j in row)


def test_seed_reproducible_and_different_seed_differs():
    q = np.arange(N_USERS).repeat(20)
    a = _sampler(1).sample(q, 2)
    b = _sampler(1).sample(q, 2)
    c = _sampler(2).sample(q, 2)
    assert (a == b).all()
    assert not (a == c).all()


def test_user_with_all_items_rejected():
    users = np.zeros(N_ITEMS, dtype=int)
    items = np.arange(N_ITEMS)
    with pytest.raises(ValueError):
        UniformNegativeSampler(users, items, N_ITEMS, seed=0)


def test_epoch_batches_cover_every_row_once_and_are_seeded():
    users, items = _train()
    batches = list(epoch_batches(users, items, _sampler(), batch_size=4, seed=5))
    assert [len(b[0]) for b in batches] == [4, 4, 4, 4, 2]
    seen = sorted(
        zip(
            np.concatenate([b[0].numpy() for b in batches]).tolist(),
            np.concatenate([b[1].numpy() for b in batches]).tolist(),
            strict=True,
        )
    )
    assert seen == sorted(zip(users.tolist(), items.tolist(), strict=True))

    again = list(epoch_batches(users, items, _sampler(), batch_size=4, seed=5))
    for x, y in zip(batches, again, strict=True):
        assert (x[0] == y[0]).all()
        assert (x[2] == y[2]).all()
