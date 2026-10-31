import numpy as np
import pytest
from src.point_sampling import d4, sample_points, stable_seed


@pytest.mark.parametrize("strategy", ["uniform", "stratified"])
@pytest.mark.parametrize("budget", [0, 1, 10, 50, 200, 1000])
@pytest.mark.parametrize("shape", [(1, 1), (7, 13), (32, 32)])
def test_budget_validity_reproducibility(strategy, budget, shape):
    valid = np.ones(shape, dtype=bool)
    valid.flat[::3] = False
    points = sample_points(valid, budget, 123, strategy)
    assert points.sum() == min(budget, valid.sum())
    assert not np.any(points & ~valid)
    np.testing.assert_array_equal(points, sample_points(valid, budget, 123, strategy))


def test_nested_uniform_and_local_rng():
    valid = np.ones((32, 32), dtype=bool)
    np.random.seed(77)
    state = np.random.get_state()
    masks = [sample_points(valid, k, 42) for k in [10, 50, 200]]
    assert np.all(~masks[0] | masks[1])
    assert np.all(~masks[1] | masks[2])
    np.testing.assert_array_equal(state[1], np.random.get_state()[1])


def test_stratified_exact_cells():
    valid = np.ones((100, 100), dtype=bool)
    points = sample_points(valid, 25, 17, "stratified")
    for row in range(0, 100, 20):
        for col in range(0, 100, 20):
            assert points[row:row + 20, col:col + 20].sum() == 1


def test_empty_and_seed_namespaces():
    assert not sample_points(np.zeros((4, 5), dtype=bool), 50, 10, "stratified").any()
    seeds = {stable_seed(17, "source", "tile", n) for n in ["uniform", "stratified", "augmentation"]}
    assert len(seeds) == 3
    assert stable_seed(17, "source", "tile", "uniform") == stable_seed(17, "source", "tile", "uniform")


@pytest.mark.parametrize("transform", range(8))
def test_joint_d4(transform):
    labels = np.arange(64).reshape(8, 8)
    mask = labels % 7 == 0
    image = np.stack([labels] * 3)
    moved = d4(labels, transform)
    np.testing.assert_array_equal(d4(image, transform)[0], moved)
    np.testing.assert_array_equal(d4(mask, transform), moved % 7 == 0)


@pytest.mark.parametrize("budget", [-1, 1.5, True])
def test_bad_budget(budget):
    with pytest.raises(ValueError):
        sample_points(np.ones((2, 2), dtype=bool), budget, 1)
