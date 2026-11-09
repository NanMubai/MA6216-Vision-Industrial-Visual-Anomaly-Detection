from itertools import combinations

import pytest
import torch

from ad.coreset import greedy_coreset


@pytest.mark.parametrize("ratio", [0.001, 0.1, 0.45, 1.0])
def test_unique_indices_and_length(ratio):
    features = torch.randn(31, 12, generator=torch.Generator().manual_seed(2))
    indices = greedy_coreset(features, ratio=ratio, proj_dim=4)
    assert indices.dtype == torch.long
    assert indices.device == features.device
    assert len(indices) == max(1, int(len(features) * ratio))
    assert len(indices.unique()) == len(indices)
    assert ((0 <= indices) & (indices < len(features))).all()


def test_deterministic_with_projection_and_local_rng():
    features = torch.randn(60, 24, generator=torch.Generator().manual_seed(7))
    state = torch.random.get_rng_state().clone()
    first = greedy_coreset(features, ratio=0.2, proj_dim=5, seed=17)
    second = greedy_coreset(features, ratio=0.2, proj_dim=5, seed=17)
    assert torch.equal(first, second)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert not torch.equal(first, greedy_coreset(features, ratio=0.2, proj_dim=5, seed=18))


def test_duplicate_features_still_produce_unique_rows():
    indices = greedy_coreset(torch.ones(20, 8), ratio=0.75, proj_dim=3)
    assert len(indices) == len(indices.unique()) == 15


def test_covers_separated_clusters():
    centers = torch.tensor([[-10., -10.], [-10., 10.], [10., -10.], [10., 10.]])
    jitter = torch.randn(4, 15, 2, generator=torch.Generator().manual_seed(5)) * 0.05
    features = (centers[:, None, :] + jitter).reshape(-1, 2)
    indices = greedy_coreset(features, ratio=4 / 60, proj_dim=2, seed=9)
    assert set((indices // 15).tolist()) == {0, 1, 2, 3}


@pytest.mark.parametrize("seed", [0, 1, 8])
def test_greedy_covering_radius_is_two_approximation(seed):
    features = torch.tensor([[0., 0.], [0.2, 0.1], [1., 0.], [4., 1.],
                             [4., 2.], [7., 4.], [8., 4.], [8., 5.]])
    distances = torch.cdist(features, features)
    count = 3
    optimum = min(distances[:, subset].min(dim=1).values.max().item()
                  for subset in combinations(range(len(features)), count))
    indices = greedy_coreset(features, ratio=count / len(features), proj_dim=2, seed=seed)
    radius = distances[:, indices].min(dim=1).values.max().item()
    assert radius <= 2 * optimum + 1e-6


def test_matches_seeded_farthest_point_reference_without_projection():
    features = torch.randn(17, 3, generator=torch.Generator().manual_seed(8))
    generator = torch.Generator().manual_seed(3)
    expected = [torch.randint(len(features), (1,), generator=generator).item()]
    for _ in range(5):
        distances = torch.cdist(features, features[expected]).min(dim=1).values
        distances[expected] = -1
        expected.append(distances.argmax().item())
    actual = greedy_coreset(features, ratio=6 / 17, proj_dim=3, seed=3)
    assert actual.tolist() == expected


@pytest.mark.parametrize("features,ratio,proj_dim", [
    (torch.empty(0, 4), 0.1, 2),
    (torch.empty(4, 0), 0.1, 2),
    (torch.empty(4), 0.1, 2),
    (torch.ones(4, 2), 0, 2),
    (torch.ones(4, 2), 1.1, 2),
    (torch.ones(4, 2), 0.1, 0),
])
def test_rejects_invalid_inputs(features, ratio, proj_dim):
    with pytest.raises(ValueError):
        greedy_coreset(features, ratio=ratio, proj_dim=proj_dim)


@pytest.mark.gpu
def test_cuda_coreset_stays_on_device_and_is_deterministic():
    features = torch.randn(2500, 32, device="cuda")
    first = greedy_coreset(features, ratio=0.02, proj_dim=8, seed=2)
    second = greedy_coreset(features, ratio=0.02, proj_dim=8, seed=2)
    assert first.device == features.device
    assert len(first.unique()) == len(first) == 50
    assert torch.equal(first, second)
