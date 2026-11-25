import numpy as np

from src.loso_features import context_features, fit_station_profiles, station_embedding
from src.loso_experiment import blend_weight


def test_new_context_is_window_local_and_preserves_missing_inputs():
    cube = np.random.default_rng(42).normal(75, 8, (3, 15, 25))
    cube[0, 3] = np.nan
    cube[1, :, 21:] = np.nan
    cube[2] = np.nan
    features, profiles = context_features(cube)
    alone, alone_profiles = context_features(cube[1:2])
    np.testing.assert_allclose(features[1:2], alone)
    np.testing.assert_allclose(profiles[1:2], alone_profiles)
    assert not np.isinf(features).any()
    assert np.isnan(features[2]).all()
    assert np.isnan(profiles[2]).all()


def test_station_embedding_excludes_training_row_own_station():
    signatures = np.array([[0, 0], [0, 0], [9, 6], [9, 6], [20, -12], [20, -12]])
    sites = np.repeat(['one', 'two', 'three'], 2)
    basis = fit_station_profiles(signatures, sites)
    from_known = station_embedding(signatures, basis)
    cross_station = station_embedding(signatures, basis, sites)
    np.testing.assert_array_equal(from_known[:, -2], np.zeros(len(sites)))
    assert (cross_station[:, -2] > 0).all()
    unknown = station_embedding(signatures, basis, np.repeat('unseen', len(sites)))
    np.testing.assert_array_equal(unknown, from_known)


def test_convex_weight_minimizes_squared_error_without_extrapolation():
    first = np.array([.9, .2])
    second = np.array([.1, .8])
    assert np.isclose(blend_weight(.75 * first + .25 * second, first, second), .75)
    assert blend_weight(first + first - second, first, second) == 1
