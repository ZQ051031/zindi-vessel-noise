import shutil
import warnings
import numpy as np
import pandas as pd
import pytest
from src.schema import BAND_NAMES, cube_to_wide, read_test_csv
from src.synthetic import synthetic_background, add_controlled_passages, write_synthetic_release, make_synthetic_competition
from src.data import load_release, make_windows
from src.features import window_features
from src.train import train
from src.models import predict_cube
from src.predict import predict, COLUMNS
from src.evaluate import rmse, leaderboard_score, evaluate_frames
from src.pressure_map import pressure_map
from src.explain import explain
from src.ablation import ablate
from src.evaluate import bootstrap_ci


def test_schema(tmp_path):
    cube = synthetic_background(3)
    path = tmp_path / 'test.csv'
    df = cube_to_wide(['a', 'b', 'c'], cube)
    df = df.rename(columns={'m01_TOL_1000': 'm1_TOL_1000.0', 'm02_TOL_31.6': 'm2_TOL_31.62'})
    df.to_csv(path, index=False)
    ids, actual = read_test_csv(path)
    assert ids.tolist() == ['a', 'b', 'c']
    np.testing.assert_allclose(actual, cube, equal_nan=True)
    df.rename(columns={'m1_TOL_1000.0': 'm1_TOL_99999'}).to_csv(path, index=False)
    with pytest.raises(ValueError, match='frequency'):
        read_test_csv(path)


def test_share_definition():
    rng = np.random.default_rng(42)
    bg = synthetic_background(20, rng)
    mix, share, presence, _ = add_controlled_passages(bg, rng, p_none=0)
    eb, em = 10 ** (bg.astype(float) / 10), 10 ** (mix / 10)
    expected = np.nansum(em - eb, axis=(1, 2)) / np.nansum(em, axis=(1, 2))
    np.testing.assert_allclose(share, expected, atol=1e-12)
    assert presence.all()
    assert np.array_equal(np.isnan(bg), np.isnan(mix))
    none, zero, p, _ = add_controlled_passages(bg, rng, p_none=1)
    np.testing.assert_allclose(none, bg, equal_nan=True)
    assert not zero.any() and not p.any()
    empty = np.full((15, 25), np.nan)
    assert add_controlled_passages(empty, rng, p_none=0)[1] == 0


def test_features_nan_safe():
    cube = synthetic_background(3)
    cube[:, :, 2] = np.nan
    cube[0] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        features = window_features(cube)
    assert not np.isinf(features.to_numpy()).any()
    assert features.b2_mean.isna().all()
    np.testing.assert_allclose(window_features(cube[1:]).to_numpy(), features.iloc[1:].to_numpy(), equal_nan=True)


def test_two_added_units_in_ten(monkeypatch):
    import src.synthetic as synthetic
    class FixedDraw:
        def random(self):
            return .9

        def beta(self, a, b):
            return .2 / .95

        def choice(self, values):
            return values[0]
    monkeypatch.setattr(synthetic, 'vessel_passage_reference', lambda rng, kind: np.ones((15, 25)))
    bg = np.full((15, 25), np.nan)
    bg[0, 0] = 10 * np.log10(8)
    mix, share, _, _ = add_controlled_passages(bg, FixedDraw(), p_none=0)
    assert share == pytest.approx(.2)
    assert 10 ** (mix[0, 0] / 10) == pytest.approx(10)


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp('pipeline')
    release = write_synthetic_release(root / 'release', days=.12)
    a, _ = train(release, root / 'model_a', augment_k=2, n_estimators=30)
    b, _ = train(release, root / 'model_b', augment_k=2, n_estimators=30)
    return root, release, a, b


def test_loader_folder_zip(trained):
    root, release, _, _ = trained
    df = load_release(release)
    archive = shutil.make_archive(str(root / 'archive'), 'zip', release)
    pd.testing.assert_frame_equal(df, load_release(archive))
    cube, meta = make_windows(df)
    assert cube.shape == (44, 15, 25)
    assert meta.site.nunique() == 4
    file = next((release / 'acoustics').glob('*.parquet'))
    raw = pd.read_parquet(file)
    raw.loc[0, 'dur'] = 5
    raw.loc[1, 'qc_flag'] = 1
    raw.to_parquet(root / 'bad.parquet')
    bad_release = root / 'bad_release' / 'acoustics'
    bad_release.mkdir(parents=True)
    shutil.copy(root / 'bad.parquet', bad_release / 'site.parquet')
    assert len(load_release(bad_release.parent, drop_bad_flags=True)) == len(raw) - 2


def test_loader_ignores_export_bands_outside_common_range(tmp_path):
    acoustics = tmp_path / 'acoustics'
    acoustics.mkdir()
    path = acoustics / 'site.parquet'
    raw = pd.DataFrame({
        't': ['2026-01-01'], 'dur': [60],
        'TOL_25.1': [10.], 'TOL_31.6': [20.],
        'TOL_7943.3': [30.], 'TOL_10000': [40.],
    })
    raw.to_parquet(path)
    loaded = load_release(tmp_path)
    assert loaded.loc[0, ['TOL_31.6', 'TOL_7943']].tolist() == [20., 30.]
    assert 'TOL_25.1' not in loaded and 'TOL_10000' not in loaded
    raw.assign(TOL_900=50.).to_parquet(path)
    with pytest.raises(ValueError, match='frequency'):
        load_release(tmp_path)


def test_loader_masks_flat_export_floor(tmp_path):
    acoustics = tmp_path / 'acoustics'
    acoustics.mkdir()
    raw = pd.DataFrame(-156.5, index=range(2), columns=BAND_NAMES)
    raw['t'] = ['2026-01-01 00:00', '2026-01-01 00:01']
    raw['dur'] = 60
    raw.loc[1, 'TOL_31.6'] = 40.
    raw.to_parquet(acoustics / 'site.parquet')
    loaded = load_release(tmp_path)
    assert loaded.loc[0, BAND_NAMES].isna().all()
    assert loaded.at[1, 'TOL_31.6'] == 40.
    assert loaded.at[1, 'TOL_39.8'] == -156.5


def test_pipeline_determinism_and_independence(trained):
    root, _, a, b = trained
    comp = make_synthetic_competition(root / 'comp', n_test=12)
    template = pd.read_csv(comp / 'sample_submission.csv').iloc[::-1]
    template.to_csv(comp / 'sample_submission.csv', index=False)
    sub = predict(comp / 'Test.csv', a, comp / 'sample_submission.csv', root / 'submission.csv')
    assert list(sub) == COLUMNS
    assert sub.ID.tolist() == template.ID.tolist()
    assert np.isfinite(sub.iloc[:, 1:]).all().all()
    assert ((sub.iloc[:, 1:] >= 0) & (sub.iloc[:, 1:] <= 1)).all().all()
    _, cube = read_test_csv(comp / 'Test.csv')
    p, score = predict_cube(a, cube)
    p2, s2 = predict_cube(b, cube)
    np.testing.assert_array_equal(p, p2)
    np.testing.assert_array_equal(score, s2)
    single, ss = predict_cube(a, cube[[3]])
    np.testing.assert_allclose(single, p[[3]], rtol=0, atol=1e-14)
    np.testing.assert_allclose(ss, score[[3]], rtol=0, atol=1e-14)
    with pytest.raises(ValueError, match='Sample'):
        predict(comp / 'Test.csv', a, _bad_sample(root, template), root / 'invalid.csv')


def _bad_sample(root, template):
    path = root / 'bad_sample.csv'
    template.iloc[:-1].to_csv(path, index=False)
    return path


def test_evaluation():
    assert rmse([0, 1], [0, 0]) == np.sqrt(.5)
    assert leaderboard_score([0, 1], [0, 0]) == 1 - np.sqrt(.5)
    labels = pd.DataFrame({'ID': ['a', 'b'], 'share_added_vessel': [0, 1], 'site': ['s', 't']})
    sub = pd.DataFrame({'ID': ['a', 'b'], 'share_added_vessel': [0, 0], 'score_added_vessel': [.1, .9]})
    result = evaluate_frames(sub, labels)
    assert result['auc'] == 1
    assert 'site_cluster_ci' in result


def test_pressure_maps(trained):
    root, release, _, _ = trained
    summary = pressure_map(release, root / 'maps', root / 'model_a' / 'bundle.joblib')
    assert summary.SYNTHETIC.all()
    for filename in ['station_map.png', 'daily_levels.png', 'daily_relative.png', 'inferred_share_map.png', 'station_summary.csv', 'daily_inferred_share.png', 'daily_excess_energy.png']:
        assert (root / 'maps' / filename).stat().st_size > 0


def test_explanations_and_ablation(trained):
    root, _, bundle, _ = trained
    x = pd.read_parquet(root / 'model_a/SYNTHETIC_addition_features.parquet')
    oof = pd.read_parquet(root / 'model_a/SYNTHETIC_addition_oof.parquet')
    assert oof.groupby('background_window_id').size().eq(2).all()
    assert oof.groupby('background_window_id').fold.nunique().eq(1).all()
    tables = explain(bundle, x, root / 'explanations', synthetic=True)
    assert len(tables['contributions_by_band']) == 25
    assert tables['contributions_by_feature'].mean_absolute_contribution.sum() == pytest.approx(tables['contributions_by_family'].mean_absolute_contribution.sum())
    assert len(tables['lightgbm_gain_top20']) == 20
    assert (root / 'explanations/ridge_coefficients_top20.png').stat().st_size > 0
    controls = ablate(x, oof, root / 'ablations', n_estimators=15)
    assert len(controls) == 5
    assert np.isfinite(controls.rmse).all()
    with pytest.raises(ValueError, match='at least one'):
        explain(bundle, x.iloc[:0], root / 'empty')
    with pytest.raises(ValueError, match='Aligned'):
        ablate(x.iloc[:0], oof, root / 'empty')


def test_cluster_bootstrap():
    y = np.tile([0., .2, .8, 1.], 3)
    pred = y * .8
    groups = np.tile(np.arange(4), 3)
    ci = bootstrap_ci(y, pred, groups, n_boot=100)
    assert 0 <= ci[0] <= leaderboard_score(y, pred) <= ci[1] <= 1
    assert ci == bootstrap_ci(y, pred, groups, n_boot=100)
    assert bootstrap_ci(y, y, groups) == [1., 1.]
    with pytest.raises(ValueError, match='nonempty'):
        bootstrap_ci([], [])
    with pytest.raises(ValueError, match='aligned'):
        bootstrap_ci(y, pred, [0])
