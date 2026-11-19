"""Tests of the reuse of the saved k-fold split when a k-fold training is
resumed (kfold_reuse_split): the folds are taken from the record in
kfold_split_dir, after checking that the events loaded now are the same."""

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf
dataset_mod = pytest.importorskip("ml_pytorch.utils.dataset")

DEFAULTS = os.path.join(
    os.path.dirname(dataset_mod.__file__), "..", "defaults", "default_configs.yml"
)
N, K = 30, 3


def _cfg(split_dir, reuse=False, **overrides):
    cfg = OmegaConf.load(DEFAULTS)
    cfg.k_folds = K
    cfg.kfold_split = "random"
    cfg.kfold_split_dir = str(split_dir)
    cfg.kfold_reuse_split = reuse
    for key, val in overrides.items():
        cfg[key] = val
    return cfg


def _events(seed=0):
    gen = torch.Generator().manual_seed(seed)
    fts = torch.randn(N, 4, generator=gen)
    X_k = torch.zeros(N, 3)
    X_k[:, 0] = torch.arange(N) % 2  # kl
    X_k[:, 1] = torch.arange(N) % 5  # dataset index
    return fts, X_k


def _prepare(cfg, fts, X_k, cache_dir):
    """What load_data does with the events of the k-fold training."""
    X_k = dataset_mod.assign_fold_ids(X_k, cfg)
    X_k = dataset_mod.apply_saved_kfold_split(X_k, fts, cfg)
    arrays = {
        "fts": fts,
        "lbl": torch.zeros(N),
        "clsw": torch.ones(N, 1),
        "clsw_bal": torch.ones(N, 1),
        "k": X_k,
    }
    meta = {"input_size": 3, "num_classes": 2, "is_binary": True, "n_events": N}
    dataset_mod.save_kfold_cache(str(cache_dir), arrays, meta, cfg)
    return X_k


@pytest.fixture
def first_run(tmp_path):
    split_dir = tmp_path / "kfold_split"
    fts, X_k = _events()
    X_k = _prepare(_cfg(split_dir), fts, X_k, tmp_path / "cache0")
    return split_dir, fts, X_k


def test_record_saved_with_fingerprint(first_run):
    split_dir, fts, X_k = first_run
    assert dataset_mod.kfold_split_record_exists(str(split_dir))
    with open(split_dir / dataset_mod.KFOLD_CACHE_META) as f:
        record = json.load(f)
    assert record["fts_sha256"] == dataset_mod.kfold_events_fingerprint(fts)
    np.testing.assert_array_equal(np.load(split_dir / "k.npy"), X_k.numpy())


def test_reuse_gives_the_saved_folds(first_run, tmp_path):
    split_dir, fts, X_k = first_run
    _, X_k_new = _events()
    out = _prepare(_cfg(split_dir, reuse=True), fts, X_k_new, tmp_path / "cache1")
    np.testing.assert_array_equal(out.numpy(), X_k.numpy())


def test_reuse_takes_the_saved_folds_if_recomputed_differ(first_run, caplog):
    split_dir, fts, X_k = first_run
    saved = np.load(split_dir / "k.npy")
    saved[:, 2] = (saved[:, 2] + 1) % K  # as if the assignment had changed
    np.save(split_dir / "k.npy", saved)
    _, X_k_new = _events()
    X_k_new = dataset_mod.assign_fold_ids(X_k_new, _cfg(split_dir))
    with caplog.at_level("WARNING"):
        out = dataset_mod.apply_saved_kfold_split(X_k_new, fts, _cfg(split_dir, reuse=True))
    np.testing.assert_array_equal(out[:, 2].numpy(), saved[:, 2])
    assert "using the saved folds" in caplog.text


def test_reuse_does_not_rewrite_the_record(first_run, tmp_path):
    split_dir, fts, _ = first_run
    before = {f: os.path.getmtime(split_dir / f) for f in os.listdir(split_dir)}
    _, X_k_new = _events()
    _prepare(_cfg(split_dir, reuse=True), fts, X_k_new, tmp_path / "cache1")
    after = {f: os.path.getmtime(split_dir / f) for f in os.listdir(split_dir)}
    assert before == after


def test_without_reuse_unchanged(first_run):
    split_dir, fts, _ = first_run
    _, X_k_new = _events()
    X_k_new[:, 2] = 7
    out = dataset_mod.apply_saved_kfold_split(X_k_new, fts, _cfg(split_dir))
    assert out is X_k_new


def test_reuse_without_record_warns(tmp_path, caplog):
    fts, X_k = _events()
    with caplog.at_level("WARNING"):
        out = dataset_mod.apply_saved_kfold_split(
            X_k, fts, _cfg(tmp_path / "missing", reuse=True)
        )
    assert out is X_k
    assert "no record" in caplog.text


def test_reuse_different_config(first_run):
    split_dir, fts, X_k = first_run
    with pytest.raises(ValueError, match="different config"):
        dataset_mod.apply_saved_kfold_split(
            X_k, fts, _cfg(split_dir, reuse=True, seed=12345)
        )


def test_reuse_different_number_of_events(first_run):
    split_dir, fts, X_k = first_run
    with pytest.raises(ValueError, match="events loaded now"):
        dataset_mod.apply_saved_kfold_split(
            X_k[:-1], fts[:-1], _cfg(split_dir, reuse=True)
        )


def test_reuse_different_events(first_run):
    split_dir, fts, X_k = first_run
    X_k = X_k.clone()
    X_k[[0, 1]] = X_k[[1, 0]]  # two events in a different order
    X_k[0, 0], X_k[1, 0] = 0.0, 1.0
    with pytest.raises(ValueError, match="not the events"):
        dataset_mod.apply_saved_kfold_split(X_k, fts, _cfg(split_dir, reuse=True))


def test_reuse_different_input_variables(first_run):
    split_dir, fts, X_k = first_run
    fts = fts.clone()
    fts[3, 1] += 1.0
    with pytest.raises(ValueError, match="input variables"):
        dataset_mod.apply_saved_kfold_split(X_k, fts, _cfg(split_dir, reuse=True))


def test_old_record_without_fingerprint(first_run):
    # records saved before the fingerprint existed are checked without it
    split_dir, fts, X_k = first_run
    meta_file = split_dir / dataset_mod.KFOLD_CACHE_META
    with open(meta_file) as f:
        record = json.load(f)
    del record["fts_sha256"]
    with open(meta_file, "w") as f:
        json.dump(record, f)
    out = dataset_mod.apply_saved_kfold_split(X_k, fts, _cfg(split_dir, reuse=True))
    np.testing.assert_array_equal(out.numpy(), X_k.numpy())
