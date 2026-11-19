"""Tests of the GPU memory budget of data_on_gpu (gpu_data_fits) and of the
fallback to RAM of loaders_to_device (no GPU needed: the CUDA calls are
mocked)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
dataset_mod = pytest.importorskip("ml_pytorch.utils.dataset")

gpu_data_fits = dataset_mod.gpu_data_fits
loaders_to_device = dataset_mod.loaders_to_device
DeviceBatchLoader = dataset_mod.DeviceBatchLoader
IndexedBatchLoader = dataset_mod.IndexedBatchLoader

GB = 1e9
TOTAL = 11.70 * GB
FREE = 11.5 * GB
DATA = 2.13 * GB


@pytest.mark.parametrize(
    "trainings_per_gpu, expected_fits",
    [(1, True), (3, True), (4, True), (5, False)],
)
def test_share_per_training(trainings_per_gpu, expected_fits):
    fits, need, budget = gpu_data_fits(DATA, TOTAL, FREE, trainings_per_gpu)
    assert fits is expected_fits
    assert need == pytest.approx(DATA + 0.5 * GB)
    assert budget == pytest.approx(min(TOTAL / trainings_per_gpu, FREE))


def test_single_training_limited_by_free_memory():
    # with one training the share is the whole GPU, so the free memory decides
    _, _, budget = gpu_data_fits(DATA, TOTAL, FREE, 1)
    assert budget == pytest.approx(FREE)


def test_margin_frac():
    fits, need, _ = gpu_data_fits(DATA, TOTAL, FREE, 4, margin_frac=0.2)
    assert need == pytest.approx(DATA * 1.2 + 0.5 * GB)
    assert fits is False  # need above the share TOTAL / 4
    fits, _, _ = gpu_data_fits(DATA, TOTAL, FREE, 3, margin_frac=0.2)
    assert fits is True


def test_free_memory_decides_when_below_share():
    # another process uses most of the GPU: the free memory is below the share
    free = 2.0 * GB
    fits, _, budget = gpu_data_fits(DATA, TOTAL, free, 2)
    assert budget == pytest.approx(free)
    assert fits is False
    fits, _, _ = gpu_data_fits(1.0 * GB, TOTAL, free, 2)
    assert fits is True


def test_large_margin_keeps_in_ram():
    fits, _, _ = gpu_data_fits(DATA, TOTAL, FREE, 1, margin_gb=100)
    assert fits is False


def test_zero_margin_exact_fit():
    fits, _, _ = gpu_data_fits(4 * GB, 8 * GB, 8 * GB, 2, margin_gb=0.0)
    assert fits is True


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(trainings_per_gpu=0),
        dict(trainings_per_gpu=1, margin_gb=-0.1),
        dict(trainings_per_gpu=1, margin_frac=-0.1),
    ],
)
def test_invalid_arguments(kwargs):
    with pytest.raises(ValueError):
        gpu_data_fits(DATA, TOTAL, FREE, **kwargs)


# ─── loaders_to_device with mocked CUDA ──────────────────────────────────────


class _Props:
    total_memory = int(TOTAL)


@pytest.fixture
def fake_cuda(monkeypatch):
    """Mock the CUDA memory queries; the copy to "cuda" raises an OOM error if
    `state["oom"]` is set, otherwise the tensors stay on the CPU."""
    state = {"oom": False, "empty_cache": 0, "copies": 0}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device=None: (int(FREE), int(TOTAL)))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device=None: _Props())

    def empty_cache():
        state["empty_cache"] += 1

    monkeypatch.setattr(torch.cuda, "empty_cache", empty_cache)

    orig_init = DeviceBatchLoader.__init__

    def init(self, dataset, batch_size, device, drop_last=True, shuffle=False):
        if torch.device(device).type == "cuda":
            state["copies"] += 1
            if state["oom"]:
                raise torch.OutOfMemoryError("CUDA out of memory (mocked)")
            device = torch.device("cpu")
        orig_init(self, dataset, batch_size, device, drop_last, shuffle)

    monkeypatch.setattr(DeviceBatchLoader, "__init__", init)
    return state


def _dataloaders(n=100):
    ds = torch.utils.data.TensorDataset(torch.randn(n, 3), torch.randn(n, 1))
    train = torch.utils.data.DataLoader(ds, batch_size=10, drop_last=True)
    val = torch.utils.data.DataLoader(ds, batch_size=10, drop_last=False)
    return [train, val, None]


def _indexed_loaders(n=100):
    sources = [np.random.randn(n, 3).astype(np.float32)]
    return [
        IndexedBatchLoader(sources, np.arange(0, n // 2), 10),
        IndexedBatchLoader(sources, np.arange(n // 2, n), 10, drop_last=False),
        None,
    ]


def test_cpu_device_unchanged():
    loaders = _dataloaders()
    assert loaders_to_device(loaders, torch.device("cpu")) is loaders


def test_fits_copies_to_gpu(fake_cuda):
    out = loaders_to_device(_dataloaders(), torch.device("cuda", 0), trainings_per_gpu=4)
    assert fake_cuda["copies"] == 2
    assert isinstance(out[0], DeviceBatchLoader) and out[2] is None


def test_does_not_fit_keeps_in_ram(fake_cuda):
    out = loaders_to_device(
        _dataloaders(), torch.device("cuda", 0), trainings_per_gpu=1, margin_gb=100
    )
    assert fake_cuda["copies"] == 0
    assert isinstance(out[0], DeviceBatchLoader)
    assert out[0].tensors[0].device.type == "cpu"


def test_does_not_fit_kfold_loaders_unchanged(fake_cuda):
    loaders = _indexed_loaders()
    out = loaders_to_device(loaders, torch.device("cuda", 0), margin_gb=100)
    assert out is loaders


@pytest.mark.parametrize("make_loaders", [_dataloaders, _indexed_loaders])
def test_oom_during_copy_falls_back_to_ram(fake_cuda, make_loaders, monkeypatch, caplog):
    # the budget says it fits, but the copy runs out of memory
    monkeypatch.setattr(dataset_mod, "gpu_data_fits", lambda *a, **k: (True, 1.0, 2.0))
    fake_cuda["oom"] = True
    loaders = make_loaders()
    with caplog.at_level("WARNING"):
        out = loaders_to_device(loaders, torch.device("cuda", 0))
    assert fake_cuda["empty_cache"] == 1
    assert "out of GPU memory" in caplog.text
    if make_loaders is _indexed_loaders:
        assert out is loaders
    else:
        assert isinstance(out[0], DeviceBatchLoader)
        assert out[0].tensors[0].device.type == "cpu"
        assert len(list(out[0])) == 10


def test_non_oom_error_is_raised(fake_cuda, monkeypatch):
    def init(*args, **kwargs):
        raise TypeError("not an OOM")

    monkeypatch.setattr(DeviceBatchLoader, "__init__", init)
    with pytest.raises(TypeError):
        loaders_to_device(_dataloaders(), torch.device("cuda", 0))


def test_old_style_oom_runtime_error():
    assert dataset_mod._is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert not dataset_mod._is_cuda_oom(RuntimeError("something else"))
