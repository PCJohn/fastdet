"""Out-of-core fitting: the booster trained block by block on more cells than fit in memory."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from fastdet import Detector, ModelConfig, TrainConfig, training
from fastdet.training import (
    auto_block_gib,
    bin_bits,
    cuda_device_id,
    fit_booster,
    fit_booster_blocks,
    gpu_free_gib,
    host_available_gib,
    plan_blocks,
    unused_blocks,
)

if TYPE_CHECKING:
    from catboost import CatBoost
    from numpy.typing import NDArray

    from fastdet import Config

_WIDTH = 12
_COARSE_COLUMNS = 6  # the first columns stand for tile-constant (coarse) features


def _cells(n: int, seed: int = 0) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, _WIDTH)).astype(np.float32)
    logit = 1.5 * x[:, 0] * (x[:, 1] > 0) + x[:, 2] - 0.8 * x[:, 3] ** 2 - 2.5
    return x, rng.random(n) < 1.0 / (1.0 + np.exp(-logit))


def _model_cfg(**overrides: object) -> ModelConfig:
    fields = {
        "n_trees": 64,
        "depth": 3,
        "leaf_bits": 4,
        "leaf_chunk": 8,
        "coarse_fraction": 0.5,
        "task_type": "CPU",
    }
    fields.update(overrides)
    return ModelConfig(**fields)  # type: ignore[arg-type]


_SIDES = [8] * _COARSE_COLUMNS + [64] * (_WIDTH - _COARSE_COLUMNS)


def test_plan_blocks_sizes_blocks_by_float32_gib() -> None:
    gib = 2**30
    assert plan_blocks(10, 100, None) == 1  # no budget: one matrix
    assert plan_blocks(gib // 400, 100, 1.0) == 1  # exactly 1 GiB fits
    assert plan_blocks(gib // 400 + 1, 100, 1.0) == 2
    assert plan_blocks(5_700_000, 3122, 4.0) == 17  # every cell of the reference data, 4 GiB blocks
    assert plan_blocks(0, 100, 1.0) == 1
    assert plan_blocks(10**9, 100, 0.0) == 1  # 0: never block


def test_block_gib_is_validated() -> None:
    assert TrainConfig().block_gib is None  # auto by default
    assert TrainConfig(block_gib=0.0).block_gib == 0.0  # never block
    with pytest.raises(ValueError, match="block_gib"):
        TrainConfig(block_gib=-1.0)


def test_auto_block_size_is_the_tighter_of_host_and_gpu_memory() -> None:
    host, why = auto_block_gib(64.0, None)
    assert host == pytest.approx(0.8 * 64.0 / 2.3)
    assert "64.0 GiB host RAM" in why
    gpu, _ = auto_block_gib(None, 11.0)
    assert gpu == pytest.approx(0.4 * 11.0 * 8.0)  # 4-bit bins: 8 float32 bytes per pool byte
    wide, _ = auto_block_gib(None, 11.0, gpu_bits=8)
    assert gpu is not None
    assert wide == pytest.approx(gpu / 2)
    both, why = auto_block_gib(64.0, 1.0)  # a GPU that is nearly full wins
    assert both == pytest.approx(0.4 * 1.0 * 8.0)
    assert "host RAM" in why
    assert "GPU memory" in why
    none, why = auto_block_gib(None, None)
    assert none is None
    assert "could not be read" in why


def test_bin_bits_follow_border_count() -> None:
    assert bin_bits(15) == 4  # the default, and the 4-bit blob's ceiling
    assert bin_bits(1) == 1
    assert bin_bits(7) == 3
    assert bin_bits(8) == 4
    assert bin_bits(254) == 8


def test_unused_blocks_counts_the_blocks_no_chunk_starts_in() -> None:
    cfg = _model_cfg(n_trees=64, leaf_chunk=8, coarse_fraction=0.5)  # chunks start every 8
    assert unused_blocks(cfg, 1) == 0
    assert unused_blocks(cfg, 8) == 0  # one chunk start per block
    assert unused_blocks(cfg, 9) == 1  # nine blocks, eight chunk starts
    assert unused_blocks(cfg, 64) == 56
    # the fine stage restarts its chunks at the coarse boundary: 24 coarse trees in chunks of
    # 16 start at 0 and 16, the fine ones at 24, 40, 56 -- five starts for 64 trees, but as
    # many starts as blocks is not enough: over 5 blocks they fall in blocks 0, 1, 1, 3, 4
    uneven = _model_cfg(n_trees=64, leaf_chunk=16, coarse_fraction=0.375)
    assert uneven.coarse_trees == 24
    assert unused_blocks(uneven, 4) == 0
    assert unused_blocks(uneven, 5) == 1
    # without quantisation-aware chunks a block serves one fit, so only a block with no tree is lost
    plain = _model_cfg(n_trees=10, quantisation_aware=False)
    assert unused_blocks(plain, 10) == 0
    assert unused_blocks(plain, 12) == 2


def test_cuda_device_id_follows_cuda_visible_devices() -> None:
    assert cuda_device_id(0, visible=None) == "0"
    assert cuda_device_id(0, visible="") == "0"
    assert cuda_device_id(0, visible="1") == "1"  # CatBoost's device 0 is physical GPU 1
    assert cuda_device_id(1, visible="3, 1") == "1"
    assert cuda_device_id(0, visible="GPU-8a3f") == "GPU-8a3f"  # UUIDs pass through
    assert cuda_device_id(2, visible="0,1") == "2"  # CUDA sees no such device


def test_host_memory_is_read_and_capped_by_a_cgroup_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    free = host_available_gib()
    assert free is None or free > 0.0
    limit, used = tmp_path / "memory.max", tmp_path / "memory.current"
    limit.write_text(str(8 * 2**30))
    used.write_text(str(6 * 2**30))
    assert training._cgroup_headroom_gib(str(limit), str(used)) == pytest.approx(2.0)
    limit.write_text("max")  # no limit
    assert training._cgroup_headroom_gib(str(limit), str(used)) is None
    assert training._cgroup_headroom_gib(str(tmp_path / "missing"), str(used)) is None
    monkeypatch.setattr(training, "_cgroup_available_gib", lambda: 0.5)
    assert host_available_gib() == pytest.approx(0.5) or free is None


def test_host_memory_probes_fall_back_and_parse_their_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Done:
        stdout = (
            "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
            "Pages free:                               100000.\n"
            "Pages active:                             900000.\n"
            "Pages inactive:                           200000.\n"
            "Pages speculative:                         20000.\n"
        )

    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/vm_stat")
    monkeypatch.setattr("subprocess.run", lambda *_a, **_k: _Done())
    assert training._vm_stat_available_gib() == pytest.approx(320000 * 16384 / 2**30)
    _Done.stdout = "garbage"
    assert training._vm_stat_available_gib() is None
    monkeypatch.setattr("shutil.which", lambda _name: None)
    assert training._vm_stat_available_gib() is None
    # a Linux host without /proc still has sysconf; nothing at all gives None, not a crash
    monkeypatch.setattr(training, "_meminfo_available_gib", lambda: None)
    monkeypatch.setattr(training, "_sysconf_available_gib", lambda: 3.0)
    monkeypatch.setattr(training, "_cgroup_available_gib", lambda: None)
    monkeypatch.setattr("sys.platform", "linux")
    assert host_available_gib() == pytest.approx(3.0)
    monkeypatch.setattr(training, "_sysconf_available_gib", lambda: None)
    assert host_available_gib() is None


def test_gpu_free_memory_comes_from_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[list[str]] = []

    class _Done:
        stdout = "9216\n"

    def run(args: list[str], **_kwargs: object) -> _Done:
        calls.append(args)
        return _Done()

    monkeypatch.setattr("shutil.which", lambda _name: None)
    monkeypatch.setattr("sys.platform", "linux")
    assert gpu_free_gib() is None  # no driver tools: nothing to go on
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr("subprocess.run", run)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    assert gpu_free_gib() == pytest.approx(9.0)  # MiB -> GiB
    assert calls[-1][-2:] == ["-i", "2"]  # the card CatBoost will train on, not physical 0
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    gpu_free_gib()
    assert calls[-1][-2:] == ["-i", "0"]
    _Done.stdout = "not a number"
    assert gpu_free_gib() is None


def test_blocked_fit_with_the_same_cells_reproduces_the_single_pool_fit() -> None:
    """The carry-over score is exact: every block is the full set, so nothing may change."""
    x, y = _cells(6000)
    cfg = _model_cfg()
    single = fit_booster(x.copy(), y, cfg, feature_sides=_SIDES)
    blocked = fit_booster_blocks(lambda _b: (x.copy(), y), 4, cfg, feature_sides=_SIDES)
    probe, _ = _cells(2000, seed=1)
    np.testing.assert_allclose(
        blocked.predict(probe, prediction_type="RawFormulaVal"),
        single.predict(probe, prediction_type="RawFormulaVal"),
        rtol=0,
        atol=1e-9,
    )


def test_blocks_are_built_once_in_order_and_share_one_set_of_borders() -> None:
    x, y = _cells(8000)
    cfg = _model_cfg()
    edges = np.linspace(0, len(y), 5).astype(int)
    seen: list[int] = []

    def provider(b: int) -> tuple[NDArray[np.float32], NDArray[np.bool_]]:
        seen.append(b)
        return x[edges[b] : edges[b + 1]].copy(), y[edges[b] : edges[b + 1]]

    booster = fit_booster_blocks(provider, 4, cfg, feature_sides=_SIDES)
    assert seen == [0, 1, 2, 3]
    assert booster.tree_count_ == cfg.n_trees
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.json"
        booster.save_model(str(path), format="json")
        features = json.loads(path.read_text())["features_info"]["float_features"]
    # a 4-bit bin: no column may have more than border_count cut points, whatever the block
    assert max(len(f.get("borders", [])) for f in features) <= cfg.border_count
    # the trees still learned the signal from the blocks
    probe_x, probe_y = _cells(4000, seed=2)
    logit = booster.predict(probe_x, prediction_type="RawFormulaVal")
    assert np.corrcoef(logit, probe_y)[0, 1] > 0.3


def test_blocked_coarse_trees_only_split_on_coarse_columns() -> None:
    x, y = _cells(6000)
    cfg = _model_cfg()
    booster = fit_booster_blocks(lambda _b: (x.copy(), y), 2, cfg, feature_sides=_SIDES)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.json"
        booster.save_model(str(path), format="json")
        trees = json.loads(path.read_text())["oblivious_trees"]
    coarse = {
        split["float_feature_index"]
        for tree in trees[: cfg.coarse_trees]
        for split in tree["splits"]
    }
    assert coarse <= set(range(_COARSE_COLUMNS))


def test_a_fit_in_blocks_matches_a_single_block_fit_in_quality() -> None:
    x, y = _cells(40_000)
    cfg = _model_cfg(n_trees=96)
    n_fit = 30_000
    single = fit_booster(x[:n_fit].copy(), y[:n_fit], cfg, feature_sides=_SIDES)
    edges = np.linspace(0, n_fit, 4).astype(int)
    blocked = fit_booster_blocks(
        lambda b: (x[edges[b] : edges[b + 1]].copy(), y[edges[b] : edges[b + 1]]),
        3,
        cfg,
        feature_sides=_SIDES,
    )

    def loss(model: CatBoost) -> float:
        logit = model.predict(x[n_fit:], prediction_type="RawFormulaVal")
        p = np.clip(1.0 / (1.0 + np.exp(-logit)), 1e-6, 1 - 1e-6)
        t = y[n_fit:]
        return float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))

    assert loss(blocked) < loss(single) * 1.05  # blocks of a third of the cells cost a little


def test_detector_fits_in_blocks_and_exports(
    tiny_dataset: tuple[Path, Path], small_config: Config, tmp_path: Path
) -> None:
    """``block_gib`` below the matrix size fits in blocks; the export is an ordinary model."""
    images_dir, masks_dir = tiny_dataset
    small_config.model.n_trees = 48
    small_config.model.leaf_chunk = 8
    small_config.model.coarse_fraction = 0.5
    small_config.train.block_gib = 0.02  # the ~20k sampled cells are ~0.09 GiB: several blocks
    detector = Detector(small_config).fit(images_dir, masks_dir, evaluate=False)
    assert detector.booster is not None
    assert detector.booster.tree_count_ == 48
    assert detector.runtime is not None
    assert detector.runtime.coarse_trees == 24
    path = detector.export(tmp_path / "blocks.fdt")
    reloaded = Detector.load(path)
    sample = min(images_dir.glob("*.png"))
    np.testing.assert_array_equal(reloaded.predict_proba(sample), detector.predict_proba(sample))
    reloaded.close()
    detector.close()


def test_a_block_with_one_class_is_refused(
    tiny_dataset: tuple[Path, Path], small_config: Config
) -> None:
    images_dir, masks_dir = tiny_dataset
    small_config.train.block_gib = 1e-6  # thousands of one-sample blocks
    with pytest.raises(ValueError, match="only one class"):
        Detector(small_config).fit(images_dir, masks_dir, evaluate=False)


def _fit_and_report(
    images_dir: Path, masks_dir: Path, config: Config, capsys: pytest.CaptureFixture[str]
) -> str:
    config.model.n_trees = 48
    config.model.leaf_chunk = 8
    Detector(config).fit(images_dir, masks_dir, evaluate=False).close()
    out: str = capsys.readouterr().out
    return out


def test_auto_block_size_fits_whole_when_memory_is_plentiful(
    tiny_dataset: tuple[Path, Path],
    small_config: Config,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("fastdet.detector.host_available_gib", lambda: 1000.0)
    out = _fit_and_report(*tiny_dataset, small_config, capsys)
    assert "block size, auto: 1000.0 GiB host RAM free" in out
    assert "fitting in" not in out


def test_auto_block_size_blocks_when_memory_is_short(
    tiny_dataset: tuple[Path, Path],
    small_config: Config,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("fastdet.detector.host_available_gib", lambda: 0.12)
    budget, _why = auto_block_gib(0.12, None)  # 0.8 * 0.12 / 2.3 = 0.04 GiB a block
    out = _fit_and_report(*tiny_dataset, small_config, capsys)
    assert "block size, auto: 0.1 GiB host RAM free" in out
    shape = re.search(r"training matrix: ([\d,]+) cells x (\d+)", out)
    assert shape is not None
    cells, width = int(shape[1].replace(",", "")), int(shape[2])
    expected = plan_blocks(cells, width, budget)
    assert expected > 1
    assert f"fitting in {expected} blocks" in out


def test_auto_block_size_uses_the_gpu_only_when_training_on_it(
    tiny_dataset: tuple[Path, Path],
    small_config: Config,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[int] = []

    def tiny_gpu() -> float:
        calls.append(1)
        return 0.001

    monkeypatch.setattr("fastdet.detector.host_available_gib", lambda: 1000.0)
    monkeypatch.setattr("fastdet.detector.gpu_free_gib", tiny_gpu)
    small_config.model.task_type = "CPU"
    out = _fit_and_report(*tiny_dataset, small_config, capsys)
    assert not calls
    assert "GPU memory" not in out


def test_the_fit_records_how_the_matrix_was_fitted(
    tiny_dataset: tuple[Path, Path],
    small_config: Config,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Blocked boosting is not the one-pool fit, so the artifact says which it got."""
    images_dir, masks_dir = tiny_dataset
    small_config.model.n_trees = 48
    small_config.model.leaf_chunk = 8
    small_config.train.block_gib = 0.02
    detector = Detector(small_config).fit(images_dir, masks_dir, evaluate=False)
    sampled = re.search(r"sampled cells=([\d,]+)", capsys.readouterr().out)
    assert sampled is not None
    assert detector.metrics["train_cells"] == int(sampled[1].replace(",", ""))
    assert detector.metrics["train_blocks"] > 1
    assert detector.metrics["block_gib"] == 0.02
    reloaded = Detector.load(detector.export(tmp_path / "blocks.fdt"))
    assert reloaded.metrics["train_blocks"] == detector.metrics["train_blocks"]
    assert reloaded.metrics["block_gib"] == 0.02
    reloaded.close()
    detector.close()
    small_config.train.block_gib = 0.0
    whole = Detector(small_config).fit(images_dir, masks_dir, evaluate=False)
    assert whole.metrics["train_blocks"] == 1
    assert whole.metrics["block_gib"] == 0.0
    whole.close()


def test_more_blocks_than_chunk_starts_is_refused(
    tiny_dataset: tuple[Path, Path], small_config: Config
) -> None:
    """A block no chunk starts in would never be seen: the fit says so instead of training."""
    images_dir, masks_dir = tiny_dataset
    small_config.model.n_trees = 16  # one chunk of 16 trees
    small_config.model.leaf_chunk = 16
    small_config.train.block_gib = 0.02  # ~0.09 GiB of cells: several blocks
    with pytest.raises(ValueError, match="never be seen"):
        Detector(small_config).fit(images_dir, masks_dir, evaluate=False)


def test_block_gib_zero_never_blocks_and_skips_the_memory_probe(
    tiny_dataset: tuple[Path, Path],
    small_config: Config,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("fastdet.detector.host_available_gib", lambda: 0.001)
    small_config.train.block_gib = 0.0
    out = _fit_and_report(*tiny_dataset, small_config, capsys)
    assert "block size, auto" not in out
    assert "fitting in" not in out


def test_unreadable_memory_falls_back_to_one_matrix(
    tiny_dataset: tuple[Path, Path],
    small_config: Config,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("fastdet.detector.host_available_gib", lambda: None)
    out = _fit_and_report(*tiny_dataset, small_config, capsys)
    assert "could not be read" in out
    assert "fitting in" not in out
