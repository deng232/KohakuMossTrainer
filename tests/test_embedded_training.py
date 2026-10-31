from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from kohakumoss.training.drivers import native
from kohakumoss.training.library import session, train
from kohakumoss.training.state import TrainState
from kohakumoss_common.events import Cancelled, Session, current


def test_stop_saves_only_completed_optimizer_step(monkeypatch):
    driver = native.NativeDriver(
        5,
        1,
        None,
        None,
        torch.device("cpu"),
        Mock(),
        Mock(),
        lambda state: {"step": state.step},
        count_fn=lambda batch: 1,
    )
    driver.checkpoints.saved.return_value = False
    steps = []
    driver.run_group = lambda *args: steps.append(1)

    def finish(state, *args):
        state.step += 1

    monkeypatch.setattr(native, "finish_step", finish)
    events = []
    with Session(events.append, lambda: len(steps) == 1), pytest.raises(Cancelled):
        driver.run(None, [object()], None, None, TrainState())
    driver.checkpoints.save.assert_called_once_with(1, {"step": 1})
    driver.checkpoints.finish.assert_not_called()
    assert any(e["phase"] == "training" and e["step"] == 1 for e in events)


def test_embedded_session_restores_rng_and_logger_on_failed_build():
    import logging
    import random

    root = logging.getLogger("kohakumoss")
    before = (
        torch.get_rng_state().clone(),
        random.getstate(),
        list(root.handlers),
        root.level,
    )
    with pytest.raises(ValueError):
        with session(lambda event: None):
            torch.manual_seed(123)
            random.seed(456)
            root.addHandler(logging.NullHandler())
            root.setLevel(3)
            raise ValueError("failed model build")
    assert torch.equal(torch.get_rng_state(), before[0])
    assert random.getstate() == before[1]
    assert root.handlers == before[2]
    assert root.level == before[3]
    assert current() is None


def test_library_never_calls_cli_entry(monkeypatch):
    from kohakumoss.training import entry, knobs

    monkeypatch.setattr(knobs, "check", lambda k: k)
    worker = Mock(return_value=TrainState(step=2))
    monkeypatch.setattr(entry, "worker", worker)
    monkeypatch.setattr(
        entry.shutdown, "install", Mock(side_effect=AssertionError("CLI hook"))
    )
    assert train({"PARALLEL": "none", "ENCODE_DEVICES": [], "DEVICES": [0]}).step == 2
    with pytest.raises(ValueError):
        train({"PARALLEL": "ddp", "ENCODE_DEVICES": [], "DEVICES": [0, 1]})
    assert worker.call_count == 1


def test_failed_adapter_build_closes_already_created_engines(monkeypatch):
    from kohakumoss.training.build import model

    engine = Mock()
    monkeypatch.setattr(model, "resolve_memory", lambda *args: (Mock(), []))
    monkeypatch.setattr(model, "is_full", lambda k: False)
    monkeypatch.setattr(model, "build_backbone", lambda *args: Mock())
    monkeypatch.setattr(model, "apply_fp16_fix", lambda *args, **kwargs: None)
    monkeypatch.setattr(model, "apply_to_backbone", lambda *args, **kwargs: engine)
    monkeypatch.setattr(
        model, "build_trainable", Mock(side_effect=RuntimeError("adapter OOM"))
    )
    prec = SimpleNamespace(
        compute=torch.float32, keep_fp32=(), fp16_fix=False, fp16_fix_kwargs={}
    )
    with pytest.raises(RuntimeError, match="adapter OOM"):
        model.build_model({}, torch.device("cpu"), prec)
    engine.close.assert_called_once()
    from kohakumoss_common.events import CleanupFailed

    engine.close.side_effect = RuntimeError("close failed")
    with pytest.raises(CleanupFailed, match="close failed"):
        model.build_model({}, torch.device("cpu"), prec)


def test_session_converts_system_exit_and_cleans_context():
    with pytest.raises(RuntimeError, match="no memory"):
        with session(lambda event: None):
            raise SystemExit("no memory")
    assert current() is None


def test_cleanup_still_runs_when_observer_fails(monkeypatch):
    from kohakumoss.training import entry

    built = SimpleNamespace(
        engines=Mock(), trainable=Mock(), backbone=Mock(), plan=Mock()
    )
    loader = Mock()
    layout = Mock()
    layout.data_rank.return_value = 0
    layout.data_world.return_value = 1
    monkeypatch.setattr(entry, "layout_of", lambda k: layout)
    monkeypatch.setattr(entry, "_start", lambda *args: ("run", Mock(), False))
    monkeypatch.setattr(entry, "_data", lambda *args: ({}, None, loader))
    monkeypatch.setattr(entry, "build_device_use", lambda device: Mock())
    monkeypatch.setattr(entry, "build_model", lambda *args: built)
    monkeypatch.setattr(entry, "placement_lines", lambda *args: [])
    monkeypatch.setattr(entry, "export_files", lambda *args: [])
    monkeypatch.setattr(entry, "export_bytes", lambda *args: 0)
    monkeypatch.setattr(entry, "log_export_plan", lambda *args: None)
    monkeypatch.setattr(
        entry, "_prewarm", Mock(side_effect=RuntimeError("warm failure"))
    )

    def observer(event):
        if event["phase"] == "cleaning_up":
            raise OSError("disk full")

    with Session(observer), pytest.raises(OSError, match="disk full"):
        entry.worker({"SEED": 0}, device=torch.device("cpu"))
    built.engines.close.assert_called_once()


def test_integration_composes_native_configs_without_cli(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path

    from kohakumoss.training import library

    root = Path(__file__).resolve().parents[1]
    plugin = root / "integrations/ComfyUI-KohakuMoss-Trainer/kohaku_comfy"
    spec = importlib.util.spec_from_file_location(
        "embedded_test",
        plugin / "__init__.py",
        submodule_search_locations=[str(plugin)],
    )
    package = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "embedded_test", package)
    spec.loader.exec_module(package)
    from embedded_test.config import DEFAULTS
    from embedded_test.embedded import run

    model = tmp_path / "model.safetensors"
    model.touch()
    data = tmp_path / "data"
    data.mkdir()
    config = tmp_path / "train.py"
    settings = {
        **DEFAULTS,
        "dataset_path": str(data),
        "transformer": str(model),
        "vae": str(model),
        "text_encoder": str(model),
        "steps": 2,
    }
    job = {
        "settings": settings,
        "config": str(config),
        "run_name": "smoke",
        "run_dir": str(tmp_path / "runs/smoke"),
        "cache_dir": str(tmp_path / "cache"),
        "log": str(tmp_path / "worker.log"),
    }
    caches, trains = [], []
    monkeypatch.setattr(library, "cache", lambda k, kind: caches.append((k, kind)))
    monkeypatch.setattr(library, "train", lambda k: trains.append(k))
    runtime = SimpleNamespace(repo=root)
    run(runtime, job, None, lambda event: None, lambda: False)
    assert [kind for k, kind in caches] == ["text", "latents"]
    assert caches[0][0]["TEXT_ENCODER_KWARGS"]["path"] == str(model)
    assert caches[0][0]["TEXT_ENCODER_KWARGS"]["plan"]["memory"]
    assert trains[0]["LR"] == 0.0001
    assert trains[0]["NUM_WORKERS"] == trains[0]["DATA_SCAN_WORKERS"] == 0
    assert trains[0]["RUN_NAME"] == "smoke"
    run(runtime, job, tmp_path / "step_00000002", lambda event: None, lambda: False)
    assert trains[1]["RESUME"].endswith("step_00000002")
