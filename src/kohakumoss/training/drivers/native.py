"""Native driver: the plain training loop (the default driver)."""

import time
from collections.abc import Callable

import torch

from kohakumoss.diffusion.objective import TokenNormalizer
from kohakumoss.training.callbacks.runner import CallbackRunner
from kohakumoss.training.checkpoint import CheckpointManager
from kohakumoss.training.drivers.common import (
    Parts,
    finish_step,
    grouped,
    last_flags,
    loss_count,
    micro_steps,
    no_accumulate,
)
from kohakumoss.training.state import TrainState
from kohakumoss.training.step import image_token_count
from kohakumoss_common.calls import nothing
from kohakumoss_common.events import Cancelled, emit, stopped


class NativeDriver:
    """GRAD_ACC host batches per optimiser step, normalised by their global token count.
    `end_step` runs after the step's last backward (the adapter set's `end_step`)."""

    def __init__(
        self,
        steps: int,
        grad_acc: int,
        batcher: Callable,
        normalizer: TokenNormalizer,
        device,
        runner: CallbackRunner,
        checkpoints: CheckpointManager,
        parts: Parts,
        accumulate: Callable = no_accumulate,
        count_fn: Callable = image_token_count,
        end_step: Callable[[], None] = nothing,
    ):
        self.end_step = end_step
        self.steps = steps
        self.grad_acc = grad_acc
        self.batcher = batcher
        self.normalizer = normalizer
        self.device = device
        self.runner = runner
        self.checkpoints = checkpoints
        self.parts = parts
        self.accumulate = accumulate
        self.count_fn = count_fn
        self.lasts = last_flags(grad_acc)

    def run(self, step, loader, bundle, ema, state: TrainState) -> TrainState:
        groups = grouped(loader, self.grad_acc)
        self.runner.reset(state.step)
        self.runner.start(state)
        out = None
        started, first_step = time.monotonic(), state.step
        emit("training", step=state.step, total=self.steps)
        while state.step < self.steps:
            if stopped():
                emit("saving_checkpoint", step=state.step, total=self.steps)
                if not self.checkpoints.saved(state.step):
                    self.checkpoints.save(state.step, self.parts(state))
                raise Cancelled()
            group = next(groups)
            count = loss_count(group, self.count_fn)
            out = self.run_group(step, group, count, bundle, state, ema)
            self.end_step()
            finish_step(state, group, bundle, ema, self.runner, out)
            elapsed = time.monotonic() - started
            seconds = elapsed / max(1, state.step - first_step)
            metrics = {
                **state.metrics,
                "step": state.step,
                "total": self.steps,
                "metrics_step": state.step if state.synced else None,
                "loss": state.loss if state.synced else None,
                "lr": state.lr,
                "grad_norm": state.grad_norm if state.synced else None,
                "seconds_per_step": seconds,
                "eta_seconds": seconds * (self.steps - state.step),
            }
            if self.device.type == "cuda":
                metrics.update(
                    vram_allocated=torch.cuda.memory_allocated(self.device),
                    vram_reserved=torch.cuda.memory_reserved(self.device),
                )
            emit("training", **metrics)
        if stopped():
            emit("saving_checkpoint", step=state.step, total=self.steps)
            if not self.checkpoints.saved(state.step):
                self.checkpoints.save(state.step, self.parts(state))
            raise Cancelled()
        self.checkpoints.finish(state.step, self.parts(state))
        return state

    def run_group(
        self, step, group: list, count: int, bundle, state: TrainState, ema=None
    ):
        """Forward + backward of one optimiser step's micro-batches; the last StepOut."""
        denom = self.normalizer.denominator(count, self.device)
        return micro_steps(
            step,
            self.batcher,
            self.accumulate,
            self.lasts,
            bundle,
            group,
            denom,
            state.step,
            ema,
        )
