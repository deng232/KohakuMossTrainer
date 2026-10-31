"""Single-device embedded training; the caller owns the queue and cancellation."""

import copy
import gc
import logging
import random
from contextlib import contextmanager

import numpy as np
import torch

from kohakumoss_common.events import Session, check_stop, emit


@contextmanager
def session(observer, cancelled=lambda: False):
    """Restore host RNG/logging/tuning even when configuration or model loading fails."""
    from kohakumoss.adapters.lycoris.tuning import (
        restore_lycoris_tuning,
        set_lycoris_tuning,
    )
    from kohakumoss_kernels.tuning import tuner, window

    rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    device = torch.cuda.current_device() if cuda_rng is not None else None
    linalg = (
        torch.backends.cuda.preferred_linalg_library() if cuda_rng is not None else None
    )
    tag = torch.compiler.config.cache_key_tag
    tuning, windows = copy.copy(tuner._STATE), copy.copy(window.STATE)
    lycoris = set_lycoris_tuning(False)
    roots = [
        logging.getLogger(n)
        for n in ("kohakumoss", "kohakumoss_data", "kohakumoss_kernels")
    ]
    logs = [(r, list(r.handlers), r.level, r.propagate) for r in roots]
    try:
        with Session(observer, cancelled):
            try:
                yield
            except SystemExit as error:
                raise RuntimeError(str(error)) from error
    finally:
        for root, handlers, level, propagate in logs:
            for handler in root.handlers[:]:
                if handler not in handlers:
                    root.removeHandler(handler)
                    handler.close()
            root.handlers[:] = handlers
            root.setLevel(level)
            root.propagate = propagate
        restore_lycoris_tuning(lycoris)
        tuner._STATE.update(tuning)
        window.STATE.update(windows)
        torch.compiler.config.cache_key_tag = tag
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.set_rng_state(rng[2])
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
            torch.cuda.set_device(device)
            torch.backends.cuda.preferred_linalg_library(linalg)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train(knobs):
    """Run the native loop without installing CLI signal/atexit handlers."""
    from kohakumoss.training.entry import worker
    from kohakumoss.training.knobs import check

    k = check(knobs)
    if k["PARALLEL"] != "none" or k["ENCODE_DEVICES"] or len(k["DEVICES"]) != 1:
        raise ValueError(
            "Embedded training supports one device and PARALLEL='none' only"
        )
    check_stop()
    return worker(k)


def cache(knobs, kind):
    """The native resumable offline cache pass, with batch progress and cancellation."""
    from kohakumoss.encoders.specs import latent_encoder_kwargs, text_encoder_kwargs
    from kohakumoss_data.cache.build import build_latents, build_text
    from kohakumoss_data.dataset import MossDataset
    from kohakumoss_data.scripting import build_encoder, data_config

    text = kind == "text"
    phase = "caching_text" if text else "caching_latents"
    emit(phase, detail="loading encoder")
    check_stop()
    kwargs = text_encoder_kwargs(knobs) if text else latent_encoder_kwargs(knobs)
    encode = dataset = None
    try:
        encode = build_encoder(
            knobs["TEXT_ENCODER" if text else "IMAGE_ENCODER"], kwargs
        )
        spec = getattr(encode, "encoder_spec", "")
        overrides = dict(
            text_cache="offline" if text else "off",
            latent_cache="off" if text else "offline",
        )
        overrides["text_encoder_spec" if text else "latent_encoder_spec"] = spec
        dataset = MossDataset(data_config(knobs, **overrides))

        def progress(n):
            emit(phase, completed=n)
            check_stop()

        check_stop()
        (build_text if text else build_latents)(
            dataset, encode, knobs["ENCODE_BATCH"], progress
        )
    finally:
        if dataset is not None:
            for name in ("texts", "latents"):
                cache = getattr(dataset, name, None)
                if cache is not None and cache.store is not None:
                    cache.store.flush()
        encode = dataset = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
