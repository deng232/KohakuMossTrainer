"""Sequential offline cache passes through Kohaku's native builders."""

import gc
from dataclasses import asdict

from .session import STORE


def prepare(knobs, job_id):
    from . import native

    device = native.torch.device("cuda", knobs["DEVICES"][0])
    data = native.data_knobs(knobs, device)
    for kind, knob in (("text", "TEXT_CACHE"), ("latents", "LATENT_CACHE")):
        if knobs[knob] != "offline":
            continue
        STORE.update(job_id, phase=f"CACHE_{kind.upper()}", cache_completed=0)
        encoder = dataset = None
        try:
            cfg = native.DataConfig.from_knobs(data)
            cfg.token_budget = "estimate"
            if kind == "text":
                cfg.latent_cache = "off"
                plan, text_device = native.training_text_plan(knobs, device)
                path = native.component(knobs, native.TEXT)
                on = native.torch.device("cpu") if plan.memory == "cpu" else text_device
                plan, _, _ = native.resolve_text_plan(path, plan, on, device)
                encoder = native.text_encoder(
                    path=path,
                    device=text_device,
                    plan=asdict(plan),
                )
                build = native.build_text
                batch = 16
            else:
                encoder = native.vae_encoder(
                    path=native.component(knobs, native.VAE),
                    device=device,
                    dtype=knobs["VAE_DTYPE"],
                    mode=knobs["LATENT_DRAW"],
                    plan=knobs["VAE_ENCODE"],
                )
                build = native.build_latents
                batch = 8
            dataset = native.MossDataset(cfg)
            build(
                dataset,
                encoder,
                batch_size=batch,
                progress=lambda n: STORE.update(job_id, cache_completed=n),
            )
        finally:
            del encoder, dataset
            gc.collect()
            native.torch.cuda.empty_cache()
    STORE.update(job_id, phase="INITIALIZING")
