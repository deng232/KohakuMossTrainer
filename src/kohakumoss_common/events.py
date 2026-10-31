"""Optional run-local observer for library hosts; CLI callers need no observer."""

import math
from contextvars import ContextVar
from dataclasses import dataclass

_ACTIVE = ContextVar("kohakumoss_session", default=None)


class Cancelled(Exception):
    """Cooperative stop; the training driver saves before raising this."""


class CleanupFailed(RuntimeError):
    """The host must retain GPU exclusion until its process is restarted."""


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    return value


@dataclass
class Session:
    observer: object
    cancelled: object = lambda: False

    def __enter__(self):
        self.token = _ACTIVE.set(self)
        return self

    def __exit__(self, *exc):
        _ACTIVE.reset(self.token)


def current():
    return _ACTIVE.get()


def emit(phase, **fields):
    session = current()
    if session is not None:
        session.observer(clean({"phase": phase, **fields}))


def stopped():
    session = current()
    return session is not None and session.cancelled()


def check_stop():
    if stopped():
        raise Cancelled()
