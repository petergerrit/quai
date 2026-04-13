"""Registered target families for the specialised-gate-set regime.

A `TargetFamily` is a finite-ish list of target unitaries + metadata that a
SWIFTbot user wants to synthesise efficiently with some (base, extension)
gate set. It plays the role that Q_T's "all of SU(d)" plays for the
universal regime: the thing the pipeline optimises against.

Registry pattern (same as `swiftbot.tools.groups`):
    register a TargetFamily in a module, import it here, expose `REGISTRY`.
    Lookup via `get_target_family(name)`.

Built-in families live in sibling modules:
    * lamm.py — Lamm-series lattice QFT target families (Σ(36×3), Σ(72×3), …).
"""
from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, ConfigDict, Field
import numpy as np


class TargetFamily(BaseModel):
    """Named list of target unitaries to approximate.

    Split into a *discrete* list (fixed targets) and an optional
    *parametric* generator (a callable returning targets for each sampled
    parameter value). The parametric part is sampled lazily at evaluation
    time so the same family definition covers any number of parameter
    draws.

    Use `materialize(n_samples, rng)` to produce a concrete list of
    (label, matrix) for a run.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    qudit_dim: int = Field(..., gt=0)
    discrete: list[tuple[str, np.ndarray]] = Field(default_factory=list)
    parametric: Callable[[float], list[tuple[str, np.ndarray]]] | None = None
    param_range: tuple[float, float] = (0.05, float(np.pi - 0.05))
    reference: str = ""

    def materialize(
        self,
        n_parametric_samples: int = 0,
        rng_seed: int = 0,
    ) -> list[tuple[str, np.ndarray]]:
        """Produce a concrete list of (label, matrix) targets.

        Discrete targets are always included. Parametric targets are
        sampled at `n_parametric_samples` random parameter values drawn
        from `param_range` (deterministic under `rng_seed`).
        """
        items: list[tuple[str, np.ndarray]] = list(self.discrete)
        if self.parametric is not None and n_parametric_samples > 0:
            rng = np.random.default_rng(rng_seed)
            lo, hi = self.param_range
            for i, theta in enumerate(rng.uniform(lo, hi, size=n_parametric_samples)):
                for label, mat in self.parametric(float(theta)):
                    items.append((f"{label} θ_idx={i} θ={theta:.4f}", mat))
        return items


REGISTRY: dict[str, TargetFamily] = {}


def register(family: TargetFamily) -> None:
    if family.name in REGISTRY:
        raise ValueError(f"target family {family.name!r} already registered")
    REGISTRY[family.name] = family


def get_target_family(name: str) -> TargetFamily:
    if name not in REGISTRY:
        raise KeyError(
            f"target family {name!r} not registered. Registered: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]


def list_target_families() -> list[TargetFamily]:
    return sorted(REGISTRY.values(), key=lambda f: f.name)


# Side-effect import to populate the registry. Placed at bottom so the
# `register` function is defined first.
from . import lamm  # noqa: F401, E402
