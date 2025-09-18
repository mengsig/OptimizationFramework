# Marcus Engsig --- github@mengsig
# Unified baselines for continuous, box-constrained optimization
# Includes: BHO (user implementation), PSO, CMA-ES, SciPy-DE, GA(SBX), ABC,
# Dual Annealing, Nelder-Mead. Consistent interface

# ======================================================================
#                           Internal README
# ======================================================================
# to use these functions, please interact with the run_{method} functions.
# that are shown below. See src/run.py for reference

# ======================================================================
#                           Imports
# ======================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, Tuple, List
import networkx as nx
import networkx
import sys
import numpy as np

try:
    import scipy.optimize as spo
except Exception:
    spo = None

Array = np.ndarray
Objective = Callable[[Array], float]

# ======================================================================
#                           Progress helpers
# ======================================================================


class _TextBar:
    """Lightweight TTY progress bar driven by function evals."""

    def __init__(self, total: int, label: str = "", width: int = 30, stream=sys.stdout):
        self.total = max(1, int(total))
        self.label = label
        self.width = width
        self.stream = stream
        self._last = -1
        self.best = np.inf

    def update(self, n: int, best: Optional[float] = None):
        n = int(min(max(0, n), self.total))
        if best is not None:
            self.best = min(self.best, float(best))
        # Throttle to avoid spam: redraw only when percent step changes
        pct = int(100 * n / self.total)
        if pct == self._last and n < self.total:
            return
        self._last = pct
        filled = int(self.width * n / self.total)
        bar = "#" * filled + "-" * (self.width - filled)
        best_str = f"  best={self.best:.6f}" if np.isfinite(self.best) else ""
        self.stream.write(f"\r[{self.label:<12}] [{bar}] {pct:3d}%{best_str}")
        self.stream.flush()

    def close(self):
        # Force 100% and newline
        self.update(self.total, self.best)
        self.stream.write("\n")
        self.stream.flush()


# ======================================================================
#                           Common helpers
# ======================================================================


@dataclass
class OptResult:
    x: Array
    fx: float
    nfev: int
    trace: List[float]  # best-so-far curve (optional; may be empty)


def _rng(seed: Optional[int]) -> np.random.default_rng:
    return np.random.default_rng(None if seed is None else int(seed))


class EvalCounter:
    """Wrap an objective to count evals, hard-cap the budget, and drive a progress bar."""

    def __init__(
        self,
        f: Objective,
        budget: int,
        bar: Optional[_TextBar] = None,
        progress_every: Optional[int] = None,
    ):
        self.f = f
        self.n = 0
        self.best = np.inf
        self.budget = int(budget)
        self.bar = bar
        # ~200 redraws max by default
        self.progress_every = (
            progress_every if progress_every is not None else max(1, self.budget // 200)
        )

    def __call__(self, x: Array) -> float:
        if self.n >= self.budget:
            return np.inf
        self.n += 1
        y = float(self.f(np.asarray(x, dtype=float)))
        self.best = min(self.best, y)
        # Drive progress bar on a coarse cadence (and on final eval)
        if self.bar and (self.n % self.progress_every == 0 or self.n == self.budget):
            self.bar.update(self.n, self.best)
        return y


def _sample_in_box(rng: np.random.Generator, bounds: Array, n: int) -> Array:
    lo, hi = bounds[:, 0], bounds[:, 1]
    return rng.uniform(lo, hi, size=(n, lo.size))


def _project_box(x: Array, bounds: Array) -> Array:
    lo, hi = bounds[:, 0], bounds[:, 1]
    return np.clip(x, lo, hi)


# ======================================================================
#                           CMA-ES (pycma)
# ======================================================================


def run_cma(
    f: Objective,
    bounds: Array,
    budget: int,
    seed: Optional[int] = None,
    x0: Optional[Array] = None,
    sigma0: Optional[float] = None,
    popsize: Optional[int] = None,
) -> OptResult:
    try:
        import cma
    except Exception as e:
        raise RuntimeError("pycma is not installed (pip install cma).") from e

    label = "cma"
    bar = _TextBar(total=budget, label=label)
    d = bounds.shape[0]
    rng = _rng(seed)
    if x0 is None:
        x0 = _sample_in_box(rng, bounds, 1)[0]
    if sigma0 is None:
        span = (bounds[:, 1] - bounds[:, 0]).mean()
        sigma0 = 0.3 * float(span)

    wrapped = EvalCounter(f, budget, bar=bar)

    opts = {
        "bounds": [bounds[:, 0].tolist(), bounds[:, 1].tolist()],
        "seed": seed,
        "verbose": -9,
        "tolfun": 0,
        "maxfevals": budget,
    }
    if popsize is not None:
        opts["popsize"] = int(popsize)

    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)
    trace = []
    # Use our own loop; rely on hard eval cap via wrapped
    while True:
        X = es.ask()
        vals = [wrapped(np.array(x)) for x in X]
        es.tell(X, vals)
        trace.append(float(np.nanmin(vals)))
        if wrapped.n >= budget:
            break

    try:
        xbest = np.array(es.best.x)
        fx = float(es.best.f)
    except Exception:
        xbest = np.array(es.result.xbest)
        fx = float(es.result.fbest)
    bar.close()
    return OptResult(xbest, fx, wrapped.n, trace)


# ======================================================================
#                    Differential Evolution (SciPy)
# ======================================================================


def run_de_scipy(
    f: Objective,
    bounds: Array,
    budget: int,
    seed: Optional[int] = None,
    popsize: int = 15,
    strategy: str = "best1bin",
    tol: float = 0.0,
) -> OptResult:
    if spo is None:
        raise RuntimeError("SciPy not available.")
    label = "de"
    bar = _TextBar(total=budget, label=label)
    d = bounds.shape[0]
    # Rough mapping: initial pop ~ popsize*d, then ~popsize*d per gen
    maxiter = max(1, budget // max(popsize * d, 1) - 1)

    wrapped = EvalCounter(f, budget, bar=bar)

    # Update bar each generation via callback (also driven by EvalCounter inside)
    def _cb(xk, convergence):
        # xk is current best; show wrapped.best which tracks true best f seen
        bar.update(wrapped.n, wrapped.best)
        return False

    r = spo.differential_evolution(
        func=lambda x: wrapped(x),
        bounds=bounds.tolist(),
        strategy=strategy,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        seed=seed,
        polish=False,  # pure DE for fairness
        disp=False,
        updating="deferred",
        workers=1,
        callback=_cb,
    )
    bar.close()
    return OptResult(np.array(r.x), float(r.fun), wrapped.n, trace=[])


# ======================================================================
#                     GA (SBX + polynomial mutation)
# ======================================================================


def _sbx_crossover(
    rng: np.random.Generator, p1: Array, p2: Array, eta_c: float, lo: Array, hi: Array
) -> Tuple[Array, Array]:
    u = rng.random(p1.shape)
    beta = np.where(
        u <= 0.5, (2 * u) ** (1 / (eta_c + 1)), (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
    )
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return _project_box(c1, np.c_[lo, hi]), _project_box(c2, np.c_[lo, hi])


def _poly_mutation(
    rng: np.random.Generator, x: Array, pm: float, eta_m: float, lo: Array, hi: Array
) -> Array:
    y = x.copy()
    for i in range(x.size):
        if rng.random() < pm:
            delta1 = (x[i] - lo[i]) / (hi[i] - lo[i] + 1e-12)
            delta2 = (hi[i] - x[i]) / (hi[i] - lo[i] + 1e-12)
            u = rng.random()
            mut_pow = 1.0 / (eta_m + 1.0)
            if u < 0.5:
                xy = 1.0 - delta1
                val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta_m + 1))
                deltaq = val**mut_pow - 1.0
            else:
                xy = 1.0 - delta2
                val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta_m + 1))
                deltaq = 1.0 - val**mut_pow
            y[i] = x[i] + deltaq * (hi[i] - lo[i])
    return _project_box(y, np.c_[lo, hi])


def run_ga_sbx(
    f: Objective,
    bounds: Array,
    budget: int,
    seed: Optional[int] = None,
    pop: int = 60,
    elite: int = 2,
    eta_c: float = 15.0,
    eta_m: float = 20.0,
    pm: Optional[float] = None,
    tournament: int = 2,
) -> OptResult:
    label = "ga"
    bar = _TextBar(total=budget, label=label)
    rng = _rng(seed)
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    if pm is None:
        pm = 1.0 / d

    wrapped = EvalCounter(f, budget, bar=bar)
    P = _sample_in_box(rng, bounds, pop)
    fitness = np.array([wrapped(x) for x in P])
    trace = [float(np.nanmin(fitness))]
    xbest, fbest = P[int(np.argmin(fitness))].copy(), float(np.nanmin(fitness))

    while wrapped.n < budget:
        idx_sorted = np.argsort(fitness)
        newP = [P[i].copy() for i in idx_sorted[:elite]]

        def pick():
            cand = rng.choice(P.shape[0], size=tournament, replace=False)
            return P[cand[np.argmin(fitness[cand])]]

        while len(newP) < P.shape[0]:
            p1, p2 = pick(), pick()
            c1, c2 = _sbx_crossover(rng, p1, p2, eta_c, lo, hi)
            c1 = _poly_mutation(rng, c1, pm, eta_m, lo, hi)
            newP.append(c1)
            if len(newP) < P.shape[0]:
                c2 = _poly_mutation(rng, c2, pm, eta_m, lo, hi)
                newP.append(c2)

        P = np.array(newP)
        fitness = np.array([wrapped(x) for x in P])
        bi = int(np.argmin(fitness))
        if fitness[bi] < fbest:
            xbest, fbest = P[bi].copy(), float(fitness[bi])
        trace.append(fbest)
        if wrapped.n >= budget:
            break

        # Generation-level bar refresh
        bar.update(wrapped.n, wrapped.best)

    bar.close()
    return OptResult(xbest, fbest, wrapped.n, trace)


# ======================================================================
#                   Artificial Bee Colony (compact)
# ======================================================================


def run_abc(
    f: Objective,
    bounds: Array,
    budget: int,
    seed: Optional[int] = None,
    SN: int = 40,
    limit: Optional[int] = None,
) -> OptResult:
    label = "abc"
    bar = _TextBar(total=budget, label=label)

    rng = _rng(seed)
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    if limit is None:
        limit = SN * d

    wrapped = EvalCounter(f, budget, bar=bar)
    X = _sample_in_box(rng, bounds, SN)
    F = np.array([wrapped(x) for x in X])
    trials = np.zeros(SN, dtype=int)
    trace = [float(np.nanmin(F))]
    best_idx = int(np.argmin(F))
    xbest, fbest = X[best_idx].copy(), float(F[best_idx])

    def employed_phase():
        nonlocal X, F, trials, xbest, fbest
        for i in range(SN):
            k = rng.integers(0, SN - 1)
            if k >= i:
                k += 1
            phi = rng.uniform(-1, 1, size=d)
            j = rng.integers(0, d)
            v = X[i].copy()
            v[j] = X[i, j] + phi[j] * (X[i, j] - X[k, j])
            v = _project_box(v, np.c_[lo, hi])
            fv = wrapped(v)
            if fv < F[i]:
                X[i], F[i] = v, fv
                trials[i] = 0
                if fv < fbest:
                    xbest, fbest = v.copy(), float(fv)
            else:
                trials[i] += 1

    def onlooker_phase():
        nonlocal X, F, trials, xbest, fbest
        fit = 1.0 / (1.0 + F - F.min() + 1e-12)
        p = fit / fit.sum()
        i = 0
        count = 0
        while count < SN:
            if rng.random() < p[i]:
                count += 1
                k = rng.integers(0, SN - 1)
                if k >= i:
                    k += 1
                phi = rng.uniform(-1, 1, size=d)
                j = rng.integers(0, d)
                v = X[i].copy()
                v[j] = X[i, j] + phi[j] * (X[i, j] - X[k, j])
                v = _project_box(v, np.c_[lo, hi])
                fv = wrapped(v)
                if fv < F[i]:
                    X[i], F[i] = v, fv
                    trials[i] = 0
                    if fv < fbest:
                        xbest, fbest = v.copy(), float(fv)
                else:
                    trials[i] += 1
            i = (i + 1) % SN

    def scout_phase():
        nonlocal X, F, trials, xbest, fbest
        for i in range(SN):
            if trials[i] >= limit:
                X[i] = _sample_in_box(rng, bounds, 1)[0]
                F[i] = wrapped(X[i])
                trials[i] = 0
                if F[i] < fbest:
                    xbest, fbest = X[i].copy(), float(F[i])

    while wrapped.n < budget:
        employed_phase()
        if wrapped.n >= budget:
            break
        onlooker_phase()
        if wrapped.n >= budget:
            break
        scout_phase()
        trace.append(fbest)
        bar.update(wrapped.n, wrapped.best)

    bar.close()
    return OptResult(xbest, fbest, wrapped.n, trace)


# ======================================================================
#                   Dual Annealing / Local (SciPy)
# ======================================================================


def run_dual_annealing(
    f: Objective, bounds: Array, budget: int, seed: Optional[int] = None
) -> OptResult:
    if spo is None:
        raise RuntimeError("SciPy not available.")
    label = "dual_anneal"
    bar = _TextBar(total=budget, label=label)
    wrapped = EvalCounter(f, budget, bar=bar)

    # Try to get iteration callbacks too; EvalCounter will still drive the bar by evals
    def _cb(x, fval, context):
        bar.update(wrapped.n, wrapped.best)
        return False

    r = spo.dual_annealing(
        lambda x: wrapped(x),
        bounds=bounds.tolist(),
        seed=seed,
        maxfun=budget,
        callback=_cb,
    )
    bar.close()
    return OptResult(np.array(r.x), float(r.fun), wrapped.n, trace=[])


def run_nelder_mead(
    f: Objective,
    bounds: Array,
    budget: int,
    seed: Optional[int] = None,
    x0: Optional[Array] = None,
) -> OptResult:
    if spo is None:
        raise RuntimeError("SciPy not available.")
    label = "nelder_mead"
    bar = _TextBar(total=budget, label=label)

    rng = _rng(seed)
    if x0 is None:
        x0 = _sample_in_box(rng, bounds, 1)[0]
    wrapped = EvalCounter(f, budget, bar=bar)

    def _cb(xk, *args, **kwargs):
        bar.update(wrapped.n, wrapped.best)

    r = spo.minimize(
        lambda x: wrapped(x),
        x0=x0,
        method="Nelder-Mead",
        options={"maxfev": budget, "xatol": 1e-9, "fatol": 1e-9, "disp": False},
        callback=_cb,
    )
    bar.close()
    return OptResult(np.array(r.x), float(r.fun), wrapped.n, trace=[])


def run_powell(
    f: Objective,
    bounds: Array,
    budget: int,
    seed: Optional[int] = None,
    x0: Optional[Array] = None,
) -> OptResult:
    if spo is None:
        raise RuntimeError("SciPy not available.")
    label = "powell"
    bar = _TextBar(total=budget, label=label)

    rng = _rng(seed)
    if x0 is None:
        x0 = _sample_in_box(rng, bounds, 1)[0]
    wrapped = EvalCounter(f, budget, bar=bar)

    def _cb(xk, *args, **kwargs):
        bar.update(wrapped.n, wrapped.best)

    r = spo.minimize(
        lambda x: wrapped(x),
        x0=x0,
        method="Powell",
        bounds=bounds.tolist(),
        options={"maxfev": budget, "xtol": 1e-9, "ftol": 1e-9, "disp": False},
        callback=_cb,
    )
    bar.close()
    return OptResult(np.array(r.x), float(r.fun), wrapped.n, trace=[])


# ======================================================================
#                        PSO (global-best, reflective)
# ======================================================================


def run_pso(
    f: Objective,
    bounds: Array,
    budget: int,
    seed: Optional[int] = None,
    n_particles: int = 60,
    w: float = 0.7,
    c1: float = 1.5,
    c2: float = 1.5,
    v_init_scale: float = 0.1,
    vmax_frac: float = 0.2,
) -> OptResult:
    """
    Global-best PSO with reflective bounds and velocity clamping.
    Budget mapping: initial N evals + ~N per-iteration.
    """
    label = "pso"
    bar = _TextBar(total=budget, label=label)

    rng = _rng(seed)
    d = bounds.shape[0]
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo

    wrapped = EvalCounter(f, budget, bar=bar)

    # Init
    X = _sample_in_box(rng, bounds, n_particles)
    V = rng.normal(0.0, 1.0, size=(n_particles, d)) * (span * v_init_scale)
    Pbest = X.copy()
    Fp = np.array([wrapped(x) for x in X])
    g_idx = int(np.argmin(Fp))
    Gbest = X[g_idx].copy()
    Fg = float(Fp[g_idx])

    vmax = vmax_frac * span
    trace = [float(Fg)]

    if n_particles <= 0:
        bar.close()
        return OptResult(Gbest, Fg, wrapped.n, trace)
    remaining = max(0, budget - wrapped.n)
    max_iter = max(1, remaining // max(n_particles, 1))

    for _ in range(max_iter):
        r1 = rng.random(size=(n_particles, d))
        r2 = rng.random(size=(n_particles, d))
        V = w * V + c1 * r1 * (Pbest - X) + c2 * r2 * (Gbest - X)
        V = np.clip(V, -vmax, vmax)
        X = X + V

        below = X < lo
        above = X > hi
        X = np.where(below, 2.0 * lo - X, X)
        X = np.where(above, 2.0 * hi - X, X)
        V = np.where(below | above, -0.5 * V, V)
        X = np.clip(X, lo, hi)

        F = np.array([wrapped(x) for x in X])

        improved = F < Fp
        Pbest[improved] = X[improved]
        Fp[improved] = F[improved]

        idx = int(np.argmin(Fp))
        if Fp[idx] < Fg:
            Gbest = Pbest[idx].copy()
            Fg = float(Fp[idx])

        trace.append(Fg)
        if wrapped.n >= budget:
            break

        # Iter-level refresh
        bar.update(wrapped.n, wrapped.best)

    bar.close()
    return OptResult(Gbest, Fg, wrapped.n, trace)


# ======================================================================
#                      Beehive Optimization
# ======================================================================


@dataclass
class Pheromone:
    """Represents a pheromone emitted at time T from a given bee position.
    Base strength s is defined at drop time T; strength at time t>=T is s * gamma**(t - T).
    """

    position: Array  # shape (d,)
    base_strength: float  # s = (L(T-1)-L(T)) / L(T)
    T: int  # drop time

    def strength_at(self, t: int, gamma: float) -> float:
        return float(self.base_strength) * (gamma ** max(0, t - self.T))


@dataclass
class BeehiveOptimization:
    func: Callable[[Array], float]
    bounds: Array  # shape (d,2)
    N: int = 60  # number of bees
    c: float = 0.9  # collective intelligence weight (pheromone follow)
    q: float = 0.05  # queen attraction weight
    rho: float = 0.98  # inertia (0<rho<1)
    gamma: float = 0.5  # pheromone decay (0..1)
    dt: float = 1.0  # time step for position update
    kappa: float = 1.0  # pheromone force scaling factor in Eq. (5)
    max_steps: int = 1000
    seed: Optional[int] = None
    v_init_scale: float = 0.02  # initial velocity magnitude ~ box span * scale
    pheromone_capacity: Optional[int] = None  # max # pheromones kept (None = unbounded)
    bound_strategy: str = "reflect"  # 'reflect' or 'clip'
    tol_abs: Optional[float] = None  # stop if best loss < tol_abs
    patience: Optional[int] = None  # early stop if no improvement for 'patience' steps
    earlystop: Optional[bool] = True

    # State (populated by reset)
    X: Array = field(init=False, repr=False)  # (N,d) positions
    V: Array = field(init=False, repr=False)  # (N,d) velocities
    losses: Array = field(init=False, repr=False)  # (N,) current losses
    prev_losses: Array = field(init=False, repr=False)  # (N,) previous losses
    pheromones: List[Pheromone] = field(init=False, repr=False)
    queen_pos: Array = field(init=False, repr=False)
    queen_loss: float = field(init=False, repr=False)
    queen_history: List[float] = field(init=False, repr=False)
    history_losses: List[List[float]] = field(init=False, repr=False)
    history_positions: List[List[float]] = field(init=False, repr=False)
    history_pheromones: List[List[float]] = field(init=False, repr=False)
    best_pos_history: List[Array] = field(init=False, repr=False)
    rng: np.random.Generator = field(init=False, repr=False)
    t: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not (
            0.0 <= self.c < 1.0 and 0.0 <= self.q < 1.0 and (self.c + self.q) < 1.0
        ):
            raise ValueError("Require 0<=c<1, 0<=q<1 and c+q<1 (see paper).")
        if not (0.0 < self.rho < 1.0):
            raise ValueError("Require 0<rho<1 (inertia; see paper).")
        self.bounds = np.asarray(self.bounds, dtype=float)
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise ValueError("bounds must be an array of shape (d,2).")
        if np.any(self.bounds[:, 1] <= self.bounds[:, 0]):
            raise ValueError("Each row of bounds must satisfy hi > lo.")
        self.reset()

    def reset(self) -> None:
        self.d = self.bounds.shape[0]
        self.rng = _rng(self.seed)
        self.X = _sample_in_box(self.rng, self.bounds, self.N)
        # velocities
        span = (self.bounds[:, 1] - self.bounds[:, 0]).astype(float)
        std = np.where(span * self.v_init_scale <= 0.0, 1.0, span * self.v_init_scale)
        self.V = self.rng.normal(0.0, 1.0, size=(self.N, self.d)) * std

        self.losses = np.apply_along_axis(self.func, 1, self.X).astype(float)
        self.prev_losses = self.losses.copy()
        best_idx = int(np.argmin(self.losses))
        self.queen_pos = self.X[best_idx].copy()
        self.queen_loss = float(self.losses[best_idx])
        self.pheromones = []
        self.queen_history = [self.queen_loss]
        self.best_pos_history = [self.queen_pos.copy()]
        self.history_losses = []
        if self.d == 2:
            self.history_positions = []
            self.history_pheromones = []
        self.t = 0
        # stochastic per-bee weights (from your code)
        self.c = np.random.normal(self.c, 0.1, (self.X.shape[0], 1))
        self.q = np.random.normal(self.q, 0.02, (self.X.shape[0], 1))

    def optimize(self) -> Dict[str, Any]:
        import sys

        no_improve_steps = 0
        for step in range(1, self.max_steps + 1):
            self.t = step
            self._drop_pheromones()
            attraction = self._pheromone_attraction()

            self.V = (
                self.rho * self.V
                + self.c * attraction
                + self.q * (self.queen_pos - self.X)
            )
            self.X = self.X + self.V * self.dt
            self._apply_bounds()

            self.prev_losses, self.losses = (
                self.losses,
                np.apply_along_axis(self.func, 1, self.X).astype(float),
            )
            self.history_losses.append(self.losses)
            if self.d == 2:
                self.history_positions.append(self.X)
                self.history_pheromones.append(self.pheromones)
            best_idx = int(np.argmin(self.losses))
            best_loss = float(self.losses[best_idx])
            if best_loss < self.queen_loss:
                self.queen_pos = self.X[best_idx].copy()
                self.queen_loss = best_loss
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            self.queen_history.append(self.queen_loss)
            self.best_pos_history.append(self.queen_pos.copy())

            # progress
            fraction = (self.t + 1) / self.max_steps
            bar_length = 30
            filled_length = int(bar_length * fraction)
            bar = "#" * filled_length + "-" * (bar_length - filled_length)
            msg = f"\r[bho          ] [{bar}] Best Loss: {self.queen_loss:.6f}"
            sys.stdout.write(msg)
            sys.stdout.flush()

            if (
                self.tol_abs is not None
                and self.queen_loss <= self.tol_abs
                and self.earlystop
            ):
                break
            if self.patience is not None and no_improve_steps >= self.patience:
                no_improve_steps = 0
                dt_old = self.dt
                self.dt /= 2
                print(f"\n [OVERHEATED]: Reducing dt: {dt_old} --> {self.dt}")

        sys.stdout.write("\n")
        return {
            "best_x": self.queen_pos.copy(),
            "best_loss": float(self.queen_loss),
            "loss_curve": np.array(self.queen_history, dtype=float),
            "best_pos_history": np.vstack(self.best_pos_history),
            "history_losses": np.array(self.history_losses, dtype=float),
            "steps": self.t,
        }

    def _drop_pheromones(self) -> None:
        improved = self.prev_losses > self.losses + 0.0
        if not np.any(improved):
            return
        eps = 1e-12
        for i in np.where(improved)[0]:
            L_prev = float(self.prev_losses[i])
            L_curr = float(self.losses[i])
            if L_curr <= 0.0:
                s = (L_prev - L_curr) / (abs(L_curr) + eps)
            else:
                s = (L_prev - L_curr) / (L_curr + eps)
            pher = Pheromone(
                position=self.X[i].copy(), base_strength=float(max(0.0, s)), T=self.t
            )
            self.pheromones.append(pher)

        if (
            self.pheromone_capacity is not None
            and len(self.pheromones) > self.pheromone_capacity
        ):
            strengths = np.array(
                [p.strength_at(self.t, self.gamma) for p in self.pheromones]
            )
            keep_idx = np.argsort(strengths)[-self.pheromone_capacity :]
            self.pheromones = [self.pheromones[i] for i in keep_idx]

    def _pheromone_attraction(self) -> Array:
        N, d = self.X.shape
        if len(self.pheromones) == 0:
            return np.zeros_like(self.X)
        P = np.vstack([p.position for p in self.pheromones])
        S = np.array(
            [p.strength_at(self.t, self.gamma) for p in self.pheromones], dtype=float
        )
        eps = 1e-12
        attraction = np.zeros_like(self.X)
        for i in range(N):
            diff = P - self.X[i]
            dist2 = np.maximum(np.sum(diff * diff, axis=1), eps)
            F = S / dist2
            F_sum = float(np.sum(F))
            if F_sum <= 0.0 or not np.isfinite(F_sum):
                continue
            Psi = F / F_sum
            attraction[i] = np.einsum("m,md->d", Psi, diff, optimize=True)
        return attraction * self.kappa

    def _apply_bounds(self) -> None:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        if self.bound_strategy == "clip":
            self.X = np.clip(self.X, lo, hi)
            return
        if self.bound_strategy == "reflect":
            for i in range(self.X.shape[0]):
                for d in range(self.X.shape[1]):
                    if self.X[i, d] < lo[d]:
                        over = lo[d] - self.X[i, d]
                        self.X[i, d] = lo[d] + over
                        self.V[i, d] = -0.5 * self.V[i, d]
                    elif self.X[i, d] > hi[d]:
                        over = self.X[i, d] - hi[d]
                        self.X[i, d] = hi[d] - over
                        self.V[i, d] = -0.5 * self.V[i, d]
            self.X = np.clip(self.X, lo, hi)
            return
        raise ValueError("Unknown bound_strategy: %r" % (self.bound_strategy,))


# ---- End user's BHO class ----


def run_bho(
    f: Objective,
    bounds: Array,
    budget: int,
    seed: Optional[int] = None,
    n_particles: int = 60,
    n_iterations: Optional[int] = None,
    c: float = 0.9499,
    q: float = 0.05,
    rho: float = 0.99,
    gamma: float = 0.7,
    dt: float = 0.5,
    kappa: float = 1.0,
    v_init_scale: float = 0.10,
    pheromone_capacity: Optional[int] = None,
    tol_abs: float = 1e-100,
    patience: int = 200,
    earlystop: bool = True,
) -> OptResult:
    """
    Wraps the provided BHO to respect a function-evaluation budget.
    Each step evaluates ~N points; initialization evaluates N points.
    """
    N = int(n_particles)
    if N <= 0:
        raise ValueError("n_particles must be > 0")

    # Map budget -> steps: budget ≈ N (init) + steps * N
    if n_iterations is None:
        steps = max(1, (budget // N) - 1)
    else:
        steps = int(n_iterations)
        # If user sets steps too high for the budget, the EvalCounter will cap further evals

    wrapped = EvalCounter(f, budget)
    # Instantiate user BHO with the wrapped objective
    bho = BeehiveOptimization(
        func=lambda x: wrapped(x),
        bounds=np.asarray(bounds, dtype=float),
        N=N,
        c=c,
        q=q,
        rho=rho,
        gamma=gamma,
        dt=dt,
        kappa=kappa,
        max_steps=steps,
        seed=seed,
        v_init_scale=v_init_scale,
        pheromone_capacity=(
            pheromone_capacity if pheromone_capacity is not None else N * 10
        ),
        tol_abs=tol_abs,
        patience=patience,
        earlystop=earlystop,
    )
    res = bho.optimize()
    xbest = res["best_x"]
    fbest = float(res["best_loss"])
    return OptResult(xbest, fbest, wrapped.n, trace=list(res["loss_curve"]))


class GBO:
    def __init__(
        self,
        loss,
        dim,
        budget,
        final_epsilon=0.001,
        init_radius=3,
        init_temperature=2,
        graph_max_size=50,
        prune_every=None,
        bounds=None,
        seed=None,
    ):
        self.loss_func = loss
        self.dim = dim
        self.budget = budget
        self.bounds = bounds
        self.radius = init_radius
        self.epsilon_decay = (final_epsilon) ** (1 / budget)
        self.epsilon = 1
        self.graph_max_size = graph_max_size
        if prune_every is None:
            self.prune_every = graph_max_size // 10
        else:
            self.prune_every = prune_every

        self.temperature = init_temperature
        self.best_loss = np.inf
        self.best_pos = np.zeros(dim)
        self.best_node_id = 0
        self.loss_history = []
        self.graph_history = []
        self.best_pos_history = []
        self.edge_list = []
        self.losses = []
        self.G = nx.DiGraph()
        self.node_count = 0
        if seed is not None:
            np.random.seed(seed)

    def _make_node_attributes(self, pos):
        loss = self.loss_func(pos)
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_pos = pos
            self.best_node_id = len(self.G)
        return {"pos": pos, "loss": loss}

    def _add_node(self, pos):
        # creation of node attributes

        id = self.node_count
        self.node_count += 1
        loss = self.loss_func(pos)
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_pos = pos
            self.best_node_id = id
        self.losses.append(loss)
        node_attributes = {"pos": pos, "loss": loss}
        # adding node to graph
        self.G.add_node(id, **node_attributes)
        self.G.add_edge(id, id, weight=node_attributes["loss"])
        pos = node_attributes["pos"]
        nodes = self.G.nodes(data=True)
        for nodeId in self.G.nodes():
            if ((nodes[nodeId]["pos"] - pos) ** 2).sum() ** (1 / 2) < self.radius:
                self.G.add_edge(id, nodeId, weight=nodes[nodeId]["loss"])

    def _random_guess(self):
        #        if np.random.uniform() < 0.5:
        #            centrality = nx.degree_centrality(self.G)
        #            print(centrality)
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        return np.random.uniform(low, high, self.dim)

    def _project(self, x):
        if self.bounds is None:
            return x
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return np.minimum(hi, np.maximum(lo, x))

    def _smart_guess(self):
        # this will proportionaly pick larger clusters.
        gen = getattr(self, "_gen", np.random.default_rng())
        # pick a start node, mildly favoring good ones
        nodes = np.array(list(self.G.nodes()))
        losses = np.array([self.G.nodes[n]["loss"] for n in nodes], float)
        prob = 1.0 / (losses - losses.min() + 1.0)
        prob /= prob.sum()
        random_node = int(gen.choice(nodes, p=prob))
        edges = self.G.edges(random_node, data=True)  # [random_node]
        force = np.zeros(self.dim)
        weights = np.zeros(len(edges))
        targets = np.zeros(len(edges))
        counter = 0
        for u, v, weight in edges:
            weights[counter] = weight["weight"]
            targets[counter] = v
            counter += 1
        weights = 1 / weights
        weights = weights - weights.min() + 1
        weights = weights / weights.sum()
        counter = 0
        nodes = self.G.nodes()
        for u, v, weight in edges:
            vec = nodes[v]["pos"] - nodes[u]["pos"]
            if (vec == np.zeros(self.dim)).sum() == self.dim:
                force += np.random.uniform(-1, 1, self.dim)
            force += vec * weights[counter] * np.random.uniform(0, 2, self.dim)
            counter += 1
        final_pos = nodes[random_node]["pos"] + force * self.temperature
        return self._project(final_pos)

    def _initialize(self):
        if self.bounds is None:
            first_node_pos = np.random.uniform(-10, 10, self.dim)
        else:
            first_node_pos = self._random_guess()
        self._add_node(first_node_pos)

    def _prune_graph(self):
        nodes = self.G.nodes(data=True)
        losses = []
        node_id = []
        for node in nodes:
            node_id.append(node[0])
            losses.append(node[1]["loss"])
        remove_nodes = np.argsort(losses)[self.graph_max_size :]
        node_id = np.array(node_id)
        for remove_node in remove_nodes:
            self.G.remove_node(node_id[remove_node])
        nodes = self.G.nodes(data=True)
        remove_edges = []
        for (
            u,
            v,
        ) in self.G.edges():
            if ((nodes[u]["pos"] - nodes[v]["pos"]) ** 2).sum() ** (
                1 / 2
            ) > self.radius:
                remove_edges.append((u, v))
        self.G.remove_edges_from(remove_edges)
        self.losses = []
        for node in self.G.nodes(data=True):
            self.losses.append(node[1]["loss"])

    def optimize(self):
        self._initialize()
        for i in range(self.budget):
            if np.random.uniform() < self.epsilon:
                guess = self._random_guess()
            else:
                guess = self._smart_guess()
            self._add_node(guess)
            self.temperature *= self.epsilon_decay
            self.epsilon *= self.epsilon_decay
            if i % self.prune_every == 0:
                self._prune_graph()


def run_gbo_rgg(
    f: Objective,
    bounds: Array,
    budget: int,
    seed: Optional[int] = None,
    *,
    # expose the knobs your GBO supports:
    final_epsilon: float = 0.001,
    init_radius: float = 2.0,
    graph_max_size: int = 100,
    prune_every: Optional[int] = None,
    # If your GBO lives in a different module, adjust the import below
    _gbo_ctor=None,  # for dependency injection / testing (optional)
) -> OptResult:
    """
    Wrapper for the new graph-based optimizer (class GBO) with a consistent interface.

    Notes
    -----
    - Your GBO does 1 evaluation in _initialize() + 1 per step in the for-loop.
      We therefore run it for `budget - 1` steps so the total budget ≈ `budget`.
    - We pass the objective through EvalCounter to hard-cap the budget.
    - We do NOT attach our own progress bar here because your GBO already prints one.
      (If you prefer the unified bar, just set `bar=_TextBar(... )` below.)
    """
    # Lazy import here so this wrapper file doesn't hard-depend on your class location

    d = int(bounds.shape[0])

    # Drive budget & progress consistently with the rest of your wrappers
    # (we avoid a second bar because GBO already shows one)
    bar = _TextBar(total=budget, label="gbo_rgg")
    wrapped = EvalCounter(f, budget, bar=bar)

    # Make the internal loop use budget-1 so total evals ≈ budget
    internal_budget = max(1, int(budget) - 1)

    opt = GBO(
        loss=lambda x: wrapped(x),  # EvalCounter enforces the eval budget
        dim=d,
        budget=internal_budget,
        final_epsilon=final_epsilon,
        init_radius=init_radius,
        graph_max_size=graph_max_size,
        prune_every=prune_every,
        bounds=np.asarray(bounds, dtype=float),
        seed=seed,
    )

    opt.optimize()

    # Collect results in the unified format
    xbest = np.asarray(opt.best_pos, dtype=float)
    fbest = float(opt.best_loss)
    nfev = int(wrapped.n)

    # If you ever record per-step bests inside GBO, you can return them; for now, empty.
    bar.close()
    return OptResult(xbest, fbest, nfev, trace=[])


class PGO:
    def __init__(
        self,
        loss,  # loss function
        budget,  # loss evals
        bounds,  # constrained optimization
        graph_max_size=50,  # max graph size
        init_temperature=2,  # temperature of the system -- annealing
        final_temperature=None,
        final_epsilon=0.01,  # final probability of randomly selecting a point
        seed=None,
    ):
        # set parameters
        self.loss_func = loss
        self.dim = len(bounds)
        self.budget = budget
        self.graph_max_size = graph_max_size
        self.final_epsilon = final_epsilon
        self.bounds = bounds
        if init_temperature is None:
            self.temperature = (bounds.max(axis=-1) - bounds.min(axis=-1)).max() / 5
        else:
            self.temperature = init_temperature
        if final_temperature is None:
            self.final_temperature = self.temperature / 100
        else:
            self.final_temperature = final_temperature
        self.init_temperature = init_temperature
        self.temperature_decay = (self.final_temperature / self.temperature) ** (
            1 / self.budget
        )
        # annealing parameters
        self.epsilon = 1
        self.epsilon_decay = (final_epsilon) ** (1 / budget)

        # initialization of variables
        self.best_loss = np.inf
        self.best_pos = np.zeros(self.dim)
        self.loss_history = []
        self.best_pos_history = []
        self.losses = dict()
        self.G = nx.DiGraph()
        self.current_node_id = 0
        self.best_node_id = 0
        self.num_evals = 0
        if seed is not None:
            np.random.seed(seed)
        self.time_since_improved = 0

    def _project(self, x):
        if self.bounds is None:
            return x
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        y = np.array(x, dtype=float, copy=True)
        for i in range(self.dim):
            a, b = lo[i], hi[i]
            w = b - a
            if w <= 0:  # degenerate bound
                y[i] = a
                continue
            t = (y[i] - a) % (2.0 * w)
            y[i] = (a + t) if t <= w else (b - (t - w))
        return y

    def _sample_in_cone(self, mu, sigma=0.3):
        """
        mu: unit vector (mean direction)
        sigma: spread (0 = deterministic, higher = wider cone)
        """
        d = mu.size
        z = np.random.normal(size=d)
        # remove parallel component => orthogonal noise
        z = z - mu * np.dot(mu, z)
        nz = np.linalg.norm(z)
        if nz < 1e-12:
            return mu  # degenerate, just return mean
        z /= nz
        v = mu + sigma * z
        nv = np.linalg.norm(v)
        return v / (nv + 1e-12)

    ### HELPER FUNCTIONS ###
    def _make_node_attributes(self, pos):
        loss = self.loss_func(pos)
        self.losses[self.current_node_id] = loss
        if loss < self.best_loss:
            self.time_since_improved = 0
            self.best_loss = loss
            self.best_pos = pos
            self.best_node_id = self.current_node_id
        return {"pos": pos, "loss": loss}

    def _add_node(self, pos, target_node=-1):
        pos = self._project(pos)
        node_attributes = self._make_node_attributes(pos)
        # addding node
        self.G.add_node(self.current_node_id, **node_attributes)
        # adding edge
        if target_node > 0:
            # link is pointing from attached node to target node
            gradient = self.G.nodes[target_node]["loss"] - node_attributes["loss"]
            distance = (
                np.linalg.norm(
                    self.G.nodes[target_node]["pos"] - node_attributes["pos"]
                )
                + 1e-9
            )
            self.G.add_edge(
                self.current_node_id,
                target_node,
                weight=gradient / distance,
            )
            self.G.add_edge(
                target_node,
                self.current_node_id,
                weight=-gradient / distance,
            )
        self.current_node_id += 1

    def _add_random_edge(self, random_node, weights):
        target_node = int(np.random.choice(list(self.losses.keys()), p=weights))
        while (
            target_node == random_node
        ):  # TODO: or self.G.has_edge(random_node, target_node):
            target_node = int(np.random.choice(list(self.losses.keys()), p=weights))
        weight = self.G.nodes[target_node]["loss"] - self.G.nodes[random_node]["loss"]
        distance = np.linalg.norm(
            self.G.nodes[target_node]["pos"] - self.G.nodes[random_node]["pos"]
        )
        self.G.add_edge(random_node, target_node, weight=weight / distance)
        self.G.add_edge(target_node, random_node, weight=-weight / distance)

    def _random_guess(self):
        low, high = self.bounds[:, 0], self.bounds[:, 1]
        if np.random.uniform() < 0.5 and len(self.G) > self.graph_max_size // 2:
            best_pos = self.best_pos
            sigma = self.temperature  # controls spread
            chosen_idx = np.random.choice(len(self.losses))  # or pick best_node_id
            target_pos = self.G.nodes[self.best_node_id]["pos"]
            # Sample around best_pos
            pos = np.random.normal(loc=target_pos, scale=sigma, size=self.dim)
        else:
            pos = np.random.uniform(low, high, low.shape[0])
        self._add_node(pos)

    def _local_guess(self, chosen_node):
        force_vector = np.zeros(self.dim)
        for node, data in self.G.nodes(data=True):
            vec = data["pos"] - self.G.nodes[chosen_node]["pos"]
            if np.linalg.norm(vec) < self.temperature:
                g = data["loss"] - self.G.nodes[chosen_node]["loss"]
                force_vector += -g * vec
        norm = np.linalg.norm(force_vector)
        if norm == 0:
            return False
        else:
            force_vector = (force_vector / norm) * self.temperature
        pos = self._project(self.G.nodes[chosen_node]["pos"] + force_vector)
        self._add_node(pos)
        return True

    def _smart_guess(self):
        loss_array = np.array(list(self.losses.values()))
        mag_min = np.abs(loss_array.min())
        loss_array = np.sqrt((loss_array - loss_array.min()) + mag_min)
        loss_array = 1 / loss_array
        # TODO: !!!!!!!!!!!!!!!improve this... its VERY important!!!!!!!!!!!!!!
        # loss_array = np.log1p(loss_array - loss_array.min()) + self.temperature
        # loss_array = (1 / loss_array) ** (0.5 + self.current_node_id / self.budget)
        #        if np.any(loss_array < 0):
        #            loss_array = loss_array - loss_array.min() + np.abs(loss_array).mean()
        #        loss_array = 1 / loss_array
        prob_draw_array = loss_array / loss_array.sum()
        # TODO: !!!!!!!!!!!improve this above.... its VERY important!!!!!!!!!!!

        key = np.argmax(prob_draw_array)
        if list(self.losses.keys())[key] != self.best_node_id:
            raise ValueError("oops... best node is not in our space... why...?")
        # TODO: we can convert loss_array to something like a log space / sqrt space for more thorough search
        target_node = int(np.random.choice(list(self.losses.keys()), p=prob_draw_array))
        if np.random.uniform() < 0.1:
            self._add_random_edge(target_node, prob_draw_array)

        # get pos
        target_node_pos = self.G.nodes[target_node]["pos"]
        target_node_edges = self.G.edges(target_node, data=True)
        if target_node_edges:
            # TODO: do we get equal contribution from all or should it be based on edge wegiht?
            num_edges = len(target_node_edges)
            smart_direction = np.zeros(self.dim)
            for u, v, gradient in target_node_edges:
                if u == v:
                    continue
                vec = self.G.nodes[v]["pos"] - self.G.nodes[u]["pos"]
                norm = np.linalg.norm(vec)
                if norm < 1e-12:
                    pos = np.random.normal(
                        loc=self.G.nodes[u]["pos"],
                        scale=self.temperature * 2,
                        size=self.dim,
                    )
                    smart_direction += self.G.nodes[u]["pos"]
                else:
                    vec = vec / norm
                    smart_direction += gradient["weight"] * vec
            norm = np.linalg.norm(smart_direction)

            if norm == 0:
                norm = 1
            smart_direction = smart_direction / norm
            smart_direction = self._sample_in_cone(
                smart_direction, sigma=min(self.temperature, 0.5)
            )
        else:
            smart_direction = np.random.uniform(-1, 1, self.dim) * self.temperature

        pos = (
            target_node_pos
            + smart_direction * self.temperature
            + np.random.uniform(-1, 1, self.dim) * self.temperature
        )

        self._add_node(pos, target_node)

    def _prune_graph(self):
        min_loss = -np.inf
        remove_node = -1
        for node, loss in self.losses.items():
            if loss > min_loss:
                remove_node = node
                min_loss = loss

        if remove_node == self.best_node_id:
            remove_node = list(self.losses.keys())[-1]
            self.temperature *= 5
            self.epsilon *= 5
            print(self.G.nodes(data=True))
            print("oh no... all of our nodes are the same position...")
            # raise ValueError("oh no... all of our nodes are the same position...")
        self.G.remove_node(remove_node)
        self.losses.pop(remove_node)

    def optimize(self):
        for t in range(self.budget + 1):
            self.time_since_improved += 1
            if np.random.uniform() < self.epsilon:
                self._random_guess()
            else:
                self._smart_guess()
            self.epsilon *= self.epsilon_decay
            # TODO: change to anneal with temperature rate rather than epsilon decay
            self.temperature *= self.temperature_decay
            if len(self.G) > self.graph_max_size:
                self._prune_graph()


def run_pgo(
    f: Objective,
    bounds: Array,
    budget: int,
    seed: Optional[int] = None,
    *,
    # expose the knobs your GBO supports:
    final_epsilon: float = 0.01,
    init_temperature: float = 2.0,
    graph_max_size: int = 100,
    final_temperature: Optional[float] = None,
    # If your GBO lives in a different module, adjust the import below
    _gbo_ctor=None,  # for dependency injection / testing (optional)
) -> OptResult:
    """
    Wrapper for the new graph-based optimizer (class GBO) with a consistent interface.

    Notes
    -----
    - Your GBO does 1 evaluation in _initialize() + 1 per step in the for-loop.
      We therefore run it for `budget - 1` steps so the total budget ≈ `budget`.
    - We pass the objective through EvalCounter to hard-cap the budget.
    - We do NOT attach our own progress bar here because your GBO already prints one.
      (If you prefer the unified bar, just set `bar=_TextBar(... )` below.)
    """
    # Lazy import here so this wrapper file doesn't hard-depend on your class location

    d = int(bounds.shape[0])

    # Drive budget & progress consistently with the rest of your wrappers
    # (we avoid a second bar because GBO already shows one)
    bar = _TextBar(total=budget, label="pgo")
    wrapped = EvalCounter(f, budget, bar=bar)

    # Make the internal loop use budget-1 so total evals ≈ budget
    internal_budget = max(1, int(budget) - 1)

    opt = PGO(
        loss=lambda x: wrapped(x),  # EvalCounter enforces the eval budget
        bounds=np.asarray(bounds, dtype=float),
        budget=internal_budget,
        final_epsilon=final_epsilon,
        graph_max_size=graph_max_size,
        init_temperature=init_temperature,
        final_temperature=final_temperature,
        seed=seed,
    )

    opt.optimize()

    # Collect results in the unified format
    xbest = np.asarray(opt.best_pos, dtype=float)
    fbest = float(opt.best_loss)
    nfev = int(wrapped.n)

    # If you ever record per-step bests inside GBO, you can return them; for now, empty.
    bar.close()
    return OptResult(xbest, fbest, nfev, trace=[])
