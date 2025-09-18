from __future__ import annotations
from typing import Callable, Dict, Any, List, Tuple
import numpy as np
import time

Array = np.ndarray
Objective = Callable[[Array], float]

# ======================================================================
#                           Method registry
# ======================================================================

from OptimizationFramework.optimizers import (
    run_cma,
    run_de_scipy,
    run_ga_sbx,
    run_abc,
    run_dual_annealing,
    run_nelder_mead,
    run_powell,
    run_pso,
    run_bho,
    run_pgo,
)

from OptimizationFramework.lossFunctions import (
    rastrigin,
    ackley,
    rosenbrock,
    ellipsoid,
    sphere,
)

METHODS: Dict[str, Callable[..., Any]] = {
    "cma": run_cma,
    "de": run_de_scipy,
    "ga": run_ga_sbx,
    "abc": run_abc,
    "dual_anneal": run_dual_annealing,
    "nelder_mead": run_nelder_mead,
    "powell": run_powell,
    "pso": run_pso,
    "bho": run_bho,
    "pgo": run_pgo,
}


# Pretty nfev formatting (handles methods that don't return it)
def _fmt_nfev(res: Any) -> str:
    nfev = getattr(res, "nfev", None)
    try:
        return f"{int(nfev):6d}"
    except Exception:
        return f"{'n/a':>6s}"


# Optional: plot best (circle-packing context) if helpers are available
def _maybe_plot_best(best_name: str, best_score: float, best_res: Any) -> None:
    try:
        # these must exist in the global namespace for plotting to work
        decode_centers  # type: ignore[name-defined]
        plot_solution  # type: ignore[name-defined]
        weights  # type: ignore[name-defined]
        RADII  # type: ignore[name-defined]
        DIM  # type: ignore[name-defined]
    except NameError:
        return
    try:
        best_centers = decode_centers(best_res.x, DIM)  # type: ignore[name-defined]
        title = f"{best_name} — union weight={best_score:.4f}"
        plot_solution(  # type: ignore[name-defined]
            weights, best_centers, RADII, title=title, name=best_name
        )
    except Exception as e:
        print(f"[plot] skipped ({e})")


# ======================================================================
#                           Quick demo
# ======================================================================

if __name__ == "__main__":
    # TODO: preferential attachment (barabasi-albert kbar 1).
    # we do preferential attachment according to proportional loss,
    # we remove nodes via degree-based domicoarse every n (or maybe even every single)
    # iteration. Selfloops will be the value to do preferential attachment for. and it should be something like 1/loss. That way, if a new point is found, even though it hasn't absorbed any nodes, it can explode in terms of importance and become the sought-after point. Now, with the radius of exploration, it should either be a simulated annealing kind of thing or it should be somehow based on 1/selfloop value. (gravitational attraction kind of thing).

    # Problem
    seed = np.random.randint(100000)
    d = 5
    bounds = np.array([[-5.12, 5.12]] * d, dtype=float)
    f: Objective = rastrigin
    budget = 1000

    to_run = [
        "cma",
        "de",
        "ga",
        "abc",
        "dual_anneal",
        "nelder_mead",
        "powell",
        "pso",
        "pgo",
        "bho",
    ]

    # Collect results as (name, score, res, dt)
    results: List[Tuple[str, float, Any, float]] = []

    for name in to_run:
        runner = METHODS[name]
        t0 = time.time()
        try:
            if name == "pgo":
                res = runner(
                    f,
                    bounds,
                    budget,
                    graph_max_size=50,
                    init_temperature=2,
                    final_temperature=0.01,
                )
            elif name == "pso":
                res = runner(
                    f,
                    bounds,
                    budget,
                    seed=seed,
                    n_particles=20,
                    w=0.7,
                    c1=1.5,
                    c2=1.5,
                    v_init_scale=0.10,
                    vmax_frac=0.2,
                )
            elif name == "bho":
                # default parameter setting
                n_particles = max(2, int(np.sqrt(budget)) // 2)
                rho = (0.95) ** (200 / max(1, (budget / max(1, n_particles))))
                if rho < 0.90:
                    rho = 0.90
                res = runner(
                    f,
                    bounds,
                    budget,
                    seed=seed,
                    n_particles=20,
                    c=0.9499,
                    q=0.05,
                    rho=rho,
                    gamma=0.7,
                    dt=0.5,
                    kappa=1.0,
                    v_init_scale=0.50,
                    patience=200,
                )
            else:
                res = runner(f, bounds, budget, seed=seed)
        except RuntimeError as e:
            print(f"[{name}] skipped: {e}")
            continue

        dt = time.time() - t0
        fx = float(getattr(res, "fx", np.nan))
        # Leaderboard uses "higher is better" => score = -fx (we minimize fx)
        score = -fx

        results.append((name, score, res, dt))
        print(
            f"[{name:12s}] fx={fx:.6g}  score={score:.6g}  nfev={_fmt_nfev(res)}  time={dt:6.2f}s"
        )

    if not results:
        raise SystemExit("No successful runs to rank.")

    # -----------------------------
    # Leaderboard (higher is better)
    # -----------------------------
    results.sort(key=lambda t: t[1], reverse=True)
    print("\n=== Scoreboard (higher is better) ===")
    for i, (name, score, res, dt) in enumerate(results, 1):
        print(
            f"{i:2d}. {name:12s}  score={score:.6f}  nfev={_fmt_nfev(res)}  time={dt:6.2f}s"
        )

    # Winner
    best_name, best_score, best_res, best_dt = results[0]
    print(
        f"\nWinner: {best_name}  score={best_score:.6f}  fx={getattr(best_res, 'fx', np.nan):.6f}"
    )

    # Optional plotting hook (only if your circle-packing context is available)
    _maybe_plot_best(best_name, best_score, best_res)
