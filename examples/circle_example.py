from __future__ import annotations
from typing import Callable, Dict, Tuple, List
import numpy as np
import time
import sys
from OptimizationFramework.plottingUtils import PlotDefaults

Array = np.ndarray
Objective = Callable[[Array], float]

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# =============================================================================
#                               Heatmap generator
# =============================================================================


def random_seeder(dim: int, time_steps: int = 10000) -> Array:
    x = np.random.uniform(0, 1, (dim, dim))
    seed_pos_x = int(np.random.uniform(0, dim))
    seed_pos_y = int(np.random.uniform(0, dim))
    tele_prob = 0.001
    for _ in range(time_steps):
        x[seed_pos_x, seed_pos_y] += np.random.uniform(0, 1)
        if np.random.uniform() < tele_prob:
            seed_pos_x = int(np.random.uniform(0, dim))
            seed_pos_y = int(np.random.uniform(0, dim))
        else:
            if np.random.uniform() < 0.5:
                seed_pos_x += 1
            if np.random.uniform() < 0.5:
                seed_pos_x += -1
            if np.random.uniform() < 0.5:
                seed_pos_y += 1
            if np.random.uniform() < 0.5:
                seed_pos_y += -1
            seed_pos_x = int(max(min(seed_pos_x, dim - 1), 0))
            seed_pos_y = int(max(min(seed_pos_y, dim - 1), 0))
    return x


# =============================================================================
#                      Objective: union coverage of disks
# =============================================================================


def disk_kernel(radius: int) -> Array:
    r = int(radius)
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= r * r


def _stamp_or(mask: np.ndarray, center: Tuple[int, int], disk: np.ndarray):
    """OR a centered binary disk into a boolean mask (handles image edges)."""
    H, W = mask.shape
    r = disk.shape[0] // 2
    cy, cx = center
    y0, y1 = cy - r, cy + r + 1
    x0, x1 = cx - r, cx + r + 1
    iy0, iy1 = max(0, y0), min(H, y1)
    ix0, ix1 = max(0, x0), min(W, x1)
    if iy0 >= iy1 or ix0 >= ix1:
        return
    my0, my1 = iy0 - y0, iy1 - y0
    mx0, mx1 = ix0 - x0, ix1 - x0
    m = disk[my0:my1, mx0:mx1]
    mask[iy0:iy1, ix0:ix1] |= m


def decode_centers(x: Array, dim: int) -> List[Tuple[int, int]]:
    """x shape (2K,), -> [(row,col), ...]; rounded and clipped to [0, dim-1]."""
    xi = np.clip(np.round(x).astype(int), 0, dim - 1)
    return [(int(xi[2 * i]), int(xi[2 * i + 1])) for i in range(xi.size // 2)]


def make_union_coverage_loss(
    weights: Array, radii: List[int]
) -> Tuple[Objective, List[Array]]:
    """
    Returns loss(x) that NEGATES the total covered weight (we minimize).
    x packs centers as [y0,x0,y1,x1,...].
    """
    H, W = weights.shape
    disks = [disk_kernel(r) for r in radii]

    def loss(x_vec: Array) -> float:
        centers = decode_centers(x_vec, H)
        covered = np.zeros_like(weights, dtype=bool)
        for (cy, cx), disk in zip(centers, disks):
            _stamp_or(covered, (cy, cx), disk)
        score = float(weights[covered].sum())  # we want to maximize this
        return -score  # minimize

    return loss, disks


def render_union_mask(
    dim: int, centers: List[Tuple[int, int]], radii: List[int]
) -> np.ndarray:
    mask = np.zeros((dim, dim), dtype=bool)
    for (cy, cx), r in zip(centers, radii):
        _stamp_or(mask, (cy, cx), disk_kernel(r))
    return mask


# =============================================================================
#                             Method registry
# =============================================================================

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

METHODS: Dict[str, Callable[..., OptResult]] = {
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


# =============================================================================
#                                  Plot
# =============================================================================


def plot_solution(
    weights: Array,
    centers: List[Tuple[int, int]],
    radii: List[int],
    title: str = "",
    name: str = "",
):
    PlotDefaults()
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.2))
    im = ax.imshow(weights, cmap="magma", origin="upper")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="weight")
    for (cy, cx), r in zip(centers, radii):
        circ = Circle((cx, cy), r, fill=False, lw=2.0, ec="cyan")
        ax.add_patch(circ)
        ax.plot([cx], [cy], "o", ms=3, color="white")
    ax.set_title(title)
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")
    ax.set_xlim(-0.5, weights.shape[1] - 0.5)
    ax.set_ylim(weights.shape[0] - 0.5, -0.5)
    plt.tight_layout()
    fig.savefig(f"circle_placement_winner_{name}.png", dpi=300)
    plt.show()


# =============================================================================
#                                    Main
# =============================================================================

if __name__ == "__main__":
    # Problem configuration
    SEED = np.random.randint(100000)
    DIM = 64
    STEPS = 100_000
    RADII = [12, 10, 10, 8, 6, 6, 4]  # 7 circles
    K = len(RADII)
    DVAR = 2 * K  # y,x per circle

    # Generate weights
    np.random.seed(SEED)
    weights = random_seeder(DIM, time_steps=STEPS)
    weights = weights / (weights.max() + 1e-12)

    # Loss: NEGATIVE union coverage
    loss, _ = make_union_coverage_loss(weights, RADII)

    # Box bounds for optimizers (continuous; loss handles rounding/clipping)
    bounds = np.array([[0, DIM - 1]] * DVAR, dtype=float)

    # Shared evaluation budget
    budget = 5000

    # Methods to run (all)
    order = [
        "pgo",
        "cma",
        "de",
        "ga",
        "abc",
        "dual_anneal",
        "nelder_mead",
        "powell",
        "pso",
        "bho",
    ]

    results = []
    for name in order:
        runner = METHODS[name]
        t0 = time.time()
        try:
            if name == "pgo":
                res = runner(
                    loss,
                    bounds,
                    budget,
                    graph_max_size=25,
                    init_temperature=5,
                    final_temperature=0.8,
                    final_epsilon=0.001,
                )
            elif name == "pso":
                res = runner(
                    loss,
                    bounds,
                    budget,
                    seed=SEED,
                    n_particles=60,
                    w=0.9,
                    c1=1.5,
                    c2=1.5,
                    v_init_scale=0.10,
                    vmax_frac=0.20,
                )
            elif name == "bho":
                # default parameter setting)
                n_particles = int(np.sqrt(budget)) // 2
                rho = (0.95) ** (200 / (budget / n_particles))
                if rho < 0.90:
                    rho = 0.90
                res = runner(
                    loss,
                    bounds,
                    budget,
                    seed=SEED,
                    n_particles=n_particles,
                    c=0.9499,
                    q=0.05,
                    rho=rho,
                    gamma=0.7,
                    dt=0.2,
                    v_init_scale=2,
                    patience=100,
                    earlystop=False,
                )
            else:
                res = runner(loss, bounds, budget, seed=SEED)
        except RuntimeError as e:
            print(f"[{name}] skipped: {e}")
            continue
        dt = time.time() - t0
        score = -float(res.fx)  # flip sign: higher is better
        centers = decode_centers(res.x, DIM)
        print(
            f"[{name:12s}] score={score:.6f}  nfev={res.nfev:6d}  time={dt:6.2f}s  centers={centers}"
        )
        results.append((name, score, res, dt))

    if not results:
        print("No methods ran (missing dependencies?).")
        sys.exit(0)

    # Rank by score (descending)
    results.sort(key=lambda t: t[1], reverse=True)
    print("\n=== Scoreboard (higher is better) ===")
    for i, (name, score, res, dt) in enumerate(results, 1):
        print(
            f"{i:2d}. {name:12s}  score={score:.6f}  nfev={res.nfev:6d}  time={dt:6.2f}s"
        )

    # Plot best
    best_name, best_score, best_res, best_dt = results[0]
    best_centers = decode_centers(best_res.x, DIM)
    print(f"\nWinner: {best_name}  score={best_score:.6f}")
    plot_solution(
        weights,
        best_centers,
        RADII,
        title=f"{best_name} — union weight={best_score:.4f}",
        name=best_name,
    )
