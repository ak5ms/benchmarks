"""Performance benchmark for the CVXPYgen/Clarabel JAX FFI trade layer."""

from __future__ import annotations

import argparse
import statistics
import time

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from trade_solver import (  # noqa: E402
    build_trade_problem,
    make_trade_cvxpylayer,
    make_trade_layer,
    solve_trade_jit,
)


def make_instance(n: int, seed: int):
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(n, n))
    s = b.T @ b + 0.2 * np.eye(n)
    s_root = np.linalg.cholesky(s).T
    return (
        rng.normal(size=n),
        0.01 + 0.05 * rng.random(n),
        0.1 * rng.normal(size=n),
        s_root,
        np.array([0.5 + rng.random()]),
    )


def direct_cvxpy_solve(args):
    r, hs, w_prev, s_root, sqrt_c = args
    n = r.size
    w = cp.Variable(n)
    dw = cp.Variable(n)
    y = cp.Variable(n)
    problem = cp.Problem(
        cp.Maximize(r @ w - hs @ cp.abs(dw)),
        [dw == w - w_prev, y == s_root @ w, cp.norm(y, 2) <= sqrt_c.item()],
    )
    problem.solve(
        solver="CLARABEL",
        tol_gap_abs=1e-9,
        tol_gap_rel=1e-9,
        tol_feas=1e-9,
        verbose=False,
    )
    return np.asarray(w.value)


class ReusableCvxpyClarabel:
    """Fast Python baseline: one DPP CVXPY problem with Parameters reused."""

    def __init__(self, n: int):
        spec = build_trade_problem(n)
        self.problem = spec.problem
        self.r, self.hs, self.w_prev, self.s_root, self.sqrt_c = spec.parameters
        (self.w,) = spec.variables

    def __call__(self, args):
        r, hs, w_prev, s_root, sqrt_c = args
        self.r.value = r
        self.hs.value = hs
        self.w_prev.value = w_prev
        self.s_root.value = s_root
        self.sqrt_c.value = sqrt_c.item()
        self.problem.solve(
            solver="CLARABEL",
            tol_gap_abs=1e-9,
            tol_gap_rel=1e-9,
            tol_feas=1e-9,
            verbose=False,
        )
        return np.asarray(self.w.value)


def time_many(label: str, fn, inputs, repeat: int):
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        for item in inputs:
            out = fn(item)
            if hasattr(out, "block_until_ready"):
                out.block_until_ready()
        durations.append(time.perf_counter() - start)

    per_solve_ms = [1000.0 * duration / len(inputs) for duration in durations]
    print(
        f"{label:18s} median_ms={statistics.median(per_solve_ms):9.3f} "
        f"min_ms={min(per_solve_ms):9.3f} solves={len(inputs)} repeats={repeat}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=25)
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--grad-instances", type=int, default=10)
    parser.add_argument("--no-grad", action="store_true")
    parser.add_argument("--no-moreau", action="store_true")
    args = parser.parse_args()

    np_inputs = [make_instance(args.n, seed) for seed in range(args.instances)]
    jax_inputs = [tuple(jnp.asarray(x) for x in item) for item in np_inputs]
    reusable_cvxpy = ReusableCvxpyClarabel(args.n)
    layer = make_trade_layer(args.n, n_shards=args.shards)
    jitted_layer = jax.jit(lambda r, hs, wp, sr, sc: layer(r, hs, wp, sr, sc)[0])
    jitted_moreau = None
    if not args.no_moreau:
        moreau_layer = make_trade_cvxpylayer(args.n, solver="MOREAU")
        jitted_moreau = jax.jit(
            lambda r, hs, wp, sr, sc: moreau_layer(r, hs, wp, sr, sc[0])[0]
        )

    # Compile outside the timed region.
    solve_trade_jit(*jax_inputs[0], n_shards=args.shards)[0].block_until_ready()
    jitted_layer(*jax_inputs[0]).block_until_ready()
    if jitted_moreau is not None:
        try:
            jitted_moreau(*jax_inputs[0]).block_until_ready()
        except Exception as exc:
            print(f"moreau_unavailable {type(exc).__name__}: {exc}", flush=True)
            jitted_moreau = None

    print(f"benchmark n={args.n} instances={args.instances} repeat={args.repeat}", flush=True)
    time_many("direct_cvxpy", direct_cvxpy_solve, np_inputs, max(1, min(args.repeat, 2)))
    time_many("cvxpy_params", reusable_cvxpy, np_inputs, args.repeat)
    time_many(
        "ffi_jit",
        lambda item: solve_trade_jit(*item, n_shards=args.shards)[0],
        jax_inputs,
        args.repeat,
    )
    time_many("layer_jit", lambda item: jitted_layer(*item), jax_inputs, args.repeat)
    if jitted_moreau is not None:
        time_many("moreau_jit", lambda item: jitted_moreau(*item), jax_inputs, args.repeat)

    if not args.no_grad:
        diff_layer = make_trade_cvxpylayer(args.n, solver=None)

        def cvxpylayers_loss(r, hs, wp, sr, sc):
            return jnp.sum(diff_layer(r, hs, wp, sr, sc[0])[0] ** 2)

        cvxpylayers_grad_fn = jax.grad(cvxpylayers_loss, argnums=0)
        cvxpylayers_grad_fn(*jax_inputs[0]).block_until_ready()
        time_many(
            "cvxpylayers_default_grad_r",
            lambda item: cvxpylayers_grad_fn(*item),
            jax_inputs[: args.grad_instances],
            1,
        )

    if not args.no_grad and jitted_moreau is not None:

        def moreau_loss(r, hs, wp, sr, sc):
            return jnp.sum(jitted_moreau(r, hs, wp, sr, sc) ** 2)

        moreau_grad_fn = jax.jit(jax.grad(moreau_loss, argnums=0))
        moreau_grad_fn(*jax_inputs[0]).block_until_ready()
        time_many(
            "moreau_grad_r_jit",
            lambda item: moreau_grad_fn(*item),
            jax_inputs[: args.grad_instances],
            1,
        )
    elif not args.no_grad:
        print("moreau_grad_r_jit skipped: Moreau unavailable", flush=True)


if __name__ == "__main__":
    main()
