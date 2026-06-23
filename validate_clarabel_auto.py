from __future__ import annotations

import argparse

import cvxpy as cp
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from trade_solver import (  # noqa: E402
    CPG_CLARABEL,
    _BUILD_COUNTS,
    build_trade_problem,
    ensure_trade_solver,
    make_trade_cvxpylayer,
    make_trade_layer,
    solve_trade_jit,
)


def direct_solve(r, hs, w_prev, s_root, sqrt_c):
    n = len(r)
    w = cp.Variable(n)
    dw = cp.Variable(n)
    y = cp.Variable(n)
    problem = cp.Problem(
        cp.Maximize(r @ w - hs @ cp.abs(dw)),
        [dw == w - w_prev, y == s_root @ w, cp.norm(y, 2) <= sqrt_c],
    )
    assert problem.is_dcp(dpp=True)
    problem.solve(
        solver="CLARABEL",
        tol_gap_abs=1e-9,
        tol_gap_rel=1e-9,
        tol_feas=1e-9,
        verbose=False,
    )
    return np.asarray(w.value), problem.value, problem.status, problem.solver_stats.num_iters


def random_instance(n: int, seed: int):
    rng = np.random.default_rng(seed)
    b = rng.normal(size=(n, n))
    s = b.T @ b + 0.2 * np.eye(n)
    s_root = np.linalg.cholesky(s).T
    r = rng.normal(size=n)
    hs = 0.01 + rng.random(n) * 0.05
    w_prev = rng.normal(size=n) * 0.1
    sqrt_c = float(0.5 + rng.random())
    return s, r, hs, w_prev, s_root, sqrt_c


def validate(n: int, seeds: int, shards: int, force_rebuild: bool) -> None:
    if force_rebuild:
        ensure_trade_solver(n, shards, True)

    layer = make_trade_layer(n, n_shards=shards)
    custom_solver_layer = make_trade_cvxpylayer(
        n,
        solver=CPG_CLARABEL,
        n_shards=shards,
    )
    diff_layer = make_trade_cvxpylayer(n, solver=None)
    layer_jit = jax.jit(lambda r, hs, wp, sr, sc: layer(r, hs, wp, sr, sc)[0])
    custom_solver_jit = jax.jit(
        lambda r, hs, wp, sr, sc: custom_solver_layer(r, hs, wp, sr, sc)[0]
    )
    custom_solver_vmap_jit = jax.jit(
        jax.vmap(
            lambda rr, hh, wp, sr, sc: custom_solver_layer(rr, hh, wp, sr, sc)[0]
        )
    )

    print("seed max_abs_w_err obj_abs_err risk_minus_c status iter grad_norm")
    for seed in range(seeds):
        s, r, hs, w_prev, s_root, sqrt_c = random_instance(n, seed)
        w_cvx, obj_cvx, status, iterations = direct_solve(r, hs, w_prev, s_root, sqrt_c)

        jax_args = tuple(
            jnp.asarray(x) for x in (r, hs, w_prev, s_root, np.array([sqrt_c]))
        )
        w_ffi, info = solve_trade_jit(*jax_args, n_shards=shards)
        w_layer = layer_jit(*jax_args)
        w_custom_solver = custom_solver_jit(*jax_args)
        batched_args = tuple(jnp.stack([arg, arg]) for arg in jax_args)
        w_custom_vmap = custom_solver_vmap_jit(*batched_args)

        w_np = np.asarray(w_ffi)
        obj_ffi = r @ w_np - hs @ np.abs(w_np - w_prev)
        risk_minus_c = w_np @ s @ w_np - sqrt_c * sqrt_c
        grad_r = jax.grad(
            lambda rr: jnp.sum(diff_layer(rr, *jax_args[1:-1], jax_args[-1][0])[0] ** 2)
        )(jax_args[0])

        np.testing.assert_allclose(np.asarray(w_layer), w_np, rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            np.asarray(w_custom_solver),
            w_np,
            rtol=0,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            np.asarray(w_custom_vmap),
            np.stack([w_np, w_np]),
            rtol=0,
            atol=1e-10,
        )

        if seed == 0:
            spec = build_trade_problem(n)
            values = (r, hs, w_prev, s_root, sqrt_c)
            for parameter, value in zip(spec.parameters, values, strict=True):
                parameter.value = value
            spec.problem.solve(solver=CPG_CLARABEL, n_shards=shards)
            np.testing.assert_allclose(
                np.asarray(spec.variables[0].value),
                w_np,
                rtol=0,
                atol=1e-10,
            )

        print(
            f"{seed:4d} {np.max(np.abs(w_np - w_cvx)):.3e} "
            f"{abs(obj_ffi - obj_cvx):.3e} {risk_minus_c:.3e} "
            f"{status} {int(info[1])} {float(jnp.linalg.norm(grad_r)):.3e}"
        )

    print("build_counts", dict(_BUILD_COUNTS))
    hlo = str(
        solve_trade_jit.lower(
            jnp.ones(n),
            jnp.ones(n) * 0.01,
            jnp.zeros(n),
            jnp.eye(n),
            jnp.ones(1),
            n_shards=shards,
        ).compiler_ir(dialect="stablehlo")
    )
    print("hlo_has_trade_solve", "trade_solve" in hlo)
    print("hlo_has_pure_callback", "pure_callback" in hlo)
    print(
        "hlo_has_python_callback",
        "xla_python_cpu_callback" in hlo or "host_callback" in hlo,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--force-rebuild", action="store_true")
    args = parser.parse_args()
    validate(args.n, args.seeds, args.shards, args.force_rebuild)
