# Trade CVXPYgen + Clarabel + JAX FFI layer

This repository demonstrates the intended architecture for solving a DPP CVXPY
problem from inside `jax.jit` without Python callbacks:

```text
CVXPY model -> CVXPYgen codegen with solver="CLARABEL" -> native shared library
-> XLA FFI custom call -> jax.jit
```

The API is deliberately close to
[`cvxpylayers`](https://github.com/cvxpy/cvxpylayers): define the optimization
problem in normal CVXPY, construct a layer with `parameters=[...]` and
`variables=[...]`, and call that layer with JAX arrays.

```python
import jax
import jax.numpy as jnp
from trade_solver import build_trade_problem, CvxpygenLayer

spec = build_trade_problem(n)
layer = CvxpygenLayer(
    spec.problem,
    parameters=spec.parameters,
    variables=spec.variables,
    n_shards=8,
)

@jax.jit
def policy(r, hs, w_prev, S_root, sqrt_c):
    w, = layer(r, hs, w_prev, S_root, sqrt_c)
    return w
```

For compatibility with earlier scripts, `solve_trade_jit` also returns solver
metadata:

```python
from trade_solver import solve_trade_jit
w, info = solve_trade_jit(r, hs, w_prev, S_root, sqrt_c, n_shards=8)
```

`w.shape == (n,)` and `info.shape == (5,)`.

## CVXPY model

`trade_solver.build_trade_problem(n)` is the only optimization-model code. It
uses the DPP-compliant reformulation:

```text
maximize r @ w - hs @ abs(dw)
subject to dw == w - w_prev
           y == S_root @ w
           norm(y, 2) <= sqrt_c
```

The CVXPY parameters are `r`, `hs`, `w_prev`, `S_root`, and `sqrt_c`. The builder
checks `problem.is_dcp(dpp=True)` before code generation.

## Auto-n cache/build behavior

`ensure_trade_solver(n, n_shards=8, force_rebuild=False)`:

1. builds the CVXPY problem for that `n`,
2. runs `cvxpygen.cpg.generate_code(..., solver="CLARABEL", prefix="trade")`,
3. patches the generated Clarabel solve path to free the per-call solver object,
4. writes a small XLA FFI C++ binding,
5. compiles `libtrade_ffi.so`, loads it with `ctypes`, and registers
   `trade_solve_n{n}` with `jax.ffi.register_ffi_target`.

Artifacts are cached under `~/.cache/trade_cpg_clarabel_auto/n{n}/`. Later calls
with the same `n` reuse the compiled artifact; calls with a new `n` generate,
build, and register a new fixed-shape target automatically.


## Custom solver alias

Importing `trade_solver` installs a trade-specific custom solver alias named
`CPG_CLARABEL` (also accepted as `"CVXPYGEN_CLARABEL"` and
`"TRADE_CPG_CLARABEL"`). The same name can be used from CVXPY-style code and
from the layer constructor:

```python
from trade_solver import CPG_CLARABEL, build_trade_problem, make_trade_cvxpylayer

spec = build_trade_problem(n)
for parameter, value in zip(spec.parameters, (r, hs, w_prev, S_root, sqrt_c)):
    parameter.value = value

# CVXPY-style solve. This populates spec.variables[0].value.
value = spec.problem.solve(solver=CPG_CLARABEL, n_shards=8)

# cvxpylayers-style solve. This returns the callback-free XLA FFI layer.
layer = make_trade_cvxpylayer(n, solver=CPG_CLARABEL, n_shards=8)
w, = layer(r_jax, hs_jax, w_prev_jax, S_root_jax, sqrt_c_jax)
```

The `CPG_CLARABEL` layer uses `jax.ffi.ffi_call(..., vmap_method="sequential")`,
so it can be called under `jax.jit` and `jax.vmap`. Other solver strings are
forwarded to the official `cvxpylayers.jax.CvxpyLayer`; for example,
`solver="MOREAU"` still selects Moreau.

## Gradients

The callback-free CVXPYgen/Clarabel FFI layer is used for forward solves inside
`jax.jit`. `solver=CPG_CLARABEL` is intentionally forward-only today: JAX can
`jit` and `vmap` the solve, but `jax.grad` through the FFI custom call raises
because no JVP/VJP rule is registered for that native solver call.

For gradients, use `make_trade_cvxpylayer(..., solver="MOREAU")`, which returns
the official `cvxpylayers.jax.CvxpyLayer` and differentiates by implicit
differentiation rather than finite differences. The Moreau-backed layer is also
JIT-compatible according to the cvxpylayers JAX API, but note that its
implementation is separate from the callback-free XLA FFI path benchmarked here.


I checked the available native-gradient options for this specific SOCP path.
Clarabel's public solver APIs expose solve/update/status interfaces, but not a
ready-to-call native derivative/JVP/VJP API for the generated C/Rust solver.
CVXPYgen has a `gradient=True` code-generation option, but for this SOCP it
currently routes through OSQP-oriented gradient code generation and fails with
`SolverError: The solver OSQP cannot solve this problem.` Therefore no performant
native JVP/VJP rule is registered for `CPG_CLARABEL` yet; Moreau remains the
registered differentiable JAX path.

## Global mutable state / concurrency

CVXPYgen's generated C workspace uses global mutable symbols such as
`trade_Canon_Params`, `trade_CPG_Prim`, and `trade_solver`. The FFI binding
serializes entry into each generated solver DSO with a native C++ mutex, and the
generated `cpg_solve.c` is patched to call
`clarabel_DefaultSolver_free(trade_solver); trade_solver = 0;`. That avoids
corrupted global workspace state and solver leaks. The `n_shards` argument is
kept in the API for compatibility and future replacement with a multi-DSO shard
pool.

## Why this is not a Python callback

`solve_trade` uses `jax.ffi.ffi_call` with sequential vmap lowering, not
`jax.pure_callback`,
`jax.experimental.host_callback`, or Python callbacks. Validation lowers the
jitted function to StableHLO and checks that a native custom call target such as
`trade_solve_n3` appears while callback markers are absent.

## Install/run

System packages needed by CVXPYgen/Clarabel builds include a C/C++ compiler,
CMake/Ninja, Rust/Cargo, and Eigen headers.

```bash
python -m pip install numpy scipy cvxpy clarabel cvxpygen cvxpylayers moreau jax jaxlib pybind11 nanobind cmake ninja
python validate_clarabel_auto.py --n 3 --seeds 8 --shards 4 --force-rebuild
python validate_clarabel_auto.py --n 9 --seeds 8 --shards 4 --force-rebuild
python test_auto_n_concurrency.py
python benchmark_trade.py --n 25 --instances 100 --repeat 3 --shards 4
python benchmark_trade.py --n 50 --instances 100 --repeat 3 --shards 4
# If you do not have a Moreau license key, add --no-moreau to skip Moreau timings.
```


### Moreau license setup

Moreau can be installed with `python -m pip install moreau`, but `solve()` needs
an active license key. The archived Moreau installation guide documents these
lookup locations, in order:

1. `MOREAU_LICENSE_KEY` environment variable,
2. `.moreau_key` in the current working directory,
3. `~/.moreau/key`,
4. `~/.moreau_key`.

For local benchmarking, prefer an environment variable or a user key file rather
than committing secrets:

```bash
export MOREAU_LICENSE_KEY="moreau_v1_..."
# or:
mkdir -p ~/.moreau
printf '%s\n' "moreau_v1_..." > ~/.moreau/key
```

The repository `.gitignore` excludes `.moreau_key` in case you use the
per-project key-file option.

The benchmark prints median milliseconds per solve for direct CVXPY+Clarabel,
a reused DPP CVXPY+Clarabel Parameter baseline (`cvxpy_params`), the raw
`solve_trade_jit` FFI path, the cvxpylayers-style `CvxpygenLayer` FFI path,
the Moreau-backed `cvxpylayers.jax.CvxpyLayer` path (`moreau_jit`), and
optional gradient smoke benchmarks for the standard cvxpylayers
default-solver path and the Moreau path. Moreau requires a license key at
runtime (`MOREAU_LICENSE_KEY` or `~/.moreau/key`); without it the script reports
`moreau_unavailable` and continues with the open-source baselines.


### Forward-only CPG_CLARABEL vs Moreau benchmark

With the Moreau license installed in `~/.moreau/key`, CPU forward-solve timings
from this environment were:

| command | CPG_CLARABEL raw FFI JIT | CPG_CLARABEL layer JIT | Moreau JIT |
| --- | ---: | ---: | ---: |
| `python benchmark_trade.py --n 50 --instances 100 --repeat 3 --shards 4 --no-grad` | 4.531 ms | **4.385 ms** | 7.644 ms |
| `python benchmark_trade.py --n 100 --instances 30 --repeat 3 --shards 4 --no-grad` | **19.666 ms** | 20.493 ms | 68.176 ms |

So the current callback-free CVXPYgen/Clarabel FFI path was about 1.7x faster
than Moreau at `n=50` and about 3.3x faster at `n=100` for forward solves on
this CPU run. Compilation and JIT warmup are excluded from the timed region.

## Local benchmark results with Moreau license

These runs used the CPU-only Moreau package with the license supplied through
`MOREAU_LICENSE_KEY`. Compilation and JIT warmup are excluded from the timed
region; values are median milliseconds per solve.

| command | direct CVXPY | reused CVXPY params | raw FFI JIT | CVXPYgen layer JIT | Moreau JIT | cvxpylayers default grad wrt `r` | Moreau grad wrt `r` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `python benchmark_trade.py --n 25 --instances 100 --repeat 3 --shards 4 --grad-instances 3` | 11.937 | 4.060 | 1.584 | **1.471** | 3.165 | 36.122 | 6.130 |
| `python benchmark_trade.py --n 50 --instances 100 --repeat 3 --shards 4 --grad-instances 3` | 14.417 | 7.553 | 5.278 | **4.807** | 7.637 | 45.509 | 11.981 |

On these CPU runs, the callback-free CVXPYgen/Clarabel XLA FFI path is faster
than Moreau for forward solves. The standard cvxpylayers gradient path is
reported as `cvxpylayers_default_grad_r`; Moreau remains the faster
differentiable path used by `make_trade_cvxpylayer(..., solver="MOREAU")`
for gradients. The custom CVXPYgen/Clarabel XLA FFI solver is the
forward-only path in this table; it is not what `cvxpylayers_default_grad_r` times.
