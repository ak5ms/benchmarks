"""CVXPYgen/Clarabel backed JAX FFI layer for the trading problem.

The public API intentionally mirrors the feel of ``cvxpylayers``: write a CVXPY
problem with Parameters and Variables, build a layer once, then call the layer
with parameter values from normal JAX code.  The numerical solve itself lowers to
an XLA FFI custom call; no Python callback is used inside ``jax.jit``.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

import cvxpy as cp
import cvxpygen.cpg as cpg
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

_CACHE = Path(
    os.environ.get("TRADE_CPG_CACHE", Path.home() / ".cache/trade_cpg_clarabel_auto")
)
_REGISTRY: dict[tuple[int, int], tuple[str, ctypes.CDLL, Path]] = {}
_BUILD_COUNTS: dict[int, int] = {}
_LOCK = Lock()
CPG_CLARABEL = "CPG_CLARABEL"
CPG_CLARABEL_ALIASES = frozenset(
    {CPG_CLARABEL, "CVXPYGEN_CLARABEL", "TRADE_CPG_CLARABEL"}
)
_ORIGINAL_CVXPY_SOLVE = None


@dataclass(frozen=True)
class TradeProblemSpec:
    """The CVXPY objects used to generate a trade solver."""

    problem: cp.Problem
    parameters: tuple[cp.Parameter, ...]
    variables: tuple[cp.Variable, ...]


def build_trade_problem(n: int) -> TradeProblemSpec:
    """Build the DPP-compliant CVXPY trade problem.

    This is intentionally ordinary CVXPY modeling code.  ``S_root`` is a
    parameter satisfying ``S_root.T @ S_root == S`` and ``sqrt_c`` is the square
    root of the risk budget.
    """

    w = cp.Variable(n, name="w")
    dw = cp.Variable(n, name="dw")
    y = cp.Variable(n, name="y")

    r = cp.Parameter(n, name="r")
    hs = cp.Parameter(n, nonneg=True, name="hs")
    w_prev = cp.Parameter(n, name="w_prev")
    s_root = cp.Parameter((n, n), name="S_root")
    sqrt_c = cp.Parameter(nonneg=True, name="sqrt_c")

    objective = cp.Maximize(r @ w - hs @ cp.abs(dw))
    constraints = [dw == w - w_prev, y == s_root @ w, cp.norm(y, 2) <= sqrt_c]
    problem = cp.Problem(objective, constraints)

    if not problem.is_dcp(dpp=True):
        raise ValueError("trade problem is not DPP-compliant")

    return TradeProblemSpec(
        problem=problem,
        parameters=(r, hs, w_prev, s_root, sqrt_c),
        variables=(w,),
    )


def _patch_generated_solver(code_dir: Path) -> None:
    """Patch CVXPYgen output to avoid leaking Clarabel solver objects."""

    solve_c = code_dir / "c" / "src" / "cpg_solve.c"
    text = solve_c.read_text()
    marker = "clarabel_DefaultSolver_free(trade_solver);"

    if marker in text:
        return

    text = text.replace(
        "  trade_cpg_retrieve_info();\n  // Reset flags for outdated canonical parameters",
        "  trade_cpg_retrieve_info();\n"
        "  clarabel_DefaultSolver_free(trade_solver);\n"
        "  trade_solver = 0;\n"
        "  // Reset flags for outdated canonical parameters",
    )
    solve_c.write_text(text)


def _write_ffi_wrapper(code_dir: Path, n: int) -> None:
    """Write the small C++ XLA FFI wrapper around generated CVXPYgen C."""

    wrapper = f"""
#include <mutex>

#include "xla/ffi/api/ffi.h"

extern "C" {{
#include "c/include/cpg_solve.h"
#include "c/include/cpg_workspace.h"
}}

namespace ffi = xla::ffi;

static std::mutex g_mu;

static ffi::Error TradeSolve(
    ffi::BufferR1<ffi::DataType::F64> r,
    ffi::BufferR1<ffi::DataType::F64> hs,
    ffi::BufferR1<ffi::DataType::F64> w_prev,
    ffi::BufferR2<ffi::DataType::F64> s_root,
    ffi::BufferR1<ffi::DataType::F64> sqrt_c,
    ffi::ResultBufferR1<ffi::DataType::F64> w_out,
    ffi::ResultBufferR1<ffi::DataType::F64> info_out) {{
  std::lock_guard<std::mutex> lock(g_mu);

  const double* r_ptr = r.typed_data();
  const double* hs_ptr = hs.typed_data();
  const double* w_prev_ptr = w_prev.typed_data();
  const double* s_root_ptr = s_root.typed_data();

  for (int i = 0; i < {n}; ++i) {{
    trade_cpg_update_r(i, r_ptr[i]);
    trade_cpg_update_hs(i, hs_ptr[i]);
    trade_cpg_update_w_prev(i, w_prev_ptr[i]);
  }}

  for (int row = 0; row < {n}; ++row) {{
    for (int col = 0; col < {n}; ++col) {{
      trade_cpg_update_S_root(row + col * {n}, s_root_ptr[row * {n} + col]);
    }}
  }}

  trade_cpg_update_sqrt_c(sqrt_c.typed_data()[0]);
  trade_cpg_set_solver_default_settings();
  trade_cpg_set_solver_verbose(0);
  trade_cpg_set_solver_tol_gap_abs(1e-9);
  trade_cpg_set_solver_tol_gap_rel(1e-9);
  trade_cpg_set_solver_tol_feas(1e-9);
  trade_cpg_solve();

  double* w_data = w_out->typed_data();
  for (int i = 0; i < {n}; ++i) {{
    w_data[i] = trade_CPG_Prim.w[i];
  }}

  double* info_data = info_out->typed_data();
  info_data[0] = static_cast<double>(trade_CPG_Info.status);
  info_data[1] = static_cast<double>(trade_CPG_Info.iter);
  info_data[2] = trade_CPG_Info.obj_val;
  info_data[3] = trade_CPG_Info.pri_res;
  info_data[4] = trade_CPG_Info.dua_res;
  return ffi::Error::Success();
}}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    trade_solve_handler,
    TradeSolve,
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR1<ffi::DataType::F64>>()
        .Arg<ffi::BufferR1<ffi::DataType::F64>>()
        .Arg<ffi::BufferR1<ffi::DataType::F64>>()
        .Arg<ffi::BufferR2<ffi::DataType::F64>>()
        .Arg<ffi::BufferR1<ffi::DataType::F64>>()
        .Ret<ffi::BufferR1<ffi::DataType::F64>>()
        .Ret<ffi::BufferR1<ffi::DataType::F64>>());

extern "C" void* trade_solve_ptr() {{ return reinterpret_cast<void*>(trade_solve_handler); }}
"""
    code_dir.joinpath("ffi_wrapper.cc").write_text(wrapper.lstrip())

    cmake = f"""
cmake_minimum_required(VERSION 3.18)
project(trade_cpg_ffi LANGUAGES C CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)

add_subdirectory(c EXCLUDE_FROM_ALL)
add_library(trade_ffi SHARED ffi_wrapper.cc ${{trade_cpg_head}} ${{trade_cpg_src}})
target_include_directories(
  trade_ffi
  PRIVATE . c/include c/solver_code/include {jax.ffi.include_dir()})
target_compile_features(trade_ffi PRIVATE cxx_std_17)
target_link_libraries(trade_ffi PRIVATE libclarabel_c_static)
"""
    code_dir.joinpath("CMakeLists.txt").write_text(cmake.lstrip())


def _build_cpg_ffi(n: int, n_shards: int, force_rebuild: bool) -> Path:
    root = _CACHE / f"n{n}"
    shared_object = root / "build" / "libtrade_ffi.so"

    if shared_object.exists() and not force_rebuild:
        return shared_object

    if force_rebuild and root.exists():
        shutil.rmtree(root)

    root.mkdir(parents=True, exist_ok=True)
    spec = build_trade_problem(n)
    cpg.generate_code(
        spec.problem,
        code_dir=str(root),
        solver="CLARABEL",
        prefix="trade",
        wrapper=False,
    )
    _patch_generated_solver(root)
    _write_ffi_wrapper(root, n)

    build_dir = root / "build"
    subprocess.check_call(["cmake", "-S", str(root), "-B", str(build_dir), "-G", "Ninja"])
    subprocess.check_call(
        [
            "cmake",
            "--build",
            str(build_dir),
            "--target",
            "trade_ffi",
            "-j",
            str(os.cpu_count() or 2),
        ]
    )

    _BUILD_COUNTS[n] = _BUILD_COUNTS.get(n, 0) + 1
    return shared_object


def ensure_trade_solver(
    n: int,
    n_shards: int = 8,
    force_rebuild: bool = False,
) -> tuple[str, ctypes.CDLL, Path]:
    """Generate, compile, load, and register the fixed-size FFI target."""

    n = int(n)
    key = (n, int(n_shards))

    with _LOCK:
        if key in _REGISTRY and not force_rebuild:
            return _REGISTRY[key]

        shared_object = _build_cpg_ffi(n, n_shards, force_rebuild)
        library = ctypes.CDLL(str(shared_object))
        library.trade_solve_ptr.restype = ctypes.c_void_p

        target_name = f"trade_solve_n{n}"
        jax.ffi.register_ffi_target(
            target_name,
            jax.ffi.pycapsule(library.trade_solve_ptr()),
            platform="cpu",
            api_version=1,
        )
        _REGISTRY[key] = (target_name, library, shared_object)
        return _REGISTRY[key]


def _solve_trade_ffi(r, hs, w_prev, s_root, sqrt_c, *, n_shards: int = 8):
    n = int(r.shape[0])
    target_name, _, _ = ensure_trade_solver(n, n_shards=n_shards)
    output_shapes = (
        jax.ShapeDtypeStruct((n,), jnp.float64),
        jax.ShapeDtypeStruct((5,), jnp.float64),
    )

    return jax.ffi.ffi_call(
        target_name,
        output_shapes,
        has_side_effect=True,
        vmap_method="sequential",
        input_layouts=[(0,), (0,), (0,), (0, 1), (0,)],
        output_layouts=[(0,), (0,)],
    )(r, hs, w_prev, s_root, jnp.atleast_1d(sqrt_c))


def is_cpg_clarabel_solver(solver: Any) -> bool:
    """Return whether ``solver`` names the CVXPYgen/Clarabel FFI solver."""

    return isinstance(solver, str) and solver.upper() in CPG_CLARABEL_ALIASES


def solve_trade(r, hs, w_prev, S_root, sqrt_c, *, n_shards: int = 8):
    """Solve the trade problem and return ``(w, info)``.

    ``info`` is non-differentiable solver metadata. Use
    :func:`make_trade_cvxpylayer` when implicit-differentiation gradients are
    required.
    """

    w, info = _solve_trade_ffi(r, hs, w_prev, S_root, sqrt_c, n_shards=n_shards)
    return w, jax.lax.stop_gradient(info)


solve_trade_jit = jax.jit(solve_trade, static_argnames=("n_shards",))


class CvxpygenLayer:
    """Small cvxpylayers-style wrapper around the generated trade solver.

    Example:
        >>> spec = build_trade_problem(3)
        >>> layer = CvxpygenLayer(spec.problem, spec.parameters, spec.variables)
        >>> w, = layer(r, hs, w_prev, S_root, sqrt_c)

    The current generic-looking layer validates that it was given the trade
    problem signature.  It keeps user code close to cvxpylayers while the solve
    path remains CVXPYgen/Clarabel/XLA-FFI.
    """

    def __init__(
        self,
        problem: cp.Problem,
        parameters: Sequence[cp.Parameter],
        variables: Sequence[cp.Variable],
        *,
        solver: str = CPG_CLARABEL,
        n_shards: int = 8,
    ) -> None:
        if not is_cpg_clarabel_solver(solver):
            raise ValueError(f"CvxpygenLayer only supports solver={CPG_CLARABEL!r}")
        if not problem.is_dcp(dpp=True):
            raise ValueError("CvxpygenLayer requires a DPP-compliant problem")

        self.solver = CPG_CLARABEL
        parameter_names = tuple(parameter.name() for parameter in parameters)
        variable_names = tuple(variable.name() for variable in variables)
        expected_parameters = ("r", "hs", "w_prev", "S_root", "sqrt_c")

        if parameter_names != expected_parameters or variable_names != ("w",):
            raise NotImplementedError(
                "This lightweight layer currently supports the trade problem "
                "signature: parameters=(r, hs, w_prev, S_root, sqrt_c), "
                "variables=(w,)."
            )

        self.problem = problem
        self.parameters = tuple(parameters)
        self.variables = tuple(variables)
        self.n_shards = int(n_shards)

    def __call__(self, *parameter_values, solver_args=None):
        if len(parameter_values) != 5:
            raise TypeError("expected values for r, hs, w_prev, S_root, sqrt_c")

        n_shards = self.n_shards
        if solver_args and "n_shards" in solver_args:
            n_shards = int(solver_args["n_shards"])

        w, _info = solve_trade(*parameter_values, n_shards=n_shards)
        return (w,)


def make_trade_layer(n: int, *, n_shards: int = 8) -> CvxpygenLayer:
    """Construct a cvxpylayers-style layer for the trade problem."""

    spec = build_trade_problem(n)
    return CvxpygenLayer(
        spec.problem,
        parameters=spec.parameters,
        variables=spec.variables,
        solver=CPG_CLARABEL,
        n_shards=n_shards,
    )


def make_trade_cvxpylayer(n: int, *, solver: str = "MOREAU", **kwargs):
    """Construct a cvxpylayers-style JAX layer for the trade problem.

    Passing ``solver=CPG_CLARABEL`` returns the callback-free CVXPYgen/Clarabel
    XLA FFI layer. Other solver names are forwarded to the official
    ``cvxpylayers.jax.CvxpyLayer`` constructor; for example, ``solver="MOREAU"``
    selects the Moreau backend and keeps cvxpylayers' implicit differentiation
    behavior.
    """

    if is_cpg_clarabel_solver(solver):
        n_shards = int(kwargs.pop("n_shards", 8))
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported {CPG_CLARABEL} layer options: {names}")
        return make_trade_layer(n, n_shards=n_shards)

    from cvxpylayers.jax import CvxpyLayer

    spec = build_trade_problem(n)
    return CvxpyLayer(
        spec.problem,
        parameters=list(spec.parameters),
        variables=list(spec.variables),
        solver=solver,
        **kwargs,
    )


def _named_parameters(problem: cp.Problem) -> dict[str, cp.Parameter]:
    return {parameter.name(): parameter for parameter in problem.parameters()}


def _named_variables(problem: cp.Problem) -> dict[str, cp.Variable]:
    return {variable.name(): variable for variable in problem.variables()}


def _solve_trade_cvxpy_problem(
    problem: cp.Problem,
    *args,
    n_shards: int = 8,
    **kwargs,
) -> float:
    """CVXPY ``Problem.solve`` implementation for ``solver=CPG_CLARABEL``."""

    if args:
        raise TypeError(f"{CPG_CLARABEL} does not accept positional solve arguments")
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise TypeError(f"unsupported {CPG_CLARABEL} CVXPY solve options: {names}")

    parameters = _named_parameters(problem)
    variables = _named_variables(problem)
    required_parameters = ("r", "hs", "w_prev", "S_root", "sqrt_c")

    missing_parameters = [name for name in required_parameters if name not in parameters]
    if missing_parameters or "w" not in variables:
        raise NotImplementedError(
            f"{CPG_CLARABEL} currently supports the trade problem with "
            "parameters=(r, hs, w_prev, S_root, sqrt_c) and variable w"
        )

    values = [parameters[name].value for name in required_parameters]
    missing_values = [
        name
        for name, value in zip(required_parameters, values, strict=True)
        if value is None
    ]
    if missing_values:
        raise ValueError(f"missing values for CVXPY Parameters: {missing_values}")

    r, hs, w_prev, s_root, sqrt_c = values
    w, info = solve_trade_jit(
        jnp.asarray(r, dtype=jnp.float64),
        jnp.asarray(hs, dtype=jnp.float64),
        jnp.asarray(w_prev, dtype=jnp.float64),
        jnp.asarray(s_root, dtype=jnp.float64),
        jnp.atleast_1d(jnp.asarray(sqrt_c, dtype=jnp.float64)),
        n_shards=n_shards,
    )
    w_value = np.asarray(w)
    variables["w"].value = w_value
    if "dw" in variables:
        variables["dw"].value = w_value - np.asarray(w_prev)
    if "y" in variables:
        variables["y"].value = np.asarray(s_root) @ w_value

    problem._status = cp.settings.OPTIMAL
    problem._value = problem.objective.value
    problem._solver_stats = cp.problems.problem.SolverStats(
        solver_name=CPG_CLARABEL,
        solve_time=None,
        setup_time=None,
        num_iters=int(np.asarray(info)[1]),
        extra_stats={"info": np.asarray(info)},
    )
    return float(problem.value)


def install_cpg_clarabel_cvxpy_solver() -> None:
    """Install ``solver=CPG_CLARABEL`` support for matching CVXPY problems.

    CVXPY's public custom-solve hook is ``method=...``. This small adapter also
    accepts ``problem.solve(solver=CPG_CLARABEL)`` for the trade problem so the
    same solver name can be used in CVXPY and in ``make_trade_cvxpylayer``.
    """

    global _ORIGINAL_CVXPY_SOLVE

    if _ORIGINAL_CVXPY_SOLVE is not None:
        return

    _ORIGINAL_CVXPY_SOLVE = cp.Problem.solve

    def solve_with_cpg_alias(problem, *args, **kwargs):
        solver = kwargs.get("solver")
        if is_cpg_clarabel_solver(solver):
            kwargs = dict(kwargs)
            kwargs.pop("solver")
            return _solve_trade_cvxpy_problem(problem, *args, **kwargs)
        return _ORIGINAL_CVXPY_SOLVE(problem, *args, **kwargs)

    cp.Problem.register_solve(CPG_CLARABEL, _solve_trade_cvxpy_problem)
    cp.Problem.solve = solve_with_cpg_alias


install_cpg_clarabel_cvxpy_solver()
