# Copyright 2026 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Optional SymEngine support, used internally to accelerate sympy resolution.

SymEngine is an optional dependency. When it is installed, parameter
resolution of sympy expressions can be routed through SymEngine's C++
substitution engine: expressions are converted once (cached), substituted
natively, and converted back, so the speedup is invisible to callers.
"""

from __future__ import annotations

import functools
from typing import Any

import sympy

try:
    import symengine

    HAVE_SYMENGINE = True
except ImportError:  # pragma: no cover
    symengine = None
    HAVE_SYMENGINE = False


@functools.lru_cache(maxsize=65536)
def _to_symengine(expr: sympy.Basic) -> Any:
    """Converts a sympy expression to symengine, or returns None if unsupported.

    SymEngine has no equivalent of some sympy constructs (e.g. `Indexed`), in
    which case `symengine.sympify` raises and the caller should fall back to
    the sympy code path. Results are cached because conversion costs ~10x a
    single substitution; expressions are typically shared across a whole sweep.
    """
    try:
        return symengine.sympify(expr)
    except Exception:
        return None


def _symengine_to_number(se_value: Any) -> Any:
    """Coerces a free-symbol-free symengine expression to a Python number.

    Returns NotImplemented when symengine cannot evaluate the expression
    numerically (e.g. its `sign` does not evaluate), so the caller can fall
    back to the sympy code path.
    """
    try:
        return float(se_value)
    except (RuntimeError, TypeError):
        pass
    try:
        result = complex(se_value)
        return result if result.imag else result.real
    except (RuntimeError, TypeError):
        return NotImplemented


def resolve_via_symengine(
    value: sympy.Basic, str_keyed_params: dict[str, Any], recursive: bool
) -> Any:
    """Resolves a sympy expression using symengine's C++ substitution.

    Converts `value` to symengine (cached), substitutes with the string-keyed
    parameter dict natively, and converts the result back so callers never see
    a symengine object: fully resolved results become Python floats/complexes
    (matching what the sympy path returns) and partially resolved results are
    converted back to sympy expressions.

    Args:
        value: The sympy expression to resolve.
        str_keyed_params: Parameter dict with string keys.
        recursive: Whether to substitute until a fixpoint is reached.

    Returns:
        The resolved value, or NotImplemented if the expression or the
        parameter values use sympy features symengine does not support; the
        caller should fall back to the sympy code path.

    Raises:
        RecursionError: If recursive substitution does not reach a fixpoint.
    """
    se_value = _to_symengine(value)
    if se_value is None:
        return NotImplemented
    if not se_value.free_symbols:
        # A constant expression (e.g. Integer leaves of a larger formula):
        # nothing to substitute, coerce to a number without calling subs.
        # subs cost scales with the parameter dict size, so skipping it here
        # matters for large sweeps.
        return _symengine_to_number(se_value)
    try:
        if recursive:
            seen = {se_value}
            while True:
                resolved = se_value.subs(str_keyed_params)
                if resolved == se_value:
                    break
                if resolved in seen:
                    raise RecursionError(f'Evaluation of {value} indirectly contains itself.')
                seen.add(resolved)
                se_value = resolved
        else:
            se_value = se_value.subs(str_keyed_params)
    except RecursionError:
        raise
    except Exception:
        # Parameter values that symengine cannot sympify (e.g. Indexed).
        return NotImplemented
    if se_value.free_symbols:
        # Partially resolved: convert back so callers never see a symengine object.
        return sympy.sympify(se_value)
    # Fully resolved: coerce to a Python float/complex (or NotImplemented to
    # trigger the sympy fallback when symengine cannot evaluate numerically).
    return _symengine_to_number(se_value)
