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

"""Optional support for SymEngine symbolic expressions.

Cirq's symbolic parameter handling is built on SymPy. If the optional
`symengine` package is installed, Cirq additionally accepts `symengine.Basic`
expressions anywhere a sympy expression is accepted (gate exponents, parameter
resolvers, sweeps, ...) and evaluates them natively, which is significantly
faster. SymEngine is not a required dependency; this module degrades
gracefully when it is absent, and the sympy code paths are unaffected.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, TYPE_CHECKING

import numpy as np
import sympy

if TYPE_CHECKING:
    import symengine

try:
    import symengine

    HAVE_SYMENGINE = True
except ImportError:
    symengine = None
    HAVE_SYMENGINE = False

# Runtime tuple for isinstance checks covering every supported symbolic
# backend. Use is_symbolic()/is_symbol() below rather than using this tuple
# directly, so call sites read clearly.
SymbolicTypes: tuple[type, ...] = (
    (sympy.Basic, symengine.Basic) if HAVE_SYMENGINE else (sympy.Basic,)
)

SymbolTypes: tuple[type, ...] = (
    (sympy.Symbol, symengine.Symbol) if HAVE_SYMENGINE else (sympy.Symbol,)
)

# Union of symbolic expression types, for use in type annotations. Equals
# just sympy.Basic when symengine is not installed, so annotations never
# reference a missing module at runtime.
if TYPE_CHECKING:
    SymbolicExpr = sympy.Basic | symengine.Basic
else:
    SymbolicExpr = sympy.Basic | symengine.Basic if HAVE_SYMENGINE else sympy.Basic


def is_symbolic(val: Any) -> bool:
    """Returns whether val is a sympy or symengine symbolic expression.

    This covers symbolic constants such as `sympy.pi` and `symengine.pi`,
    as well as exact numbers from either backend (matching the behavior of
    isinstance checks against sympy.Basic elsewhere in cirq).
    """
    return isinstance(val, SymbolicTypes)


def is_symbol(val: Any) -> bool:
    """Returns whether val is a sympy.Symbol or symengine.Symbol."""
    return isinstance(val, SymbolTypes)


def is_symengine_expr(val: Any) -> bool:
    """Returns whether val is a symengine expression.

    Always returns False when symengine is not installed, so call sites do
    not need to check HAVE_SYMENGINE separately.
    """
    return HAVE_SYMENGINE and isinstance(val, symengine.Basic)


def symbolic_pi(val: Any) -> Any:
    """Returns the pi constant of the symbolic backend matching val.

    Use this instead of a hardcoded sympy.pi when building an expression
    that involves a user-provided symbolic value, so that symengine values
    combine with symengine.pi and are not silently promoted to sympy.
    Returns sympy.pi for sympy values and for anything non-symengine.
    """
    return symengine.pi if is_symengine_expr(val) else sympy.pi


def symbolic_lib(val: Any) -> ModuleType:
    """Returns the symbolic math module (sympy or symengine) matching val.

    Symengine exposes the elementary functions cirq needs (pi, sin, cos,
    exp, sqrt, acos, ...), so module-level selections like
    `lib = sympy if is_parameterized(x) else np` can use this to keep
    symengine inputs on the symengine backend.
    """
    return symengine if is_symengine_expr(val) else sympy


def symbol_like(name: str, val: Any) -> Any:
    """Returns a Symbol with the given name from the backend matching val."""
    return symengine.Symbol(name) if is_symengine_expr(val) else sympy.Symbol(name)


def symengine_number_value(val: Any) -> Any:
    """Converts a symengine number or numeric constant to a Python number.

    Returns NotImplemented if val is not a symengine number (including when
    symengine is not installed, or val is a symengine expression with free
    symbols), mirroring how exact sympy numbers are converted in
    cirq.study.resolver.
    """
    if not is_symengine_expr(val):
        return NotImplemented
    if val is symengine.pi:
        return np.pi
    if val.is_Number:
        if val.is_integer:
            return int(val)
        c = complex(val)
        return c.real if c.imag == 0 else c
    if not val.free_symbols:
        # Expressions over symbolic constants (e.g. 1.0/pi) are not is_Number
        # but can be evaluated numerically, matching the sympy resolver path.
        try:
            c = complex(val)
        except (TypeError, ValueError, RuntimeError):
            return NotImplemented
        return c.real if c.imag == 0 else c
    return NotImplemented
