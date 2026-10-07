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

"""Tests for the optional SymEngine acceleration (cirq/_compat_symbolic.py).

The acceleration is internal: callers always pass and receive sympy objects
(or Python numbers); symengine only speeds up the substitution in between.
All tests in this file are skipped when symengine is not installed, in which
case the plain sympy code paths are used (covered by the existing suite).
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy

import cirq
from cirq._compat_symbolic import HAVE_SYMENGINE, resolve_via_symengine

symengine = pytest.importorskip('symengine')

t = sympy.Symbol('t')
s = sympy.Symbol('s')


def test_have_symengine():
    assert HAVE_SYMENGINE


@pytest.mark.parametrize(
    'expr',
    [
        t + 1,
        t * 2,
        t**2,
        t**s,
        sympy.sqrt(t),
        sympy.Abs(t - 2),
        sympy.sin(t),
        sympy.cos(t),
        sympy.tan(t),
        sympy.exp(t),
        sympy.log(t + 2),
        sympy.Rational(1, 3) * t,
        sympy.pi * t,
        sympy.E * t,
        sympy.I * t,
        sympy.floor(t + s),
        sympy.ceiling(t * s),
        sympy.Mod(s, t),
        sympy.Min(t, s),
        sympy.Max(t, s),
        sympy.Float('0.123456789012345678901234567890', 30) * t,
        sympy.sin(t + s * sympy.pi / 7) ** 3,
    ],
)
def test_value_of_matches_sympy_path(expr):
    # The symengine fast path must return the same values as the sympy path.
    resolver = cirq.ParamResolver({'t': 0.7, 's': 1.3})
    via_symengine = resolve_via_symengine(expr, {'t': 0.7, 's': 1.3}, recursive=True)
    expected = cirq.ParamResolver({'t': 0.7, 's': 1.3}).value_of(expr)
    assert via_symengine == pytest.approx(expected)
    assert type(via_symengine) in (float, complex) or isinstance(via_symengine, (int, np.floating))
    assert resolver.value_of(expr) == pytest.approx(expected)


def test_full_resolution_returns_python_float():
    result = resolve_via_symengine(sympy.sin(t * sympy.pi), {'t': 0.5}, recursive=True)
    assert isinstance(result, float)
    assert result == pytest.approx(1.0)


def test_complex_result():
    result = resolve_via_symengine(sympy.exp(sympy.I * t * sympy.pi), {'t': 0.5}, recursive=True)
    assert isinstance(result, complex)
    assert result == pytest.approx(1j)


def test_zero_imaginary_complex_folds_to_float():
    result = resolve_via_symengine(sympy.I * t, {'t': 0.0}, recursive=True)
    assert result == 0.0
    assert not isinstance(result, complex)


def test_partial_resolution_returns_sympy():
    # Callers must never see a symengine object.
    result = resolve_via_symengine(sympy.sin(t * s), {'t': 0.7}, recursive=True)
    assert isinstance(result, sympy.Basic)
    assert sympy.simplify(result - sympy.sin(0.7 * s)) == 0


def test_constant_expression_skips_subs():
    result = resolve_via_symengine(sympy.Integer(2), {'t': 0.5}, recursive=True)
    assert result == 2.0
    assert isinstance(result, float)


def test_recursive_chain_resolution():
    params = {'a': t + 1, 't': 2.0}
    result = resolve_via_symengine(
        sympy.sin(sympy.Symbol('a') * sympy.pi / 2), params, recursive=True
    )
    assert result == pytest.approx(-1.0)


def test_recursive_loop_detection():
    params = {'a': sympy.Symbol('b'), 'b': sympy.Symbol('a')}
    with pytest.raises(RecursionError, match='indirectly contains itself'):
        resolve_via_symengine(sympy.sin(sympy.Symbol('a')), params, recursive=True)


def test_non_recursive_single_step():
    params = {'a': t, 't': 2.0}
    result = resolve_via_symengine(sympy.Symbol('a') + 1, params, recursive=False)
    assert sympy.simplify(result - (t + 1)) == 0


def test_unsupported_expression_returns_not_implemented():
    # Indexed has no symengine equivalent; callers fall back to the sympy path.
    expr = sympy.IndexedBase('m')[0] + t
    assert resolve_via_symengine(expr, {'t': 0.5}, recursive=True) is NotImplemented


def test_unevaluable_constant_returns_not_implemented():
    # symengine's sign() does not evaluate numerically when its argument
    # contains a symbolic constant like pi; sympy's does.
    expr = sympy.sign(t * sympy.pi + 1e-9)
    assert resolve_via_symengine(expr, {'t': 1.0}, recursive=True) is NotImplemented
    # The resolver falls back to the sympy path and gets the right answer.
    assert cirq.ParamResolver({'t': 1.0}).value_of(expr) == 1.0


def test_resolver_falls_back_for_indexed():
    resolver = cirq.ParamResolver({'t': 0.5})
    result = resolver.value_of(sympy.IndexedBase('m')[0] + t)
    assert sympy.simplify(result - (sympy.IndexedBase('m')[0] + 0.5)) == 0


def test_resolve_gate_and_circuit_end_to_end():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(
        cirq.XPowGate(exponent=sympy.sin(t * sympy.pi)).on(q), cirq.measure(q, key='m')
    )
    results = cirq.Simulator().run_sweep(
        circuit, cirq.Linspace('t', start=0.0, stop=1.0, length=3), repetitions=5
    )
    assert len(results) == 3
    resolved = cirq.resolve_parameters(circuit, {'t': 0.5})
    assert not cirq.is_parameterized(resolved)
    np.testing.assert_allclose(cirq.unitary(resolved[0][q].gate), cirq.unitary(cirq.X**1.0))
