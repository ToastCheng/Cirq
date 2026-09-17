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

"""Tests for optional SymEngine support (cirq/_compat_symbolic.py).

All tests in this file are skipped when symengine is not installed; the
sympy code paths are covered by the existing test suite.
"""

from __future__ import annotations

import typing

import numpy as np
import pytest
import sympy

import cirq
from cirq._compat import proper_repr
from cirq._compat_symbolic import (
    HAVE_SYMENGINE,
    is_symbol,
    is_symbolic,
    symbolic_lib,
    symbolic_pi,
    SymbolicExpr,
)

symengine = pytest.importorskip('symengine')

SETUP_CODE = (
    'import cirq\nimport numpy as np\nimport sympy\nimport pandas as pd\n'
    'import datetime\nimport symengine\n'
)


def test_have_symengine():
    assert HAVE_SYMENGINE


def test_is_symbolic():
    assert is_symbolic(symengine.Symbol('t'))
    assert is_symbolic(symengine.Symbol('t') * 2)
    assert is_symbolic(symengine.pi)
    assert is_symbolic(sympy.Symbol('t'))
    assert is_symbolic(sympy.pi)
    assert not is_symbolic(1.0)
    assert not is_symbolic('t')
    assert not is_symbolic(None)


def test_is_symbol():
    assert is_symbol(symengine.Symbol('t'))
    assert is_symbol(sympy.Symbol('t'))
    assert not is_symbol(symengine.Symbol('t') * 2)
    assert not is_symbol(sympy.Symbol('t') * 2)
    assert not is_symbol('t')


def test_symbolic_expr_annotation_union():
    # The annotation alias must include symengine.Basic when installed.
    assert symengine.Basic in typing.get_args(SymbolicExpr)


def test_type_aliases_include_symengine():
    assert symengine.Basic in typing.get_args(cirq.TParamVal)
    assert symengine.Basic in typing.get_args(cirq.TParamValComplex)
    assert symengine.Basic in typing.get_args(cirq.TParamKey)


def test_is_parameterized():
    t = symengine.Symbol('t')
    assert cirq.is_parameterized(t)
    assert cirq.is_parameterized(t * 2 + 1)
    assert cirq.is_parameterized(symengine.pi)
    assert not cirq.is_parameterized(1.0)
    assert not cirq.is_parameterized('t')


def test_parameter_names():
    t = symengine.Symbol('t')
    a = symengine.Symbol('a')
    assert cirq.parameter_names(t * a + 1) == {'t', 'a'}
    assert cirq.parameter_names(symengine.pi) == set()
    assert cirq.parameter_names(t) == {'t'}


def test_resolver_value_of_basic():
    t = symengine.Symbol('t')
    resolver = cirq.ParamResolver({'t': 0.5})
    assert resolver.value_of(t) == 0.5
    assert resolver.value_of('t') == 0.5
    assert resolver.value_of(t * 2 + 1) == 2.0
    assert isinstance(resolver.value_of(t * 2 + 1), float)


def test_resolver_symengine_keys():
    t = symengine.Symbol('t')
    resolver = cirq.ParamResolver({t: 0.25})
    assert resolver.value_of(t) == 0.25
    assert resolver.value_of('t') == 0.25


def test_resolver_rejects_symengine_formula_keys():
    with pytest.raises(TypeError, match='non-symbol'):
        cirq.ParamResolver({symengine.Symbol('t') * 2: 0.5})


def test_resolver_numeric_coercion():
    resolver = cirq.ParamResolver(
        {'i': symengine.Integer(3), 'r': symengine.Rational(1, 2), 'c': symengine.sympify(3 + 4j)}
    )
    assert resolver.value_of('i') == 3
    assert isinstance(resolver.value_of('i'), int)
    assert resolver.value_of('r') == 0.5
    assert isinstance(resolver.value_of('r'), float)
    assert resolver.value_of('c') == 3 + 4j
    assert isinstance(resolver.value_of('c'), complex)
    assert resolver.value_of(symengine.pi) == np.pi


def test_resolver_folds_symbolic_constants():
    # 1.0/pi has no free symbols but is not is_Number; it must still be
    # evaluated numerically, matching the sympy resolution path.
    t = symengine.Symbol('t')
    resolver = cirq.ParamResolver({'t': 0.5})
    resolved = resolver.value_of(t * 2 / symengine.pi)
    assert resolved == pytest.approx(1.0 / np.pi)
    assert isinstance(resolved, float)


def test_resolver_unresolved_symbol_passthrough():
    t = symengine.Symbol('t')
    z = symengine.Symbol('z')
    resolver = cirq.ParamResolver({'t': 0.5})
    # Unresolved symengine symbols stay symengine symbols.
    assert resolver.value_of(z) is z
    # Unresolvable formulas keep free symbols and stay symengine expressions.
    unresolved = resolver.value_of(t + z)
    assert isinstance(unresolved, symengine.Basic)
    assert {s.name for s in unresolved.free_symbols} == {'z'}


def test_resolver_recursive():
    t = symengine.Symbol('t')
    resolver = cirq.ParamResolver({'a': t, 't': 2.0})
    assert resolver.value_of(symengine.Symbol('a')) == 2.0
    assert resolver.value_of(symengine.Symbol('a') * 3) == 6.0


def test_resolver_non_recursive():
    resolver = cirq.ParamResolver({'a': symengine.Symbol('b'), 'b': 1.0})
    # One simultaneous substitution step only.
    resolved = resolver.value_of(symengine.Symbol('a') * 2, recursive=False)
    assert resolved == symengine.Symbol('b') * 2


def test_resolver_loop_detection():
    resolver = cirq.ParamResolver({'a': symengine.Symbol('b'), 'b': symengine.Symbol('a')})
    with pytest.raises(RecursionError):
        resolver.value_of(symengine.Symbol('a'))
    with pytest.raises(RecursionError):
        resolver.value_of(symengine.Symbol('a') * 2)


def test_resolver_cross_backend_resolution():
    # Resolution is keyed by symbol name, so a sympy-keyed resolver resolves
    # symengine expressions and vice versa.
    t = symengine.Symbol('t')
    assert cirq.ParamResolver({sympy.Symbol('t'): 3.0}).value_of(t) == 3.0
    assert cirq.ParamResolver({t: 3.0}).value_of(sympy.Symbol('t') * 2) == 6.0


def test_resolve_gate_and_circuit():
    q = cirq.LineQubit(0)
    t = symengine.Symbol('t')
    gate = cirq.XPowGate(exponent=t)
    assert cirq.is_parameterized(gate)
    assert cirq.parameter_names(gate) == {'t'}

    circuit = cirq.Circuit(gate.on(q))
    resolved = cirq.resolve_parameters(circuit, {'t': 0.5})
    assert not cirq.is_parameterized(resolved)
    np.testing.assert_allclose(cirq.unitary(resolved), cirq.unitary(cirq.X**0.5))

    # Partial resolution keeps the remaining symbol symbolic.
    formula_circuit = cirq.Circuit(cirq.XPowGate(exponent=t + 1).on(q))
    partially_resolved = cirq.resolve_parameters(formula_circuit, {'t': symengine.Symbol('s')})
    assert cirq.is_parameterized(partially_resolved)
    assert cirq.parameter_names(partially_resolved) == {'s'}


def test_simulator_sweep():
    q = cirq.LineQubit(0)
    t = symengine.Symbol('t')
    circuit = cirq.Circuit(cirq.XPowGate(exponent=t).on(q), cirq.measure(q, key='m'))
    results = cirq.Simulator().run_sweep(circuit, cirq.ParamResolver({'t': 1.0}), repetitions=10)
    assert np.all(results[0].measurements['m'] == 1)


def test_json_roundtrip_symengine_expression():
    expr = symengine.Symbol('t') * 2 + symengine.pi
    restored = cirq.read_json(json_text=cirq.to_json(expr))
    assert isinstance(restored, symengine.Basic)
    assert restored == expr


def test_json_roundtrip_circuit():
    q = cirq.LineQubit(0)
    t = symengine.Symbol('t')
    circuit = cirq.Circuit(cirq.XPowGate(exponent=t).on(q))
    restored = cirq.read_json(json_text=cirq.to_json(circuit))
    assert cirq.is_parameterized(restored)
    assert isinstance(restored[0][q].gate.exponent, symengine.Basic)
    assert cirq.resolve_parameters(restored, {'t': 0.25}) == cirq.resolve_parameters(
        circuit, {'t': 0.25}
    )


def test_json_roundtrip_param_resolver():
    resolver = cirq.ParamResolver({'t': symengine.Symbol('s') * 2})
    restored = cirq.read_json(json_text=cirq.to_json(resolver))
    assert restored == resolver


def test_proper_repr():
    expr = symengine.Symbol('t') * 2 + symengine.pi
    r = proper_repr(expr)
    assert r == "symengine.sympify('2*t + pi')"
    global_vals: dict = {}
    exec(SETUP_CODE, global_vals)
    assert eval(r, global_vals) == expr


def test_repr_param_resolver_with_symengine_values():
    resolver = cirq.ParamResolver({'a': symengine.Symbol('t')})
    cirq.testing.assert_equivalent_repr(resolver, setup_code=SETUP_CODE)


def test_symbolic_pi():
    assert symbolic_pi(symengine.Symbol('t')) is symengine.pi
    assert symbolic_pi(sympy.Symbol('t')) is sympy.pi
    assert symbolic_pi(1.0) is sympy.pi
    assert symbolic_pi('t') is sympy.pi


def test_symbolic_lib():
    assert symbolic_lib(symengine.Symbol('t')) is symengine
    assert symbolic_lib(sympy.Symbol('t')) is sympy
    assert symbolic_lib(1.0) is sympy


def test_rx_keeps_backend():
    # Gate math (rads / pi) must not promote the exponent to the other backend.
    assert isinstance(cirq.rx(symengine.Symbol('t')).exponent, symengine.Basic)
    assert isinstance(cirq.rx(sympy.Symbol('t')).exponent, sympy.Basic)


@pytest.mark.parametrize('gate_type', [cirq.XPowGate, cirq.YPowGate, cirq.ZPowGate])
def test_pauli_expansion_keeps_backend_and_matches_sympy(gate_type):
    t = symengine.Symbol('t')
    expansion = gate_type(exponent=t)._pauli_expansion_()
    assert expansion
    assert all(isinstance(c, symengine.Basic) for c in expansion.values())

    # Substituting t in the symbolic expansion must match expanding the
    # numerically resolved gate.
    resolved_expansion = cirq.resolve_parameters(gate_type(exponent=t), {'t': 0.3})
    numeric = resolved_expansion._pauli_expansion_()
    for key, coeff in expansion.items():
        assert np.isclose(complex(coeff.subs({'t': 0.3})), complex(numeric[key]))


def test_linear_dict_symengine_coefficients():
    # Numeric symengine coefficients are coerced to Python numbers.
    linear_dict = cirq.LinearDict({'X': symengine.Integer(2), 'Y': symengine.Rational(1, 3)})
    assert linear_dict['X'] == 2
    assert isinstance(linear_dict['X'], int)
    assert linear_dict['Y'] == pytest.approx(1 / 3)
    # Symbolic coefficients are kept symbolic and are not dropped by clean().
    symbolic_dict = cirq.LinearDict({'X': symengine.Symbol('t')})
    symbolic_dict.clean(atol=0)
    assert symbolic_dict['X'] == symengine.Symbol('t')


def test_pauli_string_symengine_coefficient():
    q = cirq.LineQubit(0)
    t = symengine.Symbol('t')
    pauli_string = cirq.PauliString({q: 'X'}, coefficient=t)
    assert cirq.is_parameterized(pauli_string)
    resolved = cirq.resolve_parameters(pauli_string, {'t': 2.0})
    assert not cirq.is_parameterized(resolved)
    assert complex(resolved.coefficient) == 2.0


def test_symengine_condition_wrapped():
    # A symengine condition must be wrapped in a SympyCondition like sympy
    # conditions are; a raw symengine relational stored unwrapped fails later
    # with AttributeError during Circuit construction.
    q = cirq.LineQubit(0)
    op = cirq.X(q).with_classical_controls(symengine.Symbol('m') > 0)
    condition = next(iter(op.classical_controls))
    assert isinstance(condition, cirq.SympyCondition)
    assert isinstance(condition.expr, symengine.Basic)
    # Circuit construction queries condition.keys; this used to fail.
    circuit = cirq.Circuit(cirq.measure(q, key='m'), op)
    assert len(circuit) == 2


def test_symengine_condition_simulation():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(
        cirq.X(q),
        cirq.measure(q, key='m'),
        # m == 1 always, so the conditional X undoes the first X.
        cirq.X(q).with_classical_controls(symengine.Symbol('m') > 0),
        cirq.measure(q, key='out'),
    )
    result = cirq.Simulator().run(circuit, repetitions=5)
    assert np.all(result.measurements['m'] == 1)
    assert np.all(result.measurements['out'] == 0)


def test_symengine_condition_keys_and_replace_key():
    condition = cirq.SympyCondition(symengine.Symbol('m') > 0)
    assert condition.keys == (cirq.MeasurementKey('m'),)
    replaced = condition.replace_key(cirq.MeasurementKey('m'), cirq.MeasurementKey('m2'))
    assert replaced.keys == (cirq.MeasurementKey('m2'),)
    assert isinstance(replaced.expr, symengine.Basic)


def test_symengine_condition_json_roundtrip():
    q = cirq.LineQubit(0)
    op = cirq.X(q).with_classical_controls(symengine.Symbol('m') > 0)
    restored = cirq.read_json(json_text=cirq.to_json(op))
    assert restored == op
