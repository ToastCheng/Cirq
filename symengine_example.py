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

"""Demonstrates the optional SymEngine support (POC for Cirq issue #7900).

Requires: pip install symengine

Each section shows one code path that accepts symengine expressions in
addition to sympy. Run: python symengine_example.py
"""

import numpy as np
import symengine as se

import cirq


def section(title: str) -> None:
    print(f'\n=== {title} ===')


def main() -> None:
    q = cirq.LineQubit(0)
    t = se.Symbol('t')

    section('1. Parametrized gates with symengine symbols')
    gate = cirq.XPowGate(exponent=t)
    print('gate:', gate)
    print('is_parameterized:', cirq.is_parameterized(gate))
    print('parameter_names:', cirq.parameter_names(gate))

    section('2. Formulas as exponents (gate math stays on symengine)')
    rx_gate = cirq.rx(t * 2)
    print('rx(2t) exponent:', rx_gate.exponent, f'({type(rx_gate.exponent).__module__})')
    circuit = cirq.Circuit(rx_gate.on(q), cirq.XPowGate(exponent=t + 1).on(q))
    print('circuit:', circuit, sep='\n')

    section('3. Resolution (native C++ subs, no sympy involved)')
    resolver = cirq.ParamResolver({'t': 0.5})
    print('value_of(2t + 1):', resolver.value_of(t * 2 + 1))
    resolved = cirq.resolve_parameters(circuit, resolver)
    print('resolved circuit:', resolved, sep='\n')
    print(
        'unitary matches sympy-built gate:',
        np.allclose(cirq.unitary(resolved[0][q]), cirq.unitary(cirq.rx(1.0))),
    )

    section('4. Recursive resolution and loop detection')
    chained = cirq.ParamResolver({'a': t, 't': 0.25})
    print('a -> t -> 0.25:', chained.value_of(se.Symbol('a')))
    looped = cirq.ParamResolver({'a': se.Symbol('b'), 'b': se.Symbol('a')})
    try:
        looped.value_of(se.Symbol('a'))
    except RecursionError as err:
        print('loop detected:', err)

    section('5. Sweeps and simulation')
    meas_circuit = circuit + cirq.measure(q, key='m')
    sweep = cirq.Linspace('t', start=0.0, stop=1.0, length=3)
    results = cirq.Simulator().run_sweep(meas_circuit, sweep, repetitions=5)
    for params, result in zip(sweep, results):
        print(f't={params["t"]}: measurements={result.measurements["m"].flatten()}')

    section('6. Symbolic coefficients (PauliString / LinearDict)')
    ps = cirq.PauliString({q: 'X'}, coefficient=t)
    print('parameterized PauliString:', ps)
    print('resolved:', cirq.resolve_parameters(ps, {'t': 2.0}))
    expansion = cirq.XPowGate(exponent=t)._pauli_expansion_()
    print('pauli expansion of X^t:', expansion)

    section('7. JSON serialization round trip')
    json_text = cirq.to_json(circuit)
    restored = cirq.read_json(json_text=json_text)
    print('round trip preserves parameterization:', cirq.is_parameterized(restored))
    print(
        'resolved equality:',
        cirq.resolve_parameters(restored, {'t': 0.5})
        == cirq.resolve_parameters(circuit, {'t': 0.5}),
    )

    section('8. Cross-backend resolution (resolver keys are names)')
    import sympy

    sympy_keyed = cirq.ParamResolver({sympy.Symbol('t'): 3.0})
    print('sympy-keyed resolver resolves symengine expr t*2:', sympy_keyed.value_of(t * 2))

    section('9. Mixing hazard: do not mix backends in one workflow')
    print('sympy.Symbol("t") == se.Symbol("t"):', sympy.Symbol('t') == t)
    print('but hashes differ:', hash(sympy.Symbol('t')) != hash(t))
    print('-> dict/set semantics break; pick one backend per workflow.')


if __name__ == '__main__':
    main()
