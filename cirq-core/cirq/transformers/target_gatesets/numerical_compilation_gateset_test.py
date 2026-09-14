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

from __future__ import annotations

import numpy as np
import pytest

import cirq
from cirq.testing import random_special_unitary

_Q0, _Q1, _Q2 = cirq.LineQubit.range(3)
_FSIM = cirq.FSimGate(theta=0.4, phi=0.1)


def _decomposition_fidelity(circuit: cirq.AbstractCircuit, target: np.ndarray) -> float:
    return abs(np.trace(cirq.unitary(circuit).conj().T @ target)) / 4


def test_invalid_base_gates() -> None:
    with pytest.raises(ValueError, match='non-empty sequence'):
        cirq.NumericalCompilationTargetGateset([])
    with pytest.raises(ValueError, match='two-qubit gates with a unitary'):
        cirq.NumericalCompilationTargetGateset(cirq.H)
    with pytest.raises(ValueError, match='two-qubit gates with a unitary'):
        cirq.NumericalCompilationTargetGateset(cirq.DepolarizingChannel(0.1, n_qubits=2))
    with pytest.raises(ValueError, match='one error rate per base gate'):
        cirq.NumericalCompilationTargetGateset([cirq.CZ, _FSIM], base_gate_error_rates=[0.01])


def test_numerical_path_custom_gate() -> None:
    """A base gate without analytical decomposition uses numerical optimization."""
    gateset = cirq.NumericalCompilationTargetGateset(_FSIM, random_state=3)
    assert gateset.base_gates == (_FSIM,)
    target = random_special_unitary(4, random_state=np.random.RandomState(2))
    circuit = cirq.Circuit(cirq.MatrixGate(target).on(_Q0, _Q1))
    out = cirq.optimize_for_target_gateset(circuit, gateset=gateset)
    assert gateset.validate(out)
    assert _decomposition_fidelity(out, target) > 1 - 1e-6
    assert {op.gate for op in out.all_operations() if len(op.qubits) == 2} == {_FSIM}


def test_noise_adaptive_gate_selection() -> None:
    """With error rates, the base gate maximizing the overall fidelity is selected."""
    gateset = cirq.NumericalCompilationTargetGateset(
        [cirq.CZ, _FSIM], base_gate_error_rates=[0.01, 0.001], random_state=5
    )
    target = random_special_unitary(4, random_state=np.random.RandomState(2))
    circuit = cirq.Circuit(cirq.MatrixGate(target).on(_Q0, _Q1))
    out = cirq.optimize_for_target_gateset(circuit, gateset=gateset)
    assert gateset.validate(out)
    # The much more reliable FSim gate wins over CZ.
    assert {op.gate for op in out.all_operations() if len(op.qubits) == 2} == {_FSIM}


def test_base_gates_with_equal_unitaries() -> None:
    """Base gates with equal unitaries are distinguished by index, not by
    re-matching unitaries; the compiler tries them in order, so the first
    gate is selected."""
    gateset = cirq.NumericalCompilationTargetGateset(
        [cirq.CZ, cirq.MatrixGate(cirq.unitary(cirq.CZ))], random_state=1
    )
    target = random_special_unitary(4, random_state=np.random.RandomState(2))
    circuit = cirq.Circuit(cirq.MatrixGate(target).on(_Q0, _Q1))
    out = cirq.optimize_for_target_gateset(circuit, gateset=gateset)
    assert gateset.validate(out)
    assert {op.gate for op in out.all_operations() if len(op.qubits) == 2} == {cirq.CZ}


def test_optimize_for_target_gateset_end_to_end() -> None:
    """A circuit with several two-qubit unitaries and a measurement is compiled."""
    gateset = cirq.NumericalCompilationTargetGateset(_FSIM, random_state=7)
    u01 = random_special_unitary(4, random_state=np.random.RandomState(8))
    u12 = random_special_unitary(4, random_state=np.random.RandomState(9))
    circuit = cirq.Circuit(
        cirq.H(_Q0),
        cirq.MatrixGate(u01).on(_Q0, _Q1),
        cirq.MatrixGate(u12).on(_Q1, _Q2),
        cirq.measure(_Q0, _Q1, _Q2),
    )
    out = cirq.optimize_for_target_gateset(circuit, gateset=gateset)
    assert gateset.validate(out)
    assert cirq.is_measurement(out[-1])
    unitary_circuit = cirq.Circuit(op for op in circuit.all_operations() if len(op.qubits) <= 2)
    unitary_out = cirq.Circuit(op for op in out.all_operations() if len(op.qubits) <= 2)
    assert _decomposition_fidelity(unitary_out, cirq.unitary(unitary_circuit)) > 1 - 1e-6


def test_decompose_non_unitary_operation() -> None:
    gateset = cirq.NumericalCompilationTargetGateset(cirq.CZ, random_state=1)
    assert gateset._decompose_two_qubit_operation(cirq.measure(_Q0, _Q1), 0) is NotImplemented


def test_repr() -> None:
    cirq.testing.assert_equivalent_repr(
        cirq.NumericalCompilationTargetGateset(cirq.CZ, random_state=5)
    )
    cirq.testing.assert_equivalent_repr(
        cirq.NumericalCompilationTargetGateset(
            [cirq.CZ, _FSIM], base_gate_error_rates=[0.01, 0.001], random_state=5
        )
    )


def test_equality() -> None:
    gateset = cirq.NumericalCompilationTargetGateset(
        [cirq.CZ, _FSIM], base_gate_error_rates=[0.01, 0.001], random_state=5
    )
    same = cirq.NumericalCompilationTargetGateset(
        [cirq.CZ, _FSIM], base_gate_error_rates=[0.01, 0.001], random_state=5
    )
    assert gateset == same
    assert hash(gateset) == hash(same)
    assert gateset != cirq.NumericalCompilationTargetGateset(cirq.CZ, random_state=5)
    assert gateset != cirq.NumericalCompilationTargetGateset(
        [cirq.CZ, _FSIM], base_gate_error_rates=[0.001, 0.01], random_state=5
    )
    assert gateset != 'not a gateset'


def test_json_roundtrip() -> None:
    cirq.testing.assert_json_roundtrip_works(
        cirq.NumericalCompilationTargetGateset(cirq.CZ, random_state=5)
    )
    cirq.testing.assert_json_roundtrip_works(
        cirq.NumericalCompilationTargetGateset(
            [cirq.CZ, _FSIM], base_gate_error_rates=[0.01, 0.001], random_state=5
        )
    )


def test_json_rejects_live_rng() -> None:
    gateset = cirq.NumericalCompilationTargetGateset(cirq.CZ, random_state=np.random.RandomState(5))
    with pytest.raises(ValueError, match='None or an integer seed'):
        cirq.to_json(gateset)
