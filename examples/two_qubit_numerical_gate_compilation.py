# pylint: disable=wrong-or-nonexistent-copyright-notice
"""Example application of numerical two-qubit gate compilation (NuOp).

This script demonstrates the numerical compilation technique of
arXiv:2106.15490, which decomposes an arbitrary two-qubit unitary U_t into
layers of a fixed hardware "base" gate G sandwiched between single-qubit
gates,

    k_i . G . k_{i-1} . G . ... . G . k_0  ~  U_t

where the angles of the single-qubit gates k_j are found by numerical
optimization.

The script shows the three ways an end user interacts with the compiler:

1. Compiling a single two-qubit unitary onto a custom hardware gate (here: a
   calibrated FSimGate) with `cirq.two_qubit_gate_numerical_compilation`.
2. Noise-adaptive compilation: given several base gates with hardware error
   rates, the compiler picks the decomposition with the highest overall
   fidelity, which can deliberately be approximate.
3. Compiling a whole circuit with `cirq.optimize_for_target_gateset` using
   `cirq.NumericalCompilationTargetGateset`.
"""

from __future__ import annotations

import numpy as np

import cirq
from cirq.testing import random_special_unitary


def main(random_state: int = 5):
    """Demonstration of the NuOp numerical two-qubit gate compiler.

    Args:
        random_state: Seed for the random target unitaries and the optimizer's
            starting points.
    """
    # A custom hardware gate without an analytical decomposition in cirq,
    # e.g. a calibrated FSim gate.
    hardware_gate = cirq.FSimGate(theta=0.4, phi=0.1)

    # --- 1. Compile a single two-qubit unitary onto the hardware gate. ---
    target = random_special_unitary(4, random_state=np.random.RandomState(2))
    result = cirq.two_qubit_gate_numerical_compilation(
        target, cirq.unitary(hardware_gate), random_state=random_state
    )
    print('=== Single-gate compilation onto a custom FSim gate ===')
    print(f'success: {result.success}')
    print(f'base gates used: {result.num_base_gates}')
    print(f'decomposition fidelity: {result.decomposition_fidelity:.9f}')
    print()

    # --- 2. Noise-adaptive compilation with several noisy base gates. ---
    # Given hardware error rates, the compiler maximizes the overall fidelity
    # Fu = Fd * Fh over all (base gate, layer count) combinations.
    noisy_cz, reliable_fsim = 0.01, 0.001
    noisy_result = cirq.two_qubit_gate_numerical_compilation(
        target,
        [cirq.unitary(cirq.CZ), cirq.unitary(hardware_gate)],
        base_gate_error_rates=[noisy_cz, reliable_fsim],
        random_state=random_state,
    )
    picked = ('CZ', 'FSim')[noisy_result.base_gate_index]
    print('=== Noise-adaptive compilation (CZ @ 1% vs FSim @ 0.1% error) ===')
    assert noisy_result.hardware_fidelity is not None
    print(f'picked base gate: {picked}')
    print(f'base gates used: {noisy_result.num_base_gates}')
    print(f'decomposition fidelity Fd: {noisy_result.decomposition_fidelity:.6f}')
    print(f'hardware fidelity Fh: {noisy_result.hardware_fidelity:.6f}')
    overall = noisy_result.decomposition_fidelity * noisy_result.hardware_fidelity
    print(f'overall fidelity Fu = Fd * Fh: {overall:.6f}')
    print()

    # --- 3. Compile a whole circuit with a target gateset. ---
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(
        cirq.H(q0),
        cirq.MatrixGate(random_special_unitary(4, random_state=np.random.RandomState(8)))(q0, q1),
        cirq.MatrixGate(random_special_unitary(4, random_state=np.random.RandomState(9)))(q1, q2),
        cirq.measure(q0, q1, q2),
    )
    gateset = cirq.NumericalCompilationTargetGateset(
        [cirq.CZ, hardware_gate],
        base_gate_error_rates=[noisy_cz, reliable_fsim],
        random_state=random_state,
    )
    compiled = cirq.optimize_for_target_gateset(circuit, gateset=gateset)
    print('=== Circuit compiled with NumericalCompilationTargetGateset ===')
    print(compiled)
    print(f'output uses only gates from the gateset: {gateset.validate(compiled)}')


if __name__ == '__main__':
    main()
