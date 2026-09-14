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

"""Target gateset compiling two-qubit operations via numerical optimization."""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from typing import Any, TYPE_CHECKING

from cirq import ops, protocols
from cirq.transformers.heuristic_decompositions import two_qubit_numerical_optimization
from cirq.transformers.target_gatesets import compilation_target_gateset

if TYPE_CHECKING:
    import cirq


class NumericalCompilationTargetGateset(
    compilation_target_gateset.TwoQubitCompilationTargetGateset
):
    r"""A target gateset that compiles circuits with the NuOp numerical compiler.

    This is a `cirq.CompilationTargetGateset` adapter around
    `cirq.two_qubit_gate_numerical_compilation`, so that
    `cirq.optimize_for_target_gateset` can compile whole circuits onto
    arbitrary hardware base gates. Use it for base gates that have no
    analytical decomposition in cirq (e.g. a calibrated `cirq.FSimGate`),
    or for noise-adaptive compilation via `base_gate_error_rates`. See
    `cirq.two_qubit_gate_numerical_compilation` for the decomposition
    algorithm, the fidelity definitions, and how the best decomposition
    is selected.

    Numerical optimization costs ~a second per two-qubit unitary, so
    results are memoized: circuits that repeat identical unitaries (QAOA
    layers, Trotter steps) compile each distinct unitary only once. For
    `cirq.CZ` or `cirq.SQRT_ISWAP` base gates without error rates, prefer
    `cirq.CZTargetGateset` or `cirq.SqrtIswapTargetGateset`, which use
    analytical decompositions and are exact and much faster.
    """

    def __init__(
        self,
        base_gates: cirq.Gate | Sequence[cirq.Gate],
        *,
        base_gate_error_rates: Sequence[float] | None = None,
        target_fidelity: float = 1 - 1e-8,
        max_layers: int = 3,
        single_qubit_error_rates: float | tuple[float, float] = 0.0,
        num_restarts: int = 3,
        maxiter: int = 1000,
        random_state: cirq.RANDOM_STATE_OR_SEED_LIKE = None,
        preserve_moment_structure: bool = True,
        reorder_operations: bool = False,
    ) -> None:
        """Initializes `cirq.NumericalCompilationTargetGateset`.

        Args:
            base_gates: A single two-qubit gate, or a sequence of them, onto
                which operations are compiled. Each gate must have a unitary
                and be entangling for arbitrary targets to be compilable.
            base_gate_error_rates: Optional hardware error rate of each base
                gate, one per entry of `base_gates`. When given, compilation
                is noise-adaptive; see
                `cirq.two_qubit_gate_numerical_compilation`.
            target_fidelity: Decomposition fidelity threshold.
            max_layers: Maximum number of base gate applications allowed per
                two-qubit operation.
            single_qubit_error_rates: Optional hardware error rate of each
                single-qubit gate, folded into the hardware fidelity when
                `base_gate_error_rates` is given. Either a single rate
                applied to both qubits, or a pair of rates (one per qubit).
            num_restarts: Number of random restarts of the numerical
                optimizer per (base gate, layer count) pair.
            maxiter: Maximum number of iterations of each optimizer run.
            random_state: Random state or seed used to generate the
                optimizer's starting points.
            preserve_moment_structure: Whether to preserve the moment
                structure of the circuit during compilation or not.
            reorder_operations: Whether to attempt to reorder the operations
                in order to reduce circuit depth or not (can be True only if
                preserve_moment_structure=False).

        Raises:
            ValueError: If `base_gates` is empty, contains gates that are not
                two-qubit unitary gates, or `base_gate_error_rates` does not
                match `base_gates`.
        """
        if isinstance(base_gates, ops.Gate):
            base_gates = (base_gates,)
        self._base_gates = tuple(base_gates)
        if not self._base_gates:
            raise ValueError('base_gates must be a non-empty sequence of two-qubit gates')
        for gate in self._base_gates:
            if (
                not isinstance(gate, ops.Gate)
                or protocols.num_qubits(gate) != 2
                or not protocols.has_unitary(gate)
            ):
                raise ValueError(f'base gates must be two-qubit gates with a unitary, got {gate!r}')
        if base_gate_error_rates is not None and len(base_gate_error_rates) != len(
            self._base_gates
        ):
            raise ValueError(
                f'Expected one error rate per base gate ({len(self._base_gates)}), '
                f'got {len(base_gate_error_rates)}'
            )
        self._base_gate_unitaries = tuple(protocols.unitary(gate) for gate in self._base_gates)
        self._base_gate_error_rates = (
            None if base_gate_error_rates is None else tuple(base_gate_error_rates)
        )
        self._target_fidelity = target_fidelity
        self._max_layers = max_layers
        self._single_qubit_error_rates = single_qubit_error_rates
        self._num_restarts = num_restarts
        self._maxiter = maxiter
        self._random_state = random_state
        self._compiler = two_qubit_numerical_optimization.TwoQubitNumericalCompiler(
            base_gates=self._base_gate_unitaries,
            base_gate_error_rates=self._base_gate_error_rates,
            target_fidelity=target_fidelity,
            max_layers=max_layers,
            single_qubit_error_rates=single_qubit_error_rates,
            num_restarts=num_restarts,
            maxiter=maxiter,
            random_state=random_state,
        )
        super().__init__(
            *self._base_gates,
            ops.MeasurementGate,
            ops.PhasedXZGate,
            ops.GlobalPhaseGate,
            name='NumericalCompilationTargetGateset',
            preserve_moment_structure=preserve_moment_structure,
            reorder_operations=reorder_operations,
        )

    @property
    def base_gates(self) -> tuple[cirq.Gate, ...]:
        """The base two-qubit gates onto which operations are compiled."""
        return self._base_gates

    def _decompose_two_qubit_operation(self, op: cirq.Operation, moment_idx: int) -> cirq.OP_TREE:
        if not protocols.has_unitary(op):
            return NotImplemented
        result = self._compiler.compile_two_qubit_gate(protocols.unitary(op))
        base_gate = self._base_gates[result.base_gate_index]
        q0, q1 = op.qubits
        decomposition: list[cirq.Operation] = []
        for i, (k0, k1) in enumerate(result.local_unitaries):
            decomposition.append(ops.PhasedXZGate.from_matrix(k0).on(q0))
            decomposition.append(ops.PhasedXZGate.from_matrix(k1).on(q1))
            if i < result.num_base_gates:
                decomposition.append(base_gate.on(q0, q1))
        return decomposition

    def _value_equality_values_(self) -> Any:
        return (
            self._base_gates,
            self._base_gate_error_rates,
            self._target_fidelity,
            self._max_layers,
            self._single_qubit_error_rates,
            self._num_restarts,
            self._maxiter,
            self._random_state,
        )

    def __repr__(self) -> str:
        gates = ', '.join(repr(gate) for gate in self._base_gates)
        if len(self._base_gates) == 1:
            gates += ','
        return (
            f'cirq.NumericalCompilationTargetGateset(base_gates=({gates}), '
            f'base_gate_error_rates={self._base_gate_error_rates!r}, '
            f'target_fidelity={self._target_fidelity!r}, '
            f'max_layers={self._max_layers!r}, '
            f'single_qubit_error_rates={self._single_qubit_error_rates!r}, '
            f'num_restarts={self._num_restarts!r}, '
            f'maxiter={self._maxiter!r}, '
            f'random_state={self._random_state!r})'
        )

    def _json_dict_(self) -> dict[str, Any]:
        if self._random_state is not None and not isinstance(self._random_state, numbers.Integral):
            raise ValueError(
                'NumericalCompilationTargetGateset can only be JSON-serialized when '
                'random_state is None or an integer seed, got '
                f'{type(self._random_state).__name__}.'
            )
        return {
            'base_gates': list(self._base_gates),
            'base_gate_error_rates': self._base_gate_error_rates,
            'target_fidelity': self._target_fidelity,
            'max_layers': self._max_layers,
            'single_qubit_error_rates': self._single_qubit_error_rates,
            'num_restarts': self._num_restarts,
            'maxiter': self._maxiter,
            'random_state': self._random_state,
        }

    @classmethod
    def _from_json_dict_(
        cls,
        base_gates,
        base_gate_error_rates,
        target_fidelity,
        max_layers,
        single_qubit_error_rates,
        num_restarts,
        maxiter,
        random_state,
        **kwargs,
    ):
        return cls(
            base_gates,
            base_gate_error_rates=base_gate_error_rates,
            target_fidelity=target_fidelity,
            max_layers=max_layers,
            single_qubit_error_rates=single_qubit_error_rates,
            num_restarts=num_restarts,
            maxiter=maxiter,
            random_state=random_state,
        )
