#!/usr/bin/env python3
"""Benchmark PauliString/PauliSum sparse matrix implementations.

Usage:
    python benchmark_sparse_matrix.py
"""

import time
import tracemalloc
import numpy as np
import cirq


def benchmark(func, *args, target_time=0.1):
    """Time and memory for a function call.

    Runs the function enough times to reach at least ``target_time`` seconds
    for a stable average.  Peak memory is from the final batch of runs.
    """
    # Warm-up
    result = func(*args)

    # Single-run timing to decide how many repeats we need
    t0 = time.perf_counter()
    func(*args)
    single = time.perf_counter() - t0

    n_runs = max(1, int(target_time / single))

    tracemalloc.start()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        result = func(*args)
    elapsed = (time.perf_counter() - t0) / n_runs
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak


def random_pauli_string(n_qubits, seed=0):
    """Generate a random PauliString on n_qubits."""
    rng = np.random.RandomState(seed)
    qubits = cirq.LineQubit.range(n_qubits)
    paulis = [cirq.X, cirq.Y, cirq.Z, cirq.I]
    n_act = rng.randint(1, n_qubits + 1)
    qs = rng.choice(qubits, size=n_act, replace=False)
    pstr = cirq.PauliString({q: rng.choice(paulis) for q in qs})
    return (rng.randn() + 1j * rng.randn()) * pstr, qubits


def random_pauli_sum(n_qubits, n_terms, seed=0):
    """Generate a random PauliSum with n_terms on n_qubits."""
    rng = np.random.RandomState(seed)
    qubits = cirq.LineQubit.range(n_qubits)
    paulis = [cirq.X, cirq.Y, cirq.Z, cirq.I]
    terms = []
    for _ in range(n_terms):
        n_act = rng.randint(1, n_qubits + 1)
        qs = rng.choice(qubits, size=n_act, replace=False)
        pstr = cirq.PauliString({q: rng.choice(paulis) for q in qs})
        terms.append((rng.randn() + 1j * rng.randn()) * pstr)
    return sum(terms, cirq.PauliSum()), qubits


def run_pauli_string(label, n_qubits):
    """Benchmark PauliString.sparse_matrix()."""
    print(f"\n{'='*70}")
    print(f"  PauliString: {label} ({n_qubits} qubits)")
    print(f"{'='*70}")

    pstr, qubits = random_pauli_string(n_qubits, seed=42)
    dim = 2**n_qubits

    # ---- Naive ----
    naive_result, naive_t, naive_peak = benchmark(pstr.sparse_matrix_naive, qubits)
    print(f"Naive (kron)  {naive_t*1000:8.2f} ms | " f"peak mem {naive_peak/1024**2:7.2f} MB | ")

    # ---- Direct ----
    direct_result, direct_t, direct_peak = benchmark(pstr.sparse_matrix, qubits)
    print(
        f"Direct (bit)  {direct_t*1000:8.2f} ms | "
        f"peak mem {direct_peak/1024**2:7.2f} MB | "
        # f"speedup {naive_t/direct_t:.1f}x"
    )

    # Dense baseline
    if dim <= 4096:
        _, dense_t, dense_peak = benchmark(pstr.matrix, qubits)
        dense_mb = (dim * dim * 16) / (1024 * 1024)
        print(
            f"Dense         {dense_t*1000:8.2f} ms | "
            f"peak mem {dense_peak/1024**2:7.2f} MB | "
            # f"result {dense_mb:6.2f} MB"
        )

    # Correctness
    assert np.allclose(naive_result.toarray(), direct_result.toarray())
    dense = pstr.matrix(qubits)
    assert np.allclose(dense, direct_result.toarray())
    print("  Correctness: PASS (sparse == naive == dense)")


def run_pauli_sum(label, n_qubits, n_terms):
    """Benchmark PauliSum.sparse_matrix()."""
    print(f"\n{'='*70}")
    print(f"  PauliSum: {label} ({n_qubits} qubits, {n_terms} terms)")
    print(f"{'='*70}")

    ps, qubits = random_pauli_sum(n_qubits, n_terms, seed=42)
    dim = 2**n_qubits

    # ---- OpenFermion ----
    of_t = None
    try:
        from openfermion import QubitOperator
        from openfermion.linalg import qubit_operator_sparse

        qop = QubitOperator()
        for pstr in ps:
            of_term = []
            for q, pauli in sorted(pstr.items(), key=lambda x: qubits.index(x[0])):
                of_term.append((qubits.index(q), pauli._name))
            qop += QubitOperator(of_term, complex(pstr.coefficient))

        of_result, of_t, of_peak = benchmark(qubit_operator_sparse, qop, n_qubits)
        print(f"OpenFermion   {of_t*1000:8.2f} ms | " f"peak mem {of_peak/1024**2:7.2f} MB | ")
    except Exception as e:
        print(f"OpenFermion   SKIP: {e}")

    # ---- Naive ----
    naive_result, naive_t, naive_peak = benchmark(ps.sparse_matrix_naive, qubits)
    print(f"Naive (kron)  {naive_t*1000:8.2f} ms | " f"peak mem {naive_peak/1024**2:7.2f} MB | ")

    # ---- Direct ----
    direct_result, direct_t, direct_peak = benchmark(ps.sparse_matrix, qubits)
    print(
        f"Direct (bit)  {direct_t*1000:8.2f} ms | "
        f"peak mem {direct_peak/1024**2:7.2f} MB | "
        # f"speedup {naive_t/direct_t:.1f}x"
    )

    # Dense baseline
    if dim <= 4096:
        _, dense_t, dense_peak = benchmark(ps.matrix, qubits)
        dense_mb = (dim * dim * 16) / (1024 * 1024)
        print(
            f"Dense         {dense_t*1000:8.2f} ms | "
            f"peak mem {dense_peak/1024**2:7.2f} MB | "
            # f"result {dense_mb:6.2f} MB"
        )

    # Correctness
    assert np.allclose(naive_result.toarray(), direct_result.toarray())
    if of_t is not None:
        assert np.allclose(of_result.toarray(), direct_result.toarray())
    dense = ps.matrix(qubits)
    assert np.allclose(dense, direct_result.toarray())
    print("  Correctness: PASS (sparse == naive == OpenFermion == dense)")

    # if of_t is not None:
        # print(f"  vs OpenFermion: {of_t/direct_t:.1f}x faster")


if __name__ == "__main__":
    print("Benchmarking PauliString.sparse_matrix()")
    for n in [
        # 4,
        # 6,
        # 8,
        # 10,
        12
    ]:
        run_pauli_string(f"n={n}", n)

    print("\n" + "=" * 70)
    print("Benchmarking PauliSum.sparse_matrix()")
    configs = [
        # ("Tiny", 4, 10),
        # ("Small", 6, 50),
        # ("Medium", 8, 100),
        # ("Large", 10, 500),
        # ("Heavy", 12, 1000),
        ("", 12, 1000)
    ]
    for label, n_qubits, n_terms in configs:
        run_pauli_sum(label, n_qubits, n_terms)
