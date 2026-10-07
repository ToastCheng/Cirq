# Benchmark: internal symengine acceleration of ParamResolver (issue #7900).
#
# Compares sympy-only resolution (HAVE_SYMENGINE toggled off) against the
# internal symengine fast path, on sympy inputs only — the speedup is
# invisible to users. Run: python symengine_internal_bench.py

import timeit

import sympy as sp

import cirq
import cirq._compat_symbolic as compat
from cirq.study import resolver as resolver_mod

N_QUBITS = 50
N_POINTS = 80


def make_circuit(exponent_fn):
    qubits = cirq.GridQubit.rect(1, N_QUBITS)
    return cirq.Circuit(
        [cirq.X(q) ** exponent_fn(i) for i, q in enumerate(qubits)], cirq.measure_each(*qubits)
    )


def bench(label, fn, number=3):
    # Best-of-N wall time per full sweep.
    best = min(timeit.repeat(fn, number=1, repeat=number))
    print(f'  {label:<44} {best * 1e3:9.2f} ms', flush=True)
    return best


def compare(title, make_resolver_circuit, resolvers):
    print(f'\n=== {title} ===', flush=True)
    circuit = make_resolver_circuit()

    def sweep():
        for r in resolvers:
            cirq.resolve_parameters(circuit, r)

    resolver_mod.HAVE_SYMENGINE = False
    t_sympy = bench('sympy only (upstream behavior)', sweep)

    # Conversion cache disabled: every resolution pays full sympy->symengine
    # conversion, isolating symengine's substitution speed from caching.
    cached_to_symengine = compat._to_symengine
    compat._to_symengine = compat._to_symengine.__wrapped__
    resolver_mod.HAVE_SYMENGINE = True
    t_nocache = bench('internal symengine, NO conversion cache', sweep)

    compat._to_symengine = cached_to_symengine
    compat._to_symengine.cache_clear()
    t_se = bench('internal symengine, cached conversion', sweep)
    print(f'  speedup: {t_sympy / t_nocache:.2f}x uncached, {t_sympy / t_se:.2f}x cached')


def main():
    print(f'symengine available: {resolver_mod.HAVE_SYMENGINE}')
    print(f'circuit: {N_QUBITS} qubits x {N_POINTS} sweep points')
    syms = [sp.Symbol(f'a_{i}') for i in range(N_QUBITS)]
    points = [i / N_POINTS for i in range(N_POINTS)]

    # A. Pure Add/Mul arithmetic, fully resolving (today: recursive descent,
    #    no sympy.subs; the symengine hook is not expected to help here).
    compare(
        'A. a_i*2 + 1  (Add/Mul, full resolution)',
        lambda: make_circuit(lambda i: syms[i] * 2 + 1),
        [dict.fromkeys(syms, p) for p in points],
    )

    # B. Transcendental formulas, fully resolving (today: slow sympy.subs).
    compare(
        'B. sin(a_i * pi / 2)  (trig, full resolution)',
        lambda: make_circuit(lambda i: sp.sin(syms[i] * sp.pi / 2)),
        [dict.fromkeys(syms, p) for p in points],
    )

    # C. Partial resolution: sin(a_i + b_i), only a_i resolved — the result
    #    stays symbolic and must be handed back as a sympy expression.
    b_syms = [sp.Symbol(f'b_{i}') for i in range(N_QUBITS)]
    compare(
        'C. sin(a_i + b_i), resolve a only  (partial)',
        lambda: make_circuit(lambda i: sp.sin(syms[i] + b_syms[i])),
        [dict.fromkeys(syms, p) for p in points],
    )

    # D. Chained mappings: a_i -> t_i + 1, t_i -> number.
    t_syms = [sp.Symbol(f't_{i}') for i in range(N_QUBITS)]
    compare(
        'D. sin(a_i * pi), a_i -> t_i + 1 -> number  (chained)',
        lambda: make_circuit(lambda i: sp.sin(syms[i] * sp.pi)),
        [
            dict(zip(syms, [t + 1 for t in t_syms])) | dict(zip(t_syms, [p] * N_QUBITS))
            for p in points
        ],
    )

    # E. Mixed large formula: sin(a)*cos(b) + a**2/3 (today: sympy.subs).
    compare(
        'E. sin(a_i)*cos(b_i) + a_i**2/3  (compound, full resolution)',
        lambda: make_circuit(lambda i: sp.sin(syms[i]) * sp.cos(b_syms[i]) + syms[i] ** 2 / 3),
        [dict(zip(syms, [p] * N_QUBITS)) | dict(zip(b_syms, [2 * p] * N_QUBITS)) for p in points],
    )

    # F. Existing repo benchmark shape: bare symbols (dict lookup only).
    compare(
        'F. bare a_i  (baseline repo benchmark shape)',
        lambda: make_circuit(lambda i: syms[i]),
        [dict.fromkeys(syms, p) for p in points],
    )


if __name__ == '__main__':
    main()
