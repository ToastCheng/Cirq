# SymEngine Support in Cirq — POC Status

Tracking document for [quantumlib/Cirq#7900](https://github.com/quantumlib/Cirq/issues/7900):
accept `symengine` expressions anywhere sympy expressions are accepted, evaluating them
natively (C++ backend, much faster `subs`), as an **optional** dependency.

Scope decision (per issue discussion): this is an **acceptance layer**, not a replacement.
User code using sympy performs exactly as before; users who want faster symbolic evaluation
`pip install symengine` and use `symengine.Symbol`. No internal conversion between backends.

## Design

Single compat module, everything else imports predicates from it:

- `cirq/_compat_symbolic.py`
  - `HAVE_SYMENGINE` — True iff symengine is importable.
  - `is_symbolic(x)` / `is_symbol(x)` — isinstance over both backends (sympy always,
    symengine when installed).
  - `is_symengine_expr(x)` — False when symengine is absent; call sites never need
    `HAVE_SYMENGINE and ...`.
  - `symbolic_pi(x)` / `symbolic_lib(x)` — return the pi constant / symbolic module
    **matching the backend of x**, so library-built expressions don't mix backends.
  - `symbol_like(name, x)` — returns a `Symbol` named `name` from the backend matching
    `x` (used by `SympyCondition.replace_key` so substitutions stay on-backend).
  - `symengine_number_value(x)` — converts symengine numbers/constants to Python
    `int`/`float`/`complex`/`np.pi`, mirroring sympy number coercion in the resolver.
    Also folds constant expressions that are not `is_Number` but have no free symbols
    (e.g. `1.0/pi`) via `complex()`, matching what the sympy resolver path produces.
  - `SymbolicExpr` — annotation alias (`sympy.Basic | symengine.Basic`, degrading to
    just `sympy.Basic` without symengine). Note `sympy.Expr` ⊂ `sympy.Basic`, so it is
    already covered; the alias also admits sympy booleans, which Cirq conditions use.

## Changed files and why

| File | Role in the parameter lifecycle | Change |
|---|---|---|
| `value/type_alias.py` | public type vocabulary | `TParamKey/TParamVal/TParamValComplex` use `SymbolicExpr` |
| `protocols/resolve_parameters.py` | detection | `is_parameterized`, `parameter_names`, `resolve_parameters` accept symengine |
| `study/resolver.py` | evaluation (the perf point) | `ParamResolver` accepts symengine keys, native `subs` fast path, fixpoint recursion with `RecursionError` loop detection, numeric coercion |
| `ops/eigen_gate.py` | equality | `_value_equality_values_` no longer assumes non-sympy ⇒ floatable; diagram-angle `pi` uses `symbolic_pi` |
| `ops/common_gates.py` | gate math | `_pi()` and 3× `lib = sympy if parameterized` sites use `symbolic_pi`/`symbolic_lib` |
| `value/linear_dict.py` | symbolic coefficients | symbolic-vs-complex coercion via `is_symbolic`; symengine numbers coerced on `update`; `str()` formatting |
| `ops/pauli_string.py` | symbolic coefficients | coefficient checks via `is_symbolic` (2 sites) |
| `ops/classically_controlled_operation.py` | classical control | wraps symengine condition exprs in `SympyCondition` (was: stored raw → `AttributeError` in Circuit construction) |
| `value/condition.py` | classical control | `SympyCondition` generalized: `is_symbol` for key extraction, backend-matching `replace_key` via `symbol_like`; bitwise `Indexed` conditions and QASM export remain sympy-only |
| `protocols/json_serialization.py` + `json_resolver_cache.py` | persistence | symengine exprs round-trip via `str()` / `symengine.sympify` (`cirq_type: 'symengine.Basic'`, registered only when installed) |
| `_compat.py` | repr convention | `proper_repr` emits `symengine.sympify('...')` so resolver reprs stay evaluable |
| `pyproject.toml` | typing | `symengine.*` in mypy `ignore_missing_imports` (no py.typed upstream) |
| `_compat_symbolic_test.py` (new) | tests | 36 tests, all skipped when symengine absent |
| `symengine_example.py` (new, repo root) | demo | runnable walkthrough of every supported code path |

## Verified behavior (experiments)

- `symengine.subs` accepts **string keys** → no conversion layer needed in the resolver.
- Resolution is name-keyed, so cross-backend resolution works: a sympy-keyed resolver
  resolves symengine expressions and vice versa.
- End-to-end: parametrized gates → circuit → resolve → unitary → `run_sweep` → JSON
  round-trip → repr round-trip, all on symengine exponents.
- Backend stays consistent through gate math: `cirq.rx(se.Symbol('t')).exponent` is
  symengine; `_pauli_expansion_()` coefficients are symengine; numerically equal to the
  sympy path after substitution (X/Y/Z pow gates tested).

## Mixing hazards (documented, not fixed)

- **eq without hash agreement, demonstrable at every level of the stack**:
  `se.Symbol('t') == sympy.Symbol('t')` is True but hashes differ — and this
  propagates: `XPowGate(exponent=se.Symbol('t')) == XPowGate(exponent=sympy.Symbol('t'))`
  is True, yet `{g_se, g_sp}` has size 2 and dict lookups miss. Same for Operations
  and FrozenCircuits. Gates and circuits are dict keys/set members all over Cirq
  (caches, dedup, transformers), so this is a correctness footgun, not a curiosity.
  **Supported contract: one backend per workflow.**
  - Normalization was considered and rejected: hashing by `str(x)` is unsound
    (`1 + t**2` symengine vs `t**2 + 1` sympy — equal objects, different strings);
    hashing by a sympy str-round-trip canonical form is sound but puts string
    parsing on the gate-hashing hot path. The eq/hash mismatch itself lives in
    the two third-party libraries and cannot be fixed from Cirq.
- **Operand-order promotion**: `se * sympy` → symengine, `sympy * se` → sympy (sympy's
  `__mul__` never returns NotImplemented, so reflected dispatch can't be forced without
  monkeypatching — rejected). Mitigation: library code uses `symbolic_pi`/`symbolic_lib`
  so it never *creates* mixed expressions; user-level mixing is unsupported.
- `parameter_symbols()` always returns `sympy.Symbol` (rebuilt from names) — by design.
- `cirq.flatten` works on symengine formulas but produces a mixed representation
  (sympy combined `<a + b>` placeholder symbol + symengine ExpressionMap entry).
  Verified benign: sweeps keyed by `se.Symbol` construct and run fine.

## Known remaining gaps

- ~42 non-test files still contain sympy-specific `isinstance` checks (sweeps,
  duration, angle, quirk/quantikz exporters, clifford sim, ...). Symengine inputs there
  are **untested, not necessarily broken**; migrate on demand using `is_symbolic`.
- Classical control: bitwise conditions (`sympy.IndexedBase`) and the `qasm` property
  of `SympyCondition` remain sympy-only (no symengine equivalent of Indexed).
- `assert_equivalent_repr` default setup lacks `import symengine`; tests pass a custom
  `setup_code`.
- No CI job with symengine installed yet.
- No benchmark numbers yet (`benchmarks/parameter_resolution_perf.py` with symengine
  expressions is the obvious next measurement).

## Fixed after initial POC

- **Classical control silently misbehaved** (was the most dangerous uncovered call
  site): `cirq.X(q).with_classical_controls(se.Symbol('m') > 0)` constructed fine and
  `is_parameterized` returned False — but that part is parity with sympy, which also
  reports False (conditions reference measurement keys, not sweep parameters). The real
  bug: the wrap check `isinstance(c, sympy.Basic)` missed symengine, so the raw
  relational was stored unwrapped and Circuit construction died with
  `AttributeError: 'StrictLessThan' object has no attribute 'keys'`. Fixed by wrapping
  via `is_symbolic` and generalizing `SympyCondition` (class name kept for JSON compat).

## Tooling gotchas hit (for the write-up)

- `no_implicit_reexport = true` in pyproject.toml: call sites must import **functions**
  from `_compat_symbolic`, not the `symengine` module object.
- symengine has no `py.typed` → mypy overrides entry required.
- symengine numbers are **not** `numbers.Number` instances — ordering of isinstance
  checks matters.
- symengine `pi` is a symbolic `Constant`, not a Number: a resolved expression like
  `1.0/pi` has no free symbols yet fails `is_Number`. Numeric coercion must fall back
  to `complex()` for constant expressions (found via `cirq.rx(t*2)` resolving to a
  symbolic `1.0/pi` instead of a float).
- Repo formats with black (`skip-string-normalization`, line-length 100), not ruff format.

## How to run

```bash
export PATH="$HOME/.venv/cirq-dev/bin:$PATH"
python -m pytest cirq-core/cirq/_compat_symbolic_test.py -q   # 36 tests
python symengine_example.py                                    # usage demo
```
