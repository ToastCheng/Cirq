import pytest
import sympy
import numpy as np

from cirq.protocols import is_parameterized
from cirq.value.type_alias import _SYMENGINE_AVAILABLE

def test_is_parameterized_sympy():
    assert is_parameterized(sympy.Symbol('x'))
    assert is_parameterized(sympy.Symbol('y') + 1)

def test_is_parameterized_primitives():
    assert not is_parameterized("str")
    assert not is_parameterized(1.0)
    assert not is_parameterized(1)
    assert not is_parameterized(1j)
    assert not is_parameterized(np.int32(1))
    assert not is_parameterized(None)

@pytest.mark.skipif(not _SYMENGINE_AVAILABLE, reason="symengine not installed")
def test_is_parameterized_symengine():
    import symengine
    assert is_parameterized(symengine.Symbol('x'))
    assert is_parameterized(symengine.Symbol('y') + 1)

def test_is_parameterized_without_symengine(monkeypatch):
    import cirq.protocols.resolve_parameters as rp
    monkeypatch.setattr(rp, '_SYMENGINE_AVAILABLE', False)
    
    # Should still work for sympy
    assert rp.is_parameterized(sympy.Symbol('x'))
    
    if _SYMENGINE_AVAILABLE:
        import symengine
        # Even if passed a symengine object, if _SYMENGINE_AVAILABLE is False, it returns False.
        # This simulates the environment where symengine is not installed.
        assert not rp.is_parameterized(symengine.Symbol('x'))
