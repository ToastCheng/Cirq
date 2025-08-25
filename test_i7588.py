"""Test file to reproduce and verify fix for issue #7588: Inconsistent Pauli gate multiplication."""

import sys
import os

# Use local cirq-core for testing the fix
sys.path.insert(0, os.path.join(os.getcwd(), 'cirq-core'))

try:
    import cirq
    using_local = True
    print("Using local cirq-core")
except ImportError:
    import cirq
    using_local = False
    print("Using system cirq")


def test_pauli_multiplication_consistency():
    """Test that Pauli gate multiplication works consistently in all orders."""
    x = cirq.X(cirq.LineQubit(0))
    
    print(f"Testing with x = {x}")
    print(f"Expected result for all cases: {x**5}")
    print()
    
    # Test all four cases from the issue
    cases = [
        ("x**2 * x * x * x", lambda: x**2 * x * x * x),
        ("x * x**2 * x * x", lambda: x * x**2 * x * x),
        ("x * x * x**2 * x", lambda: x * x * x**2 * x),
        ("x * x * x * x**2", lambda: x * x * x * x**2),
    ]
    
    results = []
    expected = x**5
    
    for i, (description, operation) in enumerate(cases, 1):
        try:
            result = operation()
            print(f"Case {i}: {description} = {result} ✓")
            results.append(result)
            
            # Verify it equals expected
            if result == expected:
                print(f"  ✓ Equals expected: {expected}")
            else:
                print(f"  ✗ Expected {expected}, got {result}")
                
        except TypeError as e:
            print(f"Case {i}: {description} FAILED: {e} ✗")
            results.append(None)
        print()
    
    # Summary
    successful_cases = sum(1 for r in results if r is not None)
    print(f"Summary: {successful_cases}/4 cases successful")
    
    if using_local and successful_cases == 4:
        print("🎉 All cases work with the fix!")
    elif not using_local and successful_cases < 4:
        print("❌ Some cases still fail (expected without the fix)")
    
    return results


def test_other_pauli_gates():
    """Test multiplication consistency with other Pauli gates."""
    q = cirq.LineQubit(0)
    
    print("\n" + "="*50)
    print("Testing other Pauli gates:")
    print("="*50)
    
    gates = [
        ("Y", cirq.Y(q)),
        ("Z", cirq.Z(q)),
    ]
    
    for gate_name, gate in gates:
        print(f"\nTesting {gate_name} gate:")
        try:
            # Test the problematic case: gate * gate * gate**2 * gate
            result = gate * gate * gate**2 * gate
            expected = gate**5
            print(f"  {gate_name} * {gate_name} * {gate_name}**2 * {gate_name} = {result}")
            print(f"  Expected: {expected}")
            
            if result == expected:
                print(f"  ✓ Correct!")
            else:
                print(f"  ✗ Mismatch")
                
        except TypeError as e:
            print(f"  ✗ Failed: {e}")


if __name__ == "__main__":
    print("Testing Pauli gate multiplication consistency...")
    print("Issue #7588: https://github.com/quantumlib/Cirq/issues/7588")
    print("="*60)
    
    test_pauli_multiplication_consistency()
    test_other_pauli_gates()
    
    print("\n" + "="*60)
    print("Test completed.")