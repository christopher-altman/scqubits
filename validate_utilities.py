#!/usr/bin/env python
"""
Simple validation script for adiabatic utilities.
Runs without requiring full scqubits installation.
"""

import sys
import numpy as np

print("=" * 70)
print("Adiabatic Utilities Validation Script")
print("=" * 70)
print()

# Test 1: Import utilities
print("Test 1: Importing utilities...")
try:
    from scqubits.utils.state_tracking import (
        track_dressed_states,
        _compute_overlap_matrix,
        _find_optimal_mapping
    )
    from scqubits.utils.adiabatic_validator import (
        validate_adiabatic_ramp,
        _compute_energy_gaps,
        _compute_gap_derivative,
        _compute_landau_zener_parameter
    )
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

print()

# Test 2: Create mock sweep object
print("Test 2: Creating mock data...")

class MockSweep:
    def __init__(self, evals, evecs=None, params=None):
        self._data = {'evals': evals}
        if evecs is not None:
            self._data['evecs'] = evecs
        if params is not None:
            self._parameters = type('obj', (object,), {
                'paramvals_list': [params]
            })()
        else:
            self._parameters = None
        self._evals_count = evals.shape[-1]

# Create two-level system with avoided crossing
n_points = 50
params = np.linspace(0, 1, n_points)
gap = 0.1

E1_bare = params
E2_bare = 1 - params

eigenvalues = np.zeros((n_points, 2))
eigenvectors = np.zeros((n_points, 2, 2), dtype=complex)

for i, p in enumerate(params):
    H = np.array([[E1_bare[i], gap], [gap, E2_bare[i]]])
    evals, evecs = np.linalg.eigh(H)
    eigenvalues[i] = evals
    eigenvectors[i] = evecs.T

sweep = MockSweep(eigenvalues, eigenvectors, params)
print(f"  ✓ Created {n_points}-point sweep with avoided crossing")
print()

# Test 3: State tracking
print("Test 3: Testing state tracking...")
try:
    tracked = track_dressed_states(sweep, initial_labels=[0, 1], overlap_threshold=0.3)
    print(f"  ✓ State tracking completed")
    print(f"    - Eigenvalues shape: {tracked.eigenvalues.shape}")
    print(f"    - State labels shape: {tracked.state_labels.shape}")
    print(f"    - Continuity breaks: {len(tracked.continuity_breaks)}")
    print(f"    - Mean overlap: {np.mean(tracked.overlap_history):.4f}")
except Exception as e:
    print(f"  ✗ State tracking failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 4: Overlap matrix computation
print("Test 4: Testing overlap matrix computation...")
try:
    evecs1 = np.array([[1, 0], [0, 1]], dtype=complex)
    evecs2 = np.array([[1, 0], [0, 1]], dtype=complex)
    overlap = _compute_overlap_matrix(evecs1, evecs2)
    expected = np.eye(2)
    assert overlap.shape == (2, 2)
    assert np.allclose(overlap, expected, atol=1e-10)
    print("  ✓ Overlap matrix computation correct")
except AssertionError as e:
    print(f"  ✗ Overlap matrix test failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Overlap matrix computation failed: {e}")
    sys.exit(1)

print()

# Test 5: Optimal mapping
print("Test 5: Testing optimal state mapping...")
try:
    overlap_identity = np.eye(3)
    perm, max_overlaps, min_overlap = _find_optimal_mapping(overlap_identity, 0.5)
    assert np.array_equal(perm, [0, 1, 2])
    assert np.allclose(max_overlaps, [1.0, 1.0, 1.0])
    assert min_overlap == 1.0
    print("  ✓ Optimal mapping correct")
except AssertionError as e:
    print(f"  ✗ Optimal mapping test failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Optimal mapping failed: {e}")
    sys.exit(1)

print()

# Test 6: Energy gap computation
print("Test 6: Testing energy gap computation...")
try:
    test_evals = np.array([[0.0, 1.0, 2.0]])
    gaps = _compute_energy_gaps(test_evals, 0, 1)
    assert np.allclose(gaps, [1.0])
    print("  ✓ Energy gap computation correct")
except AssertionError as e:
    print(f"  ✗ Energy gap test failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Energy gap computation failed: {e}")
    sys.exit(1)

print()

# Test 7: Gap derivative
print("Test 7: Testing gap derivative computation...")
try:
    test_gaps = np.ones(10)
    test_params = np.linspace(0, 1, 10)
    dgap = _compute_gap_derivative(test_gaps, test_params)
    assert len(dgap) == 10
    assert np.allclose(dgap, 0.0, atol=1e-10)
    print("  ✓ Gap derivative computation correct")
except AssertionError as e:
    print(f"  ✗ Gap derivative test failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Gap derivative computation failed: {e}")
    sys.exit(1)

print()

# Test 8: Landau-Zener parameter
print("Test 8: Testing Landau-Zener parameter...")
try:
    test_gaps = np.array([1.0, 1.0])
    test_dgap_dt = np.array([0.01, 0.01])
    gamma = _compute_landau_zener_parameter(test_gaps, test_dgap_dt)
    # γ = πΔ²/(2|dΔ/dt|) = π*1/(2*0.01) ≈ 157
    assert np.all(gamma > 100)
    print(f"  ✓ Landau-Zener parameter correct (γ ≈ {gamma[0]:.1f})")
except AssertionError as e:
    print(f"  ✗ Landau-Zener parameter test failed: {e}")
    sys.exit(1)
except Exception as e:
    print(f"  ✗ Landau-Zener parameter computation failed: {e}")
    sys.exit(1)

print()

# Test 9: Full adiabatic validation
print("Test 9: Testing full adiabatic validation...")
try:
    report = validate_adiabatic_ramp(
        sweep,
        state_indices=(0, 1),
        threshold=0.1,
        use_tracked_states=False
    )
    print(f"  ✓ Adiabatic validation completed")
    print(f"    - Adiabatic: {report.is_adiabatic}")
    print(f"    - Violations: {len(report.violation_points)}/{n_points}")
    print(f"    - Min gap: {np.min(report.min_gap_trajectory):.6f}")
    print(f"    - Suggested ramp time: {report.suggested_ramp_time:.4f}")
except Exception as e:
    print(f"  ✗ Adiabatic validation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Test 10: Integration test - combined usage
print("Test 10: Testing combined usage (tracking + validation)...")
try:
    # First track states
    tracked = track_dressed_states(sweep, initial_labels=[0, 1])

    # Then validate with tracking
    report = validate_adiabatic_ramp(
        sweep,
        state_indices=(0, 1),
        threshold=0.05,
        use_tracked_states=True
    )

    print(f"  ✓ Combined usage successful")
    print(f"    - States tracked with {len(tracked.continuity_breaks)} breaks")
    print(f"    - Adiabatic validation: {report.is_adiabatic}")
except Exception as e:
    print(f"  ✗ Combined usage failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
print()
print("Summary:")
print("  - State tracking utility: WORKING")
print("  - Adiabatic validator utility: WORKING")
print("  - All core functions tested and validated")
print("  - Ready for integration with scqubits")
print()
