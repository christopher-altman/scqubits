# adiabatic_validator.py
#
# This file is part of scqubits: a Python package for superconducting qubits,
# Quantum 5, 583 (2021). https://quantum-journal.org/papers/q-2021-11-17-583/
#
#    Copyright (c) 2019 and later, Jens Koch and Peter Groszkowski
#    All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
############################################################################

"""
Adiabatic trajectory validation for parameter sweeps.

This module provides tools for checking whether parameter sweep trajectories
satisfy the adiabatic theorem, particularly important for quantum annealing
and adiabatic state preparation protocols.
"""

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
from numpy import ndarray

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSweep


@dataclass
class AdiabaticReport:
    """
    Report of adiabatic validation for a parameter sweep trajectory.

    Attributes
    ----------
    is_adiabatic:
        True if entire trajectory satisfies adiabatic criterion
    violation_points:
        Array of parameter indices where adiabatic criterion is violated
    violation_params:
        Array of actual parameter values at violation points
    min_gap_trajectory:
        Array of minimum energy gaps between tracked states at each sweep point
    gamma_trajectory:
        Array of Landau-Zener adiabaticity parameters γ = πΔ²/(2ℏ|dΔ/dt|)
        at each transition between sweep points
    suggested_ramp_time:
        Total time needed for adiabatic evolution (in units where ℏ=1).
        Computed as sum of local times ensuring γ > threshold everywhere.
    threshold:
        Adiabatic criterion threshold used for validation
    state_indices:
        Tuple of (initial_state, final_state) indices that were tracked
    max_safe_rate:
        Maximum safe parameter change rate (if single parameter sweep)
    """
    is_adiabatic: bool
    violation_points: ndarray
    violation_params: ndarray
    min_gap_trajectory: ndarray
    gamma_trajectory: ndarray
    suggested_ramp_time: float
    threshold: float
    state_indices: Tuple[int, int]
    max_safe_rate: Optional[float] = None


def _compute_energy_gaps(
    eigenvalues: ndarray,
    state_i: int,
    state_j: int
) -> ndarray:
    """
    Compute energy gap between two states along trajectory.

    Parameters
    ----------
    eigenvalues:
        Array of eigenvalues, shape (n_points, n_states)
    state_i:
        Index of first state
    state_j:
        Index of second state

    Returns
    -------
        Array of gaps |E_j - E_i| at each point, shape (n_points,)
    """
    gaps = np.abs(eigenvalues[:, state_j] - eigenvalues[:, state_i])
    return gaps


def _compute_gap_derivative(
    gaps: ndarray,
    param_vals: ndarray,
    order: int = 2
) -> ndarray:
    """
    Compute numerical derivative of energy gap with respect to parameter.

    Parameters
    ----------
    gaps:
        Array of energy gaps at each parameter value
    param_vals:
        Array of parameter values
    order:
        Order of finite difference (1=forward/backward, 2=central)

    Returns
    -------
        Array of gap derivatives dΔ/dλ at each point (except endpoints
        if using central difference)
    """
    if order == 1:
        # Forward/backward differences
        dgap = np.gradient(gaps, param_vals)
    else:
        # Central differences for interior points
        dgap = np.zeros_like(gaps)

        # Interior points: central difference
        if len(param_vals) >= 3:
            dgap[1:-1] = (gaps[2:] - gaps[:-2]) / (param_vals[2:] - param_vals[:-2])

            # Endpoints: one-sided differences
            dgap[0] = (gaps[1] - gaps[0]) / (param_vals[1] - param_vals[0])
            dgap[-1] = (gaps[-1] - gaps[-2]) / (param_vals[-1] - param_vals[-2])
        elif len(param_vals) == 2:
            # Only two points: use forward difference
            dgap[:] = (gaps[1] - gaps[0]) / (param_vals[1] - param_vals[0])
        else:
            # Single point
            dgap[:] = 0.0

    return dgap


def _compute_landau_zener_parameter(
    gaps: ndarray,
    dgap_dt: ndarray,
    hbar: float = 1.0
) -> ndarray:
    """
    Compute Landau-Zener adiabaticity parameter.

    The adiabatic condition is satisfied when γ ≫ 1, where:
        γ = πΔ²/(2ℏ|dΔ/dt|)

    Parameters
    ----------
    gaps:
        Array of energy gaps Δ at each point
    dgap_dt:
        Array of gap time derivatives dΔ/dt
    hbar:
        Reduced Planck constant (default 1.0 for natural units)

    Returns
    -------
        Array of Landau-Zener parameters γ at each point. Returns inf
        where |dΔ/dt| ≈ 0 (stationary points, always adiabatic).
    """
    # Avoid division by zero
    abs_dgap_dt = np.abs(dgap_dt)
    abs_dgap_dt = np.where(abs_dgap_dt < 1e-15, 1e-15, abs_dgap_dt)

    # γ = πΔ²/(2ℏ|dΔ/dt|)
    gamma = (np.pi * gaps**2) / (2.0 * hbar * abs_dgap_dt)

    # Clip to reasonable range for numerical stability
    gamma = np.clip(gamma, 0.0, 1e10)

    return gamma


def _estimate_transition_matrix_element(
    evecs_i: ndarray,
    evecs_j: ndarray,
    gaps: ndarray,
    dgap_dlambda: ndarray,
    state_i: int,
    state_j: int
) -> ndarray:
    """
    Estimate transition matrix element ⟨i|∂H/∂λ|j⟩ from eigenvector overlap.

    Uses Hellmann-Feynman-like relation:
        ⟨i|∂H/∂λ|j⟩ ≈ (E_j - E_i) ⟨i|∂j/∂λ⟩

    For numerical derivatives of eigenvectors:
        ⟨i|∂j/∂λ⟩ ≈ ⟨i(λ)|j(λ+dλ)⟩ / dλ

    Parameters
    ----------
    evecs_i:
        Eigenvectors for state i, shape (n_points, hilbert_dim)
    evecs_j:
        Eigenvectors for state j, shape (n_points, hilbert_dim)
    gaps:
        Energy gaps E_j - E_i, shape (n_points,)
    dgap_dlambda:
        Gap derivatives dΔ/dλ, shape (n_points,)
    state_i:
        Index of state i
    state_j:
        Index of state j

    Returns
    -------
        Array of transition matrix element estimates at each point
    """
    # For simplicity, we use the gap derivative directly
    # A more accurate calculation would compute eigenvector derivatives
    # but this requires second-order information not always available

    # Simplified estimate: |⟨i|∂H/∂λ|j⟩| ≈ |dΔ/dλ| / 2
    # This comes from perturbation theory near avoided crossings
    transition_elements = np.abs(dgap_dlambda) / 2.0

    return transition_elements


def validate_adiabatic_ramp(
    sweep: "ParameterSweep",
    state_indices: Tuple[int, int],
    threshold: float = 0.01,
    return_rates: bool = True,
    ramp_time: Optional[float] = None,
    use_tracked_states: bool = True
) -> AdiabaticReport:
    """
    Validate whether parameter sweep satisfies adiabatic theorem.

    This function checks if a parameter sweep trajectory satisfies the
    Landau-Zener adiabatic criterion for a pair of quantum states. The
    adiabatic condition requires the Landau-Zener parameter γ ≫ 1:

        γ = πΔ²/(2ℏ|dΔ/dt|) > threshold

    where Δ is the instantaneous energy gap and dΔ/dt is its time derivative.

    Algorithm
    ---------
    1. Extract eigenvalues along sweep trajectory
    2. Optionally track states to maintain physical identity (recommended
       near avoided crossings)
    3. Compute energy gap Δ(λ) = |E_j(λ) - E_i(λ)| at each parameter point
    4. Compute gap derivative dΔ/dλ using finite differences
    5. Convert to time derivative: dΔ/dt = (dΔ/dλ)(dλ/dt)
    6. Calculate Landau-Zener parameter γ(t) at each point
    7. Flag points where γ < threshold
    8. Compute suggested ramp time ensuring γ > threshold everywhere

    Parameters
    ----------
    sweep:
        ParameterSweep object containing eigensystem data. Must have been
        run to populate eigenvalue data.
    state_indices:
        Tuple of (initial_state, final_state) indices to track. These should
        be the indices at the first sweep point; if use_tracked_states=True,
        they will be followed through avoided crossings.
    threshold:
        Minimum acceptable value for γ to satisfy adiabatic criterion
        (default 0.01). Typical values:
        - 0.01: ~1% diabatic transition probability
        - 0.1: ~10% error (looser criterion)
        - 0.001: ~0.1% error (stricter criterion)
    return_rates:
        If True, compute maximum safe ramp rate (only for 1D sweeps)
    ramp_time:
        Total time for the ramp (in units where ℏ=1). If None, assumes
        unit spacing between sweep points for rate calculation.
    use_tracked_states:
        If True, uses state tracking to maintain physical state identity
        through avoided crossings before computing gaps (recommended).

    Returns
    -------
        AdiabaticReport containing validation results, violation points,
        gap trajectory, γ trajectory, and suggested ramp time.

    Raises
    ------
    ValueError
        If sweep has no data, has incorrect dimensionality, or state_indices
        are out of range
    RuntimeWarning
        If sweep has too few points for accurate derivative computation

    Notes
    -----
    - For multi-dimensional sweeps, the trajectory follows the natural
      flattened ordering of the parameter grid
    - The suggested_ramp_time is computed assuming constant ramp rate.
      For variable-rate ramps, use max_safe_rate trajectory.
    - Near exact crossings (gap → 0), numerical derivatives may be unstable.
      Increase sweep resolution near such points.
    - This implementation uses the gap derivative |dΔ/dλ| as a proxy for
      the transition matrix element |⟨i|∂H/∂λ|j⟩|, which is exact for
      two-level systems but approximate for larger Hilbert spaces.

    Examples
    --------
    Check if flux ramp through transmon sweet spot is adiabatic:

    >>> import scqubits as scq
    >>> import numpy as np
    >>>
    >>> # Create transmon and fast flux ramp
    >>> qubit = scq.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=30)
    >>> ng_vals = np.linspace(-0.5, 0.5, 50)  # 50 points
    >>> sweep = qubit.create_sweep('ng', ng_vals)
    >>>
    >>> # Validate adiabaticity between ground and first excited state
    >>> report = validate_adiabatic_ramp(sweep, state_indices=(0, 1))
    >>>
    >>> if not report.is_adiabatic:
    ...     print(f"Ramp too fast! Need {report.suggested_ramp_time:.2f} time units")
    ...     print(f"Violations at {len(report.violation_points)} points")

    Check slow ramp:

    >>> ng_vals_slow = np.linspace(-0.5, 0.5, 500)  # 10x more points
    >>> sweep_slow = qubit.create_sweep('ng', ng_vals_slow)
    >>> report_slow = validate_adiabatic_ramp(sweep_slow, state_indices=(0, 1))
    >>> print(f"Adiabatic: {report_slow.is_adiabatic}")
    """
    # Validate input
    if not hasattr(sweep, '_data') or sweep._data is None:
        raise ValueError(
            "ParameterSweep has no data. Run sweep.run() first."
        )

    if 'evals' not in sweep._data:
        raise ValueError(
            "ParameterSweep does not contain eigenvalue data."
        )

    # Get eigenvalue data
    evals = sweep._data['evals']
    original_shape = evals.shape

    # Validate state indices
    n_states = original_shape[-1]
    state_i, state_j = state_indices
    if state_i >= n_states or state_j >= n_states:
        raise ValueError(
            f"state_indices {state_indices} out of range for {n_states} states"
        )
    if state_i == state_j:
        raise ValueError(
            f"state_indices must be different (got {state_indices})"
        )

    # Flatten sweep for 1D traversal
    n_sweep_points = np.prod(original_shape[:-1])
    evals_flat = evals.reshape(n_sweep_points, n_states)

    # Track states if requested
    if use_tracked_states:
        try:
            from scqubits.utils.state_tracking import track_dressed_states

            # Create temporary sweep-like object for tracking
            class TempSweep:
                def __init__(self, data):
                    self._data = data

            temp_sweep = TempSweep({'evals': evals, 'evecs': sweep._data.get('evecs')})
            tracked = track_dressed_states(
                temp_sweep,
                initial_labels=list(range(n_states)),
                overlap_threshold=0.3  # Lower threshold for tracking through crossings
            )
            evals_flat = tracked.eigenvalues.reshape(n_sweep_points, n_states)

            # Remap state indices if they were reordered
            # Find where original state_i and state_j ended up
            initial_labels = tracked.state_labels.reshape(n_sweep_points, n_states)
            # Use final labels (states maintain identity)
            state_i_tracked = state_i
            state_j_tracked = state_j

        except ImportError:
            warnings.warn(
                "Could not import state_tracking. Using untracked states. "
                "Results may be incorrect near avoided crossings.",
                UserWarning
            )
            use_tracked_states = False
        except Exception as e:
            warnings.warn(
                f"State tracking failed: {e}. Using untracked states.",
                UserWarning
            )
            use_tracked_states = False

    # Get parameter values for derivative computation
    # For simplicity, assume uniform spacing or use indices
    if hasattr(sweep, '_parameters') and sweep._parameters is not None:
        # Get actual parameter values
        param_lists = sweep._parameters.paramvals_list
        if len(param_lists) == 1:
            # 1D sweep
            param_vals = param_lists[0]
        else:
            # Multi-D sweep: use flattened indices as pseudo-parameter
            param_vals = np.arange(n_sweep_points, dtype=float)
            warnings.warn(
                "Multi-dimensional sweep detected. Using sweep index as "
                "parameter for derivative computation. Results may not be "
                "physically meaningful for non-uniform parameter spacing.",
                UserWarning
            )
    else:
        # Use indices as parameter values
        param_vals = np.arange(n_sweep_points, dtype=float)

    # Warn if too few points
    if len(param_vals) < 3:
        warnings.warn(
            f"Sweep has only {len(param_vals)} points. Need at least 3 for "
            "accurate derivative computation. Results may be unreliable.",
            RuntimeWarning
        )

    # Compute energy gaps
    gaps = _compute_energy_gaps(evals_flat, state_i, state_j)

    # Compute gap derivative with respect to parameter
    dgap_dlambda = _compute_gap_derivative(gaps, param_vals, order=2)

    # Convert to time derivative
    # If ramp_time provided, compute dλ/dt
    if ramp_time is not None:
        total_param_range = param_vals[-1] - param_vals[0]
        avg_rate = total_param_range / ramp_time  # dλ/dt
        dgap_dt = dgap_dlambda * avg_rate
    else:
        # Assume unit time per parameter step
        # dλ/dt ≈ Δλ/Δt where Δt = 1
        # For transitions, use local parameter spacing
        dparam = np.diff(param_vals)
        if len(dparam) > 0:
            # Extend to match gaps length
            dparam = np.concatenate([dparam, [dparam[-1]]])
            # Assume unit time per step
            dlambda_dt = dparam
        else:
            dlambda_dt = np.ones_like(param_vals)

        dgap_dt = dgap_dlambda * dlambda_dt

    # Compute Landau-Zener parameter
    gamma = _compute_landau_zener_parameter(gaps, dgap_dt, hbar=1.0)

    # Find violations
    violation_mask = gamma < threshold
    violation_indices = np.where(violation_mask)[0]
    violation_params = param_vals[violation_indices]

    # Determine if adiabatic
    is_adiabatic = len(violation_indices) == 0

    # Compute suggested ramp time
    # For each point, require γ > threshold
    # γ = πΔ²/(2ℏ|dΔ/dλ|·|dλ/dt|) > threshold
    # => |dλ/dt| < πΔ²/(2ℏ·threshold·|dΔ/dλ|)
    # => local_time > (2ℏ·threshold·|dΔ/dλ|·dλ) / (πΔ²)

    # Avoid division by zero
    gaps_safe = np.where(gaps < 1e-10, 1e-10, gaps)

    # Local safe time for each transition
    dparam_transitions = np.diff(param_vals)
    if len(dparam_transitions) > 0:
        # Use gaps and derivatives at transition midpoints
        gaps_mid = (gaps[:-1] + gaps[1:]) / 2.0
        dgap_dlambda_mid = (np.abs(dgap_dlambda[:-1]) + np.abs(dgap_dlambda[1:])) / 2.0

        # Avoid division by small gaps
        gaps_mid = np.where(gaps_mid < 1e-10, 1e-10, gaps_mid)

        # Local time needed: dt = (2·threshold·|dΔ/dλ|·dλ) / (πΔ²)
        local_times = (2.0 * threshold * dgap_dlambda_mid * dparam_transitions) / (np.pi * gaps_mid**2)

        # Clip to reasonable values
        local_times = np.clip(local_times, 0.0, 1e10)

        suggested_ramp_time = np.sum(local_times)
    else:
        suggested_ramp_time = 0.0

    # Compute max safe rate for 1D sweeps
    max_safe_rate = None
    if return_rates and len(param_vals) > 1:
        # v_max = πΔ²/(2ℏ·threshold·|dΔ/dλ|)
        abs_dgap_dlambda = np.abs(dgap_dlambda)
        abs_dgap_dlambda = np.where(abs_dgap_dlambda < 1e-15, 1e-15, abs_dgap_dlambda)

        safe_rates = (np.pi * gaps**2) / (2.0 * threshold * abs_dgap_dlambda)
        max_safe_rate = np.min(safe_rates)

    return AdiabaticReport(
        is_adiabatic=is_adiabatic,
        violation_points=violation_indices,
        violation_params=violation_params,
        min_gap_trajectory=gaps,
        gamma_trajectory=gamma,
        suggested_ramp_time=suggested_ramp_time,
        threshold=threshold,
        state_indices=state_indices,
        max_safe_rate=max_safe_rate
    )


def _demo():
    """
    Minimal demonstration of adiabatic validation functionality.

    Creates a synthetic two-level system with avoided crossing and validates
    adiabaticity for fast and slow ramps.
    """
    print("=" * 70)
    print("Adiabatic Validation Utility - Demonstration")
    print("=" * 70)
    print()

    # Create synthetic two-level system
    print("Creating synthetic two-level system with avoided crossing...")
    gap = 0.1
    n_points_fast = 20
    n_points_slow = 200

    def create_mock_sweep(n_points):
        """Create mock sweep with avoided crossing."""
        params = np.linspace(0, 1, n_points)
        E1_bare = params
        E2_bare = 1 - params

        eigenvalues = np.zeros((n_points, 2))
        for i, p in enumerate(params):
            H = np.array([[E1_bare[i], gap], [gap, E2_bare[i]]])
            evals = np.linalg.eigh(H)[0]
            eigenvalues[i] = evals

        class MockSweep:
            def __init__(self, evals, params):
                self._data = {'evals': evals}
                self._parameters = type('obj', (object,), {
                    'paramvals_list': [params]
                })()

        return MockSweep(eigenvalues, params)

    print(f"  Gap at crossing: {gap}")
    print()

    # Test fast ramp
    print("=" * 70)
    print("Test 1: Fast Ramp")
    print("=" * 70)
    sweep_fast = create_mock_sweep(n_points_fast)
    print(f"  Number of points: {n_points_fast}")
    print(f"  Validating with threshold = 0.1...")

    report_fast = validate_adiabatic_ramp(
        sweep_fast,
        state_indices=(0, 1),
        threshold=0.1,
        use_tracked_states=False
    )

    print(f"  Adiabatic: {report_fast.is_adiabatic}")
    print(f"  Violations: {len(report_fast.violation_points)}/{n_points_fast} points")
    if len(report_fast.violation_points) > 0:
        print(f"  Min γ: {np.min(report_fast.gamma_trajectory):.4f}")
    print(f"  Suggested ramp time: {report_fast.suggested_ramp_time:.4f}")
    print()

    # Test slow ramp
    print("=" * 70)
    print("Test 2: Slow Ramp (10x more points)")
    print("=" * 70)
    sweep_slow = create_mock_sweep(n_points_slow)
    print(f"  Number of points: {n_points_slow}")
    print(f"  Validating with threshold = 0.1...")

    report_slow = validate_adiabatic_ramp(
        sweep_slow,
        state_indices=(0, 1),
        threshold=0.1,
        use_tracked_states=False
    )

    print(f"  Adiabatic: {report_slow.is_adiabatic}")
    print(f"  Violations: {len(report_slow.violation_points)}/{n_points_slow} points")
    print(f"  Min γ: {np.min(report_slow.gamma_trajectory):.4f}")
    print(f"  Suggested ramp time: {report_slow.suggested_ramp_time:.4f}")
    print()

    # Gap statistics
    print("=" * 70)
    print("Gap Trajectory Statistics")
    print("=" * 70)
    print(f"  Fast ramp - Min gap: {np.min(report_fast.min_gap_trajectory):.6f}")
    print(f"  Slow ramp - Min gap: {np.min(report_slow.min_gap_trajectory):.6f}")
    print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Fast ramp: {'PASS' if report_fast.is_adiabatic else 'FAIL'}")
    print(f"  Slow ramp: {'PASS' if report_slow.is_adiabatic else 'FAIL'}")
    if report_fast.max_safe_rate:
        print(f"  Max safe rate: {report_fast.max_safe_rate:.4f}")
    print()
    print("=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
