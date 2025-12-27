# state_tracking.py
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
Dressed-state tracking utility for maintaining physical state identity across
avoided crossings in parameter sweeps.

This module provides functionality to reorder eigenvalues and eigenvectors along
a parameter sweep trajectory to follow adiabatic state evolution, preventing
discontinuities at avoided level crossings.
"""

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from numpy import ndarray
from scipy.optimize import linear_sum_assignment

if TYPE_CHECKING:
    from scqubits.core.param_sweep import ParameterSweep


@dataclass
class TrackedStates:
    """
    Container for state tracking results along a parameter sweep.

    Attributes
    ----------
    eigenvalues:
        2D array [sweep_point, state] of eigenvalues reordered to follow
        physical state evolution
    eigenvectors:
        3D array [sweep_point, state, basis] of eigenvectors (if available)
        reordered to match eigenvalues
    state_labels:
        2D array [sweep_point, state] tracking which original index each
        physical state corresponds to at each sweep point
    continuity_breaks:
        List of sweep point indices where maximum overlap between consecutive
        points falls below threshold, indicating potential discontinuity
    overlap_history:
        2D array [sweep_point-1, state] of maximum overlaps used for tracking
        at each transition
    """
    eigenvalues: ndarray
    eigenvectors: Optional[ndarray]
    state_labels: ndarray
    continuity_breaks: List[int]
    overlap_history: ndarray


def _compute_overlap_matrix(
    evecs1: ndarray,
    evecs2: ndarray,
    hermitian: bool = True
) -> ndarray:
    """
    Compute overlap matrix between two sets of eigenvectors.

    Parameters
    ----------
    evecs1:
        First set of eigenvectors, shape (n_states, hilbert_dim) or (n_states,)
        for 1D systems
    evecs2:
        Second set of eigenvectors, same shape as evecs1
    hermitian:
        If True, assumes eigenvectors are real or properly normalized complex
        and uses optimized computation

    Returns
    -------
        Overlap matrix M[i,j] = |⟨ψ_i|ψ_j⟩|² with shape (n_states, n_states)
    """
    # Handle both 1D and 2D eigenvector arrays
    if evecs1.ndim == 1:
        evecs1 = evecs1[:, np.newaxis]
    if evecs2.ndim == 1:
        evecs2 = evecs2[:, np.newaxis]

    # Compute inner products: ⟨ψ_i(t)|ψ_j(t+dt)⟩
    # Shape: (n_states1, n_states2)
    if hermitian and np.iscomplexobj(evecs1):
        inner_products = np.abs(evecs1.conj() @ evecs2.T)
    else:
        inner_products = np.abs(evecs1 @ evecs2.T)

    # Square to get |⟨ψ_i|ψ_j⟩|²
    overlap_matrix = inner_products ** 2

    # Numerical safety: clip to [0, 1]
    overlap_matrix = np.clip(overlap_matrix, 0.0, 1.0)

    return overlap_matrix


def _find_optimal_mapping(
    overlap_matrix: ndarray,
    overlap_threshold: float
) -> Tuple[ndarray, ndarray, float]:
    """
    Find optimal state mapping using Hungarian algorithm.

    Parameters
    ----------
    overlap_matrix:
        Matrix of overlaps |⟨ψ_i|ψ_j⟩|² between consecutive parameter points
    overlap_threshold:
        Minimum overlap to consider states connected

    Returns
    -------
    permutation:
        Permutation array mapping old indices to new physical ordering
    max_overlaps:
        Array of maximum overlaps for each state in the optimal mapping
    min_overlap:
        Minimum overlap in the optimal mapping (for discontinuity detection)
    """
    # Hungarian algorithm minimizes cost, so we maximize overlap by
    # converting to cost: cost = 1 - overlap
    cost_matrix = 1.0 - overlap_matrix

    # Find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Extract permutation (col_ind is the permutation array)
    permutation = col_ind

    # Get corresponding overlaps
    max_overlaps = overlap_matrix[row_ind, col_ind]
    min_overlap = np.min(max_overlaps)

    # Check if any overlaps are below threshold
    if min_overlap < overlap_threshold:
        warnings.warn(
            f"State tracking found overlap {min_overlap:.4f} below threshold "
            f"{overlap_threshold:.4f}. Discontinuity may exist in state evolution.",
            UserWarning
        )

    return permutation, max_overlaps, min_overlap


def track_dressed_states(
    sweep: "ParameterSweep",
    initial_labels: Optional[List[int]] = None,
    overlap_threshold: float = 0.5,
    use_eigenvectors: bool = True
) -> TrackedStates:
    """
    Track dressed states through parameter sweep to maintain physical identity.

    This function reorders eigenvalues and eigenvectors along a parameter sweep
    trajectory to follow adiabatic state evolution. At each step, states are
    matched to their continuations using maximum overlap criterion.

    Algorithm
    ---------
    1. Extract eigenvectors at each sweep point from ParameterSweep
    2. Initialize tracking with initial_labels ordering
    3. For each consecutive pair of parameter points:
       - Compute overlap matrix M[i,j] = |⟨ψ_i(t)|ψ_j(t+dt)⟩|²
       - Use Hungarian algorithm to find maximum-overlap bipartite matching
       - Verify overlaps exceed threshold (warn if discontinuity detected)
       - Apply permutation to reorder states at second point
    4. Return reordered eigenvalue/eigenvector arrays with tracking metadata

    Parameters
    ----------
    sweep:
        ParameterSweep object containing eigensystem data. Must have been
        run with `bare_esys=True` to ensure eigenvectors are available.
    initial_labels:
        List of state indices at first sweep point in desired physical order.
        If None, uses natural ordering [0, 1, 2, ...]. Length must match
        number of states being tracked.
    overlap_threshold:
        Minimum overlap |⟨ψ_i|ψ_j⟩|² to consider states connected (default 0.5).
        Lower values allow tracking through weaker connections; higher values
        detect discontinuities more sensitively.
    use_eigenvectors:
        If True, uses eigenvector overlaps for tracking. If False, falls back
        to energy proximity matching (useful if eigenvectors unavailable, but
        less robust near crossings).

    Returns
    -------
        TrackedStates object containing reordered eigenvalues, eigenvectors,
        state labels showing original indices, and discontinuity information.

    Raises
    ------
    ValueError
        If sweep doesn't contain required eigenvector data or if initial_labels
        has incorrect length
    RuntimeWarning
        If overlap falls below threshold at any point (potential discontinuity)

    Notes
    -----
    - For multi-dimensional parameter sweeps, tracking follows the flattened
      parameter trajectory in the order sweep data was computed
    - At exact crossings (overlap → 0 for multiple states), the algorithm
      may produce arbitrary assignment. Consider refining sweep resolution
      near such points.
    - Symmetry-protected crossings (e.g., different parity) maintain identity
      automatically if eigenvectors respect symmetry
    - Performance: O(n³) per sweep point for n states (Hungarian algorithm)

    Examples
    --------
    Track ground and first excited state through flux sweep:

    >>> import scqubits as scq
    >>> import numpy as np
    >>> qubit = scq.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=30)
    >>> param_vals = np.linspace(-0.5, 0.5, 100)
    >>> sweep = qubit.create_sweep('ng', param_vals)
    >>> tracked = track_dressed_states(sweep, initial_labels=[0, 1])
    >>>
    >>> # Plot tracked energies
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(param_vals, tracked.eigenvalues[:, 0], label='Ground state')
    >>> plt.plot(param_vals, tracked.eigenvalues[:, 1], label='1st excited')
    >>> plt.xlabel('ng')
    >>> plt.ylabel('Energy (GHz)')
    >>> plt.legend()
    """
    # Validate input
    if not hasattr(sweep, '_data') or sweep._data is None:
        raise ValueError(
            "ParameterSweep has no data. Run sweep.run() first."
        )

    # Get eigenvalue data
    if 'evals' not in sweep._data:
        raise ValueError(
            "ParameterSweep does not contain eigenvalue data."
        )

    evals = sweep._data['evals']
    evecs = None

    # Try to get eigenvector data if requested
    if use_eigenvectors:
        if 'evecs' in sweep._data and sweep._data['evecs'] is not None:
            evecs = sweep._data['evecs']
        else:
            warnings.warn(
                "Eigenvectors not available in sweep data. Falling back to "
                "energy-based tracking. For better results, run sweep with "
                "appropriate settings to store eigenvectors.",
                UserWarning
            )
            use_eigenvectors = False

    # Get sweep shape and flatten for 1D traversal
    original_shape = evals.shape
    n_sweep_points = np.prod(original_shape[:-1])  # All dims except last (states)
    n_states = original_shape[-1]

    # Reshape to (n_sweep_points, n_states)
    evals_flat = evals.reshape(n_sweep_points, n_states)
    if evecs is not None:
        # evecs shape: (*param_dims, n_states, hilbert_dim)
        evecs_shape = evecs.shape
        hilbert_dim = evecs_shape[-1]
        evecs_flat = evecs.reshape(n_sweep_points, n_states, hilbert_dim)

    # Initialize tracking arrays
    if initial_labels is None:
        initial_labels = list(range(n_states))
    else:
        if len(initial_labels) != n_states:
            raise ValueError(
                f"initial_labels length {len(initial_labels)} does not match "
                f"number of states {n_states}"
            )

    # Apply initial ordering
    current_permutation = np.array(initial_labels, dtype=int)
    evals_tracked = np.zeros_like(evals_flat)
    evals_tracked[0] = evals_flat[0, current_permutation]

    if evecs is not None:
        evecs_tracked = np.zeros_like(evecs_flat)
        evecs_tracked[0] = evecs_flat[0, current_permutation]
    else:
        evecs_tracked = None

    # Track original indices at each point
    state_labels = np.zeros((n_sweep_points, n_states), dtype=int)
    state_labels[0] = current_permutation

    # Track discontinuities and overlaps
    continuity_breaks = []
    overlap_history = np.zeros((n_sweep_points - 1, n_states))

    # Iterate through sweep points
    for i in range(1, n_sweep_points):
        if use_eigenvectors and evecs is not None:
            # Compute overlap matrix between consecutive points
            # evecs_tracked[i-1] has already been reordered
            # evecs_flat[i] is in original ordering
            overlap_matrix = _compute_overlap_matrix(
                evecs_tracked[i - 1],  # Previous (reordered) eigenvectors
                evecs_flat[i]          # Current (original ordering) eigenvectors
            )
        else:
            # Fall back to energy proximity matching
            # Create overlap matrix based on energy differences
            energy_diff = np.abs(
                evals_tracked[i - 1, :, np.newaxis] - evals_flat[i, np.newaxis, :]
            )
            # Convert to pseudo-overlap (closer energies → higher overlap)
            # Use exponential decay: overlap ≈ exp(-diff/scale)
            energy_scale = np.std(evals_flat[i]) + 1e-10  # Avoid division by zero
            overlap_matrix = np.exp(-energy_diff / energy_scale)

        # Find optimal mapping
        permutation, max_overlaps, min_overlap = _find_optimal_mapping(
            overlap_matrix, overlap_threshold
        )

        # Apply permutation to current point
        evals_tracked[i] = evals_flat[i, permutation]
        if evecs is not None:
            evecs_tracked[i] = evecs_flat[i, permutation]

        # Update state labels (compose permutations)
        state_labels[i] = state_labels[i - 1][permutation]

        # Record overlap history
        overlap_history[i - 1] = max_overlaps

        # Check for discontinuity
        if min_overlap < overlap_threshold:
            continuity_breaks.append(i)

    # Reshape back to original sweep dimensions
    evals_reshaped = evals_tracked.reshape(original_shape)
    if evecs_tracked is not None:
        evecs_reshaped = evecs_tracked.reshape(evecs_shape)
    else:
        evecs_reshaped = None

    state_labels_reshaped = state_labels.reshape(original_shape)

    return TrackedStates(
        eigenvalues=evals_reshaped,
        eigenvectors=evecs_reshaped,
        state_labels=state_labels_reshaped,
        continuity_breaks=continuity_breaks,
        overlap_history=overlap_history
    )


def _demo():
    """
    Minimal demonstration of state tracking functionality.

    Creates a simple two-level system with an avoided crossing and demonstrates
    how state tracking maintains physical state identity.
    """
    print("=" * 70)
    print("State Tracking Utility - Demonstration")
    print("=" * 70)
    print()

    # Create synthetic data: two-level system with avoided crossing
    print("Creating synthetic two-level system with avoided crossing...")
    n_points = 50
    params = np.linspace(0, 1, n_points)

    # Define two energy levels that anticross
    gap = 0.1
    E1_bare = params
    E2_bare = 1 - params

    # Avoided crossing via diagonalization of 2x2 Hamiltonian
    eigenvalues = np.zeros((n_points, 2))
    eigenvectors = np.zeros((n_points, 2, 2), dtype=complex)

    for i, p in enumerate(params):
        H = np.array([[E1_bare[i], gap], [gap, E2_bare[i]]])
        evals, evecs = np.linalg.eigh(H)
        eigenvalues[i] = evals
        eigenvectors[i] = evecs.T  # Transpose for (state, basis) ordering

    # Create mock sweep object
    class MockSweep:
        def __init__(self, evals, evecs):
            self._data = {
                'evals': evals,
                'evecs': evecs
            }

    sweep = MockSweep(eigenvalues, eigenvectors)

    print(f"  Parameter range: {params[0]:.2f} to {params[-1]:.2f}")
    print(f"  Number of points: {n_points}")
    print(f"  Gap at crossing: {gap}")
    print()

    # Track states
    print("Tracking states through avoided crossing...")
    tracked = track_dressed_states(sweep, initial_labels=[0, 1], overlap_threshold=0.3)

    print(f"  Continuity breaks detected: {len(tracked.continuity_breaks)}")
    if tracked.continuity_breaks:
        print(f"  Break locations (indices): {tracked.continuity_breaks}")

    # Compute minimum gap
    gap_trajectory = np.abs(tracked.eigenvalues[:, 1] - tracked.eigenvalues[:, 0])
    min_gap_idx = np.argmin(gap_trajectory)
    min_gap = gap_trajectory[min_gap_idx]

    print(f"  Minimum gap: {min_gap:.4f} at parameter {params[min_gap_idx]:.4f}")
    print()

    # Show state label evolution at key points
    print("State label evolution (original index at each parameter point):")
    print("  Parameter | State 0 (phys) | State 1 (phys)")
    print("  " + "-" * 45)
    for i in [0, n_points // 4, n_points // 2, 3 * n_points // 4, n_points - 1]:
        print(f"  {params[i]:9.3f} | {tracked.state_labels[i, 0]:14d} | {tracked.state_labels[i, 1]:14d}")
    print()

    # Summary statistics
    print("Overlap statistics:")
    mean_overlap = np.mean(tracked.overlap_history)
    min_overlap = np.min(tracked.overlap_history)
    print(f"  Mean overlap: {mean_overlap:.4f}")
    print(f"  Min overlap:  {min_overlap:.4f}")
    print()

    print("=" * 70)
    print("Demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
