# Quick Start Guide: Adiabatic Utilities for scqubits

## Installation
These utilities are now integrated into scqubits. No additional installation needed beyond scqubits dependencies.

## Quick Usage

### 1. State Tracking

Track physical state identity through parameter sweeps:

```python
import scqubits as scq
import numpy as np

# Create sweep
qubit = scq.Transmon(EJ=15.0, EC=0.3, ng=0.0, ncut=30)
sweep = qubit.create_sweep('ng', np.linspace(-0.5, 0.5, 100))

# Track states
tracked = sweep.reorder_by_state_evolution(
    initial_labels=[0, 1],  # Track ground and first excited
    overlap_threshold=0.5
)

# Access results
eigenvalues = tracked.eigenvalues      # Reordered to follow physical states
state_labels = tracked.state_labels    # Original index mapping
breaks = tracked.continuity_breaks     # Discontinuity locations
```

### 2. Adiabatic Validation

Check if your parameter ramp is slow enough:

```python
# Validate adiabaticity
report = sweep.validate_adiabatic(
    state_indices=(0, 1),    # States to check
    threshold=0.01,          # γ threshold (0.01 = ~1% error)
    use_tracked_states=True  # Use tracking for robustness
)

# Check results
if report.is_adiabatic:
    print("Ramp is adiabatic ✓")
else:
    print(f"Too fast! Need {report.suggested_ramp_time:.2f} time units")
    print(f"Violations at {len(report.violation_points)} points")

# Access detailed info
min_gap = np.min(report.min_gap_trajectory)
min_gamma = np.min(report.gamma_trajectory)
```

### 3. Combined Workflow

```python
# Full analysis
sweep = qubit.create_sweep('ng', np.linspace(-0.5, 0.5, 100))

# Step 1: Track states
tracked = sweep.reorder_by_state_evolution()

# Step 2: Validate with tracking
report = sweep.validate_adiabatic(state_indices=(0, 1))

# Step 3: Visualize
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1)

# Tracked eigenvalues
ax1.plot(tracked.eigenvalues[:, 0], label='Ground')
ax1.plot(tracked.eigenvalues[:, 1], label='Excited')
ax1.legend()

# Landau-Zener parameter
ax2.plot(report.gamma_trajectory)
ax2.axhline(y=report.threshold, color='r', linestyle='--')
ax2.set_ylabel('γ (adiabaticity)')

plt.show()
```

## API Reference

### `ParameterSweep.reorder_by_state_evolution()`

**Parameters:**
- `initial_labels`: List[int] - State indices at first point (default: [0,1,2,...])
- `overlap_threshold`: float - Minimum overlap to consider connected (default: 0.5)
- `use_eigenvectors`: bool - Use eigenvector overlaps vs energy proximity (default: True)

**Returns:** `TrackedStates` with:
- `eigenvalues`: Reordered eigenvalues following physical states
- `eigenvectors`: Reordered eigenvectors (if available)
- `state_labels`: Original index at each point
- `continuity_breaks`: List of discontinuity points
- `overlap_history`: Overlap values at transitions

### `ParameterSweep.validate_adiabatic()`

**Parameters:**
- `state_indices`: Tuple[int, int] - (initial_state, final_state) to track
- `threshold`: float - Minimum γ for adiabatic (default: 0.01)
- `return_rates`: bool - Compute max safe rate (default: True)
- `ramp_time`: Optional[float] - Total time for ramp (default: None)
- `use_tracked_states`: bool - Track states before validation (default: True)

**Returns:** `AdiabaticReport` with:
- `is_adiabatic`: bool - Whether criterion satisfied
- `violation_points`: Indices where γ < threshold
- `min_gap_trajectory`: Energy gaps along path
- `gamma_trajectory`: Landau-Zener parameters
- `suggested_ramp_time`: Time needed for adiabatic evolution
- `max_safe_rate`: Maximum parameter change rate

## Examples

### Example 1: Find Minimum Safe Ramp Time

```python
# Binary search for minimum adiabatic ramp time
def find_min_ramp_time(qubit, param_range, threshold=0.01):
    n_points_min = 20
    n_points_max = 1000

    while n_points_max - n_points_min > 10:
        n_points = (n_points_min + n_points_max) // 2
        sweep = qubit.create_sweep('ng', np.linspace(*param_range, n_points))
        report = sweep.validate_adiabatic(state_indices=(0,1), threshold=threshold)

        if report.is_adiabatic:
            n_points_max = n_points
        else:
            n_points_min = n_points

    return n_points_max

min_points = find_min_ramp_time(qubit, (-0.5, 0.5))
print(f"Minimum points for adiabatic ramp: {min_points}")
```

### Example 2: Compare Different Parameters

```python
# Compare adiabaticity for different parameter sweeps
params_to_test = ['ng', 'EJ', 'EC']
results = {}

for param in params_to_test:
    sweep = qubit.create_sweep(param, np.linspace(start, end, 100))
    report = sweep.validate_adiabatic(state_indices=(0, 1))
    results[param] = report.is_adiabatic

print("Adiabatic sweeps:", [p for p, v in results.items() if v])
```

### Example 3: Track Multiple State Pairs

```python
# Track several state pairs through avoided crossings
state_pairs = [(0, 1), (1, 2), (2, 3)]

for i, j in state_pairs:
    report = sweep.validate_adiabatic(state_indices=(i, j))
    print(f"States {i}↔{j}: {'✓' if report.is_adiabatic else '✗'}")
    print(f"  Min gap: {np.min(report.min_gap_trajectory):.6f} GHz")
    print(f"  Min γ: {np.min(report.gamma_trajectory):.4f}")
```

## Understanding the Results

### Landau-Zener Parameter (γ)

The adiabatic criterion requires γ ≫ 1, where:
```
γ = πΔ²/(2ℏ|dΔ/dt|)
```

**Interpretation:**
- γ > 100: Excellent adiabaticity (<1% diabatic transitions)
- γ > 10: Good adiabaticity (~1-5% transitions)
- γ > 1: Marginal (10-30% transitions)
- γ < 1: Non-adiabatic (>30% transitions)

**Threshold recommendations:**
- Quantum annealing: threshold = 0.1 (10% error acceptable)
- State preparation: threshold = 0.01 (1% error)
- High-fidelity gates: threshold = 0.001 (0.1% error)

### State Overlap

Overlap between consecutive eigenvectors indicates state continuity:

**Values:**
- Overlap > 0.9: Excellent continuity
- Overlap > 0.7: Good tracking
- Overlap > 0.5: Marginal (use lower threshold)
- Overlap < 0.5: Discontinuity likely (check for crossing)

## Troubleshooting

### "Eigenvectors not available"
**Solution:** Ensure sweep was run with settings that store eigenvectors. Falls back to energy-based tracking (less robust).

### "Continuity breaks detected"
**Cause:** States may have exact crossing or numerical discontinuity.
**Solution:**
- Increase sweep resolution near crossing
- Lower `overlap_threshold`
- Check for symmetry-protected crossings

### "Multi-dimensional sweep detected"
**Cause:** Sweep has >1 parameter varying.
**Solution:** Analysis treats as flattened 1D trajectory. Results may not be physically meaningful for grid sweeps.

### Violations at high γ
**Cause:** Gap becomes very small (near exact crossing).
**Solution:**
- Increase sweep resolution
- Use `use_tracked_states=True`
- Consider if crossing is physical or numerical artifact

## Performance Tips

1. **Large sweeps:** State tracking is O(n³) per point. For >100 points with >10 states, expect several seconds.

2. **Memory:** Both utilities reuse sweep data. Memory overhead is minimal.

3. **Accuracy:** Use at least 50-100 points for accurate derivative computation.

4. **Parallel sweeps:** Both utilities work on completed sweeps. Run sweep with multi-CPU first.

## Further Reading

- **Example notebook:** `examples/adiabatic_analysis.ipynb`
- **Implementation details:** `ADIABATIC_UTILS_IMPLEMENTATION.md`
- **Test suite:** `scqubits/tests/test_adiabatic_utils.py`
- **API docs:** Docstrings in `state_tracking.py` and `adiabatic_validator.py`

## Citation

If you use these utilities in your research, please cite:

```bibtex
@article{scqubits,
  title={scqubits: a Python package for superconducting qubits},
  author={Koch, Jens and others},
  journal={Quantum},
  volume={5},
  pages={583},
  year={2021},
  publisher={Verein zur F{\"o}rderung des Open Access Publizierens in den Quantenwissenschaften}
}
```

And for the adiabatic utilities:
```
Adiabatic validation and state tracking utilities for scqubits (2025)
Implementation available at: https://github.com/scqubits/scqubits
```
