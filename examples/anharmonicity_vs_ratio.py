"""
Example: Transmon anharmonicity as a function of EJ/EC ratio
----------------------------------------------------------------

This example illustrates how the anharmonicity of a transmon qubit changes
as the ratio between the Josephson energy (``EJ``) and the charging energy
(``EC``) varies.  The transmon qubit is characterized by large ``EJ/EC``
ratios to suppress charge dispersion while retaining a small but non‑zero
anharmonicity.  Understanding the dependence of anharmonicity on the
``EJ/EC`` ratio is useful for designing devices with desired spectral
properties.

The script sweeps over a range of ``EJ/EC`` ratios, constructs a ``Transmon``
instance for each ratio, computes the fundamental splitting ``E01`` and the
anharmonicity using built‑in methods provided by ``scqubits``, and then
stores the results for plotting.  Finally, it produces a simple plot
showing the anharmonicity as a function of ``EJ/EC``.  The plot uses
dimensionless units; to express energies in frequency units (GHz), one can
divide by Planck's constant.  The example is written to be easy to adapt
for further analysis, such as exploring the effect of offset charge ``ng``
or the cutoff ``ncut``.

Usage
-----
Run this script from a Python environment with ``scqubits`` and
``matplotlib`` installed.  No command line arguments are required.  The
generated plot will appear in a new window.  Users can modify the
``EC`` value, the sweep range of ``EJ/EC``, and the number of points to
adjust resolution and computational cost.

References
----------

Peter Groszkowski and Jens Koch,
scqubits: a Python package for superconducting qubits,
Quantum 5, 583 (2021).
https://quantum-journal.org/papers/q-2021-11-17-583/

Sai Pavan Chitta, Tianpu Zhao, Ziwen Huang, Ian Mondragon-Shem, and Jens Koch,
Computer-aided quantization and numerical analysis of superconducting circuits,
New J. Phys. 24 103020 (2022).
https://iopscience.iop.org/article/10.1088/1367-2630/ac94f2

"""

import numpy as np
import matplotlib.pyplot as plt

import scqubits as scq


def main() -> None:
    """Compute and plot transmon anharmonicity vs EJ/EC ratio.

    The function defines a fixed charging energy and sweeps the Josephson
    energy over a range that spans weakly anharmonic to strongly anharmonic
    regimes.  For each ratio, it constructs a ``Transmon`` instance,
    computes the fundamental transition frequency ``E01`` and the
    anharmonicity via ``scqubits``, and stores the ratio and anharmonicity.
    At the end, it plots the results.

    """

    # Set a fixed charging energy (in the same units as EJ).  For
    # superconducting circuits EJ is typically expressed in gigahertz units
    # (assuming ℏ = 1).  Here we use EC = 1.0 as a baseline.  Changing EC
    # will shift the absolute energy scale but does not affect the trend of
    # anharmonicity vs EJ/EC.
    EC = 1.0

    # Define the range of EJ/EC ratios to explore.  Transmon devices often
    # operate at EJ/EC ratios between 20 and 100; we sweep a broader range
    # here for illustration.
    ratio_min = 1.0  # weakly transmon (large anharmonicity)
    ratio_max = 100.0  # strongly transmon (small anharmonicity)
    num_points = 50

    ratios = np.linspace(ratio_min, ratio_max, num_points)
    anharmonicities = []

    # We fix ng = 0 to avoid charge dispersion effects and choose a
    # reasonable cutoff ncut.  Increasing ncut improves convergence at the
    # expense of computation time.
    ng = 0.0
    ncut = 30

    for ratio in ratios:
        EJ = ratio * EC
        # Construct a Transmon with the given EJ and EC.  We keep other
        # parameters at their defaults.  truncated_dim determines the number
        # of energy levels included; 6 is sufficient for computing the
        # first few levels and the anharmonicity.
        tmon = scq.Transmon(EJ=EJ, EC=EC, ng=ng, ncut=ncut, truncated_dim=6)
        # Compute the anharmonicity using the base class method.  The
        # ``anharmonicity`` method returns (E2 - E1) - (E1 - E0).
        anh = tmon.anharmonicity()
        anharmonicities.append(anh)

    # Convert results to numpy arrays for plotting.
    ratios = np.array(ratios)
    anharmonicities = np.array(anharmonicities)

    # Plotting: we use a log scale on the x-axis to better display the
    # variation across several decades of EJ/EC.  The y-axis shows the
    # anharmonicity in the same units as EC.  Negative anharmonicity is
    # expected for transmons and indicates that higher transition frequencies
    # decrease with increasing energy level.
    fig, ax = plt.subplots()
    ax.plot(ratios, anharmonicities, marker="o", linestyle="-")
    ax.set_xscale("log")
    ax.set_xlabel("EJ/EC")
    ax.set_ylabel("Anharmonicity (dimensionless)")
    ax.set_title("Transmon anharmonicity vs EJ/EC ratio")
    ax.grid(True, which="both", ls="--", alpha=0.5)

    plt.show()


if __name__ == "__main__":
    main()