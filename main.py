#########################################################################

"""
main.py

This script runs the measurement and plotting pipeline for the Symphony simulation suites.
It processes halo data at a specified target redshift, computes various halo profiles 
(density, velocity dispersion, mass), calculates pseudo phase-space density (PPSD) profiles,
applies smoothing to PPSD slopes, computes additional halo properties, and generates plots 
to visualize the results.

Author: Michael Bocheng Feng 2025 @Peking Univ.
"""

######################## set up the environment #########################

import symlib
import numpy as np
import os
from sym_measure import basic_measure, halo_properties, ppsd_measure
from sym_plot import basic_plot, ppsd_plot

########################### user config ################################
# List of simulation suites to analyze, representing different host halo mass scales
suite_names = ["SymphonyLMC", "SymphonyMilkyWay",  "SymphonyLCluster",]

# Path to the root directory containing Symphony simulation data (particle and halo catalogs)
simul_dir = '/Volumes/Atlas/Symphony'

# Target redshift at which the analysis is performed
redshift = 0.5

# Radial binning parameters for profile measurements:
# n_bins = number of radial bins
# r_min = minimum radius in units of virial radius (r_vir)
# r_max = maximum radius in units of virial radius (r_vir)
n_bins, r_min, r_max = 40, 1e-2, 1.5

############################### measure #################################
if __name__ == "__main__":
    # Loop over each simulation suite to process data and produce measurements
    for suite in suite_names:
        # ---------------------------------------------------------
        # 1. Load scale factors for the given suite and compute redshift
        #    The scale factor a(t) relates to redshift z by z = 1/a - 1.
        #    This allows us to identify snapshots corresponding to desired redshifts.
        # ---------------------------------------------------------
        sim_dir = symlib.get_host_directory(simul_dir, suite, 0)  # Get directory for suite host halo 0
        scale = symlib.scale_factors(sim_dir)                      # Array of scale factors for snapshots
        z = 1/scale - 1                                            # Convert scale factors to redshifts

        # ---------------------------------------------------------
        # 2. Find the snapshot whose redshift is closest to the target redshift
        #    This ensures we analyze data at the redshift nearest to our desired epoch.
        # ---------------------------------------------------------
        idx = np.argmin(np.abs(z - redshift))    # Find index of snapshot closest to target redshift
        closest_snapshot = idx                   # Snapshot ID corresponding to closest redshift
        closest_redshift = z[idx]                # Actual redshift of the chosen snapshot
        snap = closest_snapshot                  # Rename for clarity in downstream code

        # ---------------------------------------------------------
        # 3. Create output directory for all processed data
        #    Organize outputs by redshift for easier management and later retrieval.
        # ---------------------------------------------------------
        data_dir = f'output/z_{redshift}/data'  # Directory to save measurement data
        os.makedirs(data_dir, exist_ok=True)    # Create directory if it doesn't exist

        # ---------------------------------------------------------
        # 4. Compute and save basic halo profiles:
        #    These include density profiles, velocity dispersion profiles, and enclosed mass profiles.
        #    Such profiles characterize the structure and dynamics of dark matter halos.
        # ---------------------------------------------------------
        basic_measure.density_velocity_mass(simul_dir, data_dir, suite, snap, n_bins, r_min, r_max)

        # ---------------------------------------------------------
        # 5. Compute Pseudo Phase-Space Density (PPSD) profiles:
        #    Q_r and Q_tot = ρ / σ³, where ρ is density and σ velocity dispersion.
        #    PPSD profiles provide insight into the dynamical state of halos.
        #    Results are saved individually for each halo.
        # ---------------------------------------------------------
        ppsd_measure.ppsd_profiles(data_dir, suite)

        # ---------------------------------------------------------
        # 6. Smooth PPSD slope profiles using a selected derivative method:
        #    Default method is 'constant_jerk' smoothing with optional total variation regularization.
        #    Smoothing helps reduce noise in derivative estimates of PPSD slopes.
        # ---------------------------------------------------------
        ppsd_measure.smooth_ppsd_slopes(data_dir, suite, method='constant_jerk')

        # ---------------------------------------------------------
        # 7. Compute additional halo properties (e.g., virial radius r_vir, virial mass M_vir):
        #    These properties are fundamental for understanding halo scaling relations and comparisons.
        #    Computed properties are saved for subsequent analyses.
        # ---------------------------------------------------------
        halo_properties.process_halo_properties(simul_dir, data_dir, suite, snap)

################################ plot ###################################
    # Create directory for saving figures corresponding to the target redshift
    fig_dir = f'output/z_{redshift}/figure'
    os.makedirs(fig_dir, exist_ok=True)

    base_dir = f'output/z_{redshift}'

    # Plotting calls:
    # - basic_plot.plot_density_velocity: plots density and velocity profiles with optional Einasto fits
    # - basic_plot.plot_anisotropy: plots velocity anisotropy profiles
    # - ppsd_plot.plot_normalized_ppsd: plots normalized PPSD profiles
    # - ppsd_plot.plot_ppsd_slope: plots slopes of PPSD profiles
    #
    # Parameters:
    # - mask_range: radial range over which data is masked/selected (in r_vir units)
    # - plot_range: radial range displayed in plots (in r_vir units)
    # - einasto_fit: whether to include Einasto profile fits in density/velocity plots

    basic_plot.plot_density_velocity(base_dir, suite_names, einasto_fit=False, mask_range=[1e-2,1], plot_range=[1e-2,1])
    basic_plot.plot_anisotropy(base_dir, suite_names, mask_range=[1e-2,1], plot_range=[1e-2,1])
    ppsd_plot.plot_normalized_ppsd(base_dir, suite_names, plot_range=[1e-2,1])
    ppsd_plot.plot_ppsd_slope(base_dir, suite_names, plot_range=[1e-2,1])
