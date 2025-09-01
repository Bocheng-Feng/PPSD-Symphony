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
#suite_names = ["SymphonyGroup", "SymphonyCluster"]

# Path to the root directory containing Symphony simulation data (particle and halo catalogs)
#simul_dir = '/Volumes/Carol/Symphony'

# Target redshift at which the analysis is performed
redshifts = [0.05, 0.1 , 5, 9]

# Radial binning parameters for profile measurements:
# n_bins = number of radial bins
# r_min = minimum radius in units of virial radius (r_vir)
# r_max = maximum radius in units of virial radius (r_vir)
n_bins, r_min, r_max = 40, 1e-2, 1.5

############################### measure #################################
if __name__ == "__main__":
    for redshift in redshifts:
        print(f'------processing z={redshift}-------')
        # Loop over each simulation suite to process data and produce measurements
        for suite in ["SymphonyGroup", "SymphonyCluster"]:
            simul_dir = '/Volumes/Carol/Symphony'
            sim_dir = symlib.get_host_directory(simul_dir, suite, 0)  # Get directory for suite host halo 0
            scale = symlib.scale_factors(sim_dir)                      # Array of scale factors for snapshots
            z = 1/scale - 1                                            # Convert scale factors to redshifts

            idx = np.argmin(np.abs(z - redshift))    # Find index of snapshot closest to target redshift
            closest_snapshot = idx                   # Snapshot ID corresponding to closest redshift
            closest_redshift = z[idx]                # Actual redshift of the chosen snapshot
            snap = closest_snapshot                  # Rename for clarity in downstream code

            data_dir = f'output/z_{redshift}/data'  # Directory to save measurement data
            os.makedirs(data_dir, exist_ok=True)    # Create directory if it doesn't exist

            basic_measure.density_velocity_mass(simul_dir, data_dir, suite, snap, n_bins, r_min, r_max)
            ppsd_measure.ppsd_profiles(data_dir, suite)
            ppsd_measure.smooth_ppsd_slopes(data_dir, suite, method='constant_jerk')
            halo_properties.process_halo_properties(simul_dir, data_dir, suite, snap)

        for suite in ["SymphonyLMC", "SymphonyMilkyWay", "SymphonyLCluster"]:
            simul_dir = '/Volumes/Atlas/Symphony'
            sim_dir = symlib.get_host_directory(simul_dir, suite, 0)  # Get directory for suite host halo 0
            scale = symlib.scale_factors(sim_dir)                      # Array of scale factors for snapshots
            z = 1/scale - 1                                            # Convert scale factors to redshifts

            idx = np.argmin(np.abs(z - redshift))    # Find index of snapshot closest to target redshift
            closest_snapshot = idx                   # Snapshot ID corresponding to closest redshift
            closest_redshift = z[idx]                # Actual redshift of the chosen snapshot
            snap = closest_snapshot                  # Rename for clarity in downstream code

            data_dir = f'output/z_{redshift}/data'  # Directory to save measurement data
            os.makedirs(data_dir, exist_ok=True)    # Create directory if it doesn't exist

            basic_measure.density_velocity_mass(simul_dir, data_dir, suite, snap, n_bins, r_min, r_max)
            ppsd_measure.ppsd_profiles(data_dir, suite)
            ppsd_measure.smooth_ppsd_slopes(data_dir, suite, method='constant_jerk')
            halo_properties.process_halo_properties(simul_dir, data_dir, suite, snap)

################################ plot ###################################
        # Create directory for saving figures corresponding to the target redshift
        fig_dir = f'output/z_{redshift}/figure'
        os.makedirs(fig_dir, exist_ok=True)

        base_dir = f'output/z_{redshift}'

        suite_names=["SymphonyLMC", "SymphonyMilkyWay", "SymphonyGroup", "SymphonyLCluster", "SymphonyCluster"]
        basic_plot.plot_density_velocity(base_dir, suite_names, einasto_fit=False, mask_range=[1e-2,1], plot_range=[1e-2,1])
        basic_plot.plot_anisotropy(base_dir, suite_names, mask_range=[1e-2,1], plot_range=[1e-2,1])
        ppsd_plot.plot_normalized_ppsd(base_dir, suite_names, plot_range=[1e-2,1])
        ppsd_plot.plot_ppsd_slope(base_dir, suite_names, plot_range=[1e-2,1])
