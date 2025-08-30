#########################################################################

# This is the program that 

# Author: Michael Bocheng Feng 2025 @Peking Univ.

######################## set up the environment #########################

import symlib
import numpy as np
import os
from sym_measure.basic_measure import density_velocity_mass
from sym_measure.ppsd_measure import ppsd_profiles, smooth_ppsd_slopes
from sym_measure.halo_properties import process_halo_properties

########################### user config ################################
# List of simulation suites to analyze
suite_names = ["SymphonyLMC", "SymphonyMilkyWay", "SymphonyLCluster"]

# Path to Symphony simulation data (particle + halo information)
simul_dir = '/Volumes/Atlas/Symphony'

# Target redshift for analysis
redshift = 1

# Radial binning parameters: number of bins, min radius, max radius (in r_vir units)
n_bins, r_min, r_max = 40, 1e-3, 1.5

# PPSD quantity to analyze: 'Q_tot' for total PPSD, 'Q_r' for radial-only PPSD
quantity = 'Q_r'

############################### measure #################################
if __name__ == "__main__":
    for suite in suite_names:
    # ---------------------------------------------------------
    # 1. Load scale factors for the given suite and compute redshift
    #    scale = a(t), so redshift z = 1/a - 1
    # ---------------------------------------------------------
        sim_dir = symlib.get_host_directory(simul_dir, suite, 0)
        scale = symlib.scale_factors(sim_dir)
        z = 1/scale - 1

    # ---------------------------------------------------------
    # 2. Find the snapshot whose redshift is closest to the target redshift
    # ---------------------------------------------------------
        idx = np.argmin(np.abs(z - redshift))    # index of snapshot closest to target z
        closest_snapshot = idx                   # snapshot ID
        closest_redshift = z[idx]                # actual redshift at that snapshot
        snap = closest_snapshot                  # rename for clarity

    # ---------------------------------------------------------
    # 3. Create output directory for all processed data
    #    Each redshift gets its own folder: output/z_<z>/data
    # ---------------------------------------------------------
        data_dir = f'output/z_{redshift}/data'
        os.makedirs(data_dir, exist_ok=True)

    # ---------------------------------------------------------
    # 4. Compute and save basic halo profiles:
    #    - Density profiles
    #    - Velocity dispersion profiles
    #    - Enclosed mass profiles
    # ---------------------------------------------------------
        #density_velocity_mass(simul_dir, data_dir, suite, snap, n_bins, r_min, r_max)

    # ---------------------------------------------------------
    # 5. Compute PPSD profiles Q_r and Q_tot = ρ / σ³
    #    and save them for each halo
    # ---------------------------------------------------------
        ppsd_profiles(data_dir, suite)

    # ---------------------------------------------------------
    # 6. Smooth PPSD slope profiles using chosen derivative method
    #    Default: constant_jerk smoothing with optional TV regularization
    # ---------------------------------------------------------
        smooth_ppsd_slopes(data_dir, suite, method='constant_jerk')

    # ---------------------------------------------------------
    # 7. Compute additional halo properties (e.g., r_vir, M_vir)
    #    and save them for later analysis
    # ---------------------------------------------------------
        process_halo_properties(simul_dir, data_dir, suite, snap)

################################ plot ###################################
    fig_dir = os.makedirs(f'output/z_{redshift}/figure')
    
