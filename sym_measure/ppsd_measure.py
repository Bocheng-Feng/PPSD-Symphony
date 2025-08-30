import numpy as np
import pandas as pd
import os
import pynumdiff
import pynumdiff.optimize

def ppsd_profiles(base_dir, suite_name):
    """
    Compute and save pseudo phase-space density (PPSD) profiles for each host halo 
    in a given simulation suite. 

    The PPSD is calculated in two ways:
        - Q_r   = rho / sigma_rad^3, using the radial velocity dispersion
        - Q_tot = rho / sigma_total^3, using the total velocity dispersion

    Profiles are saved per halo in CSV format.

    Parameters
    ----------
    base_dir : str
        Root directory containing simulation suite's pre-computed profiles.

    suite_name : str
        Name of the simulation suite (e.g., 'SymphonyLMC').

    Notes
    -----
    Requires the following pre-computed profiles for each halo in base_dir:
        - Density profiles (rho_scaled) in 'density_profiles'
        - Enclosed mass profiles (m_scaled) in 'mass_profiles'
        - Velocity dispersion profiles (sigma_rad_scaled, sigma_total_scaled) in 'velocity_profiles'
    """

    # ----- Define directories for input profiles and output PPSD -----
    density_dir = os.path.join(base_dir, suite_name, "density_profiles")
    mass_dir = os.path.join(base_dir, suite_name, "mass_profiles")
    velocity_dir = os.path.join(base_dir, suite_name, "velocity_profiles")
    output_dir = os.path.join(base_dir, suite_name, "ppsd_profiles")
    os.makedirs(output_dir, exist_ok=True)

    # ----- Collect and sort all profile files -----
    density_files = sorted([f for f in os.listdir(density_dir) if f.endswith(".csv")])
    mass_files = sorted([f for f in os.listdir(mass_dir) if f.endswith(".csv")])
    velocity_files = sorted([f for f in os.listdir(velocity_dir) if f.endswith(".csv")])

    # ----- Loop over halos and compute PPSD -----
    for halo_idx, (f_rho, f_mass, f_vel) in enumerate(zip(density_files, mass_files, velocity_files)):
        # Load CSV data for the current halo
        df_rho = pd.read_csv(os.path.join(density_dir, f_rho))      # Density profile
        df_mass = pd.read_csv(os.path.join(mass_dir, f_mass))       # Enclosed mass profile
        df_vel = pd.read_csv(os.path.join(velocity_dir, f_vel))     # Velocity dispersion profile

        # Extract relevant profile arrays
        r = df_rho["r_scaled"].values           # Scaled radius
        rho = df_rho["rho_scaled"].values       # Scaled density
        m = df_mass["m_scaled"].values          # Scaled enclosed mass
        sigma_rad = df_vel["sigma_rad_scaled"].values   # Radial velocity dispersion
        sigma_tot = df_vel["sigma_total_scaled"].values # Total velocity dispersion

        # ----- Compute Pseudo Phase-Space Density (PPSD) -----
        # Q_r uses radial dispersion, Q_tot uses total dispersion
        Q_r = np.where(sigma_rad > 0, rho / sigma_rad**3, np.nan)
        Q_tot = np.where(sigma_tot > 0, rho / sigma_tot**3, np.nan)

        # ----- Save PPSD profile to CSV -----
        df_out = pd.DataFrame({
            "r_scaled": r,
            "m_scaled": m,
            "Q_r": Q_r,
            "Q_tot": Q_tot,
        })
        df_out.to_csv(f"{output_dir}/halo_{halo_idx:03d}_profile.csv", index=False)

    print(f"[Saved] PPSD profiles for {suite_name} saved to {output_dir}")

def smooth_ppsd_slopes(base_dir, suite_name, method='constant_jerk', tvgamma=None):
    """
    Compute smoothed slopes of pseudo phase-space density (PPSD) profiles for each halo 
    in a simulation suite using the pynumdiff derivative methods.

    The slopes are computed in log-log space as:
        - d(log Q_r) / d(log r)
        - d(log Q_tot) / d(log r)
        - d(log Q_r) / d(log m)
        - d(log Q_tot) / d(log m)

    Parameters
    ----------
    base_dir : str
        Root directory containing suite's profiles.

    suite_name : str
        Name of the simulation suite (e.g., 'SymphonyLMC').

    method : str, optional
        Derivative computation method in pynumdiff (default: 'constant_jerk').

    tvgamma : float or None, optional
        Regularization parameter for total variation methods. Only used if 
        the chosen optimizer supports it.
    """

    # ----- Helper function: find derivative and optimizer in pynumdiff -----
    def get_diff_and_optimize_funcs(method):
        """
        Search pynumdiff submodules for the requested derivative method
        and its corresponding optimizer function.
        """
        submodules = [
            'kalman_smooth',
            'smooth_finite_difference',
            'finite_difference',
            'total_variation_regularization',
            'linear_model'
        ]
        for submod in submodules:
            try:
                mod_optimize = getattr(pynumdiff.optimize, submod)
                mod_diff = getattr(pynumdiff, submod)
                if hasattr(mod_optimize, method) and hasattr(mod_diff, method):
                    return getattr(mod_diff, method), getattr(mod_optimize, method)
            except AttributeError:
                continue
        raise ValueError(f"Method '{method}' not found in any submodule.")

    # ----- Set up input and output directories -----
    ppsd_dir = os.path.join(base_dir, suite_name, "ppsd_profiles")
    slope_r_dir = os.path.join(base_dir, suite_name, "ppsd_slope_profiles_r")
    slope_m_dir = os.path.join(base_dir, suite_name, "ppsd_slope_profiles_m")
    os.makedirs(slope_r_dir, exist_ok=True)
    os.makedirs(slope_m_dir, exist_ok=True)

    # ----- Collect all halo PPSD files -----
    ppsd_files = sorted([f for f in os.listdir(ppsd_dir) if f.endswith(".csv")])
    n_halos = len(ppsd_files)

    # ----- Helper function: fit derivative for a profile -----
    def fit_derivative(y, dt):
        """
        Fit the derivative of a 1D array using the selected pynumdiff method.

        Parameters
        ----------
        y : array_like
            Input 1D data (e.g., log Q).
        dt : float
            Step size in the independent variable (log r or log m).

        Returns
        -------
        dydx : array_like
            Smoothed derivative of y.
        """
        try:
            diff_func, optimize_func = get_diff_and_optimize_funcs(method)
            kwargs = {'tvgamma': tvgamma} if 'tvgamma' in optimize_func.__code__.co_varnames else {}
            params, _ = optimize_func(y, dt, **kwargs)   # find optimal parameters
            _, dydx = diff_func(y, dt, params)           # compute derivative
            return dydx
        except Exception as e:
            print(f"{method} derivative fit failed: {e}")
            return None

    # ----- Loop over halos and compute slopes -----
    for halo_idx in range(n_halos):
        try:
            df_Q = pd.read_csv(os.path.join(ppsd_dir, ppsd_files[halo_idx]))
        except Exception as e:
            print(f"[Halo {halo_idx}] loading profiles failed: {e}")
            continue

        # Extract radial and mass arrays
        r = df_Q["r_scaled"].values
        m = df_Q["m_scaled"].values
        Q_tot = df_Q["Q_tot"].values
        Q_r = df_Q["Q_r"].values

        # Step size in log space for radial and mass grids
        dt_r = np.diff(np.log10(r)).mean()
        dt_m = np.diff(np.log10(m)).mean()

        # Work in log-log space
        log_Q_r = np.log10(Q_r)
        log_Q_tot = np.log10(Q_tot)

        # Compute slopes: d(log Q)/d(log r) and d(log Q)/d(log m)
        slope_Q_tot_r = fit_derivative(log_Q_tot, dt_r)
        slope_Q_rad_r = fit_derivative(log_Q_r, dt_r)
        slope_Q_tot_m = fit_derivative(log_Q_tot, dt_m)
        slope_Q_rad_m = fit_derivative(log_Q_r, dt_m)

        # ----- Save slope profiles as CSV -----
        df_r = pd.DataFrame({
            "r_scaled": r,
            "slope_Q_r": slope_Q_rad_r,
            "slope_Q_tot": slope_Q_tot_r
        })
        df_m = pd.DataFrame({
            "m_scaled": m,
            "slope_Q_r": slope_Q_rad_m,
            "slope_Q_tot": slope_Q_tot_m
        })

        df_r.to_csv(os.path.join(slope_r_dir, f"halo_{halo_idx:03d}_profile.csv"), index=False)
        df_m.to_csv(os.path.join(slope_m_dir, f"halo_{halo_idx:03d}_profile.csv"), index=False)
        print(f"[Saved] {suite_name} Halo {halo_idx:03d} PPSD slope profile saved." )
