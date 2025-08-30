import numpy as np
import pandas as pd
from astropy.constants import G
from astropy import units as u
import symlib
import os

def density_velocity_mass(input_dir, output_dir, suite_name, snap, n_bins=40, r_min=1e-3, r_max=1.5):
    """
    Compute and save density, velocity dispersion, velocity anisotropy, 
    and enclosed mass profiles for each host halo in a given simulation suite 
    at a specific snapshot.

    Profiles are normalized as follows:
        - Density: relative to background matter density rho_m = Omega_m * rho_crit at z=0
        - Velocity dispersion: relative to virial velocity at the given snapshot
        - Enclosed mass: relative to virial mass at the given snapshot

    Parameters
    ----------
    input_dir : str
        Root directory containing simulation suite's particel and halo dataset.

    suite_name : str
        Name of the simulation suite (e.g., 'SymphonyLMC').

    output_dir : str
        Directory where the computed profiles will be saved.

    snap : int
        Snapshot index for reading particle and halo data.

    n_bins : int, optional
        Number of logarithmic radial bins for profile calculation. Default: 40.

    r_min : float, optional
        Minimum radius for radial binning, in units of R_vir. Default: 1e-3.

    r_max : float, optional
        Maximum radius for radial binning, in units of R_vir. Default: 1.5.
    """

    # ----------------------------------------------------------------
    # Shared helper functions
    # ----------------------------------------------------------------
    def load_halo_and_particles(halo_idx):
        """Load host halo properties and particle snapshot for a given halo index."""
        sim_dir = symlib.get_host_directory(input_dir, suite_name, halo_idx)

        try:
            r, hist = symlib.read_rockstar(sim_dir)
            host = r[0, snap]   # host halo properties at snapshot "snap"
        except FileNotFoundError:
            print(f"[Warning] Rockstar file not found for Halo {halo_idx}")
            return None, None

        try:
            part = symlib.Particles(sim_dir)
            p = part.read(snap) # particle data at snapshot "snap"
        except FileNotFoundError:
            print(f"[Warning] Particle snapshot not found for Halo {halo_idx}")
            return host, None

        return host, p

    def create_bins():
        """Generate logarithmic radial bins in units of R_vir."""
        bins = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])
        return bins, bin_centers

    def save_profile(df, profile_type, halo_idx):
        """Save computed profile to CSV in the appropriate subdirectory."""
        out_path = f"{output_dir}/{suite_name}/{profile_type}"
        os.makedirs(out_path, exist_ok=True)
        df.to_csv(f"{out_path}/halo_{halo_idx:03d}_profile.csv", index=False)
        print(f"[Saved] {suite_name} Halo {halo_idx:03d} {profile_type} profile saved.")

    # ----------------------------------------------------------------
    # Subfunction 1: Compute density profiles
    # ----------------------------------------------------------------
    def density_profile():
        n_halos = symlib.n_hosts(suite_name)
        params = symlib.simulation_parameters(suite_name)
        h = params['h100'] # scaled Hubble parameter
        mp = params['mp'] / h # Particle mass [Msun]
        Om0 = params["Om0"] # Matter density parameter at z=0

        # Compute background matter density ρ_m = Ω_m * ρ_crit at z=0
        H0_si = params["H0"] * u.km / u.s / u.Mpc
        G_si = G.to(u.Mpc**3 / u.Msun / u.s**2)
        rho_crit = (3 * H0_si**2 / (8 * np.pi * G_si)).to(u.Msun / u.kpc**3).value
        rho_m = Om0 * rho_crit

        cvir, mvir = [], []
        bins, bin_centers = create_bins()

        for halo_idx in range(n_halos):
            host, p = load_halo_and_particles(halo_idx)
            if host is None or p is None:
                continue

            center, r_vir = host['x'], host['rvir']
            cvir.append(host['cvir'])
            mvir.append(host['m'])

            # Compute radial distances and density per bin
            radi = np.linalg.norm(p[0]['x'] - center, axis=1) / r_vir
            counts, _ = np.histogram(radi, bins=bins)
            shell_volumes = (4/3) * np.pi * ((bins[1:]*r_vir)**3 - (bins[:-1]*r_vir)**3)
            rho_scaled = (counts * mp / shell_volumes) / rho_m

            # Save density profile
            df = pd.DataFrame({"halo_idx": halo_idx, "r_scaled": bin_centers, "rho_scaled": rho_scaled})
            save_profile(df, "density_profiles", halo_idx)

        # Print mean concentration and mass
        print(f'mean cvir of {suite_name}: {np.mean(cvir)}, mean mvir: {np.mean(mvir)}')

    # ----------------------------------------------------------------
    # Subfunction 2: Compute velocity dispersion profiles
    # ----------------------------------------------------------------
    def velocity_profile():
        n_halos = symlib.n_hosts(suite_name)
        bins, bin_centers = create_bins()

        for halo_idx in range(n_halos):
            host, p = load_halo_and_particles(halo_idx)
            if host is None or p is None:
                continue

            center, v_host, r_vir = host['x'], host['v'], host['rvir']

            # Virial velocity v_vir = sqrt(G * M_vir / R_vir)
            m_vir = host['m'] * u.Msun
            r_vir_u = r_vir * u.kpc
            G_kpc = G.to(u.kpc * (u.km/u.s)**2 / u.Msun)
            v_vir = np.sqrt(G_kpc * m_vir / r_vir_u).to(u.km / u.s).value

            # Particle positions/velocities relative to host
            dx = p[0]['x'] - center
            dv = p[0]['v'] - v_host
            radi = np.linalg.norm(dx, axis=1) / r_vir
            r_hat = dx / np.linalg.norm(dx, axis=1)[:, None]
            v_rad = np.sum(dv * r_hat, axis=1)

            sigma_rad_scaled, sigma_tan_scaled, sigma_total_scaled, beta_profile = [], [], [], []

            # Loop over radial bins
            for i in range(n_bins):
                in_bin = (radi >= bins[i]) & (radi < bins[i+1])
                sigma_rad = np.std(v_rad[in_bin])
                sigma_total = np.linalg.norm([np.std(dv[in_bin][:, j]) for j in range(3)])
                sigma_tan = np.sqrt(max(0, sigma_total**2 - sigma_rad**2))
                beta = np.nan if sigma_rad == 0 else 1 - sigma_tan**2 / (2 * sigma_rad**2)

                sigma_rad_scaled.append(sigma_rad / v_vir)
                sigma_tan_scaled.append(sigma_tan / v_vir)
                sigma_total_scaled.append(sigma_total / v_vir)
                beta_profile.append(beta)

            # Save velocity profile
            df = pd.DataFrame({
                "halo_idx": halo_idx, "r_scaled": bin_centers,
                "sigma_rad_scaled": sigma_rad_scaled,
                "sigma_tan_scaled": sigma_tan_scaled,
                "sigma_total_scaled": sigma_total_scaled,
                "beta": beta_profile
            })
            save_profile(df, "velocity_profiles", halo_idx)

    # ----------------------------------------------------------------
    # Subfunction 3: Compute enclosed mass profiles
    # ----------------------------------------------------------------
    def mass_profile():
        n_halos = symlib.n_hosts(suite_name)
        bins, bin_centers = create_bins()
        params = symlib.simulation_parameters(suite_name)
        mp = params['mp'] / params['h100']

        for halo_idx in range(n_halos):
            host, p = load_halo_and_particles(halo_idx)
            if host is None or p is None:
                continue

            center, r_vir, m_vir = host['x'], host['rvir'], host['m']

            # Radial bins and enclosed mass fraction
            radi = np.linalg.norm(p[0]['x'] - center, axis=1) / r_vir
            counts, _ = np.histogram(radi, bins=bins)
            enclosed_mass = np.cumsum(counts * mp)
            m_scaled = enclosed_mass / m_vir

            # Save mass profile
            df = pd.DataFrame({"halo_idx": halo_idx, "r_scaled": bin_centers, "m_scaled": m_scaled})
            save_profile(df, "mass_profiles", halo_idx)

    # ----------------------------------------------------------------
    # Execute all profiles
    # ----------------------------------------------------------------
    density_profile()
    velocity_profile()
    mass_profile()
