import os
import csv
import warnings
import traceback
import numpy as np
import pandas as pd
from astropy.constants import G
from astropy import units as u
import symlib
import cosmology

def process_halo_properties(input_dir, output_dir, suite_name, snap):
    """
    Process a halo suite: save halo properties(including virial mass, virial radius, virial velocity, concentration), 
    compute rvmax, Jeans deviation,
    compute dynamical accretion rates.

    Parameters
    ----------
    input_dir : str
        Root directory containing Symphony simulation data.

    output_dir : str
        Directory where processed outputs will be saved.

    suite_name : str
        Name of the simulation suite (e.g., 'SymphonyLMC').

    snap : int
        Snapshot index to use for halo properties.
    """

    output_dir = os.path.join(output_dir, suite_name)
    os.makedirs(output_dir, exist_ok=True)

    # ------------------- Save halo properties -------------------
    def save_halo_properties():
        """Save halo mass, virial radius, virial velocity, and concentration."""
        n_halos = symlib.n_hosts(suite_name)

        halo_ids, mvir_list, rvir_list, vvir_list, cvir_list = [], [], [], [], []

        for halo_idx in range(n_halos):
            sim_dir = symlib.get_host_directory(input_dir, suite_name, halo_idx)
            try:
                r, _ = symlib.read_rockstar(sim_dir)
                host = r[0, snap]

                halo_ids.append(f"{halo_idx:03d}")

                # Mass
                mvir_list.append(host["m"])

                # Virial radius
                rvir_list.append(host["rvir"])

                # Virial velocity
                m = host["m"] * u.Msun
                r_kpc = host["rvir"] * u.kpc
                G_kpc = G.to(u.kpc * (u.km/u.s)**2 / u.Msun)
                vvir = np.sqrt(G_kpc * m / r_kpc).to(u.km/u.s).value
                vvir_list.append(vvir)

                # Concentration
                cvir_list.append(host["cvir"])

            except FileNotFoundError:
                print(f"[Warning] Rockstar file not found for Halo {halo_idx}")
                continue

        # Save CSVs
        pd.DataFrame({"halo_id": halo_ids, "mvir": mvir_list}).to_csv(
            os.path.join(output_dir, "halo_mass.csv"), index=False)
        pd.DataFrame({"halo_id": halo_ids, "rvir": rvir_list}).to_csv(
            os.path.join(output_dir, "virial_radius.csv"), index=False)
        pd.DataFrame({"halo_id": halo_ids, "vvir": vvir_list}).to_csv(
            os.path.join(output_dir, "virial_velocity.csv"), index=False)
        pd.DataFrame({"halo_id": halo_ids, "cvir": cvir_list}).to_csv(
            os.path.join(output_dir, "halo_concentrations.csv"), index=False)

        print(f"[Saved] Halo properties for {suite_name} to {output_dir}")

    # ------------------- Compute rvmax from mass profiles -------------------
    def save_rvmax():
        """Compute radius of maximum circular velocity from mass profiles."""
        mass_dir = os.path.join(output_dir, "mass_profiles")
        df_rvir = pd.read_csv(os.path.join(output_dir, "virial_radius.csv"), dtype={"halo_id": str})
        os.makedirs(output_dir, exist_ok=True)

        def compute_rvmax_from_mass_profile(r, m):
            vc = np.sqrt(m / r)
            return r[np.argmax(vc)]

        files = sorted([f for f in os.listdir(mass_dir) if f.endswith(".csv")])
        halo_ids, rvmax_list = [], []

        for f in files:
            try:
                df = pd.read_csv(os.path.join(mass_dir, f))
                r = df["r_scaled"].values
                m = df["m_scaled"].values
                rvmax = compute_rvmax_from_mass_profile(r, m)

                halo_id = f.split("_")[1]
                if halo_id not in df_rvir["halo_id"].values:
                    print(f"[Warning] halo_id {halo_id} not found in virial_radius.csv")
                    continue
                rvir = df_rvir.loc[df_rvir.halo_id == halo_id, "rvir"].values[0]

                halo_ids.append(halo_id)
                rvmax_list.append(rvmax * rvir)
            except Exception as e:
                print(f"[Warning] Failed to process {f}: {e}")
                continue

        pd.DataFrame({"halo_id": halo_ids, "rvmax": rvmax_list}).to_csv(
            os.path.join(output_dir, "max_radius.csv"), index=False)
        print(f"[Saved] rvmax for {suite_name}")

    # ------------------- Compute Jeans deviation -------------------
    def compute_jeans_deviation():
        """Compute Jeans deviation δ_J(r) and total δ_J for each halo."""
        density_dir = os.path.join(output_dir, suite_name, "density_profiles")
        velocity_dir = os.path.join(output_dir, suite_name, "velocity_profiles")
        mass_dir = os.path.join(output_dir, suite_name, "mass_profiles")
        jeans_dir = os.path.join(output_dir, "jeans_deviation")
        os.makedirs(jeans_dir, exist_ok=True)

        halo_files = sorted([f for f in os.listdir(density_dir) if f.endswith(".csv")])
        for f in halo_files:
            halo_id = f.split("_")[1]
            try:
                df_rho = pd.read_csv(os.path.join(density_dir, f))
                df_vel = pd.read_csv(os.path.join(velocity_dir, f))
                df_mass = pd.read_csv(os.path.join(mass_dir, f))

                r = df_rho["r_scaled"].values
                rho = df_rho["rho_scaled"].values
                sigma_r2 = df_vel["sigma_rad_scaled"].values ** 2
                beta = df_vel["beta"].values
                m = df_mass["m_scaled"].values

                P_r = rho * sigma_r2
                dPdr = np.gradient(P_r, r)
                dPhidr = m / (r ** 2)
                numer = dPdr + 2 * beta * P_r / r + rho * dPhidr
                denom = rho * dPhidr
                delta_J = np.abs(numer / denom)

                pd.DataFrame({"halo_id": int(halo_id), "r_scaled": r, "delta_J": delta_J}).to_csv(
                    os.path.join(jeans_dir, f"halo_{int(halo_id):03d}_profile.csv"), index=False)
                print(f"[Saved] Jeans deviation for halo {halo_id}")

            except Exception as e:
                print(f"[Warning] Failed to compute Jeans deviation for halo {halo_id}: {e}")

        # Compute total δ_J
        results = []
        files = sorted([f for f in os.listdir(jeans_dir) if f.endswith(".csv")])
        for f in files:
            halo_id = int(f.split("_")[1])
            df = pd.read_csv(os.path.join(jeans_dir, f))
            r = df["r_scaled"].values
            delta_J = df["delta_J"].values
            mask = r <= 1.0
            delta_J_tot = np.trapz(delta_J[mask], r[mask])
            results.append({"halo_id": halo_id, "delta_J_tot": delta_J_tot})

        pd.DataFrame(results).to_csv(os.path.join(output_dir, "jeans_deviation_total.csv"), index=False)
        print(f"[Saved] Total Jeans deviation for {suite_name}")

    # ------------------- Compute dynamical accretion rates -------------------
    def compute_accretion_rates():
        """Compute mass accretion rates over the last dynamical time."""
        import traceback
        output_csv = os.path.join(output_dir, "accretion_rates.csv")
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        with open(output_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["halo_id", "gamma"])

            n_halos = symlib.n_hosts(suite_name)
            for i in range(n_halos):
                try:
                    sim_dir = symlib.get_host_directory(input_dir, suite_name, i)
                    scale = symlib.scale_factors(sim_dir)
                    r, _ = symlib.read_rockstar(sim_dir)
                    snap_last = len(scale) - 1
                    m_now = r[0, snap_last]["m"]

                    sim_params = symlib.simulation_parameters(sim_dir)
                    cosmo = cosmology.setCosmology("custom", {
                        "flat": sim_params["flat"],
                        "H0": sim_params["H0"],
                        "Om0": sim_params["Om0"],
                        "Ob0": sim_params["Ob0"],
                        "sigma8": sim_params["sigma8"],
                        "ns": sim_params["ns"]
                    })

                    rho_m0 = cosmo.rho_m(0)
                    Delta = 99
                    rho_vir = Delta * rho_m0 * 1e9 / sim_params["h100"]**2
                    G_val = G.to(u.Mpc**3 / (u.Msun * u.Gyr**2)).value
                    t_dyn = 1.0 / np.sqrt((4/3) * np.pi * G_val * rho_vir)

                    times = cosmo.age(1 / scale - 1)
                    t0 = times[snap_last]
                    t_past = t0 - t_dyn
                    snap_past = np.argmin(np.abs(times - t_past))
                    m_past = r[0, snap_past]["m"]
                    gamma = (m_now - m_past) / t_dyn
                    writer.writerow([f"{i:03d}", gamma])

                except Exception as e:
                    traceback.print_exc()
                    warnings.warn(f"[{suite_name} - halo {i}] Error calculating accretion rate: {e}")
                    writer.writerow([i, "nan"])

        print(f"[Saved] Accretion rates for {suite_name} → {output_csv}")

    # ------------------- Run all sub-functions -------------------
    save_halo_properties()
    save_rvmax()
    compute_jeans_deviation()
    compute_accretion_rates()
