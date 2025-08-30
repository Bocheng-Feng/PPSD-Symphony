# 🌌 Symphony-PPSD

Pseudo Phase Space Density (PPSD) analysis of cold dark matter（CDM）host halos in the **[Symphony Simulation Suite](https://arxiv.org/abs/2109.04476)**.  
This repository provides tools for computing, analyzing, and visualizing the PPSD structure of halos, with a focus on connections to halo concentration, accretion history, and deviations from  **Hydrostatic Equilibrium** (HSE).  

---

## 📖 Overview
The project aims to:
- Compute **density, velocity dispersion, and PPSD profiles** of dark matter halos  
- Measure **PPSD slopes** against radius/mass and compare them to analytical models (e.g., [Nadler et al. 2017](https://arxiv.org/abs/1701.01449))  
- Quantify the impact of **halo concentration** and **accretion rate** on PPSD deviations
- ***Deviations from a power-law PPSD arise naturally when halos are *not* in a dynamically relaxed state. In such systems, ongoing mergers or rapid accretion disturb equilibrium, preventing the PPSD from settling into the approximate power-law form observed in relaxed halos***
- Perform **statistical tests** (Spearman correlation, KS test, χ²) across multiple Symphony suites  

---
![ppsd_visual]()

## 🌍 Scientific Context
The pseudo phase space density,  

$Q(r) = \frac{\rho(r)}{\sigma^3(r)}$, or $~Q_r(r) = \frac{\rho(r)}{\sigma_r^3(r)}$

has long been noted in simulation to follow an approximate power law with radius
$Q \propto r^{-\chi}, \chi \approx -1.875$
in cold dark matter (CDM) halos.  
Its main appeal has been its **apparent universality**: 

- **If the PPSD was truly universal**
  - **Theoretically:** A universal profile would suggest a fundamental emergent process in self-gravitating systems. Yet, there is currently no first-principles explanation for why such a power law should arise.  
  - **Observationally:** Universality would allow direct inference of one profile from the other. For example, measuring a halo’s density profile would immediately give its velocity dispersion profile (and vice versa). It would also allow extrapolation of halo profiles into unconstrained radial regimes.

- **Deviations from universality** could serve as probes of additional physics:
  - Non-equilibrium assembly histories  
  - Baryonic processes (e.g., feedback, central black holes)  
  - Alternative dark matter models (e.g., SIDM, warm DM)  

However, **recent literature suggests that the PPSD is unlikely to be truly fundamental**.  
The motivations above remain compelling, but the situation is more complex: rather than relying on universality, we must understand how the PPSD depends on halo properties such as **concentration** and **accretion rate**.  

👉 This is the central goal of **Symphony-PPSD**: to leverage the statistical power of the Symphony Simulation Suite to map out the *non-universality* of the PPSD, quantify its deviations, and connect them to halo formation histories and environments.

---

## 📂 Repository Structure
```bash
.
