# MOPAC interface examples and benchmark

`MOPACTheory` runs PM7, PM6, PM6-D3H4, PM3, AM1, RM1, and other supported MOPAC NDDO methods as an ASH theory. It returns the MOPAC heat of formation converted to Eh, an optional analytic gradient in Eh/Bohr, atomic charges, and the dipole vector. `MOPACTheory` can also add electrostatic point-charge embedding through MOPAC's `QMMM` keyword.

Run `example_MOPAC.py` with MOPAC available in `PATH` or set `mopacdir` in the `MOPACTheory` constructor. For reliable gradient comparisons, the example uses `PRECISE` and `SCFCRT=1e-10`.

## Native MOPAC comparison

The first two cases use the official OpenMOPAC [SCF](https://github.com/openmopac/mopac/blob/main/examples/example_scf.mop) and [COSMO](https://github.com/openmopac/mopac/blob/main/examples/example_cosmo.mop) water inputs. Native MOPAC v23.2.5 was run with `GRADIENTS PRECISE SCFCRT=1.D-10 AUX(PRECISION=9,MOS=-1,COMP)` added to expose full-precision properties and gradients. Energy was converted from kcal/mol using 627.5094740631 kcal/mol per Eh; gradients were converted from kcal/mol/Å to Eh/Bohr. The native AUX arrays and ASH results matched exactly at the precision printed by MOPAC.

| Case | Native MOPAC ΔHf (kcal/mol) | Native converted energy (Eh) | ASH energy (Eh) | ΔE (Eh) | max |Δgradient| (Eh/Bohr) | max |Δcharge| | max |Δdipole| (Debye) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PM7 SCF | −57.6903015446 | −0.09193534747940472 | −0.09193534747940472 | 0.0 | 0.0 | 0.0 | 0.0 |
| PM7 COSMO, ε = 78.4 | −65.2184243639 | −0.10393217482697238 | −0.10393217482697238 | 0.0 | 0.0 | 0.0 | 0.0 |

## Optimization, frequencies, and point-charge gradients

| Check | Native/reference result | ASH result |
|---|---|---|
| Geometry relaxation | MOPAC example stopped at gradient norm 0.561 kcal/mol/Å and −57.79930 kcal/mol | geomeTRIC converged; −0.09210993870 Eh (−57.79986 kcal/mol); max gradient component 2.18×10⁻⁵ Eh/Bohr |
| Water vibrational modes | MOPAC `FORCE`: 1395.53, 2810.70, 2859.67 cm⁻¹ | ASH `NumFreq` (0.005 Å displacement): 1397.10, 2810.32, 2859.11 cm⁻¹; largest difference 1.57 cm⁻¹ |
| QM/MM point-charge gradients | Central finite differences of MOPAC QMMM energies (0.001 Å displacement) | Max error: 1.41×10⁻⁷ Eh/Bohr for QM gradients; 6.41×10⁻¹⁰ Eh/Bohr for point-charge gradients |

The optimization and frequency checks use ASH's generic optimizer and numerical Hessian with the official [MOPAC relaxation](https://github.com/openmopac/mopac/blob/main/examples/example_relax.mop) and [vibrational](https://github.com/openmopac/mopac/blob/main/examples/example_vibe.mop) water geometries. `MOPACTheory` does not expose MOPAC's optimizer or analytic Hessian as ASH theory features.
