#!/usr/bin/env python3
"""Rebuild the result tables in README.md from results.json and phenol_energy_only.json.

Everything between the TABLES:BEGIN and TABLES:END markers in README.md is replaced.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEGIN, END = "<!-- TABLES:BEGIN -->", "<!-- TABLES:END -->"
LABEL = {
    "meoh_water": "H₂O–MeOH",
    "rubredoxin": "Rubredoxin QM-II",
    "rubredoxin_qm1": "Rubredoxin QM-I",
    "rubredoxin_qm4": "Rubredoxin QM-IV",
    "rubredoxin_qm5": "Rubredoxin QM-V",
    "lysozyme": "Lysozyme (test fixture)",
    "lysozyme_modeller": "Lysozyme (Modeller example)",
    "dhfr_charmm": "DHFR, CHARMM files",
    "gaba_water": "3FGABA in water",
    "isobutyraldehyde_water": "Isobutyraldehyde in water",
    "fecl4_water": "FeCl₄⁻ in water",
    "na2hpo4_molcrys": "Na₂HPO₄ cluster",
}


def table(header, align, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def ms(t):
    return f"{t * 1e3:.3f}"


def yes(flag):
    return "yes" if flag else "NO"


def build(results, phenol):
    R, L = "---:", "---"
    out = []

    out.append("### Table 1. Systems\n")
    out.append(table(
        ["Case", "Key in results.json", "xTB method", "QM charge / mult", "Call", "Atoms", "QM atoms",
         "Point charges", "`pcharge` bytes", "`pcgrad` bytes"],
        [L, L, L, L, L, R, R, R, R, R],
        [[LABEL[x["case"]], f"`{x['case']}`", x["xtb_method"], f"{x['qm_charge']} / {x['qm_mult']}",
          "MD force" if x["native"]["call"] == "md_force" else "E + gradient",
          f"{x['native']['system_atoms']:,}", x["native"]["qm_atoms"], f"{x['native']['point_charges']:,}",
          f"{x['repeated_io']['pcharge_bytes']:,}", f"{x['repeated_io']['pcgrad_bytes']:,}"] for x in results]))

    out.append("\n### Table 2. Source example and adaptation\n")
    out.append(table(["Case", "ASH example scripts represented", "Adaptation"], [L, L, L],
                     [[LABEL[x["case"]], x["source_example"], x["adaptation"]] for x in results]))

    for number, title, kind in ((3, "Point-charge writer (`pcharge`)", "writer"),
                                (4, "Point-charge gradient reader (`pcgrad`)", "reader")):
        out.append(f"\n### Table {number}. {title}, ms per call over "
                   f"{results[0]['repeated_io']['repetitions']} repetitions\n")
        rows = []
        for x in results:
            a, b = x["repeated_io"][f"{kind}_native"], x["repeated_io"][f"{kind}_fast_io"]
            rows.append([LABEL[x["case"]], ms(a["median_s"]), ms(a["min_s"]), ms(a["max_s"]),
                         ms(b["median_s"]), ms(b["min_s"]), ms(b["max_s"]),
                         f"{a['median_s'] / b['median_s']:.2f}×",
                         f"{(a['median_s'] - b['median_s']) * 1e3:+.3f}"])
        out.append(table(["Case", "Native median (ms)", "Native min (ms)", "Native max (ms)",
                          "Fast median (ms)", "Fast min (ms)", "Fast max (ms)",
                          "Speedup (native ÷ fast median)", "Time saved per call (ms)"], [L] + [R] * 8, rows))

    out.append("\n### Table 5. Combined I/O and full call\n")
    rows = []
    for x in results:
        io, n, f = x["repeated_io"], x["native"], x["fast_io"]
        w0, w1, r0, r1 = (io[k]["median_s"] for k in
                          ("writer_native", "writer_fast_io", "reader_native", "reader_fast_io"))
        rows.append([LABEL[x["case"]], ms(w0 + r0), ms(w1 + r1), f"{(w0 + r0) / (w1 + r1):.2f}×",
                     f"{(w0 + r0 - w1 - r1) * 1e3:+.3f}", f"{n['elapsed_s']:.3f}", f"{f['elapsed_s']:.3f}",
                     f"{n['elapsed_s'] / f['elapsed_s']:.2f}×",
                     f"{100 * (w0 + r0) / n['elapsed_s']:.2f}%",
                     f"{100 * (w0 + r0 - w1 - r1) / n['elapsed_s']:+.2f}%"])
    out.append(table(["Case", "Writer + reader, native (ms)", "Writer + reader, fast (ms)", "I/O speedup (native ÷ fast)",
                      "I/O time saved per call (ms)", "Full call, native (s)", "Full call, fast (s)",
                      "Full-call speedup (native ÷ fast)",
                      "Native I/O share of full call", "I/O saving as share of full call"], [L] + [R] * 9, rows))

    out.append("\n### Table 6. In-run I/O times, ms (single measurement inside each full call)\n")
    rows = []
    for x in results:
        n, f = x["native"], x["fast_io"]
        rows.append([LABEL[x["case"]], len(n["writer_s"]), ms(sum(n["writer_s"])), ms(sum(f["writer_s"])),
                     len(n["reader_s"]), ms(sum(n["reader_s"])), ms(sum(f["reader_s"]))])
    out.append(table(["Case", "Writer calls", "Writer, native (ms)", "Writer, fast (ms)", "Reader calls",
                      "Reader, native (ms)", "Reader, fast (ms)"], [L] + [R] * 6, rows))

    out.append("\n### Table 7. Energies (Eh)\n")
    out.append(table(["Case", "Native (Eh)", "Fast I/O (Eh)", "ΔE (Eh)", "Exactly equal"], [L, R, R, R, L],
                     [[LABEL[x["case"]], f"{x['native']['energy_Eh']:.12f}", f"{x['fast_io']['energy_Eh']:.12f}",
                       f"{x['delta_energy_Eh']:g}", yes(x["energy_exact"])] for x in results]))

    out.append("\n### Table 8. Gradients (Eh/Bohr)\n")
    out.append(table(["Case", "Shape", "Norm, native (Eh/Bohr)", "Norm, fast I/O (Eh/Bohr)", "Max abs Δ (Eh/Bohr)", "Exactly equal",
                      "SHA-256 equal", "SHA-256 (first 16 hex)"], [L, L, R, R, R, L, L, L],
                     [[LABEL[x["case"]], " × ".join(str(i) for i in x["native"]["gradient_shape"]),
                       f"{x['native']['gradient_norm']:.12f}", f"{x['fast_io']['gradient_norm']:.12f}",
                       f"{x['max_abs_delta_gradient']:g}", yes(x["gradient_exact"]),
                       yes(x["native"]["gradient_sha256"] == x["fast_io"]["gradient_sha256"]),
                       f"`{x['native']['gradient_sha256'][:16]}`"] for x in results]))

    out.append("\n### Table 9. Point-charge files\n")
    out.append(table(["Case", "Max abs Δ PC gradient (Eh/Bohr)", "PC gradient exactly equal",
                      "Max abs Δ `pcharge` record (file units)", "`pcharge` records exactly equal"], [L, R, L, R, L],
                     [[LABEL[x["case"]], f"{x['max_abs_delta_pc_gradient']:g}", yes(x["pc_gradient_exact"]),
                       f"{x['max_abs_delta_pointcharge_record']:g}", yes(x["pointcharge_records_exact"])]
                      for x in results]))

    m = phenol["modes"]
    out.append("\n### Table 10. Phenol–water energy-only check\n")
    out.append(table(["Quantity", "Native", "Fast I/O"], [L, R, R], [
        ["Source", phenol["source"], ""],
        ["xTB method", phenol["method"], ""],
        ["Atoms / QM atoms / point charges",
         f"{phenol['system_atoms']} / {phenol['qm_atoms']} / {phenol['point_charges']}", ""],
        ["Energy (Eh)", f"{m['native']['energy_Eh']:.12f}", f"{m['fast_io']['energy_Eh']:.12f}"],
        ["ΔE (Eh)", f"{phenol['delta_energy_Eh']:g}", f"exactly equal: {yes(phenol['energy_exact'])}"],
        ["Writer (ms, single call)", ms(m["native"]["writer_s"]), ms(m["fast_io"]["writer_s"])],
        ["Full call (s)", f"{m['native']['elapsed_s']:.3f}", f"{m['fast_io']['elapsed_s']:.3f}"],
        ["`pcgrad` read", "no (energy only)", "no (energy only)"]]))
    return "\n".join(out)


def main():
    results = json.loads((HERE / "results.json").read_text())["results"]
    phenol = json.loads((HERE / "phenol_energy_only.json").read_text())
    readme = HERE / "README.md"
    text = readme.read_text()
    head, rest = text.split(BEGIN)
    tail = rest.split(END)[1]
    readme.write_text(head + BEGIN + "\n" + build(results, phenol) + "\n" + END + tail)
    print(f"Tables rebuilt in {readme}")


if __name__ == "__main__":
    main()
