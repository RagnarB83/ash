"""PM6 energy and gradient for H2 using the native SCINE Sparrow interface."""

from ash import Fragment, Singlepoint, SparrowTheory

fragment = Fragment(
    coordsstring="H 0.000000 0.000000 0.000000\nH 0.000000 0.000000 0.740000",
    charge=0,
    mult=1,
)
theory = SparrowTheory(method="PM6")
result = Singlepoint(fragment=fragment, theory=theory, Grad=True)
print(f"Sparrow PM6 energy: {result.energy:.10f} Eh")
print("Sparrow PM6 gradient (Eh/Bohr):")
print(result.gradient)
