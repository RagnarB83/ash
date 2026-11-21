from ash import Theory, Singlepoint, NumGradclass
import numpy as np

# Helper functions for testing

# Translational invariance check
def transvar_check(gradient,label="", tolerance=1e-5):
    # Calculate the gradient sum along the axis
    grad_sum = np.sum(gradient, axis=0)
    print(f"Sum gradient ({label}):", grad_sum)
    assert np.allclose(grad_sum, 0.0, atol=tolerance), (
        f"""Test:{label} Gradient is not translationally invariant: Max-value {max(grad_sum, key=abs)} > tol={tolerance}"""
    )

def numgrad_check(frag, theory, TOL=1e-5, label=""):

    # Analytical
    res_ana = Singlepoint(theory=theory, fragment=frag, Grad=True, charge=0, mult=1) 
    # Numerical
    numgradtheory = NumGradclass(theory=theory)
    res_num = Singlepoint(theory=numgradtheory, fragment=frag, Grad=True, charge=0, mult=1) 

    g_ana = res_ana.gradient
    g_num = res_num.gradient

    resdict = {
        "force_sum": np.abs(g_ana.sum(axis=0)).max(),
        "fd_err": np.abs(g_ana - g_num).max(),
        "torque": np.abs(np.cross(frag.coords, g_ana).sum(axis=0)).max(),   # informational
    }
    print("resdict forcesum:", resdict["force_sum"])
    print("resdict fd_err:", resdict["fd_err"])
    print("TOL:", TOL)
    assert resdict["force_sum"] < TOL, f"{label}:{resdict}"
    assert resdict["fd_err"] < TOL, f"{label}:{resdict}"


class MockPairTheory(Theory):
    """Anharmonic pair potential over ALL atoms passed in (incl. linkatoms), analytic gradient."""
    WEIGHTS = {'C': 1.0, 'H': 0.6}

    def __init__(self, k=0.7, r0=1.5):
        super().__init__()
        self.theorynamelabel = "MockPair"
        self.theorytype = "QM"
        self.printlevel = 0
        self.filename = ""
        self.k = k
        self.r0 = r0

    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            mm_elems=None, elems=None, Grad=False, PC=False, numcores=None, restart=False,
            label=None, charge=None, mult=None):
        x = np.asarray(current_coords, dtype=float)
        els = qm_elems if qm_elems is not None else elems
        assert len(els) == len(x), "elems and coords out of sync (subset/linkatom bookkeeping bug)"
        w = np.array([self.WEIGHTS.get(e, 1.0) for e in els])
        n = len(x)
        energy = 0.0
        grad = np.zeros((n, 3))
        for i in range(n):
            for j in range(i + 1, n):
                d = x[i] - x[j]
                r = np.linalg.norm(d)
                u = r - self.r0
                ww = w[i] * w[j]
                energy += ww * (0.5 * self.k * u**2 + 0.1 * u**3)
                dEdr = ww * (self.k * u + 0.3 * u**2)
                grad[i] += dEdr * d / r
                grad[j] -= dEdr * d / r
        return (energy, grad) if Grad else energy