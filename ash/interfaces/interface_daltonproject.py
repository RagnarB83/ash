"""ASH interface to the Dalton Project Python package.

This is separate from :class:`DaltonTheory`, which accepts native Dalton input
text and runs the command-line executable directly.
"""

import numpy as np

from ash.modules.module_theory import QMTheory


class DaltonProjectTheory(QMTheory):
    """Run Dalton or LSDalton through the Dalton Project Python API.

    ``daltonproject`` and the selected external executable must be installed
    separately. ASH coordinates and charge/multiplicity are converted to the
    Dalton Project molecule and calculation objects for each run.
    """

    def __init__(self, basis, qc_method="HF", functional=None, program="dalton",
                 label="DaltonProject", numcores=1, omp_num_threads=1,
                 work_dir=None, scratch_dir=None, memory=2000, scf_threshold=None,
                 printlevel=2):
        super().__init__()
        self.theorytype = "QM"
        self.theorynamelabel = "DaltonProject"
        self.analytic_hessian = False
        self.basis = basis
        self.qc_method = qc_method
        self.functional = functional
        self.program = program.lower()
        if self.program not in ("dalton", "lsdalton"):
            raise ValueError("program must be 'dalton' or 'lsdalton'")
        self.label = label
        self.numcores = numcores
        self.omp_num_threads = omp_num_threads
        self.work_dir = work_dir
        self.scratch_dir = scratch_dir
        self.memory = memory
        self.scf_threshold = scf_threshold
        self.printlevel = printlevel

    def run(self, current_coords=None, elems=None, qm_elems=None, Grad=False,
            numcores=None, charge=None, mult=None, label=None, **kwargs):
        """Compute an energy and, when requested, an analytic gradient."""
        if current_coords is None:
            raise ValueError("DaltonProjectTheory requires current_coords")
        if charge is None or mult is None:
            raise ValueError("DaltonProjectTheory requires charge and mult")
        atom_elems = qm_elems if qm_elems is not None else elems
        if atom_elems is None:
            raise ValueError("DaltonProjectTheory requires elems or qm_elems")
        if kwargs:
            unsupported = ", ".join(sorted(kwargs))
            raise NotImplementedError(
                "DaltonProjectTheory does not yet support additional calculation "
                "arguments ({}). In particular, embedding settings must not be "
                "silently ignored.".format(unsupported)
            )
        coords = np.asarray(current_coords, dtype=float)
        if coords.shape != (len(atom_elems), 3):
            raise ValueError("current_coords must have shape (number of atoms, 3)")

        try:
            import daltonproject as dp
        except ImportError as exc:
            raise ImportError(
                "DaltonProjectTheory requires the optional 'daltonproject' "
                "Python package and a working Dalton executable."
            ) from exc

        settings = dp.ComputeSettings(
            work_dir=self.work_dir,
            scratch_dir=self.scratch_dir,
            mpi_num_procs=numcores or self.numcores,
            omp_num_threads=self.omp_num_threads,
            memory=self.memory,
        )
        qc_options = {"qc_method": self.qc_method}
        if self.functional is not None:
            qc_options["xc_functional"] = self.functional
        if self.scf_threshold is not None:
            qc_options["scf_threshold"] = self.scf_threshold
        qc_method = dp.QCMethod(**qc_options)
        properties = dp.Property(energy=True, gradients=Grad)

        run_label = label or self.label or "DaltonProject"
        safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_label)
        atom_string = "\n".join(
            "{} {:.12f} {:.12f} {:.12f}".format(element, *coord)
            for element, coord in zip(atom_elems, coords)
        )
        molecule = dp.Molecule(
            atoms=atom_string,
            charge=int(charge),
            multiplicity=int(mult),
        )
        basis = dp.Basis(basis=self.basis)
        compute = getattr(dp, self.program).compute
        result = compute(
            molecule, basis, qc_method, properties,
            compute_settings=settings,
            filename=safe_label,
            force_recompute=True,
        )

        self.energy = float(result.energy)
        if Grad:
            self.gradient = np.asarray(result.gradients, dtype=float)
            if self.gradient.shape != coords.shape:
                raise ValueError(
                    "Dalton Project returned gradient shape {}; expected {}".format(
                        self.gradient.shape, coords.shape
                    )
                )
            return self.energy, self.gradient
        return self.energy
