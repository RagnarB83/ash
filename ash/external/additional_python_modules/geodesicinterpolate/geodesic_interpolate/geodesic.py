"""Geodesic smoothing.   Minimize the path length using redundant internal coordinate
metric to find geodesics directly in Cartesian, to avoid feasibility problems associated
with redundant internals.
"""
import logging

import numpy as np
from scipy.optimize import least_squares

from .coord_utils import align_path, get_bond_list, morse_scaler, compute_wij


logger = logging.getLogger(__name__)


class Geodesic(object):
    """Optimizer to obtain geodesic in redundant internal coordinates.  Core part is the calculation
    of the path length in the internal metric."""
    def __init__(self, atoms, path, scaler=1.7, threshold=3, min_neighbors=4, log_level=logging.INFO,
                 friction=1e-3):
        """Initialize the interpolater
        Args:
            atoms:      Atom symbols, used to lookup radii
            path:       Initial geometries of the path, must be of dimension `nimage * natoms * 3`
            scaler:     Either the alpha parameter for morse potential, or an explicit scaling function.
                        It is easier to get smoother paths with small number of data points using small
                        scaling factors, as they have large range, but larger values usually give
                        better energetics because they better represent the (sharp) energy landscape.
            threshold:  Distance cut-off for constructing inter-nuclear distance coordinates.  Note that
                        any atoms linked by three or less bonds will also be added.
            min_neighbors:  Minimum number of neighbors an atom must have in the atom pair list.
            log_level:  Logging level to use.
            friction:   Friction term in the target function which regularizes the optimization step
                        size to prevent explosion.
        """
        rmsd0, self.path = align_path(path)
        logger.log(log_level, "Maximum RMSD change in initial path: %10.2f", rmsd0)
        if self.path.ndim != 3:
            raise ValueError('The path to be interpolated must have 3 dimensions')
        self.nimages, self.natoms, _ = self.path.shape
        # Construct coordinates
        self.rij_list, self.re = get_bond_list(path, atoms, threshold=threshold, min_neighbors=min_neighbors)
        if isinstance(scaler, float):
            self.scaler = morse_scaler(re=self.re, alpha=1.7)
        else:
            self.scaler = scaler
        self.nrij = len(self.rij_list)
        self.friction = friction
        # Initalize interal storages for mid points, internal coordinates and B matrices
        logger.log(log_level, "Performing geodesic smoothing")
        logger.log(log_level, "  Images: %4d  Atoms %4d Rijs %6d", self.nimages, self.natoms, len(self.rij_list))
        self.neval = 0
        self.w = [None] * len(path)
        self.dwdR = [None] * len(path)
        self.X_mid = [None] * (len(path) - 1)
        self.w_mid = [None] * (len(path) - 1)
        self.dwdR_mid = [None] * (len(path) - 1)
        self.disps = self.grad = self.segment = None
        self.conv_path = []

    def update_intc(self):
        """Adjust unknown locations of mid points and compute missing values of internal coordinates
        and their derivatives.  Any missing values will be marked with None values in internal storage,
        and this routine finds and calculates them.  This is to avoid redundant evaluation of value and
        gradients of internal coordinates."""
        for i, (X, w, dwdR) in enumerate(zip(self.path, self.w, self.dwdR)):
            if w is None:
                self.w[i], self.dwdR[i] = compute_wij(X, self.rij_list, self.scaler)
        for i, (X0, X1, w) in enumerate(zip(self.path, self.path[1:], self.w_mid)):
            if w is None:
                self.X_mid[i] = Xm = (X0 + X1) / 2
                self.w_mid[i], self.dwdR_mid[i] = compute_wij(Xm, self.rij_list, self.scaler)

    def update_geometry(self, X, start, end):
        """Update the geometry of a segment of the path, then set the corresponding internal
        coordinate, derivatives and midpoint locations to unknown"""
        X = X.reshape(self.path[start:end].shape)
        if np.array_equal(X, self.path[start:end]):
            return False
        self.path[start:end] = X
        for i in range(start, end):
            self.w_mid[i] = self.w[i] = None
        self.w_mid[start - 1] = None
        return True

    def compute_disps(self, start=1, end=-1, dx=None, friction=1e-3):
        """Compute displacement vectors and total length between two images.
        Only recalculate internal coordinates if they are unknown."""
        if end < 0:
            end += self.nimages
        self.update_intc()
        # Calculate displacement vectors in each segment, and the total length
        vecs_l = [wm - wl for wl, wm in zip(self.w[start - 1:end], self.w_mid[start - 1:end])]
        vecs_r = [wr - wm for wr, wm in zip(self.w[start:end + 1], self.w_mid[start - 1:end])]
        self.length = np.sum(np.linalg.norm(vecs_l, axis=1)) + np.sum(np.linalg.norm(vecs_r, axis=1))
        if dx is None:
            trans = np.zeros(self.path[start:end].size)
        else:
            trans = friction * dx  # Translation from initial geometry.  friction term 
        self.disps = np.concatenate(vecs_l + vecs_r + [trans])
        self.disps0 = self.disps[:len(vecs_l) * 2]

    def compute_disp_grad(self, start, end, friction=1e-3):
        """Compute derivatives of the displacement vectors with respect to the Cartesian coordinates"""
        # Calculate derivatives of displacement vectors with respect to image Cartesians
        l = end - start + 1
        self.grad = np.zeros((l * 2 * self.nrij + 3 * (end - start) * self.natoms, (end - start) * 3 * self.natoms))
        self.grad0 = self.grad[:l * 2 * self.nrij]
        grad_shape = (l, self.nrij, end - start, 3 * self.natoms)
        grad_l = self.grad[:l * self.nrij].reshape(grad_shape)
        grad_r = self.grad[l * self.nrij:l * self.nrij * 2].reshape(grad_shape)
        for i, image in enumerate(range(start, end)):
            dmid1 = self.dwdR_mid[image - 1] / 2
            dmid2 = self.dwdR_mid[image] / 2
            grad_l[i + 1, :, i, :] = dmid2 - self.dwdR[image]
            grad_l[i, :, i, :] = dmid1
            grad_r[i + 1, :, i, :] = -dmid2
            grad_r[i, :, i, :] = self.dwdR[image] - dmid1
        for idx in range((end - start) * 3 * self.natoms):
            self.grad[l * self.nrij * 2 + idx, idx] = friction

    def compute_target_func(self, X=None, start=1, end=-1, log_level=logging.INFO, x0=None, friction=1e-3):
        """Compute the vectorized target function, which is then used for least
        squares minimization."""
        if end < 0:
            end += self.nimages
        if X is not None and not self.update_geometry(X, start, end) and self.segment == (start, end):
            return
        self.segment = start, end
        dx = np.zeros(self.path[start:end].size) if x0 is None else self.path[start:end].ravel() - x0.ravel()
        self.compute_disps(start, end, dx=dx, friction=friction)
        self.compute_disp_grad(start, end, friction=friction)
        self.optimality = np.linalg.norm(np.einsum('i,i...', self.disps, self.grad), ord=np.inf)
        logger.log(log_level, "  Iteration %3d: Length %10.3f |dL|=%7.3e", self.neval, self.length, self.optimality)
        self.conv_path.append(self.path[1].copy())
        self.neval += 1

    def target_func(self, X, **kwargs):
        """Wrapper around `compute_target_func` to prevent repeated evaluation at
        the same geometry"""
        self.compute_target_func(X, **kwargs)
        return self.disps

    def target_deriv(self, X, **kwargs):
        """Wrapper around `compute_target_func` to prevent repeated evaluation at
        the same geometry"""
        self.compute_target_func(X, **kwargs)
        return self.grad

    def smooth(self, tol=1e-3, max_iter=50, start=1, end=-1, log_level=logging.INFO, friction=None,
               xref=None):
        """Minimize the path length as an overall function of the coordinates of all the images.
        This should in principle be very efficient, but may be quite costly for large systems with
        many images.

        Args:
            tol:        Convergence tolerance of the optimality. (.i.e uniform gradient of target func)
            max_iter:   Maximum number of iterations to run.
            start, end: Specify which section of the path to optimize.
            log_level:  Logging level during the optimization

        Returns:
            The optimized path.  This is also stored in self.path
        """
        X0 = np.array(self.path[start:end]).ravel()
        if xref is None:
            xref= X0
        self.disps = self.grad = self.segment = None
        logger.log(log_level, "  Degree of freedoms %6d: ", len(X0))
        if friction is None:
            friction = self.friction
        # Configure the keyword arguments that will be sent to the target function.
        kwargs = dict(start=start, end=end, log_level=log_level, x0=xref, friction=friction)
        self.compute_target_func(**kwargs)  # Compute length and optimality
        if self.optimality > tol:
            result = least_squares(self.target_func, X0, self.target_deriv, ftol=tol, gtol=tol,
                                   max_nfev=max_iter, kwargs=kwargs, loss='soft_l1')
            self.update_geometry(result['x'], start, end)
            logger.log(log_level, "Smoothing converged after %d iterations", result['nfev'])
        else:
            logger.log(log_level, "Skipping smoothing: path already optimal.")
        rmsd, self.path = align_path(self.path)
        logger.log(log_level, "Final path length: %12.5f  Max RMSD in path: %10.2f", self.length, rmsd)
        return self.path

    def sweep(self, tol=1e-3, max_iter=50, micro_iter=20, start=1, end=-1):
        """Minimize the path length by adjusting one image at a time and sweeping the optimization
        side across the chain.  This is not as efficient, but scales much more friendly with the
        size of the system given the slowness of scipy's optimizers.  Also allows more detailed
        control and easy way of skipping nearly optimal points than the overall case.

        Args:
            tol:        Convergence tolerance of the optimality. (.i.e uniform gradient of target func)
            max_iter:   Maximum number of sweeps through the path.
            micro_iter: Number of micro-iterations to be performed when optimizing each image.
            start, end: Specify which section of the path to optimize.
            log_level:  Logging level during the optimization

        Returns:
            The optimized path.  This is also stored in self.path
        """
        if end < 0:
            end = self.nimages + end
        self.neval = 0
        images = range(start, end)
        logger.info("  Degree of freedoms %6d: ", (end - start) * 3 * self.natoms)
        # Microiteration convergence tolerances are adjusted on the fly based on level of convergence.
        curr_tol = tol * 10
        self.compute_disps()    # Compute and print the initial path length
        logger.info("  Initial length: %8.3f", self.length)
        for iteration in range(max_iter):
            max_dL = 0
            X0 = self.path.copy()
            for i in images[:-1]:   # Use self.smooth() to optimize individual images
                xmid = (self.path[i - 1] + self.path[i + 1]) * 0.5
                self.smooth(curr_tol, max_iter=min(micro_iter, iteration + 6),
                            start=i, end=i + 1, log_level=logging.DEBUG,
                            friction=self.friction if iteration else 0.1,
                            xref=xmid)
                max_dL = max(max_dL, self.optimality)
            self.compute_disps()    # Compute final length after sweep
            logger.info("Sweep %3d: L=%7.2f dX=%7.2e tol=%7.3e dL=%7.3e",
                     iteration, self.length, np.linalg.norm(self.path - X0), curr_tol, max_dL)
            if max_dL < tol:    # Check for convergence.
                logger.info("Optimization converged after %d iteartions", iteration)
                break
            curr_tol = max(tol * 0.5, max_dL * 0.2) # Adjust micro-iteration threshold
            images = list(reversed(images))         # Alternate sweeping direction.
        else:
            logger.info("Optimization not converged after %d iteartions", iteration)
        rmsd, self.path = align_path(self.path)
        logger.info("Final path length: %12.5f  Max RMSD in path: %10.2f", self.length, rmsd)
        return self.path
