"""Coordinate utilities used by the interpolation program"""
import logging

import numpy as np
from scipy.spatial import KDTree


logger = logging.getLogger(__name__)


def align_path(path):
    """Rotate and translate images to minimize RMSD movements along the path.
    Also moves the geometric center of all images to the origin.
    """
    path = np.array(path)
    path[0] -= np.mean(path[0], axis=0)
    max_rmsd = 0
    for g, nextg in zip(path, path[1:]):
        rmsd, nextg[:] = align_geom(g, nextg)
        max_rmsd = max(max_rmsd, rmsd)
    return max_rmsd, path


def align_geom(refgeom, geom):
    """Find translation/rotation that moves a given geometry to maximally overlap
    with a reference geometry. Implemented with Kabsch algorithm.

    Args:
        refgeom:    The reference geometry to be rotated to
        geom:       The geometry to be rotated and shifted

    Returns:
        RMSD:       Root-mean-squared difference between the rotated geometry
                    and the reference
        new_geom:   The rotated geometry that maximumally overal with the reference
    """
    center = np.mean(refgeom, axis=0)   # Find the geometric center
    ref2 = refgeom - center
    geom2 = geom - np.mean(geom, axis=0)
    cov = np.dot(geom2.T, ref2)
    v, sv, w = np.linalg.svd(cov)
    if np.linalg.det(v) * np.linalg.det(w) < 0:
        sv[-1] = -sv[-1]
        v[:, -1] = -v[:, -1]
    u = np.dot(v, w)
    new_geom = np.dot(geom2, u) + center
    rmsd = np.sqrt(np.mean((new_geom - refgeom) ** 2))
    return rmsd, new_geom


ATOMIC_RADIUS = dict(H=0.31, He=0.28,
                     Li=1.28, Be=0.96, B=0.84, C=0.76, N=0.71, O=0.66, F=0.57, Ne=0.58,
                     Na=1.66, Mg=1.41, Al=1.21, Si=1.11, P=1.07, S=1.05, Cl=1.02, Ar=1.06)


def get_bond_list(geom, atoms=None, threshold=4, min_neighbors=4, snapshots=30, bond_threshold=1.8,
                  enforce=()):
    """Get the list of all the important atom pairs.
    Samples a number of snapshots from a list of geometries to generate all
    distances that are below a given threshold in any of them.

    Args:
        atoms:      Symbols for each atoms.
        geom:       One or a list of geometries to check for pairs
        threshold:  Threshold for including a bond in the bond list
        min_neighbors: Minimum number of neighbors to include for each atom.
                    If an atom has smaller than this number of bonds, additional
                    distances will be added to reach this number.
        snapshots:  Number of snapshots to be used in the generation, useful
                    for speeding up the process if the path is long and
                    atoms numerous.

    Returns:
        List of all the included interatomic distance pairs.
    """
    # Type casting and value checks on input parameters
    geom = np.asarray(geom)
    if len(geom.shape) < 3:
        # If there is only one geometry or it is flattened, promote to 3d
        geom = geom.reshape(1, -1, 3)
    min_neighbors = min(min_neighbors, geom.shape[1] - 1)

    # Determine which images to be used to determine distances
    snapshots = min(len(geom), snapshots)
    images = [0, len(geom) - 1]
    if snapshots > 2:
        images.extend(np.random.choice(range(1, snapshots - 1), snapshots - 2, replace=False))
    # Get neighbor list for included geometry and merge them
    rijset = set(enforce)
    for image in images:
        tree = KDTree(geom[image])
        pairs = tree.query_pairs(threshold)
        rijset.update(pairs)
        bonded = tree.query_pairs(bond_threshold)
        neighbors = {i: {i} for i in range(geom.shape[1])}
        for i, j in bonded:
            neighbors[i].add(j)
            neighbors[j].add(i)
        for i, j in bonded:
            for ni in neighbors[i]:
                for nj in neighbors[j]:
                    if ni != nj:
                        pair = tuple(sorted([ni, nj]))
                        if pair not in rijset:
                            rijset.add(pair)
    rijlist = sorted(rijset)
    # Check neighbor count to make sure `min_neighbors` is satisfied
    count = np.zeros(geom.shape[1], dtype=int)
    for i, j in rijlist:
        count[i] += 1
        count[j] += 1
    for idx, ct in enumerate(count):
        if ct < min_neighbors:
            _, neighbors = tree.query(geom[-1, idx], k=min_neighbors + 1)
            for i in neighbors:
                if i == idx:
                    continue
                pair = tuple(sorted([i, idx]))
                if pair in rijset:
                    continue
                else:
                    rijset.add(pair)
                    rijlist.append(pair)
                    count[i] += 1
                    count[idx] += 1
    if atoms is None:
        re = np.full(len(rijlist), 2.0)
    else:
        radius = np.array([ATOMIC_RADIUS.get(atom.capitalize(), 1.5) for atom in atoms])
        re = np.array([radius[i] + radius[j] for i, j in rijlist])
    logger.debug("Pair list contain %d pairs", len(rijlist))
    return rijlist, re


def compute_rij(geom, rij_list):
    """Calculate a list of distances and their derivatives

    Takes a set of cartesian geometries then calculate selected distances and their
    cartesian gradients given a list of atom pairs.

    Args:
        geom: Cartesian geometry of all the points.  Must be 2d numpy array or list
            with shape (natoms, 3)
        rij_list: list of indexes of all the atom pairs

    Returns:
        rij (array): Array of all the distances.
        bmat (3d array): Cartesian gradients of all the distances."""
    nrij = len(rij_list)
    rij = np.zeros(nrij)
    bmat = np.zeros((nrij, len(geom), 3))
    for idx, (i, j) in enumerate(rij_list):
        dvec = geom[i] - geom[j]
        rij[idx] = r = np.sqrt(dvec[0] * dvec[0] +
                               dvec[1] * dvec[1] + dvec[2] * dvec[2])
        grad = dvec / r
        bmat[idx, i] = grad
        bmat[idx, j] = -grad
    return rij, bmat


def compute_wij(geom, rij_list, func):
    """Calculate a list of scaled distances and their derivatives

    Takes a set of cartesian geometries then calculate selected distances and their
    cartesian gradients given a list of atom pairs.  The distances are scaled with
    a given scaling function.

    Args:
        geom: Cartesian geometry of all the points.  Must be 2d numpy array or list
            with shape (natoms, 3)
        rij_list: 2d numpy array of indexes of all the atom pairs
        func: A scaling function, which returns both the value and derivative.  Must
            qualify as a numpy Ufunc in order to be broadcasted to array elements.

    Returns:
        wij (array): Array of all the scaled distances.
        bmat (2d array): Cartesian gradients of all the scaled distances, with the
            second dimension flattened (need this to be used in scipy.optimize)."""
    geom = np.asarray(geom).reshape(-1, 3)
    nrij = len(rij_list)
    rij, bmat = compute_rij(geom, rij_list)
    wij, dwdr = func(rij)
    for idx, grad in enumerate(dwdr):
        bmat[idx] *= grad
    return wij, bmat.reshape(nrij, -1)


def morse_scaler(re=1.5, alpha=1.7, beta=0.01):
    """Returns a scaling function that determines the metric of the internal
    coordinates using morse potential

    Takes an internuclear distance, returns the scaled distance, and the
    derivative of the scaled distance with respect to the unscaled one.
    """
    def scaler(x):
        ratio = x / re
        val1 = np.exp(alpha * (1 - ratio))
        val2 = beta / ratio
        dval = -alpha / re * val1 - val2 / x
        return val1 + val2, dval
    return scaler


def elu_scaler(re=2, alpha=2, beta=0.01):
    """Returns a scaling function that determines the metric of the internal
    coordinates using morse potential

    Takes an internuclear distance, returns the scaled distance, and the
    derivative of the scaled distance with respect to the unscaled one.
    """
    def scaler(x):
        val1 = (1 - x / re) * alpha + 1
        dval = np.full(x.shape, -alpha / re)
        large = x > re
        v1l = np.exp(alpha * (1 - x[large] / re))
        val1[large] = v1l
        dval[large] = -alpha / re * v1l
        val2 = beta * re / x
        return val1 + val2, dval - val2 / x
    return scaler
