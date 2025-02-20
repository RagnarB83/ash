"""File IO utilities"""
import numpy as np


def read_xyz(filename):
    """Read XYZ file and return atom names and coordinates

    Args:
        filename:  Name of xyz data file

    Returns:
        atom_names: Element symbols of all the atoms
        coords: Cartesian coordinates for every frame.
    """
    coords = []
    with open(filename, 'r') as f:
        for line in f:
            try:
                natm = int(line)	# Read number of atoms
                next(f)		# Skip over comments
                atom_names = []
                geom = np.zeros((natm, 3), float)
                for i in range(natm):
                    line = next(f).split()
                    atom_names.append(line[0])
                    geom[i] = line[1:4]     # Numpy auto-converts str to float
            except (TypeError, IOError, IndexError, StopIteration):
                raise ValueError('Incorrect XYZ file format')
            coords.append(geom)
    if not coords:
        raise ValueError("File is empty")
    return atom_names, coords


def write_xyz(filename, atoms, coords):
    """Write atom names and coordinate data to XYZ file

    Args:
        filename:   Name of xyz data file
        atoms:      Iterable of atom names
        coords:     Coordinates, must be of shape nimages*natoms*3
    """
    natoms = len(atoms)
    with open(filename, 'w') as f:
        for i, X in enumerate(np.atleast_3d(coords)):
            f.write("%d\n" % natoms)
            f.write("Frame %d\n" % i)
            for a, Xa in zip(atoms, X):
                f.write(" {:3} {:21.12f} {:21.12f} {:21.12f}\n".format(a, *Xa))
