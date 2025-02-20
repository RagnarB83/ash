"""Performing geodesic interpolation or smoothing.
Optimize reaction path using geometric information by minimizing path length with metrics defined by
redundant internal coordinates.   Avoids the discontinuity and convergence problems of conventional
interpolation methods by incorporating internal coordinate structure while operating in Cartesian,
avoiding unfeasibility.

Xiaolei Zhu et al, Martinez Group, Stanford University
"""
import logging
import argparse

import numpy as np

from .fileio import read_xyz, write_xyz
from .interpolation import redistribute
from .geodesic import Geodesic


logger = logging.getLogger(__name__)


def main():
    """Main entry point of the geodesic interpolation package.
    Parse command line arguments then activate the interpolators and smoothers."""
    ps = argparse.ArgumentParser(description="Interpolates between two geometries",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ps.add_argument("filename", type=str, help="XYZ file containing geometries. If the number of images "
                    "is smaller than the desired number, interpolation points will be added.  If the "
                    "number is greater, subsampling will be performed.")
    ps.add_argument("--nimages", type=int, default=17, help="Number of images.")
    ps.add_argument("--sweep", action="store_true", help="Sweep across the path optimizing one image at "
                    "a time, instead of moving all images at the same time.  Default is to perform sweeping "
                    "updates if there are more than 30 atoms.")
    ps.add_argument("--no-sweep", dest='sweep', action="store_false", help="Do not perform sweeping.")
    ps.set_defaults(sweep=None)
    ps.add_argument("--output", default="interpolated.xyz", type=str, help="Output filename. "
                    "Default is interp.xyz")
    ps.add_argument("--tol", default=2e-3, type=float, help="Convergence tolerance")
    ps.add_argument("--maxiter", default=15, type=int, help="Maximum number of minimization iterations")
    ps.add_argument("--microiter", default=20, type=int, help="Maximum number of micro iterations for "
                    "sweeping algorithm.")
    ps.add_argument("--scaling", default=1.7, type=float, help="Exponential parameter for morse potential")
    ps.add_argument("--friction", default=1e-2, type=float, help="Size of friction term used to prevent "
                    "very large change of geometry.")
    ps.add_argument("--dist-cutoff", dest='dist_cutoff', default=3, type=float, help="Cut-off value for the "
                    "distance between a pair of atoms to be included in the coordinate system.")
    ps.add_argument("--logging", default="INFO", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                    help="Logging level to adopt [ DEBUG | INFO | WARNING | ERROR ]")
    ps.add_argument("--save-raw", dest='save_raw', default=None, type=str, help="When specified, save the "
                    "raw path after bisections be before smoothing.")
    args = ps.parse_args()
    print("args:", args)
    exit()

    # Setup logging based on designated logging level
    logging.basicConfig(format="[%(module)-12s]%(message)s", level=args.logging)

    # Read the initial geometries.
    symbols, X = read_xyz(args.filename)
    logger.info('Loaded %d geometries from %s', len(X), args.filename)
    if len(X) < 2:
        raise ValueError("Need at least two initial geometries.")

    # First redistribute number of images.  Perform interpolation if too few and subsampling if too many
    # images are given
    raw = redistribute(symbols, X, args.nimages, tol=args.tol * 5)
    if args.save_raw is not None:
        write_xyz(args.save_raw, symbols, raw)

    # Perform smoothing by minimizing distance in Cartesian coordinates with redundant internal metric
    # to find the appropriate geodesic curve on the hyperspace.
    smoother = Geodesic(symbols, raw, args.scaling, threshold=args.dist_cutoff, friction=args.friction)
    if args.sweep is None:
        args.sweep = len(symbols) > 35
    try:
        if args.sweep:
            smoother.sweep(tol=args.tol, max_iter=args.maxiter, micro_iter=args.microiter)
        else:
            smoother.smooth(tol=args.tol, max_iter=args.maxiter)
    finally:
        # Save the smoothed path to output file.  try block is to ensure output is saved if one ^C the
        # process, or there is an error
        logging.info('Saving final path to file %s', args.output)
        write_xyz(args.output, symbols, smoother.path)


if __name__ == "__main__":
    main()
