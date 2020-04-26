from KNARRcalculator.mb import MullerBrownGaussWorker, MullerBrownWorker
from KNARRcalculator.peaks import PeaksWorker
from KNARRcalculator.lepsho import LEPSHOWorker, LEPSHOGaussWorker
from KNARRcalculator.debug import DebugWorker
from KNARRcalculator.henkelman import HenkelmanWorker
from KNARRmisc.plots import PlotSurface


def DebugSurf(fname, list_of_points = None, list_of_arrays = None):
    F = DebugWorker
    xbound = [-2, 2]
    ybound = [-2, 2]
    cbound = [-2, 2]
    PlotSurface(fname, workerfunc=F, xbound=xbound, ybound=ybound,
            cbound=cbound, npts=200, filled=False,
            list_of_arrays=list_of_arrays, list_of_points=list_of_points)
    return None

def MullerBrownSurf(fname, list_of_points=None, list_of_arrays=None):
    F = MullerBrownWorker
    #fname = "mb"
    xbound = [-1.5, 1]
    ybound = [-0.42, 2.2]
    cbound = [-0.15, 0.2]
    PlotSurface(fname, workerfunc=F, xbound=xbound, ybound=ybound,
                cbound=cbound, npts=200, filled = False,
                list_of_arrays=list_of_arrays, list_of_points=list_of_points)
    return None

def HenkelmanSurf(fname, list_of_points=None, list_of_arrays=None):
    F = Henkelman
    #fname = "mb"
    xbound = [0.0, 2.0]
    ybound = [-.5, .5]
    cbound = [-2.0, 7]
    PlotSurface(fname, workerfunc=F, xbound=xbound, ybound=ybound,
                cbound=cbound, npts=200, filled = False,
                list_of_arrays=list_of_arrays, list_of_points=list_of_points)
    return None

def MullerBrownSurfGauss(fname, list_of_points=None, list_of_arrays=None):
    F = MullerBrownGaussWorker
    #fname = "mbg"
    xbound = [-1.5, 1.2]
    ybound = [-0.5, 2.0]
    cbound = [-0.15, 0.2]
    PlotSurface(fname, workerfunc=F, xbound=xbound, ybound=ybound,
                cbound=cbound, npts=100, filled = False,
                list_of_arrays=list_of_arrays, list_of_points=list_of_points)

    return None

def PeaksSurf(fname, list_of_points=None, list_of_arrays=None):
    F = PeaksWorker
    #fname = "peaks"
    xbound = [-2, 2]
    ybound = [-2, 2]
    cbound = [-4, 5]
    PlotSurface(fname, workerfunc=F, xbound=xbound, ybound=ybound,
                cbound=cbound, filled = False,
                npts=100, ncont=20,
                list_of_arrays=list_of_arrays, list_of_points=list_of_points)
    return None

    # ================================
    # LEPS + HO
    # ================================

def LEPSHOSurf(fname,list_of_points=None, list_of_arrays=None):
    F = LEPSHOWorker
    #fname = "lepsho"
    xbound = [0.5, 3.3]
    ybound = [-3.5, 3.5]
    cbound = [-4.5, 6.0]
    PlotSurface(fname, workerfunc=F, xbound=xbound, ybound=ybound,
                cbound=cbound, filled = False,
                npts=100, ncont=200,
                list_of_arrays=list_of_arrays, list_of_points=list_of_points)
    return None

def LEPSHOGaussSurf(fname,list_of_points=None, list_of_arrays=None):
    F = LEPSHOGaussWorker
    #fname = "lepshogauss"
    xbound = [0.5, 3.3]
    ybound = [-3.5, 3.5]
    cbound = [-3.0, 6.0]
    PlotSurface(fname, workerfunc=F, xbound=xbound, ybound=ybound,
                cbound=cbound, filled = False,
                npts=100, ncont=100,
                list_of_arrays=list_of_arrays, list_of_points=list_of_points)
    return None
