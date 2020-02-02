#-----------------------------------------------------------------------------
# Imports
#-----------------------------------------------------------------------------
# stdlib
import math
import time

# openmm
from simtk import unit
from simtk.openmm.app import StateDataReporter

#-----------------------------------------------------------------------------
# Classes
#-----------------------------------------------------------------------------

class ProgressReporter(StateDataReporter):
    def __init__(self, file, reportInterval, totalSteps):
        super(ProgressReporter, self).__init__(file, reportInterval, step=False, time=True,
            potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
            temperature=True)

        self._totalSteps = totalSteps

    def _initializeConstants(self, simulation, state):
        if simulation.topology.getUnitCellDimensions() is not None:
            self._volume = True
            self._density = True

        # this needs to come after _density and _volume are set so
        # that the mass gets computed, if needed.
        super(ProgressReporter, self)._initializeConstants(simulation)

        # initialize these as late as possible, so that as little initialization
        # code gets counted in the elapsed walltime. When that happens, it
        # makes it look like the simulation is getting faster and faster,
        # since that time is being amoritized out.
        self._initialWallTime = time.time()
        self._initialStep = simulation.currentStep
        self._initialSimTime = state.getTime()

    def _constructReportValues(self, simulation, state):
        progressPercent = 100 * float(simulation.currentStep - self._initialStep) / self._totalSteps
        if progressPercent > 0:
            timeLeft = (time.time() - self._initialWallTime) * (100.0 - progressPercent) / progressPercent

            elapsedSim = (state.getTime() - self._initialSimTime).value_in_unit(unit.nanoseconds)
            walltime = ((time.time() - self._initialWallTime)*unit.seconds).value_in_unit(unit.days)
            rate = elapsedSim / walltime
        else:
            timeLeft = float('nan')
            rate = 0

        values = [progressPercent, self.pretty_time(timeLeft), rate] + \
                 super(ProgressReporter, self)._constructReportValues(simulation, state)
        return values

    def _constructHeaders(self):
        headers = [('Progress', '(%)'),
                   ('WallTime Left', '(d:h:m:s)'),
                   ('Speed', '(ns/day)'),
                   ('Time',  '(ps)'),
                   ('P.E.', '(kJ/mol)'),
                   ('K.E.', '(kJ/mol)'),
                   ('Total E.', '(kJ/mol)'),
                   ('Temp', '(K)'),
                  ]

        widths =  [8,          15,      10,        13,       15,
                   15,         15,      13]
        formats = ['%7.3f%%', '%15s', '%10.2f', '%13.5f', '%15.5f',
                   '%15.5f', '%15.5f', '%13.5f']

        if self._volume:
            headers.append(('Vol', '(nm^3)'))
            formats.append('%10.4f')
            widths.append(10)
        if self._density:
            headers.append(('Rho', '(g/mL)'))
            formats.append('%10.4f')
            widths.append(10)

        self._formats = formats

        row1, row2 = zip(*headers)
        headerwidths = ['%{w}s'.format(w=w) for w in widths]
        print >>self._out, ' '.join(f % e for f, e in zip(headerwidths, row1))
        print >>self._out, ' '.join(f % e for f, e in zip(headerwidths, row2))


    def report(self, simulation, state):
        if not self._hasInitialized:
            self._initializeConstants(simulation, state)
            self._constructHeaders()
            self._hasInitialized = True

        # Check for errors.
        self._checkForErrors(simulation, state)

        # Query for the values
        values = self._constructReportValues(simulation, state)

        print >>self._out, ' '.join(f % v for f, v in zip(self._formats, values))

    def pretty_time(self, secs):
        """Format the time in a pretty way"""

        if math.isnan(secs):
            return "??"
        secs = int(secs)

        days = secs // 86400
        secs -= 86400*days

        hrs = secs // 3600
        secs -= 3600*hrs

        mins = secs // 60
        secs -= 60*mins

        if days > 0:
            s = "%d:%d:%02d:%02d" % (days, hrs, mins, secs)
        elif hrs > 0:
            s = "%d:%02d:%02d" % (hrs, mins, secs)
        elif mins > 0:
            s = "%d:%02d" % (mins, secs)
        else:
            s = "0:%02d" % secs

        return s
