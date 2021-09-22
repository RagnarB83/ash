import numpy as np
import math
from operator import itemgetter

from ase import Atoms
from ase.calculators.lj import LennardJones as LJ
from ase.io import write
from ase.geometry import find_mic


class SAFIRES:

    """
    ###################### --- SAFIRES ---- ########################
    # Scattering-Assisted Flexible Inner Region Ensemble Separator #
    #                                                              #
    # Authors: Björn Kirchhoff, Elvar Ö. Jónsson,                  #
    #          Asmus O. Dohn, Timo Jacob, Hannes Jónsson*          #
    #                                                              #
    # If you use the code, please cite: 10.1021/acs.jctc.1c00522   #
    #                                                              #
    ################################################################

    VERSION INFO
    ------------
    0.1.0: 
        2021-09-06 
        Initial release

    DESCRIPTION
    -----------
    SAFIRES is a partitioning scheme designed to divide a simulation
    supercell into two regions to be calculated separately, using
    different (or the same) computational methodologies.

    SAFIRES assumes that three necessary components are present
    in a simulation: 1) the solute, which can be a particle / molecule
    or a periodic surface; 2) an inner region close to the solute;
    3) an outer region separated from the solute by the inner region.
    SAFIRES assumes the same fundamental premise as FIRES does
    (Rowley and Roux, JCTC, 2012) by requiring that all particles in
    the inner and outer region are the same (the solute can be
    different). If particles are indistinguishable, it can be shown
    that correct average ensemble properties can be obtained despite
    the presence of the bounadary.
    The position of the boundary is flexible with respect to the
    inner region particle that is furthest away from the solute.
    If a boundary event is detected, SAFIRES will redirect the
    involved inner and outer region particles by performing an
    elastic collision between them, using the boundary as a mediator
    for the collision. SAFIRES can scale the time step of the
    simulation to match the exact moment that a collision occurs.
    SAFIRES uses a new propagator to handle these dynamic time steps.
    This propagator reduces to the Vanden-Eijnden / Cicotti
    implementation of the Langevin propagator for a constant time
    step and to the Velocity Verlet propagator for a constant time
    step and zero friction.

    "particle": any atom or molecule involved in SAFIRES.
    "solute"/"center": referring the atoms in the solute
                       or periodic surface (atom.tag = 0).
    "inner"/"INNER": referring to particles in the inner
                     region (atom.tag = 1).
    "outer"/"OUTER": referring to particles in the outer
                     region (atom.tag = 2).
    "boundary": the boundary sphere (atomic or molecular
                solute) or boundary plane (surface as
                solute) that separates inner and outer
                region. sphere radius or plane distance,
                respectively, are defined by the distance
                of the INNER particle furthest away from
                the solute / periodic surface.
    "boundary event": if any OUTER particles are detected
                      closer to the solute than the INNER
                      particle currently furthest away from
                      the solute / periodic surface.

    USAGE:
    ------
   1) atoms = Atoms([...])
       atom.tag = 0 --> solute molecule or periodic surface
       atom.tag = 1 --> inner region
       atom.tag = 2 --> outer region

    2) md = MD([...])
        where MD = Langevin or VelocityVerlet

    3) safires = SAFIRES(atoms=atoms,
                         mdobject=md,
                         natoms=int number of atoms in each
                                molecule e.g. 3 for water,
                         logfile=str custom SAFIRE logfile name,
                         debug=True/False,
                         barometer=True/False,
                         surface=True/False,
                         reflective=True/False)

    4) md.attach(safires.safires, interval=1)

    5) md.run([...])


    TODO:
    -----
    - currently fix_com is required to be False for Langevin.
      that shouldn't be the case.
    """
    
    def __init__(self, atoms, mdobject, natoms,
                 logfile="safires.log", debug=False,
                 barometer=False, surface=False, reflective=False):
        """"Initial setup of the SAFIRES class.

        KEYWORD ARGUMENTS:
        ------------------
        atoms -- ASE atoms object
        mdobject -- ASE molecular dynamics object
        natoms -- number of atoms in each molecule in the simulation
                  (excluding the solute, which can be different)
        logfile -- custom str name for logfile
        debug -- enable/disable debug mode (default: False)
        barometer -- enable/disable barometer (default: False)
        surface -- treat solute as periodic surface (default: False)

        NAMESPACE DOCUMENTATION
        -----------------------
        self.atoms --
            atoms object passed to SAFIRES in the input script.

        self.surface --
            if True, then the solute (atom.tag = 0) is considered
            a surface. this changes / simplifies a lot of the math
            since only the z direction needs to be accounted for.
            important limitation: surface must be perpendicular to
            z axis, no oblique or stepped surfaces possible.

        self.constraints --
            some SAFIRES procedures require that all constraints
            (notably: RATTLE) are momentarily removed from the
            atoms object. they are copied here and reapplied
            at the end of the procedure.

        self.natoms --
            int number of atoms in each molecule that make up the
            inner and outer region. SAFIRES requires that all
            inner and outer region particles are indistinguishable
            as the main premise (see publication).
        
        self.reflective --
            switches to an implementation of SAFIRES that performs
            reflections at the boundary (no momentum exchange!)
            instead of performing an elastic collision between the
            conflicting inner/outer region particle pair.

        self.previous_atoms --
            copy of the atoms object from previous iterations
            used to get back to a conflict free state when a
            boundary event occurs. initialized simply as a copy
            of the initial atoms object. this means that if a
            boundary event occurs during step 0, there is no viable
            configuration to go back to to resolve the conflict
            and SAFIRES will crash.

        self.previous_atoms.calc --
            the previous_atoms object requires a placeholder
            calculator to store some of the required properties
            we need to pass to it.

        self.mdobject --
            molecular dynamics object passed to SAFIRES in the
            input script.

        self.default_dt --
            default time step defined in the input script.

        self.remaining_dt --
            counting variable used to keep track of the time
            required to complete a full default_dt when SAFIRES
            performes fractional time steps during a boundary event.

        self.previous_boundary_idx --
            keeps track of which particle (by index) defined the
            flexible boundary in the previous iteration.
            necessary because SAFIRES resets to the previous
            iteration when a boundary event occurs. initializes
            to zero, updates with a useful value after first
            successful MD iteration without boundary event.

        self.tocollide --
            keeps track of which pair of INNER/OUTER particles
            are tagged for separation through collision during
            a boundary event.

        self.recent --
            keeps track of which pair of INNER/OUTER particles
            underwent separation by collision in the current
            iteration. this is necessary so that SAFIRES doesn't
            treat the same particles over and over if multiple
            boundary events occur during the same time step.

        self.logfile --
            name of  the logfile for SAFIRES specific results..
            SAFIRES logs additional specific information such
            as the location of the boundary sphere relativ to
            the solute at each iteration, and the number of
            collisions. Set to None to suppress output.
            Default: "safires.log".

        self.ncollisions --
            tracks total number of boundary events and subsequent
            collisions that have taken place over the simulation
            time.

        self.ndoubles --
            specifically tracks the number of times that more
            than one collision has occurred during the same
            time step of length self.default_dt.

        self.barometer --
            if True, this will count how many boundary events are
            caused by inner or outer particles, respectively.
            this can reveal if there's an artificial pressure in
            the system (these numbers shouldn't deviate
            dramatically).

        self.impacts --
            counter for self.barometer.

        self.debug --
            if True, verbose output is written to 'debug.log'
            during each iteration. False by default because it
            creates overhead that's usually unnessecary.
        """
        self.atoms = atoms
        self.surface = surface
        self.constraints = self.atoms.constraints.copy()
        self.natoms = natoms
        self.reflective = reflective
        self.previous_atoms = atoms.copy()
        self.previous_atoms.calc = LJ()
        self.mdobject = mdobject
        self.default_dt = self.mdobject.dt
        self.remaining_dt = 0.
        self.previous_boundary_idx = 0
        self.tocollide = list()
        self.recent = list()
        self.logfile = logfile
        self.ncollisions = 0
        self.ndoubles = 0
        self.barometer = barometer
        if self.barometer:
            self.impacts = [0, 0]
        self.debug = debug

        # keep track of how many atoms are in the solute
        # or periodic surface model
        self.nsol = len([atom.index for atom in self.atoms 
                       if atom.tag == 0])

        # if Langevin MD is using 'fix_com', we need to turn that off,
        # it's currently incompatible with SAFIRES.
        # (TODO: remove incompatibility)
        if hasattr(self.mdobject, "fix_com"):
            if self.mdobject.fix_com:
                self.mdobject.fix_com = False
        
        if hasattr(self.mdobject, "fixcm"):
            # old / deprecated version of fix_com; might remove soon
            if self.mdobject.fixcm:
                self.mdobject.fixcm = False
        
        # setup output logfiles
        if self.logfile is not None:
            self.log = open(self.logfile, "w+")
            self.log.write("iteration   boundary_idx   d_INNER / A\n")

        # create debuglog file (if required)
        if debug:
            self.db = open("debug-safires.log", "w+")
            self.db.write("SAFIRES DEBUG LOG\n"
                          "-----------------\n")
            if self.natoms > 1:
                self.db.write("\nIMPORTANT:\nYou are using molecules "
                              "(natoms > 1).\n"
                              "Values you see here will be referring\n"
                              "to the reduced pseudoparticles we use\n"
                              "in lieu of the actual molecules\n"
                              "within SAFIRES. Keep that in mind!\n")

        # say hi to the user
        print("\n"
              " ##############\n"
              " # SAFIRES    #\n"
              " # ---------- #\n"
              " # v. 0.1.0   #\n"
              " # 2021-09-06 #\n"
              " ##############\n")

    def logger(self, iteration, boundary_idx, boundary):
        """SAFIRES-specific results to log file."""

        if self.logfile is not None:
            self.log.write("{:<9d}   {:<12d}   {:<4.7f}\n"
                           .format(iteration, boundary_idx, boundary))
        return

    def debuglog(self, string):
        """Write debug output to debug.log (if activated).

        Keyword arguments:
        string -- any string

        The debug output we want to write can be fairly varied,
        so string doesn't follow any specific pattern or constriant.
        """

        if self.debug:
            self.db.write(string)
        return

    def debugtraj(self):
        """Write the last two configurations to traj object."""
        write("crashed_atoms.traj", [self.atoms, self.previous_atoms],
              format="traj")
        return
    
    def normalize(self, x):
        """Return normalized 3D vector x."""
        return x / np.linalg.norm(x)

    def calc_angle(self, n1, n2):
        """Return angle between 3D vectors n1 and n2.

        Reference: Vincenty, T. Survey Review 23, 88–93 (1975).
        """
        return(np.arctan2(np.linalg.norm(np.cross(n1, n2)), np.dot(n1, n2)))

    def rotation_matrix(self, axis, theta):
        """ Return rotation matrix for rotation by theta around axis.

        Return the rotation matrix associated with counterclockwise
        rotation about the given axis by theta radians.
        Euler-Rodrigues formula, code stolen from
        stackoverflow.com/questions/6802577.
        """
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                         [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                         [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    def extrapolate_dt(self, previous_boundary_idx, boundary_idx,
                       outer_idx, checkup):
        """Return the time step required to resolve boundary event.

        Keyword arguments:
        previous_boundary_idx -- atom index of the inner region
                                 particle that defined the boundary
                                 during the previous iteration
        boundary_idx -- atom index of the inner region particle that
                        defines the boundary on the current iteration
        outer_idx -- atom index of the outer region particle that
                     triggered the boundary event
        checkup -- True/False, is used internally to indicate if
                   this is a checkup run which occurs after a
                   successful boundary event resolution. this is done
                   to catch rare cases where a secon outer region
                   particle has entered the inner region during the
                   resolution of a first boundary event. in this case,
                   slightly different rules apply.
        """

        # results dict
        res = {}  # structure: {inner particle index: extrapolated factor}

        # convert list to set (no duplicate values)
        for inner_idx in set([boundary_idx, previous_boundary_idx]):
            # find point where outer and inner region particles
            # have the same distance from COM, i.e. both are on
            # the boundary. we need to propagate both particles
            # to this point to perform an exact elastic collision.

            if inner_idx in self.recent and outer_idx in self.recent:
                # First, check if inner and outer performed a collision
                # in the very last iteration. This can occur if
                # a collision is performed and in the subsequent
                # iteration (within remaining_dt) the same outer
                # collides with another inner. If we do the
                # extrapolation based on both these inner, the inner
                # from the previous iteration will always give
                # the smaller dt because this inner and the outer are
                # both on the boundary at the state of checking.
                # We need to ignore this pair since it's already
                # been resolved.
                continue

            # retreive forces, velocities, distances, masses
            # for image before the event
            com_atoms, forces, r, d, unused1, unused2 = (
                    self.update(self.previous_atoms))
            r_outer = r[outer_idx]
            r_inner = r[inner_idx]
            m_outer = com_atoms[outer_idx].mass
            v_outer = com_atoms[outer_idx].momentum / m_outer
            f_outer = forces[outer_idx] / m_outer
            m_inner = com_atoms[inner_idx].mass
            v_inner = com_atoms[inner_idx].momentum / m_inner
            f_inner = forces[inner_idx] / m_inner
            
            # if molecules are used, which are reduced to
            # pseudoparticles with properties centered on their COM,
            # we need to re-expand the pseudoparticle indices into the
            # "real" indices by multiplying the pseudoparticle index by
            # the number of atoms in each molecule.
            # furthermore, shift in the indexing due to the solute or
            # periodic surface model (which can have arbitrary number
            # of atoms) needs to be accounted for.
            outer_real = self.nsol + (outer_idx - 1) * self.natoms
            inner_real = self.nsol + (inner_idx - 1) * self.natoms

            # retreive Langevin-specific values (eta and xi random
            # components, friction fr).
            # if it is a Velocity Verlet based simulation,
            # all of these parameters remain zero
            fr = 0.
            xi_outer = np.asarray([0., 0., 0.])
            xi_inner = np.asarray([0., 0., 0.])
            eta_outer = np.asarray([0., 0., 0.])
            eta_inner = np.asarray([0., 0., 0.])
            sig_outer = 0.
            sig_inner = 0.

            if hasattr(self.mdobject, "fr"):
                # if it is a Langevin-based simulation
                fr = self.mdobject.fr
                
                mod = self.natoms
                if mod > 1:
                    # we need to remove the constraints again
                    # since get_masses will fail when RATTLE
                    # for example is present in the atoms object
                    self.constraints = self.atoms.constraints.copy()
                    self.atoms.constraints = []
                    
                    # if inner/outer particles are molecules
                    m_outer_list = [math.sqrt(xm) for xm in
                                    self.atoms[outer_real:outer_real + mod]
                                    .get_masses()]
                    m_inner_list = [math.sqrt(xm) for xm in
                                    self.atoms[inner_real:inner_real + mod]
                                    .get_masses()]
                    xi_outer = (np.dot(m_outer_list,
                                self.mdobject.xi[outer_real:outer_real + mod])
                                / m_outer)
                    xi_inner = (np.dot(m_inner_list,
                                self.mdobject.xi[inner_real:inner_real + mod])
                                / m_inner)
                    eta_outer = (np.dot(m_outer_list,
                                 self.mdobject.eta[outer_real:outer_real + mod])
                                 / m_outer)
                    eta_inner = (np.dot(m_inner_list,
                                 self.mdobject.eta[inner_real:inner_real + mod])
                                 / m_inner)
                    sig_outer = math.sqrt(2 * self.mdobject.temp * fr)
                    sig_inner = math.sqrt(2 * self.mdobject.temp * fr)
                
                    # re-apply the constraints
                    self.atoms.constraints = self.constraints.copy()
                
                else:
                    # if inner/outer particles are monoatomic
                    xi_outer = self.mdobject.xi[outer_real]
                    xi_inner = self.mdobject.xi[inner_real]
                    eta_outer = self.mdobject.eta[outer_real]
                    eta_inner = self.mdobject.eta[inner_real]
                    sig_outer = math.sqrt(2 * self.mdobject.temp * fr / m_outer)
                    sig_inner = math.sqrt(2 * self.mdobject.temp * fr / m_inner)

            # surface calculations: we only need z components
            if self.surface:
                v_outer[0] = 0.
                v_outer[1] = 0.
                v_inner[0] = 0.
                v_inner[1] = 0.
                f_outer[0] = 0.
                f_outer[1] = 0.
                f_inner[0] = 0.
                f_inner[1] = 0.
                xi_outer[0] = 0.
                xi_outer[1] = 0.
                xi_inner[0] = 0.
                xi_inner[1] = 0.
                eta_outer[0] = 0.
                eta_outer[1] = 0.
                eta_inner[0] = 0.
                eta_inner[1] = 0.

            # the time step extrapolation is based on solving a
            # 2nd degree polynomial of the form:
            # y = c0*x^2 + c1*x + c2.
            # a and b are velocity modifiers derived from the first
            # velocity half step in the Langevin algorithm. see
            # publication for more details.
            if not checkup:
                idt = self.default_dt
                a_outer = (idt * (f_outer - fr * v_outer) / 2
                           + math.sqrt(idt) * sig_outer * xi_outer / 2
                           - idt * idt * fr * (f_outer - fr * v_outer) / 8
                           - idt**1.5 * fr * sig_outer * (xi_outer / 2
                           + eta_outer / math.sqrt(3)) / 4)
                b_outer = math.sqrt(idt) * sig_outer * eta_outer / (2 * math.sqrt(3))

                a_inner = (idt * (f_inner - fr * v_inner) / 2
                           + math.sqrt(idt) * sig_inner * xi_inner / 2
                           - idt * idt * fr * (f_inner - fr * v_inner) / 8
                           - idt**1.5 * fr * sig_inner * (xi_inner / 2
                           + eta_inner / math.sqrt(3)) / 4)
                b_inner = math.sqrt(idt) * sig_inner * eta_inner / (2 * math.sqrt(3))
            else:
                a_outer = 0
                a_inner = 0
                b_outer = 0
                b_inner = 0

            v_outer += a_outer
            v_outer += b_outer
            v_inner += a_inner
            v_inner += b_inner

            # set up polynomial coefficients
            c0 = np.dot(r_inner, r_inner) - np.dot(r_outer, r_outer)
            c1 = 2 * np.dot(r_inner, v_inner) - 2 * np.dot(r_outer, v_outer)
            c2 = np.dot(v_inner, v_inner) - np.dot(v_outer, v_outer)

            # find roots
            roots = np.roots([c2, c1, c0])
            self.debuglog("   < TIME STEP EXTRAPOLATION >\n"
                          "   all extrapolated roots: {:s}\n"
                          .format(np.array2string(roots)))

            for val in roots:
                if np.isreal(val) and val <= self.mdobject.dt and val > 0:
                    # the quadratic polynomial yields four roots.
                    # we're only interested in the SMALLEST positive real
                    # value, which is the required time step.
                    res.update({inner_idx: np.real(val)})

                    if self.debug:
                        # debug logging
                        r_outer_new = r_outer + val * v_outer
                        d_outer_new = np.linalg.norm(r_outer_new)
                        r_inner_new = r_inner + val * v_inner
                        d_inner_new = np.linalg.norm(r_inner_new)
                        self.debuglog("   d_inner extrapolated = {:.12f}\n"
                                      .format(d_inner_new))
                        self.debuglog("   d_outer extrapolated = {:.12f}\n"
                                      .format(d_outer_new))
                        self.debuglog("   Extapolated dt for atom pair {:d}"
                                      " (INNER) - {:d} (OUTER): {:.5f}\n"
                                      .format(inner_idx, outer_idx,
                                              np.real(val)))

        if not res:
            # if none of the obtained roots fit the criteria (real,
            # positive, <= initial time step), then we have a problem.
            # this is indicative of a major issue.
            error = ("ERROR:\n\n"
                     "Unable to extrapolate time step (all real roots\n"
                     "<= 0 or > default time step). This is indicative\n"
                     "of a fundamental issue, either with your model\n"
                     "system or with SAFIRES being unable to handle\n"
                     "your simulation parameters for some reason.\n"
                     "Possible issues might include:\n"
                     "- bad starting configuration (bond lengths)\n"
                     "- using MD other than Langevin or VV\n"
                     "- exploding temperatures / starting\n"
                     "  configuration not thermalized\n"
                     "- using SAFIRES beyond its limitations\n"
                     "- using more than one type of atom  /\n"
                     "  molecule in inner and outer region\n"
                     "- bad region assignment (check tags!)\n\n"
                     "Roots: {:s}\n".format(np.array2string(roots)))
            self.debuglog(error)
            self.debugtraj()
            raise SystemExit(error)
        else:
            return res

    def propagate(self, dt, checkup, halfstep):
        """Propagate the simulation.

        Keyword arguments:
        dt -- the time step used to propagate.
        checkup -- True/False, is used internally to indicate if
                   this is a checkup run which occurs after a
                   successful boundary event resolution. this is done
                   to catch rare cases where a secon outer region
                   particle has entered the inner region during the
                   resolution of a first boundary event. in this case,
                   slightly different rules apply.
        halfstep -- 1/2, indicates if we're propagating so that the
                    conflicting inner and outer particle are at the
                    same distance from the center using the
                    extrapolated time step from extrapolate_dt()
                    to perform a collision (halfstep = 1) or if the
                    collision is already performed and we're
                    propagating to make up the remaining time to
                    complete a full default time step (halfstep = 2).

        This propagation algorithm is based on the Vanden-Eijnden /
        Cicotti implementation of the Langevin algorithm. It has been
        modified to perform the propagation in two halfsteps. It is
        used during the SAFIRES procedure in-lieu of the superordinate
        MD algorithm specified in the user input to fulfill the
        specific requirements of this approach, as outlined below.

        The first halfstep uses a fraction of the original, default
        time step as obtained from extrapolate_dt(). However, the
        velocities are updated based on forces and Langevin random
        components as if the original, default time step would be
        performed. After the end of this first halfstep, the
        atoms have been propagated so that the conflicting pair of
        inner and outer region particles are located at the same
        distance from the solute. From this obtained arrangement
        we can perform an exact elastic collision between them.

        The second halfstep used a time step that is
        (initial time step - time step used in halfstep 1).
        During this halfstep, the velocities are NOT updated as
        they were already updated during halfstep 1 based on
        the forces and Langevin random components scaled by
        the regular, default time step.

        Note that the Langevin random components are not randomized
        between halfsteps or between the time step extrapolation in
        extrapolate_dt() and the propagation since we require the
        atoms to behave deterministically while SAFIRES operates.

        The checkup switch will make it so that velocities
        are not updated in the first halfstep. Checkup is True is
        called after a boundary event has been successfully resolved,
        which means that propagate() was already run and the
        velocities were already updated based on the full default
        time step. If a subsequent boundary event needs to be
        resolved right after the resolution of a first event
        (without the regular MD being called in between, i.e.
        multiple events during the same default time step),
        the propagation during this second resolution occurs
        without any velocity updates. Otherwise we would apply the
        forces and random components multiple times during the same
        time step and thus destroy energy conservation.

        Finally, after the first and any subsequent boundary events
        have been resolved completely and the simulation has been
        propagated by a complete default time step, the second
        velocity halfstep required by Langevin is performed. After
        this, SAFIRES returns control over the simulation back to
        the superordinate MD algorithm defined  n the user input.
        Technically, the second velocity halfstep is performed after
        the initial, checkup is False cycle but is then undone if
        subsequent boundary events are detected during the
        checkup = True cycle. This ensures that the second velocity
        halfstep is performed on the absolute final atomic
        configuration obtained by SAFIRES before returning control
        over the situation to superordinate MD propagator.
        """

        # retreive parameters
        # atoms have been reset to "previous" state at this point
        x = self.atoms.get_positions()
        m = self.atoms.get_masses()[:, np.newaxis]
        v = self.atoms.get_momenta() / m
        f = self.atoms.calc.results['forces'] / m
        sqrt_of_3 = math.sqrt(3)

        if hasattr(self.mdobject, "fr"):
            # check for langevin friction, random forces
            T = self.mdobject.temp
            fr = self.mdobject.fr
            xi = self.mdobject.xi
            eta = self.mdobject.eta
            sig = math.sqrt(2 * T * fr / m)
        else:
            T = 0
            fr = 0.
            xi = np.asarray([np.asarray([0., 0., 0.])
                             for atom in self.atoms])
            eta = np.asarray([np.asarray([0., 0., 0.])
                              for atom in self.atoms])
            sig = 0.

        # pre-calculate (random) force constant
        # based on default time step
        idt = self.default_dt
        if not checkup:
            c = (idt * (f - fr * v) / 2
                 + math.sqrt(idt) * sig * xi / 2
                 - idt * idt * fr * (f - fr * v) / 8
                 - idt**1.5 * fr * sig * (xi / 2 + eta / sqrt_of_3) / 4)
            d = math.sqrt(idt) * sig * eta / (2 * sqrt_of_3)
        else:
            # if checkup is True, this means we already performed an entire
            # propagation cycle and have already updated the velocities
            # based on values for the full default time step. thus we
            # need to make sure not to update velocities a second time
            # because that would destroy energy conservation.
            c = np.asarray([np.asarray([0., 0., 0.]) for atom in self.atoms])
            d = np.asarray([np.asarray([0., 0., 0.]) for atom in self.atoms])

        if halfstep == 1:
            # friction and (random) forces should only be
            # applied during the first halfstep.
            v += c + d
            self.atoms.set_positions(x + dt * v)
            v = (self.atoms.get_positions() - x) / dt
            self.atoms.set_momenta(v * m)

        if halfstep == 2:
            # at the end of the second part of the time step,
            # do the force update and the second
            # velocity halfstep
            self.atoms.set_positions(x + dt * v)
            v = (self.atoms.get_positions() - x - dt * d) / dt
            f = self.atoms.get_forces(md=True) / m
            c = (idt * (f - fr * v) / 2
                 + math.sqrt(idt) * sig * xi / 2
                 - idt * idt * fr * (f - fr * v) / 8
                 - idt**1.5 * fr * sig * (xi / 2 + eta / sqrt_of_3) / 4)
            v += c
            self.atoms.set_momenta(v * m)

        return

    def update(self, atoms):
        """Return reduced pseudoparticle atoms object.

        Keyword arguments:
        atoms -- ASE atoms object with attached calculator containing
                 atomic positions, cell information, and results from
                 an MD iteration

        Return values:
        com_atoms -- if used with molecules: com_atoms contains new
                     pseudoparticles centered on the COM of the
                     original molecule with effective
                     velocities for the whole molecule. if monoatomic
                     species are used, this will be identical to the
                     original atoms object.
        forces -- effective force on each pseudoparticle. forces need
                  to be stored separate from the atoms object. if
                  monoatomic species are used, this array is identical
                  to that returned from the calculator.
        r -- array of vectors between the COM of the solute and all
             inner and outer region (pseudo-) particles in the system
        d -- array of absolute distances between the COM of the solute
             and all inner and outer region (pseudo-) particles
             in the system
        boundary_idx -- atom object index of the inner region (pseudo-)
                        particle that is furthest away from the COM of
                        the solute, defining the flexible boundary
        boundary -- absolute distance between the inner region (pseudo-)
                    particle and the COM of the solute, defining the
                    radius of the boundary sphere for molecular solutes
                    or the distance of the boundary plane from the
                    solute for surfaces.

        Arrays r and d have the same ordering as atoms object.
        """
        print("Inside safires update")
        print("atoms:", atoms)
        # calculate distance of the resulting COM
        i = 0
        com_atoms = Atoms()
        com_atoms.pbc = atoms.pbc
        com_atoms.cell = atoms.cell
        mod = self.natoms
                
        # need to make sure constraints are off
        # get_com method breaks GPAW fast FixBondLength
        # constraints (-> for rigid H2O)
        self.constraints = atoms.constraints.copy()
        atoms.constraints = []
        
        # calculate cumulative properties for solute / surface
        # (solute is always the first entry)
        idx_sol = [atom.index for atom in atoms if atom.tag == 0]
        sol_com = atoms[idx_sol].get_center_of_mass()
        M = np.sum(atoms[idx_sol].get_masses())
        mom = np.sum(atoms[idx_sol].get_momenta(), axis=0)
        tag = 0
        print("herex1")
        print("atoms.calc.results:", atoms.calc.results)
        print("")
        frc = np.sum(atoms.calc.results['forces'][idx_sol], axis=0)
        sym = atoms[idx_sol[0]].symbol
        tmp = Atoms(sym)
        tmp.set_positions([sol_com])
        tmp.set_momenta([mom])
        tmp.set_masses([M])
        tmp.set_tags([tag])
        forces = [frc]
        com_atoms += tmp
        
        # calculate cumulative properties for all inner/outer
        # region particles. for monoatomic inner/outer particles,
        # a 1:1 copy is created.
        for atom in atoms:
            if atom.tag in [1, 2]:
                if (atom.index - self.nsol) % mod == 0:
                    i = atom.index
                    com = atoms[i:i + mod].get_center_of_mass()
                    M = np.sum(atoms[i:i + mod].get_masses())
                    mom = np.sum(atoms[i:i + mod].get_momenta(), axis=0)
                    tag = atoms[i].tag
                    frc = np.sum(atoms.calc.results['forces'][i:i + mod], axis=0)
                    sym = atoms[i].symbol
            
                    # create a new atom
                    tmp = Atoms(sym)
                    tmp.set_positions([com])
                    tmp.set_momenta([mom])
                    tmp.set_masses([M])
                    tmp.set_tags([tag])
                    forces.append(frc)
                    
                    # append to new atoms object
                    com_atoms += tmp

        if self.surface:
            # we only need z coordinates for surface calculations
            for atom in com_atoms:
                atom.position[0] = 0.
                atom.position[1] = 0.

        # we can no reapply the constraints to the original
        # atoms object. all further processing will be done
        # on the pseudoparticle com_atoms object, which does
        # not have constraints
        atoms.constraints = self.constraints.copy()
        
        # calculate absolute distances and distance vectors between
        # COM of solute and all inner and outer region particles
        # (respect PBCs in distance calculations)
        r, d = find_mic([atom.position for atom in com_atoms] - sol_com, 
                       com_atoms.cell, com_atoms.pbc)
        
        # list all particles in the inner region
        inner_mols = [(atom.index, d[atom.index])
                      for atom in com_atoms if atom.tag == 1]

        # boundary is defined by the inner region particle that has
        # the largest absolute distance from the COM of the solute
        print("inner_mols:", inner_mols)
        boundary_idx, boundary = sorted(inner_mols, key=itemgetter(1),
                                        reverse=True)[0]

        return com_atoms, forces, r, d, boundary_idx, boundary

    def safires(self, checkup=False):
        """Check if boundary event occurred, coordinate its resolution.

        Keyword arguments:
        checkup -- True/False, default: False.
                   safires() calls itself again after successful
                   resolution of a boundary event. in rare occasions,
                   more than one boundary event can occur during the
                   same default time step, thus we need to re-run this
                   algorithm. checkup = True will change the behavior
                   of extrapolate_dt() and propagate(), see docstrings
                   therein.

        This method is called after each successful iteration of the
        superordinate MD propagator. safires() checks if any boundary
        events have occurred as a result of the propagation. If it
        detects a boundary event, the following sequence is triggered:
        0) check for boundary events.
        1) extrapolate the time step required to propagate the atoms
           so that the inner region particle particle that is furthest
           away from the COM of the solute and the outer region
           particle that is closest to the COM of the solute are
           exactly the same distance from the solute
           --> extrapolate_dt().
        2) reset the atoms object to the last state before the event.
        3) propagate by this extrapolated time step using and internal
           modified propagator --> propagate().
        4) perform elastic collision between the conflicting pair of
           inner and outer region particles to redirect them.
        5) propagate the system by the remaining time step needed to
           complete a full default time step -->  propagate().
        6) re-run safires() to check if another boundary event
           occurred as result of the propagation in step 5).
        7a) another boundary event is detected: return to step 1) but
            reset to the configuration obtained after step 3) so as to
            not undo the previous boundary event resolution.
        7b) no additional boundary event detected: return control over
            simulation to superordinate MD simulation for the next
            regular propagation step.
        """

        # determine current iteration
        iteration = self.mdobject.get_number_of_steps()

        # start writing new debugging block if debugging is enabled
        if not checkup:
            self.debuglog("\nIteration {:d}\n".format(iteration))
        else:
            self.debuglog("".join(["\nIteration {:d} ".format(iteration),
                          "POST COLLISION CHECKUP\n"]))

        # update pseudoparticle atoms object and distance (vectors)
        # between all pseudoparticles and the COM of the solute
        com_atoms, forces, r, d, boundary_idx, boundary = (
            self.update(self.atoms))

        self.debuglog("   Boundary: idx {:d} at d = {:.3f}\n"
                      .format(boundary_idx * self.natoms,
                              d[boundary_idx]))

        ###############################################################
        #                                                             #
        # 0) CHECK FOR BOUNDARY EVENTS IN CURRENT TIME STEP            #
        #                                                             #
        ###############################################################

        # make new list to save extrapolated time steps for
        # all boundary events detected in this iteration
        dt_list = []

        for atom in [tmpatm for tmpatm in com_atoms if tmpatm.tag == 2]:
            # loop over all outer region particles

            dist = d[atom.index]
            if dist < boundary:
                # check if any outer region particle is closer to the
                # solute than the inner region particle that is
                # furthest away from the solute (i.e. boundary_idx)

                if dist < boundary - 1:
                    # if the outer region particle is detected more
                    # than 1 A inside the inner region, something has
                    # gone wrong and we can't recover from that.
                    error = "".join([
                            "\nFIRES CAUSED BAD EXCEPTION\n"
                            "OUTER atom index "
                            "{:d}".format(atom.index * self.natoms),
                            " detected way inside inner region."
                            " Something went wrong.\n"
                            "Potential sources"
                            " of error include:\n"
                            "- bad starting configuration\n"
                            "- bad tag assignment (INNER: tag = 0/1,"
                            " OUTER: tag = 2)\n"
                            "- time step too large\n"
                            "- temperature too large\n"
                            "EXITING\n"])
                    self.debuglog(error)
                    self.debugtraj()
                    raise SystemExit(error)
                else:
                    # write information about the affected particle
                    # pair into debug.log
                    self.debuglog("   < BOUNDARY EVENT DETECTED (OUTER {:d}) >\n"
                                  .format(atom.index))
                    self.debuglog("   Before: Distance solute -> INNER "
                                  "(idx {:d}): {:.15f}\n"
                                  .format(boundary_idx, d[boundary_idx]))
                    self.debuglog("   Before: Distance solute -> OUTER "
                                  "(idx {:d}): {:.15f}\n"
                                  .format(atom.index, d[atom.index]))

                    ####################################################
                    #                                                  #
                    # 1) and 7a) EXTRAPOLATE NEW TIME STEP             #
                    #                                                  #
                    ####################################################

                    # extrapolate new time step
                    # by how  much do we need to propagate the system
                    # to bring the affected particle pair to the
                    # exact same distance from the solute?
                    res = self.extrapolate_dt(
                                self.previous_boundary_idx,
                                boundary_idx, atom.index,
                                checkup)

                    # get smallest entry from res dict;
                    # smallest res = earliest collision
                    inner_reflect, modified_dt = sorted(
                            res.items(), key=itemgetter(1))[0]

                    if modified_dt < 1E-12:
                        # if we propagate by time steps this small,
                        # the calculation will run into numerical
                        # issues and will get stuck. there is no
                        # reason why time steps should become this
                        # small, so this would be indicative of
                        # a general issue with the simulation.
                        error = ("".join(["Unreasonably small "
                                          "time step (< 1E-12) taken "
                                          "in step {:d}\n"
                                          .format(iteration),
                                          "This will lead to numerical"
                                          " instability, so we exit here.\n"
                                          "Check you simulation"
                                          " parameters, this shouldn't"
                                          " happen!\n"
                                          "EXITING"]))
                        self.debuglog(error)
                        self.debugtraj()
                        raise SystemExit(error)

                    # add affected particle pair and
                    # extrapolated dt to dt_list
                    dt_list.append((atom.index, modified_dt, inner_reflect))

        if dt_list:
            # if a boundary event has been detected,
            # find next time step -> smallest dt value in dt_list
            dt_list = sorted(dt_list, key=itemgetter(1))
            new_dt = dt_list[0][1]
            self.debuglog("   Using new time step for next iteration: "
                          "dt = {:.5f}\n".format(float(new_dt)))

            # choose affected particle pair corresponding to smallest dt
            outer_reflect, inner_reflect = dt_list[0][0], dt_list[0][2]

            # keep track of what fraction of current dt
            # is left to advance later
            if self.remaining_dt == 0.:
                # if the current boundary event is the first event
                # within the current MD iteration
                self.remaining_dt = self.mdobject.dt - new_dt
                self.debuglog("   Accounting for remaining time step: "
                              "dt_rem = {:.5f}\n"
                              .format(self.remaining_dt))
            elif self.remaining_dt > 0:
                # if the current boundary event is not the first
                # event within the current MD iteration
                if self.remaining_dt - new_dt > 0:
                    self.remaining_dt -= new_dt
                    self.debuglog("   Accounting for remaining time step: "
                                  "dt_rem = {:.5f}\n"
                                  .format(self.remaining_dt))
                else:
                    # catching an exception that would indicate that
                    # something went wrong in the remaining time
                    # bookkeeping or maybe because many boundary
                    # events occurred within one time step. both
                    # situations will not immediately break the
                    # simulation but are concerning, so the user
                    # will be warned in case of a complete meltdown
                    # later on.
                    self.remaining_dt = 0
                    error = ("   WARNING: remaining_dt was about to be "
                             "adjusted to a value < 0. This should not"
                             " happen. remaining_dt was now reset"
                             " to zero.\n")
                    print("<SAFIRES>" + error)
                    self.debuglog(error)

            if self.barometer:
                # barometer keeps track of how many impacts on the
                # border are caused by inner and by outer region
                # particles, respectively. helps to find out if an
                # artificial pressure occurs when two different
                # potentials are used in the inner and outer regions.
                dot_inner_c = np.dot(r[dt_list[0][2]],
                                     self.atoms[dt_list[0][2]].momentum)
                dot_outer_c = np.dot(r[dt_list[0][0]],
                                     self.atoms[dt_list[0][0]].momentum)
                if dot_inner_c > 0 and dot_outer_c > 0:
                    self.impacts[0] += 1
                if dot_inner_c < 0 and dot_outer_c < 0:
                    self.impacts[1] += 1

            ############################################################
            #                                                          #
            # 2) RESET TO CONFIGURATION BEFORE BOUNDARY EVENT          #
            #                                                          #
            ############################################################

            # reset configuration to previous image
            # before boundary event
            self.atoms.positions = self.previous_atoms.positions.copy()
            self.atoms.set_momenta(
                self.previous_atoms.get_momenta(), apply_constraint=False)
            self.atoms.calc.results['forces'] = (
                self.previous_atoms.calc.results['forces'].copy())

            ############################################################
            #                                                          #
            # 3) PROPAGATE PARTICLE PAIR TO SAME DISTANCE FROM SOLUTE  #
            #                                                          #
            ############################################################

            # propagate first part of the time step
            self.propagate(new_dt, checkup, halfstep=1)

            # update list of distances to solute, boundary
            com_atoms, forces, r, d, boundary_idx, boundary = (
                self.update(self.atoms))

            # logging collision info
            self.ncollisions += 1
            if checkup:
                self.ndoubles += 1
            self.debuglog("   < ELASTIC COLLISION >\n")

            # write boundary event info to stdout
            print("".join(["<SAFIRES> Iteration {:d}: "
                           .format(iteration),
                           "Treating atoms {:d} and {:d} at d = {:.5f}"
                           .format(outer_reflect, inner_reflect,
                                   d[outer_reflect]),
                           ". Using dt = {:.12f}"
                           .format(float(new_dt))]))
            self.debuglog("".join(["   Treating atoms "
                                   "{:d} (OUTER) and {:d}"
                                   .format(outer_reflect,
                                           inner_reflect),
                                   " (INNER)\n"]))

            # make sure that distances to the solute are actually
            # the same for INNER and OUTER
            self.debuglog("   Distance solute -> INNER (idx "
                          "{:d}): {:.15f}\n".format(inner_reflect,
                                                    d[inner_reflect]))
            self.debuglog("   Distance solute -> OUTER (idx "
                          "{:d}): {:.15f}\n".format(outer_reflect,
                                                    d[outer_reflect]))

            if abs(d[inner_reflect] - d[outer_reflect]) > 0.001:
                # minor difference can (but shouldn't). maybe T was a
                # bit too high? anyways the code is usually robust
                # enough to deal with that. but  we inform the user,
                # just in case this leads to a complete meltdown.
                print("WARNING: INNER and OUTER particle are not "
                      "exactly on the border! Difference > 0.001 "
                      "A but < 0.01 A.")

            if abs(d[inner_reflect] - d[outer_reflect]) > 0.01:
                # now this is a real issue and indicative of something
                # having gone wrong, either during time step
                # extrapolation or in the propagation itself.
                self.debugtraj()
                raise SystemExit("\nINNER and OUTER particles are not "
                                 "at the same distance from the center "
                                 "(difference > 0.01 A). This is a "
                                 "fundamental problem and we cannot "
                                 "continue. Potential failure of the "
                                 "time step extrapolation or "
                                 "propagation routines. Check your "
                                 "simulation parameters!\n")

            ############################################################
            #                                                          #
            # 4) PERFORM ELASTIC COLLISION BETWEEN PARTICLE PAIR       #
            #                                                          #
            ############################################################

            # gather required position (always wrt. to solute),
            # mass and velocity values
            r_inner = r[inner_reflect]
            r_outer = r[outer_reflect]
            m_outer = com_atoms[outer_reflect].mass
            m_inner = com_atoms[inner_reflect].mass
            v_outer = com_atoms[outer_reflect].momentum / m_outer
            v_inner = com_atoms[inner_reflect].momentum / m_inner

            # find angle between r_outer and r_inner
            theta = self.calc_angle(r_inner, r_outer)
            self.debuglog("   angle (r_outer, r_inner) is: {:.16f}\n"
                          .format(np.degrees(theta)))

            # rotate OUTER to be exactly on top of the INNER for
            # collision. this simulates the boundary mediating
            # a collision between the particles.

            # calculate rotational axis
            axis = self.normalize(np.cross(r[outer_reflect],
                                  r[inner_reflect]))

            # rotate velocity of outer region particle
            v_outer = np.dot(self.rotation_matrix(axis, theta),
                             v_outer)
            
            # Perform mass-weighted exchange of normal components of
            # velocitiy, force (, and random forces if Langevin).
            # i.e. elastic collision
            if self.reflective:
                self.debuglog("   -> hard wall reflection\n")
                n = self.normalize(r_inner)
                if np.dot(v_inner, n) > 0:
                    dV_inner = -2 * np.dot(np.dot(v_inner, n), n) 
                else:
                    dV_inner = np.array([0., 0., 0.])
                if np.dot(v_outer, n) < 0:
                    dV_outer = -2 * np.dot(np.dot(v_outer, n), n) 
                else:
                    dV_outer = np.array([0., 0., 0.])
                self.debuglog("   dV_inner = {:s}\n"
                              .format(np.array2string(dV_inner)))
                self.debuglog("   dV_outer = {:s}\n"
                              .format(np.array2string(dV_outer)))
            else:
                self.debuglog("   -> momentum exchange collision\n")
                M = m_outer + m_inner
                r12 = r_inner
                v12 = v_outer - v_inner
                self.debuglog("   r12 = {:s}\n"
                              .format(np.array2string(r12)))
                self.debuglog("   v12 = {:s}\n"
                              .format(np.array2string(v12)))
                self.debuglog("   dot(v12, r12) = {:.16f}\n"
                              .format(np.dot(v12, r12)))
                v_norm = np.dot(v12, r12) * r12 / (np.linalg.norm(r12)**2)
                dV_inner = 2 * m_inner / M * v_norm
                self.debuglog("   dV_inner = {:s}\n"
                              .format(np.array2string(dV_inner)))
                dV_outer = -2 * m_outer / M * v_norm
                self.debuglog("   dV_outer = {:s}\n"
                              .format(np.array2string(dV_outer)))

            if not self.surface and theta != 0 and theta != np.pi:
                # rotate outer particle velocity change component
                # back to inital direction after collision
                dV_outer = np.dot(self.rotation_matrix(
                                  axis, -1 * theta), dV_outer)

            if theta == np.pi:
                # flip velocity change component of outer particle
                # to the other side of the slab (if applicable)
                dV_outer = -1 * dV_outer

            # commit new momenta to pseudoparticle atoms object
            com_atoms[outer_reflect].momentum += (dV_outer * m_outer)
            com_atoms[inner_reflect].momentum += (dV_inner * m_inner)

            # expand the pseudoparticle atoms object back into the
            # real atoms object (inverse action to self.update())
            for i in range(self.natoms):
                # redistribute the velocity change normal
                # component to individual atoms
                outer_actual = (self.nsol + (outer_reflect - 1) 
                                * self.natoms + i)
                inner_actual = (self.nsol + (inner_reflect - 1) 
                                * self.natoms + i)
                self.atoms[outer_actual].momentum += (
                        dV_outer * self.atoms[outer_actual].mass)
                self.atoms[inner_actual].momentum += (
                        dV_inner * self.atoms[inner_actual].mass)

            # reset list that tracks conflicting particle pairs
            # since conflict is resolved now
            self.tocollide = list()

            # keep track of which pair of conflicting particles
            # was just resolved for future reference
            self.recent = [outer_reflect, inner_reflect]

            # safe current configuration as new "previous" state.
            # in case there is a second boundary event in this
            # iteration, we need to make sure not to reset to the
            # configuration before this collision, lest we lose
            # all the work we did here.
            self.previous_boundary_idx = boundary_idx
            self.previous_atoms.positions = self.atoms.positions.copy()
            self.previous_atoms.set_momenta(self.atoms.get_momenta(),
                                            apply_constraint=False)
            self.previous_atoms.calc.results['forces'] = (
                self.atoms.calc.results['forces'].copy())

            ############################################################
            #                                                          #
            # 5) PROPAGATE REMAINING TIME TO COMPLETE FULL TIME STEP   #
            #                                                          #
            ############################################################

            # determine new time step required to complete a full,
            # default time step
            if self.remaining_dt > self.default_dt:
                # catching an exception that would indicate that
                # something went wrong in the remaining time
                # bookkeeping or maybe because many boundary
                # events occurred within one time step. both
                # situations will not immediately break the
                # simulation but are concerning, so the user
                # will be warned in case of a complete meltdown
                # later on.
                print("<SAFIRES> Remaining dt after collision is "
                      "larger than initial dt. Resetting to intial dt.")
                self.debuglog("".join([
                              "   WARNING: Remaining dt after collision"
                              "({:.5f}) ".format(self.remaining_dt),
                              "is larger than initial dt. "
                              "Resetting to intial dt.\n"
                              ]))
                set_dt = self.default_dt
                self.remaining_dt = 0.
            elif self.remaining_dt == 0.0:
                # for the extremely unlikely case that the (sum of)
                # extrapolated dt(s in one iteration) adds up
                # exactly to the default time step.
                set_dt = self.default_dt
            else:
                # if everything is well behaved, this should be what
                # happens every time.
                set_dt = self.remaining_dt

            # debug logging
            self.debuglog("   < FINAL PROPAGATION >\n")

            # propagate by the remaining time step
            self.propagate(set_dt, checkup, halfstep=2)

            # update the pseudoparticle atoms object with the new
            # configuration, get all distances between pseudoparticles
            # and the solute.
            com_atoms, forces, r, d, unused1, unused2 = (
                self.update(self.atoms))

            # debug log distances between inner and outer particle (that
            # were conflicting) and the solute. this is a clear
            # indicator  if the collision and subsequent propagation
            # occurred as they should.
            self.debuglog("   Final distance solute -> INNER (idx "
                          "{:d}): {:.15f}\n".format(inner_reflect,
                                                    d[inner_reflect]))
            self.debuglog("   Findal distance solute -> OUTER (idx "
                          "{:d}): {:.15f}\n".format(outer_reflect,
                                                    d[outer_reflect]))

            ############################################################
            #                                                          #
            # 6) RE-RUN SEPARATE() WITH CHECKUP = TRUE                 #
            #                                                          #
            ############################################################

            # check again for new boundary events that might have
            # occurred during the second propagation halfstep.
            # checkup = True makes sure that we're not adding
            # forces and Langevin random components to the velocities
            # again, which would destroy energy conserveration.
            self.safires(checkup=True)

        else:
            # if no boundary event is detected

            ############################################################
            #                                                          #
            # 7b) EXIT SAFIRES AND RETURN TO SUPERORDINATE MD          #
            #                                                          #
            ############################################################

            # save current configuration as a "previous" state to go
            # back to incase of boundary events in the next iteration
            self.previous_boundary_idx = boundary_idx
            self.previous_atoms.positions = self.atoms.positions.copy()
            self.previous_atoms.set_momenta(self.atoms.get_momenta(), 
                                            apply_constraint=False)
            self.previous_atoms.calc.results['forces'] = (
                self.atoms.calc.results['forces'].copy())

            # reset collision tracker and remaining_dt
            self.recent = []
            self.remaining_dt = 0

            # write logfile
            self.logger(iteration, boundary_idx, boundary)

            # note that the simulation eventually always ends up
            # here after each iteration, no matter if a boundary event
            # was processed or not. this is where we can do any
            # necessary cleanup.

        if iteration == self.mdobject.max_steps:
            # in the very last MD iteration,
            # print some useful SAFIRES stats
            checkout_long = "".join(["\n... Finished.\n"
                                     "Total number of collisions: "
                                     "{:d}\n".format(self.ncollisions),
                                     "Total number of double collisions: "
                                     "{:d}\n".format(self.ndoubles)])
            checkout_short = ("\n... Finished.\n"
                              "Total number of collisions: "
                              "{:d}\n".format(self.ncollisions))
            self.debuglog(checkout_long)
            print(checkout_short)

            # print out "barometer" results (collisions caused
            # by inner and outer region particles, respectively)
            if self.barometer:
                pressure = "".join(["Barometer:\ntotal number of "
                                    "impacts from inner region "
                                    "particles: {:d}.\n"
                                    .format(self.impacts[0]),
                                    "Total number of impacts from "
                                    "outer region particles: {:d}.\n\n"
                                    .format(self.impacts[1]),
                                    "(Note that impact values reflect "
                                    "only collisions that are caused "
                                    "by either the inner or the outer "
                                    "region particle; symmetric, "
                                    "\"heads-on\" collisions of both "
                                    "particles are not counted."])
                print(pressure)
                self.debuglog(pressure)

            # close logfiles
            if self.logfile is not None:
                self.log.close()
            if self.debug:
                self.db.close()
