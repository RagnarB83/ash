==============
Things to todo
==============


##############
General things
##############
- Go over PyCHARM warnings and fix. Adapt to PEP8 as much as possible
- Write docstrings for all the functions
- Go over this: https://levelup.gitconnected.com/write-better-python-code-today-3c7566a695e2
- Start using f-strings more
- Figure out best way to deal with some global variables.


##############
Small things
##############

- ORCA interface:
    - Todo: Add ORCA extrabasis-feature on specific atoms
    - Todo: Add embedding ECP atoms

- Basic CIF_read in, outside molcrys.
    - Idea: Read in CIF, get main molecule?
    - TODO: Read in CIF_file. Create XYZ file of both types of fragments???


#######################
Big projects: UPCOMING
#######################
- TODO: Allow Numfreq module to do QM/MM Hessian. Requires some thinking.
- TODO: Project out translational and rotational modes.
- TODO: Add PySCF interface
- TODO: Write Psi4 interface
- TODO: OpenMM interface for bonded MM interactions
- TODO: Look into using orca_mm
- Fit LJ parameters for molecules based on el-dens analysis DDEC3 approach.
    - Already started here (/home/bjornsson/Lennard-Jones-creation-program/LJ-from-chargemol)  Odin.
    - Also look into: https://github.com/awvwgk/libmbd
- TODO: Interface MEDFF??

#######################
BIG projects: one day
#######################

- TODO: NAMD QM/MM with xtB: https://github.com/fhh2626/NAMD-xtb-QMMM-interface
    https://www.ks.uiuc.edu/Research/namd/mailing_list/namd-l.2017-2018/1261.html

- Todo: Write Dalton interface
- TODO: Write Plumed interface for metadynamics stuff
- TODO: Write Janus interface for adaptive QM/MM. https://pubs.acs.org/doi/abs/10.1021/acs.jctc.9b00182
    https://github.com/CCQC/janus
- NOTE: Janus seems to have an openMM QM/MM interface

- PES-program incorporate
- Benchmarking program
- Automatic PES reaction exploration program. Uses xtb and ORCA interfaces
- Plotting data in general
- orca_vib+orca_mapsps replacement. Build on Hess-tool

- TODO: Spin projection gradients for BS-DFT geometry optimization
- Add spin projection gradients . See what has been done before:
    - Yamaguchi paper: https://reader.elsevier.com/reader/sd/pii/S0009261407006835?token=AED86C614DA6817E2973552241F73C686E1ED4C10334BB5557BC25C11B62E4CBC7EBA2C75D12A748882FF944E79F1798
    - EBS: https://www.pnas.org/content/104/52/20725
    - EBS Guidoni gradients: https://pubs.acs.org/doi/abs/10.1021/acs.jctc.7b00316
    - Guidoni: https://aip.scitation.org/doi/full/10.1063/1.4752398

- MOLECULAR DYNAMICS
    - TODO: Create initial Python-based MD code for basic functionality and understanding.
    - OpenMM might be the way for fast MM and fast MD. Would require tinkering for QM/MM
    - Look at Villi's Knarr
    - TODO: Replace with C++/Fortran written code?? Or interface something ??
    - TODO: Add thermostat and Shake functionality?
    - TODO: Maybe just do xtb MD with solute and a few xtB MD waters. Rest is frozen TIP3P.
    - TODO: Alternatively look into GFN-FF
    - xtb MD: https://xtb-docs.readthedocs.io/en/latest/md.html
    - MD program options:
        - https://github.com/openmm/openmm
        - http://openmd.org/category/examples/
        - Ab initio MD program written in Fortran. Includes ORCA interface already https://github.com/PHOTOX/ABIN
        - orca_md by Martin Brehm. Seems to be able to run as standalone via file read-in
    - MD analysis: https://github.com/MDAnalysis/mdanalysis
    - Metadynamics: https://github.com/openmm/openmm/issues/2126
        http://docs.openmm.org/latest/api-python/generated/simtk.openmm.app.metadynamics.Metadynamics.html

#######################
VAGUE IDEAS:
#######################
- STEAL IDEAS FROM: http://cuby4.molecular.cz/index.html

- Element information: https://github.com/awvwgk/QCElemental
- Inspiration: https://github.com/awvwgk/QCEngine
