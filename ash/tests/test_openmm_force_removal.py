import pytest

openmm = pytest.importorskip("openmm")

from ash.interfaces.interface_OpenMM import remove_forces_from_system


def make_system():
    # Force order on purpose: the two forces to remove sit before another force, so that
    # removing them by increasing index would delete the wrong one.
    system = openmm.System()
    for _ in range(3):
        system.addParticle(1.0)
    system.addForce(openmm.HarmonicBondForce())
    system.addForce(openmm.MonteCarloBarostat(1.0 * openmm.unit.bar, 300 * openmm.unit.kelvin))
    system.addForce(openmm.AndersenThermostat(300 * openmm.unit.kelvin, 1 / openmm.unit.picosecond))
    system.addForce(openmm.NonbondedForce())
    return system


def forcenames(system):
    return [force.__class__.__name__ for force in system.getForces()]


def test_remove_two_forces_keeps_the_others():
    system = make_system()
    removed = remove_forces_from_system(system, ["MonteCarloBarostat", "AndersenThermostat"])
    assert removed == ["MonteCarloBarostat", "AndersenThermostat"]
    assert forcenames(system) == ["HarmonicBondForce", "NonbondedForce"]


def test_remove_single_force_by_string():
    system = make_system()
    removed = remove_forces_from_system(system, "MonteCarloBarostat")
    assert removed == ["MonteCarloBarostat"]
    assert forcenames(system) == ["HarmonicBondForce", "AndersenThermostat", "NonbondedForce"]


def test_remove_force_that_is_not_present():
    system = make_system()
    removed = remove_forces_from_system(system, "CMMotionRemover")
    assert removed == []
    assert forcenames(system) == ["HarmonicBondForce", "MonteCarloBarostat",
                                  "AndersenThermostat", "NonbondedForce"]


def test_remove_duplicate_forces():
    # Two barostats (can happen when a system is reused for several MD runs)
    system = make_system()
    system.addForce(openmm.MonteCarloBarostat(1.0 * openmm.unit.bar, 300 * openmm.unit.kelvin))
    removed = remove_forces_from_system(system, "MonteCarloBarostat")
    assert removed == ["MonteCarloBarostat", "MonteCarloBarostat"]
    assert forcenames(system) == ["HarmonicBondForce", "AndersenThermostat", "NonbondedForce"]
