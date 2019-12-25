import molmod

import numpy as np
import simtk.unit as unit
import simtk.openmm as mm
import simtk.openmm.app

from yaff.pes.ext import Cell

def _align(system):
    """Aligns rvecs in system such that cell matrix is lower diagonal"""
    pos = system.pos[:]
    rvecs = system.cell._get_rvecs()[:]
    def get_angle_axis(t, s):
        cos = np.dot(s, t)/np.linalg.norm(s)/np.linalg.norm(t)
        angle = np.arccos(np.clip(cos, -1, 1))
        axis = np.cross(s, t)
        return angle, axis

    a = rvecs[0]
    b = rvecs[1]
    z = np.cross(a, b)
    z /= np.linalg.norm(z)
    angle, axis = get_angle_axis(z, np.array([0, 0, 1]))
    rotation = molmod.Rotation.from_properties(-angle, axis, False)
    rvecs_new = rotation * rvecs
    pos_new = rotation * pos
    angle, axis = get_angle_axis(rvecs_new[0], np.array([1, 0, 0]))
    rotation = molmod.Rotation.from_properties(-angle, axis, False)
    rvecs_new_new = rotation * rvecs_new
    pos_new_new = rotation * pos_new
    tol = 1e-11
    assert(np.abs(rvecs_new_new[0, 1]) < tol)
    assert(np.abs(rvecs_new_new[0, 2]) < tol)
    assert(np.abs(rvecs_new_new[1, 2]) < tol)
    rvecs_final = np.multiply(rvecs_new_new, (np.abs(rvecs_new_new) > tol))
    system.cell = Cell(rvecs_final)
    system.pos[:] = pos_new_new
    print('ALIGNED CELL TENSOR (in angstrom):')
    print(rvecs_final / molmod.units.angstrom)

def _init_openmm_system(system):
    """Creates and returns an OpenMM system object with correct cell vectors and particles"""
    system.set_standard_masses()
    pos = system.pos / molmod.units.angstrom / 10.0 * unit.nanometer
    rvecs = system.cell._get_rvecs() / molmod.units.angstrom / 10.0 * unit.nanometer
    mm_system = mm.System()
    mm_system.setDefaultPeriodicBoxVectors(*rvecs)
    for i in range(system.pos.shape[0]):
        mm_system.addParticle(system.masses[i] / molmod.units.amu * unit.dalton)
    return mm_system

def get_topology(system):
    """Creates an OpenMM topology object, necessary to run simulations"""
    top = mm.app.Topology()
    chain = top.addChain()
    res = top.addResidue('res', chain)
    elements = []
    atoms = []
    for i in range(system.natom):
        elements.append(
                mm.app.Element.getByMass(system.masses[i] / molmod.units.amu * unit.dalton),
                )
    for i in range(system.natom):
        element = elements[i]
        name = str(i)
        atoms.append(top.addAtom(
                name,
                element,
                res,
                ))
    for bond in system.bonds:
        top.addBond(atoms[bond[0]], atoms[bond[1]])
    return top

def _check_rvecs(rvecs):
    max_rcut = 0
    return max_rcut
