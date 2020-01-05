import molmod

import numpy as np
import simtk.unit as unit
import simtk.openmm as mm
import simtk.openmm.app
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from yaff.pes.ext import Cell

plt.rcParams['axes.axisbelow'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 9

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
    tol = 1e-8
    assert(np.abs(rvecs_new_new[0, 1]) < tol)
    assert(np.abs(rvecs_new_new[0, 2]) < tol)
    assert(np.abs(rvecs_new_new[1, 2]) < tol)
    rvecs_final = np.multiply(rvecs_new_new, (np.abs(rvecs_new_new) > tol))
    system.cell = Cell(rvecs_final)
    system.pos[:] = pos_new_new
    #print('ALIGNED CELL TENSOR (in angstrom):')
    #print(rvecs_final / molmod.units.angstrom)

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
    lengths = np.linalg.norm(rvecs, axis=1)
    sizes = np.array([
        rvecs[0, 0],
        rvecs[1, 1],
        rvecs[2, 2],
        ])
    return np.min(sizes) / 2

def plot_switching_functions(rcut, rswitch, width):
    """Plots the switching functions of YAFF and OpenMM

    Arguments
    ---------
        rcut (double):
            cutoff of nonbonded interactions
        rswitch (double):
            switching parameter for the OpenMM setUseSwitchingFunction member function
        width (double):
            argument for the YAFF Switch3 constructor.
    """
    sigma = 3
    epsilon = 1
    x = np.linspace(rswitch, rcut, 200)
    lj = 4 * epsilon * ((sigma / x) ** 12 - (sigma / x) ** 6)

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)

    # LJ
    ax.plot(
            x,
            lj,
            color='k',
            linewidth=1.5,
            label='standard LJ',
            )

    # YAFF
    lj_yaff = np.ones(lj.shape)
    for i in range(len(x)):
        if x[i] < rcut - width:
            lj_yaff[i] = lj[i]
        elif x[i] >= rcut - width and x[i] <= rcut:
            u = (rcut - x[i]) / width
            lj_yaff[i] = lj[i] * (3 * u ** 2 - 2 * u ** 3)
        elif x[i] > rcut:
            lj_yaff[i] = 0
    ax.plot(
            x,
            lj_yaff,
            color='b',
            linewidth=1,
            label='YAFF',
            )

    # OPENMM
    lj_mm = np.ones(lj.shape)
    for i in range(len(x)):
        if x[i] < rswitch:
            lj_mm[i] = lj[i]
        elif x[i] >= rswitch and x[i] <= rcut:
            u = (x[i] - rswitch) / (rcut - rswitch)
            assert u >= 0 and u <= 1
            lj_mm[i] = lj[i] * (1 - 6 * u ** 5 + 15 * u ** 4 - 10 * u ** 3)
        elif x[i] > rcut:
            lj_mm[i] = 0
    ax.plot(
            x,
            lj_mm,
            color='r',
            linewidth=1,
            label='OpenMM',
            )


    ax.set_xlabel('Distance [a.u]')
    ax.set_ylabel('Energy [a.u]')
    ax.grid()
    ax.legend()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax.get_yaxis().set_tick_params(which='both', direction='in')
    ax.get_xaxis().set_tick_params(which='both', direction='in')
    fig.savefig('switch.pdf', bbox_inches='tight')
