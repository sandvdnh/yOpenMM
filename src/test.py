import yaff
import molmod
import time

import numpy as np
import simtk.unit as unit
import simtk.openmm as mm
import simtk.openmm.app
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from attrdict import AttrDict
from src.utils import _align, _check_rvecs, _init_openmm_system, get_topology
from src.generator import AVAILABLE_PREFIXES, FFArgs, apply_generators, apply_generators_mm
from systems.systems import test_systems


class Test(object):
    """Base class to perform a test between OpenMM and YAFF"""
    tname = None

    def __init__(self, name, platform, use_max_rcut=False, largest_error=False):
        info = None
        for _ in test_systems:
            if _['name'] == name:
                info = AttrDict(_)
        assert(info is not None)
        _ = yaff.System.from_file(info.path_chk)
        if largest_error:
            print('LOADING ERROR CHK')
            _ = yaff.System.from_file(info.path_errorchk)
            info.supercell = [1, 1, 1]
        else:
            self.path_errorchk = info.path_errorchk
        self.system = _.supercell(*info.supercell)
        self.parameters = yaff.Parameters.from_file(info.path_pars)
        self.platform = platform
        if not use_max_rcut:
            self.rcut = info.rcut
        else:
            self.rcut = None
        self.name = info.name
        self.supercell = info.supercell

        if 'tr' in info:
            self.tr = info.tr
        else:
            self.tr = None
        if 'alpha_scale' in info:
            self.alpha_scale = info.alpha_scale
        else:
            self.alpha_scale = 3.5
        if 'gcut_scale' in info:
            self.gcut_scale = info.gcut_scale
        else:
            self.gcut_scale = 1.1
        if 'reci_ei' in info:
            self.reci_ei = info.reci_ei
        else:
            self.reci_ei = 'ewald'
        if 'tailcorrections' in info:
            self.tailcorrections = info.tailcorrections
        else:
            self.tailcorrections = False
        if 'tr' in info:
            self.tr = info.tr
        else:
            self.tr = 7.558904535685008

    def pre(self):
        """Performs a number of checks before executing the test (to save time)

        - asserts whether all prefixes in the parameter file are supported in the
          generator
        - aligns cell, checks OpenMM requirements on cell vector geometry
        - checks whether rcut is not larger than half the length of the shortest
          cell vector
        """
        for prefix, _ in self.parameters.sections.items():
            assert prefix in AVAILABLE_PREFIXES, 'prefix {} not available'.format(prefix)
        _align(self.system)
        max_rcut = _check_rvecs(self.system.cell._get_rvecs())
        if self.rcut is not None:
            assert self.rcut < max_rcut, 'chosen cutoff too large: {:.3f} > {:.3f}'.format(self.rcut, max_rcut)
        else:
            self.rcut = 0.99 * max_rcut

    def report(self):
        print('{}; {} supercell; {} atoms'.format(self.name.upper(), self.supercell, self.system.natom))
        print('')
        print(self.system.cell._get_rvecs() / molmod.units.angstrom, 'angstrom')
        if self.tr:
            print('CUTOFF: {:.5f} angstrom (SMOOTH, over {:.3f} angstrom)'.format(self.rcut / molmod.units.angstrom, self.tr / molmod.units.angstrom))
        else:
            print('CUTOFF: {:.5f} angstrom (HARD)'.format(self.rcut / molmod.units.angstrom))
        if self.tailcorrections:
            print('USING TAIL CORRECTIONS')
        else:
            print('NO TAIL CORRECTIONS')
        print('')
        pass

    def _internal_test(self, **kwargs):
        raise NotImplementedError

    def __call__(self, **kwargs):
        self.pre()
        self.report()
        self._internal_test(**kwargs)

    def _get_ffargs(self, use_yaff=True):
        if not use_yaff:
            cls = FFArgs
        else:
            cls = yaff.pes.generator.FFArgs
        tr = None
        if self.tr:
            tr = yaff.pes.ext.Switch3(self.tr)
        return cls(
                rcut=self.rcut,
                tr=tr,
                alpha_scale=self.alpha_scale,
                gcut_scale=self.gcut_scale,
                reci_ei=self.reci_ei,
                tailcorrections=self.tailcorrections,
                )

    @staticmethod
    def _add_thermostat(mm_system, T):
        """Adds a thermostat to an OpenMM system object"""
        thermo = mm.AndersenThermostat(T * unit.kelvin, 1/unit.picosecond)
        mm_system.addForce(thermo)

    @staticmethod
    def _add_barostat(mm_system, T, P):
        """Adds a barostat to an OpenMM system object"""
        P *= unit.pascal
        Pb = P.value_in_unit(unit.bar)
        baro = mm.MonteCarloAnisotropicBarostat((Pb, Pb, Pb), T * unit.kelvin)
        mm_system.addForce(baro)

    @staticmethod
    def _remove_cmmotion(mm_system):
        cmm = mm.CMMotionRemover()
        mm_system.addForce(cmm)


class SinglePoint(Test):
    """Compares energy and forces for a single state"""
    tname = 'single'

    def _section_test(self, prefix, section):
        parameters = yaff.pes.parameters.Parameters({prefix: section})
        ff_args = self._get_ffargs(use_yaff=False)
        apply_generators(self.system, parameters, ff_args)
        ff = yaff.ForceField(self.system, ff_args.parts, ff_args.nlist)
        e = ff.compute() / molmod.units.kjmol
        ff_args_ = self._get_ffargs(use_yaff=True)
        yaff.pes.generator.apply_generators(self.system, parameters, ff_args_)
        ff = yaff.ForceField(self.system, ff_args_.parts, ff_args_.nlist)
        f = np.zeros(self.system.pos.shape)
        e_ = ff.compute(f, None) / molmod.units.kjmol
        f *= molmod.units.nanometer / molmod.units.kjmol
        f *= -1.0 # gpos == -force
        assert(e == e_)
        # OPENMM
        mm_system = _init_openmm_system(self.system)
        ff_args = self._get_ffargs(use_yaff=False)
        apply_generators_mm(self.system, parameters, ff_args, mm_system)
        integrator = mm.VerletIntegrator(0.5 * unit.femtosecond)
        platform = mm.Platform.getPlatformByName(self.platform)
        context = mm.Context(mm_system, integrator, platform)
        if platform.getName() == 'CUDA':
            platform.setPropertyValue(context, "CudaPrecision", 'double')
        context.setPositions(self.system.pos / molmod.units.nanometer * unit.nanometer)
        state = context.getState(
                getPositions=True,
                getForces=True,
                getEnergy=True,
                enforcePeriodicBox=True,
                )
        pos = state.getPositions()
        mm_e = state.getPotentialEnergy()
        mm_f = state.getForces(asNumpy=True)
        #print('before: ', mm_e)
        context.setPositions(pos)
        state = context.getState(getForces=True, getEnergy=True, enforcePeriodicBox=True)
        mm_e_after = state.getPotentialEnergy()
        mm_e_ = mm_e.value_in_unit(mm_e.unit)
        mm_e_after_ = mm_e_after.value_in_unit(mm_e_after.unit)
        assert np.abs(mm_e_after_ - mm_e_) < 1e-4, 'energy before {} \t energy after: {}'.format(mm_e, mm_e_after)
        return e, mm_e.value_in_unit(mm_e.unit), f, mm_f.value_in_unit(mm_f.unit)

    def _internal_test(self):
        print(' ' * 11 + '\t{:20}'.format('(YAFF) kJ/mol') + '\t{:20}'.format('(OpenMM) kJ/mol') + '\t{:20}'.format('force MAE (kJ/(mol * nm))'))
        for prefix, section in self.parameters.sections.items():
            e, mm_e, f, mm_f = self._section_test(prefix, section)
            mae = np.mean(np.abs(f - mm_f))
            print('{:10}\t{:20}\t{:20}\t{:20}'.format(prefix, str(e), str(mm_e), str(mae)))
            #if prefix == 'OOPDIST':
            #    err = np.abs(f - mm_f)
            #    for i in range(err.shape[0]):
            #        print(i, err[i])
        return e


class VirialTest(Test):
    """Compares virial tensors between YAFF and OpenMM.

    The virial tensor is computed with YAFF both analytically and using a fourth order approximation.
    If these results are in correspondence, then the virial is also computed with OpenMM using a
    fourth order approximation.
    """
    tname = 'virial'

    def __init__(self, *args, **kwargs):
        Test.__init__(self, *args, **kwargs)
        _align(self.system)
        self.default_pos = self.system.pos.copy()
        self.default_rvecs = self.system.cell._get_rvecs().copy()
        gvecs = self.system.cell.gvecs
        self.default_reduced = np.dot(self.default_pos, gvecs.transpose())

        # INIT YAFF
        ff_args = self._get_ffargs(use_yaff=True)
        apply_generators(self.system, self.parameters, ff_args)
        self.ff = yaff.ForceField(self.system, ff_args.parts, ff_args.nlist)

        # INIT OPENMM
        mm_system = _init_openmm_system(self.system)
        ff_args = self._get_ffargs(use_yaff=False)
        apply_generators_mm(self.system, self.parameters, ff_args, mm_system)
        integrator = mm.VerletIntegrator(0.5 * unit.femtosecond)
        platform = mm.Platform.getPlatformByName(self.platform)
        self.context = mm.Context(mm_system, integrator, platform)
        if platform.getName() == 'CUDA':
            platform.setPropertyValue(self.context, "CudaPrecision", 'mixed')
        self.context.setPositions(self.system.pos / molmod.units.nanometer * unit.nanometer)

    def _set_default_pos(self):
        self.ff.update_rvecs(self.default_rvecs)
        self.ff.update_pos(self.default_pos)
        self.context.setPositions(self.default_pos / molmod.units.nanometer * unit.nanometer)
        self.context.setPeriodicBoxVectors(*(self.default_rvecs / molmod.units.nanometer * unit.nanometer))

    def _compute_yaff(self, pos, rvecs):
        """Computes and returns energy with YAFF given positions and rvecs"""
        self.ff.update_rvecs(rvecs)
        self.ff.update_pos(pos)
        return self.ff.compute()

    def _compute_mm(self, pos, rvecs):
        """Computes and returns energy with OpenMM given positions and rvecs"""
        self.context.setPositions(pos[:] / molmod.units.nanometer * unit.nanometer)
        self.context.setPeriodicBoxVectors(*(rvecs / molmod.units.nanometer * unit.nanometer))
        state = self.context.getState(getEnergy=True)
        e = state.getPotentialEnergy()
        return e.value_in_unit(e.unit) * molmod.units.kjmol

    def _compute_dE(self, component, dx, compute_func):
        rvecs = self.default_rvecs.copy()
        rvecs[component] += dx
        pos = np.dot(self.default_reduced, rvecs)
        return compute_func(self, pos, rvecs)

    def finite_difference2(self, component, compute_func, dx):
        """Computes a 2nd order finite difference approximation to the derivative of the energy

        Arguments
        ---------
            component (tuple):
                component of the cell matrix with respect to which the derivative should
                be computed.
            compute_func (function):
                function used to compute the energy. It should accept three arguments
                (self, pos, rvecs)
        """
        e1 = self._compute_dE(component, dx, compute_func)
        e0 = self._compute_dE(component, -dx, compute_func)
        return (e1 - e0) / (2 * dx)

    def finite_difference4(self, component, compute_func, dx):
        """Computes a 4th order finite difference appproximation to the derivative of the energy"""
        e2 = self._compute_dE(component, 2 * dx, compute_func)
        e1 = self._compute_dE(component, dx, compute_func)
        em1 = self._compute_dE(component, -dx, compute_func)
        em2 = self._compute_dE(component, - 2 * dx, compute_func)
        return (-e2 + 8 * e1 - 8 * em1 + em2) / (12 * dx)

    def exact(self, component):
        """Computes the derivative of the energy using the virial"""
        vtens = np.zeros((3, 3))
        self.ff.update_pos(self.default_pos)
        self.ff.update_rvecs(self.default_rvecs)
        self.ff.compute(gpos=None, vtens=vtens)
        gvecs = self.ff.system.cell.gvecs
        reduced = np.dot(self.ff.system.pos, gvecs.transpose())
        #rvecs = np.zeros((3, 3))
        dE = np.matmul(vtens, np.linalg.inv(self.system.cell._get_rvecs())).transpose()
        return dE[component]

    def _internal_test(self, dx=0.001, order=4):
        print(15 * '#')
        print('dx: {} angstrom'.format(dx))
        print('order of FD approximation: {}'.format(order))
        print(15 * '#')
        print('')
        c = molmod.units.kjmol
        print('TOTAL ENERGY [kJ/mol]')
        print('(YAFF)\t\t{:10}'.format(self._compute_yaff(self.default_pos, self.default_rvecs) / c))
        print('(OPENMM)\t{:10}'.format(self._compute_mm(self.default_pos, self.default_rvecs) / c))
        print(15 * '-')

        components = [
                (0, 0),
                (1, 1),
                (2, 2),
                (1, 0),
                (2, 0),
                (2, 1),
                ]
        dE_exact = np.zeros((3, 3))
        dE_yaff = np.zeros((3, 3))
        dE_mm = np.zeros((3, 3))
        for component in components:
            if order == 2:
                print('COMPONENT {}\t{:10}\t{:10}\t{:10}'.format(component, 'YAFF (EXACT)', 'YAFF (FD2)', 'OPENMM (FD2)'))
                fd_func = VirialTest.finite_difference2
            elif order == 4:
                print('COMPONENT {}\t{:10}\t{:10}\t{:10}'.format(component, 'YAFF (EXACT)', 'YAFF (FD4)', 'OPENMM (FD4)'))
                fd_func = VirialTest.finite_difference4

            c = molmod.units.kjmol / molmod.units.nanometer
            fd_yaff = fd_func(self, component, VirialTest._compute_yaff, dx)
            fd_mm = fd_func(self, component, VirialTest._compute_mm, dx)
            exact = self.exact(component)
            print('\t\t\t{:13}\t{:13}\t{:13}'.format(str(exact)[:13], str(fd_yaff)[:13], str(fd_mm)[:13]))
            dE_exact[component] = exact
            dE_yaff[component] = fd_yaff
            dE_mm[component] = fd_mm
        gvecs = np.linalg.inv(self.default_rvecs.transpose())
        dEs = [dE_exact, dE_yaff, dE_mm]
        virials = [np.zeros((3, 3)) for i in range(3)]
        labels = ['YAFF (EXACT)', 'YAFF (FD)', 'OPENMM (FD)']
        print(15 * '-')
        print('VIRIAL CONTRIBUTION TO PRESSURE TENSOR')
        volume = np.linalg.det(self.default_rvecs)
        for i, dE in enumerate(dEs):
            print(labels[i])
            virials[i] = -1.0 / volume * np.matmul(dE, np.linalg.inv(gvecs)).transpose()
            print(virials[i])
        print(15 * '-')
        print('VIRIAL CONTRIBUTION TO ISOTROPIC PRESSURE [MPa]')
        for i in range(3):
            c = molmod.units.pascal * 1e6
            print('{:20}'.format(labels[i]) + str(1 / 3 * np.trace(virials[i]) / c))


class VerletTest(Test):
    """Compares energy and forces over a short trajectory obtained through Verlet integration"""
    tname = 'verlet'

    @staticmethod
    def _get_energy_forces(context):
        """Extracts energy, forces and positions from state"""
        state = context.getState(
                getPositions=True,
                getForces=True,
                getEnergy=True,
                )
        mm_e = state.getPotentialEnergy()
        mm_f = state.getForces(asNumpy=True)
        mm_pos = state.getPositions(asNumpy=True)
        mm_e_ = mm_e.value_in_unit(mm_e.unit)
        mm_f_ = mm_f.value_in_unit(mm_f.unit)
        mm_pos_ = mm_pos.value_in_unit(mm_pos.unit)
        return mm_e_, mm_f_, mm_pos_

    def _simulate(self, steps):
        """Computes a trajectory using a VerletIntegrator"""
        energies = np.zeros(steps)
        forces = np.zeros((steps, self.system.natom, 3))
        positions = np.zeros((steps, self.system.natom, 3))
        mm_system = _init_openmm_system(self.system)
        ff_args = self._get_ffargs(use_yaff=False)
        apply_generators_mm(self.system, self.parameters, ff_args, mm_system)
        integrator = mm.VerletIntegrator(0.5 * unit.femtosecond)
        platform = mm.Platform.getPlatformByName(self.platform)
        topology = get_topology(self.system)
        simulation = mm.app.Simulation(
                topology,
                mm_system,
                integrator,
                platform,
                )
        simulation.context.setPositions(self.system.pos / molmod.units.nanometer * unit.nanometer)
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 5)
        simulation.reporters.append(mm.app.PDBReporter('./output.pdb', 1))
        for i in range(steps):
            simulation.step(1)
            mm_e, mm_f, mm_pos = self._get_energy_forces(simulation.context)
            energies[i] = mm_e
            forces[i, :] = mm_f
            positions[i, :] = mm_pos
        return energies, forces, positions

    def _yaff_compute(self, positions):
        """Computes energies and forces over a trajectory using YAFF"""
        energies = np.zeros(positions.shape[0])
        forces = np.zeros((positions.shape[0], self.system.natom, 3))
        ff_args = self._get_ffargs(use_yaff=False)
        apply_generators(self.system, self.parameters, ff_args)
        ff = yaff.ForceField(self.system, ff_args.parts, ff_args.nlist)
        for i in range(positions.shape[0]):
            ff.update_pos(positions[i, :] * molmod.units.nanometer)
            energies[i] = ff.compute(forces[i], None)
        forces *= molmod.units.nanometer / molmod.units.kjmol
        forces *= -1.0 # gpos == -force
        energies /= molmod.units.kjmol
        return energies, forces

    def _internal_test(self, steps=100):
        print(15 * '#')
        print('number of steps: {}'.format(steps))
        print(15 * '#')
        print('')
        print('simulating system for {} steps with OpenMM...'.format(steps))
        mm_energies, mm_forces, positions = self._simulate(steps)
        print('recomputing energies and forces wth YAFF...')
        energies, forces = self._yaff_compute(positions)

        energy_ae = np.abs(energies - mm_energies)
        energy_mae = np.mean(energy_ae)
        energy_max_ae = np.max(energy_ae)
        energy_median_ae = np.median(energy_ae)

        forces_ae = np.abs(forces - mm_forces)
        forces_mae = np.mean(forces_ae)
        forces_max_ae = np.max(forces_ae)
        forces_median_ae = np.median(forces_ae)

        energy_re = np.abs((energies - mm_energies) / mm_energies)
        energy_mre = np.mean(energy_re)
        energy_max_re = np.max(energy_re)
        energy_median_re = np.median(energy_re)

        forces_re = np.abs((forces - mm_forces) / mm_forces)
        forces_mre = np.mean(forces_re)
        forces_max_re = np.max(forces_re)
        forces_median_re = np.median(forces_re)

        print('')
        print('{:24}\t{:20}\t{:20}'.format('', 'energy [kJ/mol]', 'force [kJ/(mol * nm)]'))
        print('{:24}\t{:20}\t{:20}'.format('mean    absolute error', str(energy_mae), str(forces_mae)))
        print('{:24}\t{:20}\t{:20}'.format('max     absolute error', str(energy_max_ae), str(forces_max_ae)))
        print('{:24}\t{:20}\t{:20}'.format('median  absolute error', str(energy_median_ae), str(forces_median_ae)))
        print('{:24}\t{:20}\t{:20}'.format('mean    relative error', str(energy_mre), str(forces_mre)))
        print('{:24}\t{:20}\t{:20}'.format('max     relative error', str(energy_max_re), str(forces_max_re)))
        print('{:24}\t{:20}\t{:20}'.format('median  relative error', str(energy_median_re), str(forces_median_re)))

        largest_e = np.argmax(energy_ae)
        largest_f = np.argmax(np.mean(np.mean(forces_ae, axis=2), axis=1))
        i = np.argmax(forces_ae)
        index = np.unravel_index(i, forces_ae.shape)
        ffa_id = self.system.ffatype_ids[index[-1]]
        type_ = self.system.ffatypes[ffa_id]
        print('frame with largest energy error: {}'.format(largest_e))
        print('index of largest force error: {}  (atom type {})'.format(index, type_))
        self.save_frame(positions, largest_e)

    def save_frame(self, positions, index):
        """Saves a system file with positions[index] as pos to chk"""
        pos = positions[index] * molmod.units.nanometer
        self.system.pos[:] = pos
        self.system.to_file(self.path_errorchk)


class SimulationTest(Test):
    """Performs a simulation using OpenMM, and outputs a .pdb file with coordinates"""
    tname = 'simulate'

    def _internal_test(self, steps=1000, writer_step=100, T=None, P=None):
        print(15 * '#')
        print('number of timesteps: {}'.format(steps))
        print('writer frequency: {}'.format(writer_step))
        print('T: {}'.format(T))
        print('P: {}'.format(P))
        if T is not None:
            if P is not None:
                print('adding temperature and pressure coupling')
            else:
                print('adding temperature coupling')
        else:
            print('no additional coupling')
        print(15 * '#')
        print('')
        mm_system = _init_openmm_system(self.system)
        ff_args = self._get_ffargs(use_yaff=False)
        apply_generators_mm(self.system, self.parameters, ff_args, mm_system)
        if T is not None:
            Test._add_thermostat(mm_system, T)
            if P is not None:
                Test._add_barostat(mm_system, T, P)
        integrator = mm.VerletIntegrator(0.5 * unit.femtosecond)
        platform = mm.Platform.getPlatformByName(self.platform)
        topology = get_topology(self.system)
        simulation = mm.app.Simulation(
                topology,
                mm_system,
                integrator,
                platform,
                )
        simulation.context.setPositions(self.system.pos / molmod.units.nanometer * unit.nanometer)
        simulation.context.setVelocitiesToTemperature(300 * unit.kelvin, 5)
        simulation.reporters.append(mm.app.PDBReporter('./output.pdb', writer_step))
        print('simulation in progress')
        t0 = time.time()
        simulation.step(steps)
        t1 = time.time()
        print('elapsed time:\t\t{:.3f}s'.format(t1 - t0))


class CutoffTest(SinglePoint):
    """Sweeps the potential energy of the MM3/LJ force part"""
    tname = 'cutoff'

    def __init__(self, *args, **kwargs):
        SinglePoint.__init__(self, *args, **kwargs)
        assert not kwargs['use_max_rcut'], 'Cannot use max_rcut option in CutoffTest'

    def _internal_test(self, npoints=15, delta=0.1):
        print(15 * '#')
        print('number of grid points: {}'.format(npoints))
        print('delta: {} angstrom'.format(delta))
        print(15 * '#')
        print('')
        prefixes = [
                'MM3',
                'LJ',
                ]
        energies = np.zeros(npoints)
        energies_mm = np.zeros(energies.shape)
        forces = []
        forces_mm = []
        c = molmod.units.angstrom
        rcuts = np.linspace(self.rcut - delta * c, self.rcut + delta * c, npoints)
        for i in range(len(rcuts)):
            for prefix, section in self.parameters.sections.items():
                if prefix in prefixes:
                    self.rcut = rcuts[i]
                    e, mm_e, f, mm_f = self._section_test(prefix, section)
                    energies[i] = e
                    energies_mm[i] = mm_e
                    forces.append(f.copy())
                    forces_mm.append(mm_f.copy())
        self.plot(rcuts, energies, energies_mm)
        print('ENERGY DIFFERENCES [kJ/mol]')
        print(energies_mm - energies)
        index = np.argmax((energies_mm - energies) > 1e-1)
        if index >= 1:
            print('FORCE DIFFERENCE BEFORE')
            df_before = forces[index - 1] - forces_mm[index - 1]
            print(df_before)
        else:
            print('jump occurs before scanned rcut range')
        print('FORCE DIFFERENCE AFTER')
        df_after = forces[index] - forces_mm[index]
        print(df_after)
        if index >= 1:
            ddf = df_before - df_after
            count = 0
            for i in range(ddf.shape[0]):
                if np.linalg.norm(ddf[i]) > 1e-5:
                    print('{}: {}'.format(i, ddf[i]))
                    count += 1
            print('TOTAL NUMBER OF ATOMS WITH LARGE FORCE DEVIATIONS: {}'.format(count))



    @staticmethod
    def plot(rcuts, energies, energies_mm):
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)

        # LJ
        ax.plot(
                rcuts / molmod.units.angstrom,
                energies,
                color='b',
                linewidth=0.8,
                linestyle='--',
                label='YAFF',
                marker='.',
                markersize=10,
                markeredgecolor='k',
                markerfacecolor='b',
                )
        ax.plot(
                rcuts / molmod.units.angstrom,
                energies_mm,
                color='r',
                linewidth=0.8,
                linestyle='--',
                label='OpenMM',
                marker='.',
                markersize=10,
                markeredgecolor='k',
                markerfacecolor='r',
                )
        ax.set_xlabel('Distance [A]')
        ax.set_ylabel('Energy [kJ/mol]')
        ax.grid()
        ax.legend()
        ax.yaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        ax.get_yaxis().set_tick_params(which='both', direction='in')
        ax.get_xaxis().set_tick_params(which='both', direction='in')
        fig.savefig('rcut.pdf', bbox_inches='tight')


class NPTTest(Test):
    """Validates NPT simulations"""
    tname = 'npt'

    def __init__(self, *args, **kwargs):
        Test.__init__(self, *args, **kwargs)


def get_test(args):
    """Returns the appropriate ``Test`` object"""
    test_cls = None
    for cls in list(globals().values()):
        if hasattr(cls, 'tname'):
            pre = 'test-'
            name = args.mode[len(pre):]
            if name == cls.tname:
                test_cls = cls
    assert(test_cls is not None)
    test = test_cls(
            args.system,
            args.platform,
            use_max_rcut=args.max_rcut,
            largest_error=args.largest_error,
            )
    return test
