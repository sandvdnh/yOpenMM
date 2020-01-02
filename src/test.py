import yaff
import molmod

import numpy as np
import simtk.unit as unit
import simtk.openmm as mm
import simtk.openmm.app

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

    def pre(self):
        """Performs a number of checks before executing the test (to save time)

        - asserts whether all prefixes in the parameter file are supported in the
          generator
        - aligns cell, checks OpenMM requirements on cell vector geometry
        - checks whether rcut is not larger than half the length of the shortest
          cell vector
        """
        for prefix, _ in self.parameters.sections.items():
            assert(prefix in AVAILABLE_PREFIXES)
        _align(self.system)
        max_rcut = _check_rvecs(self.system.cell._get_rvecs())
        if self.rcut is not None:
            assert self.rcut < max_rcut, 'chosen cutoff too large: {:.3f} > {:.3f}'.format(self.rcut, max_rcut)
        else:
            self.rcut = 0.99 * max_rcut

    def report(self):
        print('{}; {} supercell; {} atoms'.format(self.name.upper(), self.supercell, self.system.natom))
        print('')
        print('CUTOFF: {:.5f} angstrom'.format(self.rcut / molmod.units.angstrom))
        print(self.system.cell._get_rvecs() / molmod.units.angstrom, 'angstrom')
        print('')
        pass

    def _internal_test(self):
        raise NotImplementedError

    def __call__(self):
        self.pre()
        self.report()
        self._internal_test()


class SinglePoint(Test):
    """Compares energy and forces for a single state"""
    tname = 'single'

    def _section_test(self, prefix, section):
        parameters = yaff.pes.parameters.Parameters({prefix: section})
        ff_args = FFArgs(
                rcut=self.rcut,
                tr=self.tr,
                alpha_scale=self.alpha_scale,
                gcut_scale=self.gcut_scale,
                reci_ei=self.reci_ei,
                )
        apply_generators(self.system, parameters, ff_args)
        ff = yaff.ForceField(self.system, ff_args.parts, ff_args.nlist)
        e = ff.compute() / molmod.units.kjmol
        ff_args_ = yaff.pes.generator.FFArgs(
                rcut=self.rcut,
                tr=self.tr,
                alpha_scale=self.alpha_scale,
                gcut_scale=self.gcut_scale,
                reci_ei=self.reci_ei,
                )
        yaff.pes.generator.apply_generators(self.system, parameters, ff_args_)
        ff = yaff.ForceField(self.system, ff_args_.parts, ff_args_.nlist)
        #ff.compute()
        #for part in ff.parts:
        #    print(part.name, part.compute() / molmod.units.kjmol)
        f = np.zeros(self.system.pos.shape)
        e_ = ff.compute(f, None) / molmod.units.kjmol
        f *= molmod.units.nanometer / molmod.units.kjmol
        f *= -1.0 # gpos == -force
        #ff__ = yaff.ForceField.generate(self.system, parameters)
        #e__ = ff__.compute()
        assert(e == e_)
        # OPENMM
        mm_system = _init_openmm_system(self.system)
        ff_args = FFArgs(
                rcut=self.rcut,
                tr=self.tr,
                alpha_scale=self.alpha_scale,
                gcut_scale=self.gcut_scale,
                reci_ei=self.reci_ei,
                )
        apply_generators_mm(self.system, parameters, ff_args, mm_system)
        integrator = mm.VerletIntegrator(0.5 * unit.femtosecond)
        platform = mm.Platform.getPlatformByName(self.platform)
        context = mm.Context(mm_system, integrator, platform)
        if platform.getName() == 'CUDA':
            platform.setPropertyValue(context, "CudaPrecision", 'single')
        context.setPositions(self.system.pos / molmod.units.nanometer * unit.nanometer)
        state = context.getState(
                getPositions=True,
                getForces=True,
                getEnergy=True,
                )
        mm_e = state.getPotentialEnergy()
        mm_f = state.getForces(asNumpy=True)
        return e, mm_e.value_in_unit(mm_e.unit), f, mm_f.value_in_unit(mm_f.unit)

    def _internal_test(self):
        print(' ' * 11 + '\t{:20}'.format('(YAFF) kJ/mol') + '\t{:20}'.format('(OpenMM) kJ/mol') + '\t{:20}'.format('force MAE (kJ/(mol * nm))'))
        for prefix, section in self.parameters.sections.items():
            e, mm_e, f, mm_f = self._section_test(prefix, section)
            mae = np.mean(np.abs(f - mm_f))
            print('{:10}\t{:20}\t{:20}\t{:20}'.format(prefix, str(e), str(mm_e), str(mae)))
            #if prefix == 'MM3':
            #    err = np.abs(f - mm_f)
            #    for i in range(err.shape[0]):
            #        print(i, err[i])
        return e


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
        ff_args = FFArgs(
                rcut=self.rcut,
                tr=self.tr,
                alpha_scale=self.alpha_scale,
                gcut_scale=self.gcut_scale,
                reci_ei=self.reci_ei,
                )
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
        ff_args = FFArgs(
                rcut=self.rcut,
                tr=self.tr,
                alpha_scale=self.alpha_scale,
                gcut_scale=self.gcut_scale,
                reci_ei=self.reci_ei,
                )
        apply_generators(self.system, self.parameters, ff_args)
        ff = yaff.ForceField(self.system, ff_args.parts, ff_args.nlist)
        for i in range(positions.shape[0]):
            ff.update_pos(positions[i, :] * molmod.units.nanometer)
            energies[i] = ff.compute(forces[i], None)
        forces *= molmod.units.nanometer / molmod.units.kjmol
        forces *= -1.0 # gpos == -force
        energies /= molmod.units.kjmol
        return energies, forces

    def _internal_test(self):
        steps = 100
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
            largest_error=args.largest_error)
    return test
