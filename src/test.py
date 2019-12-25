import yaff
import molmod

import simtk.unit as unit
import simtk.openmm as mm
import simtk.openmm.app

from attrdict import AttrDict
from src.utils import _align, _check_rvecs, _init_openmm_system
from src.generator import AVAILABLE_PREFIXES, FFArgs, apply_generators, apply_generators_mm
from systems.systems import test_systems


class Test(object):
    """Base class to perform a test between OpenMM and YAFF"""
    tname = None

    def __init__(self, name, platform, use_max_rcut=False):
        info = None
        for _ in test_systems:
            if _['name'] == name:
                info = AttrDict(_)
        assert(info is not None)
        _ = yaff.System.from_file(info.path_chk)
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
            assert(self.rcut < max_rcut)
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
        mm_system = _init_openmm_system(self.system)
        ff_args = FFArgs(
                rcut=self.rcut,
                tr=self.tr,
                alpha_scale=self.alpha_scale,
                gcut_scale=self.gcut_scale,
                )
        apply_generators(self.system, parameters, ff_args)
        ff = yaff.ForceField(self.system, ff_args.parts, ff_args.nlist)
        e = ff.compute() / molmod.units.kjmol
        ff_args_ = yaff.pes.generator.FFArgs(
                rcut=self.rcut,
                tr=self.tr,
                alpha_scale=self.alpha_scale,
                gcut_scale=self.gcut_scale,
                )
        yaff.pes.generator.apply_generators(self.system, parameters, ff_args_)
        ff = yaff.ForceField(self.system, ff_args_.parts, ff_args_.nlist)
        e_ = ff.compute() / molmod.units.kjmol
        assert(e == e_)
        # OPENMM
        mm_system = _init_openmm_system(self.system)
        ff_args = FFArgs(
                rcut=self.rcut,
                tr=self.tr,
                alpha_scale=self.alpha_scale,
                gcut_scale=self.gcut_scale,
                )
        apply_generators_mm(self.system, parameters, ff_args, mm_system)
        integrator = mm.VerletIntegrator(0.5 * unit.femtosecond)
        platform = mm.Platform.getPlatformByName(self.platform)
        context = mm.Context(mm_system, integrator, platform)
        if platform.getName() == 'CUDA':
            platform.setPropertyValue(context, "CudaPrecision", 'mixed')
        context.setPositions(self.system.pos / molmod.units.nanometer * unit.nanometer)
        state = context.getState(
                getPositions=True,
                getForces=True,
                getEnergy=True,
                )
        mm_e = state.getPotentialEnergy()
        return e, mm_e

    def _internal_test(self):
        print(' ' * 11 + '\t{:20}'.format('(YAFF)') + '\t{:20}'.format('(OpenMM)'))
        for prefix, section in self.parameters.sections.items():
            e, mm_e = self._section_test(prefix, section)
            print('{:10}\t{:20}\t{:20}'.format(prefix, str(e), str(mm_e)[:-6]))
        return e


class VerletTest(Test):
    """Compares energy and forces over a short trajectory obtained through Verlet integration"""
    tname = 'verlet'


def get_test(args):
    """Returns the appropriate ``Test`` object"""
    test_cls = None
    for cls in list(globals().values()):
        if hasattr(cls, 'tname'):
            if args.test == cls.tname:
                test_cls = cls
    assert(test_cls is not None)
    test = test_cls(args.system, args.platform, use_max_rcut=args.max_rcut)
    return test
