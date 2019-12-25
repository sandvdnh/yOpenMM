import yaff

from attrdict import AttrDict
from src.utils import _align, _check_rvecs
from src.generator import AVAILABLE_PREFIXES
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
        self.system = yaff.System.from_file(info.path_chk)
        self.parameters = yaff.Parameters.from_file(info.path_pars)
        self.platform = platform
        if not use_max_rcut:
            self.rcut = info.rcut
        else:
            self.rcut = None

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
    pass


class VerletTest(Test):
    """Compares energy and forces over a short trajectory obtained through Verlet integration"""
    tname = 'verlet'
    pass


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
