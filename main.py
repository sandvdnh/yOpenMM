import os
import yaff
import argparse

from systems.systems import test_systems
from src.test import get_test

yaff.log.set_level(yaff.log.silent)

def main(args):
    test = get_test(args)
    test()


if __name__ == '__main__':
    """
    Arguments
    ---------
        system: (string)
            name of the test system (must be defined in ./systems/systems.py)
        test: (string)
            type of test
                'single': validates the entire potential energy and forces
                'verlet'      : performs a short (200 steps) verlet integration run and
                                compares energy and forces over entire trajectory
        platform: (string)
            OpenMM platform to validate
                'Reference'
                'CPU'
                'CUDA' (requires CUDA drivers to be installed)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('system', action='store')
    parser.add_argument('test', action='store')
    parser.add_argument('platform', action='store')
    parser.add_argument('-r', '--max_rcut', action='store_true')
    args = parser.parse_args()
    main(args)
