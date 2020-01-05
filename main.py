import os
import yaff
import argparse

from systems.systems import test_systems
from src.test import get_test
from src.utils import plot_switching_functions

yaff.log.set_level(yaff.log.silent)


def main(args):
    if args.mode.startswith('test-'):
        test = get_test(args)
        test(steps=500000, writer_step=1000, T=300, P=-200e6)
    elif args.mode == 'serialize':
        raise NotImplementedError
    elif args.mode == 'calibrate-PME':
        raise NotImplementedError
    elif args.mode == 'plot-switch':
        plot_switching_functions(
                rcut=10,
                rswitch=6,
                width=4,
                )
    else:
        raise NotImplementedError


if __name__ == '__main__':
    """
    This is a suite of tests designed to validate the OpenMM implementation of YAFF system
    and force field objects.

    Arguments
    ---------
        system: (string)
            name of the test system (must be defined in ./systems/systems.py)
        mode: (string)
            mode of operation
                'test-single': validates the entire potential energy and forces
                'test-verlet': performs a short (200 steps) verlet integration run and
                          compares energy and forces over entire trajectory
                'test-virial': computes a numerical approximation to the virial tensor and
                        compares with the analytical result from YAFF.
                'serialize': save the OpenMM system object as .xml file, for later reuse.
                'calibrate-PME': uses a plugin for OpenMM to calibrate the PME parameters for
                    optimal overall accuracy.
        platform: (string)
            OpenMM platform to validate
                'Reference' (most accurate)
                'CPU' (contains optimizations for nonbonded interactions)
                'CUDA' (fastest; requires CUDA drivers to be installed)

    OPTIONAL Arguments
    ------------------
        --max_rcut (-r)
            overrides the specified rcut with the maximum possible rcut, given
            the current simulation cell
        --largest_error (-e)
            loads the chk specified by info.path_errorchk, saved by a previous
            verlet test.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('system', action='store')
    parser.add_argument('mode', action='store')
    parser.add_argument('platform', action='store')
    parser.add_argument('-r', '--max_rcut', action='store_true')
    parser.add_argument('-e', '--largest_error', action='store_true')
    args = parser.parse_args()
    main(args)
