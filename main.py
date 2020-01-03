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
        test()
        #test.yaff_test_virial(component=(0, 0))
    elif args.mode == 'serialize':
        raise NotImplementedError
    elif args.mode == 'calibrate-PME':
        raise NotImplementedError
    elif args.mode == 'switch-tail':
        rcut = 13
        rswitch = 10.5
        width = 2.5
        plot_switching_functions(rcut, rswitch, width)


if __name__ == '__main__':
    """
    Arguments
    ---------
        system: (string)
            name of the test system (must be defined in ./systems/systems.py)
        mode: (string)
            mode of operation
                'test-single': validates the entire potential energy and forces
                'test-verlet': performs a short (200 steps) verlet integration run and
                          compares energy and forces over entire trajectory
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
        --tail (-t)
            whether or not to include tail corrections.
        --use_switching (-s)
            whether or not to include a switching function for dispersion interactions.
            Since OpenMM and YAFF use different switching functions, this invalidates
            the comparison of dispersion energies.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('system', action='store')
    parser.add_argument('mode', action='store')
    parser.add_argument('platform', action='store')
    parser.add_argument('-r', '--max_rcut', action='store_true')
    parser.add_argument('-e', '--largest_error', action='store_true')
    parser.add_argument('-t', '--use_tail', action='store_true')
    parser.add_argument('-s', '--use_switching', action='store_true')
    args = parser.parse_args()
    main(args)
