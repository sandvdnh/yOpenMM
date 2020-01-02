import os
import molmod

from yaff.pes.ext import Switch3

"""
supercell: the super cell to use for the validation
rcut: cutoff distance of all nonbonded interactions (in Angstrom)
tr: switching function for dispersion interactions (default: None)
path_pars: path to YAFF Parameter .txt file
path_chk: path to .chk file
name: name of the system
"""

test_systems = []
PATH = os.path.join(
        os.getcwd(),
        'systems',
        )

def _create_dict(**kwargs):
    assert('supercell' in kwargs.keys())
    assert('rcut' in kwargs.keys())
    assert('tr' in kwargs.keys())
    assert('reci_ei' in kwargs.keys())
    assert('name' in kwargs.keys())
    assert('path_pars' in kwargs.keys())
    assert('path_chk' in kwargs.keys())
    return kwargs


## MIL53 ##
info = _create_dict(
        supercell=[1, 2, 4],
        rcut=9 * molmod.units.angstrom,
        tr=None,
        name='mil53',
        reci_ei='ewald',
        path_pars=os.path.join(PATH, 'mil53', 'pars.txt'),
        path_chk=os.path.join(PATH, 'mil53', 'init.chk'),
        path_errorchk=os.path.join(PATH, 'mil53', 'largest_e.chk'),
        )
test_systems.append(dict(info))


## COF5 ##
info = _create_dict(
        supercell=[3, 2, 9],
        rcut=12 * molmod.units.angstrom,
        #tr=Switch3(7.558904535685008),
        tr=None,
        name='cof5',
        reci_ei='ewald',
        path_pars=os.path.join(PATH, 'cof5', 'pars.txt'),
        path_chk=os.path.join(PATH, 'cof5', 'init.chk'),
        path_errorchk=os.path.join(PATH, 'cof5', 'largest_e.chk'),
        )
test_systems.append(dict(info))
