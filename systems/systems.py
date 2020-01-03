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
        supercell=[3, 2, 7],
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


## MOF808 ##
info = _create_dict(
        supercell=[3, 2, 1],
        rcut=5 * molmod.units.angstrom,
        #tr=Switch3(7.558904535685008),
        tr=None,
        name='mof808',
        reci_ei='ewald',
        path_pars=os.path.join(PATH, 'mof808', 'pars.txt'),
        path_chk=os.path.join(PATH, 'mof808', 'init.chk'),
        path_errorchk=os.path.join(PATH, 'mof808', 'largest_e.chk'),
        )
test_systems.append(dict(info))


## UIO66 ##
info = _create_dict(
        supercell=[1, 1, 1],
        rcut=10 * molmod.units.angstrom,
        #tr=Switch3(7.558904535685008),
        tr=None,
        name='uio66',
        reci_ei='ewald',
        path_pars=os.path.join(PATH, 'uio66', 'pars.txt'),
        path_chk=os.path.join(PATH, 'uio66', 'init.chk'),
        path_errorchk=os.path.join(PATH, 'uio66', 'largest_e.chk'),
        )
test_systems.append(dict(info))


## CAU13 ##
info = _create_dict(
        supercell=[3, 4, 3],
        rcut=12 * molmod.units.angstrom,
        #tr=Switch3(7.558904535685008),
        tr=None,
        name='cau13',
        reci_ei='ewald',
        path_pars=os.path.join(PATH, 'cau13', 'pars.txt'),
        path_chk=os.path.join(PATH, 'cau13', 'init.chk'),
        path_errorchk=os.path.join(PATH, 'cau13', 'largest_e.chk'),
        )
test_systems.append(dict(info))


## PPY-COF ##
info = _create_dict(
        supercell=[2, 2, 7],
        rcut=5 * molmod.units.angstrom,
        #tr=Switch3(7.558904535685008),
        tr=None,
        name='ppy-cof',
        reci_ei='ewald',
        path_pars=os.path.join(PATH, 'ppy-cof', 'pars.txt'),
        path_chk=os.path.join(PATH, 'ppy-cof', 'init.chk'),
        path_errorchk=os.path.join(PATH, 'ppy-cof', 'largest_e.chk'),
        )
test_systems.append(dict(info))
