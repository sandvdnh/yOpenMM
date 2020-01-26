import os
import molmod

from yaff.pes.ext import Switch3

"""
supercell: the super cell to use for the validation
rcut: cutoff distance of all nonbonded interactions
tr: width of the switching distance for dispersion interactions
tailcorrections: boolean that indicates whether to use tail corrections for dispersion interactions
reci_ei: string that indicates the method to evaluate the reciprocal sum of electrostatic interactions in YAFF
name: string representing the name of the system
path_pars: path to YAFF Parameter .txt file
path_chk: path to .chk file
path_errorchk (optional): path that is used to write system files for which forces contain the largest error
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
    assert('tailcorrections' in kwargs.keys())
    assert('reci_ei' in kwargs.keys())
    assert('name' in kwargs.keys())
    assert('path_pars' in kwargs.keys())
    assert('path_chk' in kwargs.keys())
    return kwargs

tr0 = 7.558904535685008
tr0 = 4.0 * molmod.units.angstrom

## MIL53 ##
info = _create_dict(
        supercell=[2, 2, 4],
        rcut=12 * molmod.units.angstrom,
        tr=4 * molmod.units.angstrom,
        tailcorrections=True,
        reci_ei='ewald',
        name='mil53',
        path_pars=os.path.join(PATH, 'mil53', 'pars.txt'),
        path_chk=os.path.join(PATH, 'mil53', 'init.chk'),
        path_errorchk=os.path.join(PATH, 'mil53', 'largest_e.chk'),
        )
test_systems.append(dict(info))


## COF5 ##
info = _create_dict(
        supercell=[3, 2, 6],
        rcut=10 * molmod.units.angstrom,
        tr=4 * molmod.units.angstrom,
        tailcorrections=False,
        reci_ei='ewald',
        name='cof5',
        path_pars=os.path.join(PATH, 'cof5', 'pars.txt'),
        path_chk=os.path.join(PATH, 'cof5', 'init.chk'),
        path_errorchk=os.path.join(PATH, 'cof5', 'largest_e.chk'),
        )
test_systems.append(dict(info))


## MOF808 ##
info = _create_dict(
        supercell=[3, 2, 1],
        rcut=6.995 * molmod.units.angstrom, ## becomes incorrect between 6.9905 and 6.9907 A
        tr=None, #2 * molmod.units.angstrom,
        tailcorrections=False,
        reci_ei='ewald',
        name='mof808',
        path_pars=os.path.join(PATH, 'mof808', 'pars.txt'),
        path_chk=os.path.join(PATH, 'mof808', 'init.chk'),
        path_errorchk=os.path.join(PATH, 'mof808', 'largest_e.chk'),
        )
test_systems.append(dict(info))


## UIO66 ##
info = _create_dict(
        supercell=[1, 1, 1],
        rcut=10 * molmod.units.angstrom,
        tr=4 * molmod.units.angstrom,
        tailcorrections=True,
        reci_ei='ewald',
        name='uio66',
        path_pars=os.path.join(PATH, 'uio66', 'pars.txt'),
        path_chk=os.path.join(PATH, 'uio66', 'init.chk'),
        path_errorchk=os.path.join(PATH, 'uio66', 'largest_e.chk'),
        )
test_systems.append(dict(info))


## CAU13 ##
info = _create_dict(
        supercell=[3, 4, 3],
        rcut=5 * molmod.units.angstrom,
        tr=None,
        tailcorrections=False,
        reci_ei='ewald',
        name='cau13',
        path_pars=os.path.join(PATH, 'cau13', 'pars.txt'),
        path_chk=os.path.join(PATH, 'cau13', 'init.chk'),
        path_errorchk=os.path.join(PATH, 'cau13', 'largest_e.chk'),
        )
test_systems.append(dict(info))


## PPY-COF ##
info = _create_dict(
        supercell=[2, 2, 7],
        rcut=5 * molmod.units.angstrom,
        tr=None,
        tailcorrections=False,
        reci_ei='ewald',
        name='ppy-cof',
        path_pars=os.path.join(PATH, 'ppy-cof', 'pars.txt'),
        path_chk=os.path.join(PATH, 'ppy-cof', 'init.chk'),
        path_errorchk=os.path.join(PATH, 'ppy-cof', 'largest_e.chk'),
        )
test_systems.append(dict(info))


## 4PE-2P ##
info = _create_dict(
        supercell=[2, 1, 10],
        rcut=15 * molmod.units.angstrom,
        tr=4 * molmod.units.angstrom,
        tailcorrections=False,
        reci_ei='ewald',
        name='4pe-2p',
        path_pars=os.path.join(PATH, '4pe-2p', 'pars.txt'),
        path_chk=os.path.join(PATH, '4pe-2p', 'init.chk'),
        path_errorchk=os.path.join(PATH, '4pe-2p', 'largest_e.chk'),
        )
test_systems.append(dict(info))


## MIL53 system for barostat ##
info = _create_dict(
        supercell=[1, 1, 1],
        rcut=2 * molmod.units.angstrom,
        tr=4 * molmod.units.angstrom,
        tailcorrections=True,
        reci_ei='ewald',
        name='mil53baro',
        path_pars=os.path.join(PATH, 'mil53', 'pars.txt'),
        path_chk=os.path.join(PATH, 'mil53', 'init.chk'),
        path_errorchk=os.path.join(PATH, 'mil53', 'largest_e.chk'),
        )
test_systems.append(dict(info))


## CoBDP system for barostat ##
info = _create_dict(
        supercell=[1, 1, 1],
        rcut=2 * molmod.units.angstrom,
        tr=4 * molmod.units.angstrom,
        tailcorrections=True,
        reci_ei='ewald',
        name='cobdpbaro',
        path_pars=os.path.join(PATH, 'cobdp', 'pars.txt'),
        path_chk=os.path.join(PATH, 'cobdp', 'init.chk'),
        path_errorchk=os.path.join(PATH, 'cobdp', 'largest_e.chk'),
        )
test_systems.append(dict(info))
