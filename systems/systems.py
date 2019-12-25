import os

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
    assert('path_pars' in kwargs.keys())
    assert('path_chk' in kwargs.keys())
    assert('name' in kwargs.keys())
    return kwargs


## MIL53 ##
info = _create_dict(
        supercell=[3, 4, 6],
        rcut=12,
        tr=None,
        name='mil53',
        path_pars=os.path.join(PATH, 'mil53', 'pars.txt'),
        path_chk=os.path.join(PATH, 'mil53', 'init.chk'),
        )
test_systems.append(dict(info))


## COF5 ##
info = _create_dict(
        supercell=[2, 1, 4],
        rcut=10,
        tr=None,
        name='cof5',
        path_pars=os.path.join(PATH, 'cof5', 'pars.txt'),
        path_chk=os.path.join(PATH, 'cof5', 'init.chk'),
        )
test_systems.append(dict(info))
