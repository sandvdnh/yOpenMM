import numpy as np
import molmod

from molmod.units import parse_unit, kjmol, angstrom

from itertools import permutations

from yaff.log import log
from yaff.pes.ext import Cell, PairPotEI, PairPotLJ, PairPotMM3, PairPotMM3CAP, PairPotExpRep, \
    PairPotQMDFFRep, PairPotDampDisp, PairPotDisp68BJDamp, Switch3, PairPotEIDip
from yaff.pes.ff import ForcePartPair, ForcePartValence, \
    ForcePartEwaldReciprocal, ForcePartEwaldCorrection, \
    ForcePartEwaldNeutralizing, ForcePartTailCorrection
from yaff.pes.iclist import Bond, BendAngle, DihedAngle, DihedCos, OopDist
from yaff.pes.vlist import Harmonic, Cosine, Chebychev1, Chebychev2, Chebychev3, Chebychev4, Chebychev6, \
        Cross
from yaff.pes.nlist import NeighborList
from yaff.pes.scaling import Scalings
from yaff.pes.parameters import Parameters
from yaff.pes.comlist import COMList
from yaff.system import System

import simtk.unit as unit
import simtk.openmm as mm



class FFArgs(object):
    '''Data structure that holds all arguments for the ForceField constructor

       The attributes of this object are gradually filled up by the various
       generators based on the data in the ParsedPars object.
    '''
    def __init__(self, rcut=18.89726133921252, tr=Switch3(7.558904535685008),
                 alpha_scale=3.5, gcut_scale=1.1, skin=0, smooth_ei=False,
                 reci_ei='ewald', nlow=0, nhigh=-1, tailcorrections=False):
        """
           **Optional arguments:**

           Some optional arguments only make sense if related parameters in the
           parameter file are present.

           rcut
                The real space cutoff used by all pair potentials.

           tr
                Default truncation model for everything except the electrostatic
                interactions. The electrostatic interactions are not truncated
                by default.

           alpha_scale
                Determines the alpha parameter in the Ewald summation based on
                the real-space cutoff: alpha = alpha_scale / rcut. Higher
                values for this parameter imply a faster convergence of the
                reciprocal terms, but a slower convergence in real-space.

           gcut_scale
                Determines the reciprocale space cutoff based on the alpha
                parameter: gcut = gcut_scale * alpha. Higher values for this
                parameter imply a better convergence in the reciprocal space.

           skin
                The skin parameter for the neighborlist.

           smooth_ei
                Flag for smooth truncations for the electrostatic interactions.

           reci_ei
                The method to be used for the reciprocal contribution to the
                electrostatic interactions in the case of periodic systems. This
                must be one of 'ignore' or 'ewald' or 'ewald_interaction'.
                The options starting with 'ewald' are only supported for 3D
                periodic systems. If 'ewald_interaction' is chosen, the
                reciprocal contribution will not be included and it should be
                accounted for by using the :class:`EwaldReciprocalInteraction`

           nlow
                Interactions between atom pairs are only included if at least
                one atom index is higher than or equal to nlow. The default
                nlow=0 means no exclusion. Valence terms involving atoms with
                index lower than or equal to nlow will not be included.

           nhigh
                Interactions between atom pairs are only included if at least
                one atom index is smaller than nhigh. The default nhigh=-1
                means no exclusion. Valence terms involving atoms with index
                higher than nhigh will not be included.
                If nlow=nhigh, the system is divided into two parts and only
                pairs involving one atom of each part will be included. This is
                useful to calculate interaction energies in Monte Carlo
                simulations

           tailcorrections
                Boolean: if true, apply a correction for the truncation of the
                pair potentials assuming the system is homogeneous in the
                region where the truncation modifies the pair potential

           The actual value of gcut, which depends on both gcut_scale and
           alpha_scale, determines the computational cost of the reciprocal term
           in the Ewald summation. The default values are just examples. An
           optimal trade-off between accuracy and computational cost requires
           some tuning. Dimensionless scaling parameters are used to make sure
           that the numerical errors do not depend too much on the real space
           cutoff and the system size.
        """
        if reci_ei not in ['ignore', 'ewald', 'ewald_interaction']:
            raise ValueError('The reci_ei option must be one of \'ignore\' or \'ewald\' or \'ewald_interaction\'.')
        self.rcut = rcut
        self.tr = tr
        self.alpha_scale = alpha_scale
        self.gcut_scale = gcut_scale
        self.skin = skin
        self.smooth_ei = smooth_ei
        self.reci_ei = reci_ei
        # arguments for the ForceField constructor
        self.parts = []
        self.nlist = None
        self.nlow = nlow
        self.nhigh = nhigh
        self.tailcorrections = tailcorrections

    def get_nlist(self, system):
        if self.nlist is None:
            self.nlist = NeighborList(system, skin=self.skin, nlow=self.nlow,
                            nhigh=self.nhigh)
        return self.nlist

    def get_part(self, ForcePartClass):
        for part in self.parts:
            if isinstance(part, ForcePartClass):
                return part

    def get_part_pair(self, PairPotClass):
        for part in self.parts:
            if isinstance(part, ForcePartPair) and isinstance(part.pair_pot, PairPotClass):
                return part

    def get_part_valence(self, system):
        part_valence = self.get_part(ForcePartValence)
        if part_valence is None:
            part_valence = ForcePartValence(system)
            self.parts.append(part_valence)
        return part_valence

    def add_electrostatic_parts(self, system, scalings, dielectric):
        if self.get_part_pair(PairPotEI) is not None:
            return
        nlist = self.get_nlist(system)
        if system.cell.nvec == 0:
            alpha = 0.0
        elif system.cell.nvec == 3:
            #TODO: the choice of alpha should depend on the radii of the
            #charge distributions. Following expression is OK for point charges.
            alpha = self.alpha_scale/self.rcut
        else:
            raise NotImplementedError('Only zero- and three-dimensional electrostatics are supported.')
        # Real-space electrostatics
        if self.smooth_ei:
            pair_pot_ei = PairPotEI(system.charges, alpha, self.rcut, self.tr, dielectric, system.radii)
        else:
            pair_pot_ei = PairPotEI(system.charges, alpha, self.rcut, None, dielectric, system.radii)
        part_pair_ei = ForcePartPair(system, nlist, scalings, pair_pot_ei)
        self.parts.append(part_pair_ei)
        if self.reci_ei == 'ignore':
            # Nothing to do
            pass
        elif self.reci_ei.startswith('ewald'):
            if system.cell.nvec == 3:
                if self.reci_ei == 'ewald_interaction':
                    part_ewald_reci = ForcePartEwaldReciprocalInteraction(system.cell, alpha, self.gcut_scale*alpha, dielectric=dielectric)
                elif self.reci_ei == 'ewald':
                    # Reciprocal-space electrostatics
                    part_ewald_reci = ForcePartEwaldReciprocal(system, alpha, self.gcut_scale*alpha, dielectric, self.nlow, self.nhigh)
                else: raise NotImplementedError
                self.parts.append(part_ewald_reci)
                # Ewald corrections
                part_ewald_corr = ForcePartEwaldCorrection(system, alpha, scalings, dielectric, self.nlow, self.nhigh)
                self.parts.append(part_ewald_corr)
                # Neutralizing background
                part_ewald_neut = ForcePartEwaldNeutralizing(system, alpha, dielectric, self.nlow, self.nhigh)
                self.parts.append(part_ewald_neut)
            elif system.cell.nvec != 0:
                raise NotImplementedError('The ewald summation is only available for 3D periodic systems.')
        else:
            raise NotImplementedError


class Generator(object):
    """Creates (part of a) ForceField object automatically.

       A generator is a class that describes how a part of a parameter file
       must be turned into a part of ForceField object. As the generator
       proceeds, it will modify and extend the current arguments of the FF. They
       should be implemented such that the order of the generators is not
       important.

       **Important class attributes:**

       prefix
            The prefix string that must match the prefix in the parameter file.
            If this is None, it is assumed that the Generator class is abstract.
            In that case it will be ignored by the apply_generators function
            at the bottom of this module.

       par_info
            A description of the parameters on a single line (PARS suffix)

       suffixes
            The supported suffixes

       allow_superpositions
            Whether multiple PARS lines with the same atom types are allowed.
            This is rarely the case, except for the TORSIONS and a few other
            weirdos.
    """
    prefix = None
    par_info = None
    suffixes = None
    allow_superposition = False

    def __call__(self, system, parsec, ff_args):
        '''Add contributions to the force field from this generator

           **Arguments:**

           system
                The System object for which a force field is being prepared

           parse
                An instance of the ParameterSection class

           ff_ars
                An instance of the FFargs class
        '''
        raise NotImplementedError

    def check_suffixes(self, parsec):
        for suffix in parsec.definitions:
            if suffix not in self.suffixes:
                parsec.complain(None, 'contains a suffix (%s) that is not recognized by generator %s.' % (suffix, self.prefix))

    def process_units(self, pardef):
        '''Load parameter conversion information

           **Arguments:**

           pardef
                An instance of the ParameterDefinition class.

           Returns a dictionary with (name, converion) pairs.
        '''
        result = {}
        expected_names = [name for name, dtype in self.par_info if dtype is float]
        for counter, line in pardef:
            words = line.split()
            if len(words) != 2:
                pardef.complain(counter, 'must have two arguments in UNIT suffix')
            name = words[0].upper()
            if name not in expected_names:
                pardef.complain(counter, 'specifies a unit for an unknown parameter. (Must be one of %s, but got %s.)' % (expected_names, name))
            try:
                result[name] = parse_unit(words[1])
            except (NameError, ValueError):
                pardef.complain(counter, 'has a UNIT suffix with an unknown unit')
        if len(result) != len(expected_names):
            raise IOError('Not all units are specified for generator %s in file %s. Got %s, should have %s.' % (
                self.prefix, pardef.complain.filename, list(result.keys()), expected_names
            ))
        return result

    def process_pars(self, pardef, conversions, nffatype, par_info=None):
        '''Load parameter and apply conversion factors

           **Arguments:**

           pardef
                An instance of the ParameterDefinition class.

           conversions
                A dictionary with (name, conversion) items.

           nffatype
                The number of ffatypes per line of parameters.

           **Optional arguments:**

           par_info
                A custom description of the parameters. If not present,
                self.par_info is used. This is convenient when this method
                is used to parse other definitions than PARS.
        '''
        if par_info is None:
            # Parsing PARS
            par_info = self.par_info
            allow_superposition = self.allow_superposition
        else:
            # Parsing other fields than PARS, so supperposition should never be allowed.
            allow_superposition = False

        # Generate a parameter table (dictionary):
        # key:
        #   Tuple of ffatype strings.
        # values:
        #   List of tuples of corresponding parameters, with multiple items only allowed
        #   in case of a superposition of energy terms of the same type with different
        #   parameters.
        par_table = {}
        for counter, line in pardef:
            words = line.split()
            num_args = nffatype + len(par_info)
            if len(words) != num_args:
                pardef.complain(counter, 'should have %s arguments' % num_args)
            # Extract the key
            key = tuple(words[:nffatype])
            # Extract the parameters
            pars = []
            for i, (name, dtype) in enumerate(par_info):
                word = words[i + nffatype]
                try:
                    if issubclass(dtype, float):
                        pars.append(float(word)*conversions[name])
                    else:
                        pars.append(dtype(word))
                except ValueError:
                    pardef.complain(counter, 'contains a parameter that can not be converted to a number: {}'.format(word))
            pars = tuple(pars)

            # Process the new key + pars pair, taking into account equivalent permutations
            # of the atom types and corresponding permutations of parameters.
            current_par_table = {}
            for alt_key, alt_pars in self.iter_equiv_keys_and_pars(key, pars):
                # When permuted keys are identical to the original, no new items are
                # added.
                if alt_key in current_par_table:
                    if current_par_table[alt_key] != alt_pars:
                        pardef.complain(counter, 'contains parameters that are not consistent with the permutational symmetry of the atom types')
                else:
                    current_par_table[alt_key] = alt_pars

            # Add the parameters and their permutations to the parameter table, checking
            # for superposition.
            for alt_key, alt_pars in current_par_table.items():
                par_list = par_table.setdefault(alt_key, [])
                if len(par_list) > 0 and not allow_superposition:
                    pardef.complain(counter, 'contains a duplicate energy term, possibly with different parameters, which is not allowed for generator %s' % self.prefix)
                par_list.append(alt_pars)
        return par_table

    def iter_equiv_keys_and_pars(self, key, pars):
        '''Iterates of all equivalent re-orderings of a tuple of ffatypes (keys) and corresponding parameters.'''
        if len(key) == 1:
            yield key, pars
        else:
            raise NotImplementedError


class ValenceGenerator(Generator):
    '''All generators for diagonal valence terms derive from this class.

       **More important attributes:**

       nffatype
            The number of atoms involved in the internal coordinates. Hence
            this is also the number ffatypes in a single row in the force field
            parameter file.

       ICClass
            The ``InternalCoordinate`` class. See ``yaff.pes.iclist``.

       VClass
            The ``ValenceTerm`` class. See ``yaff.pes.vlist``.
    '''

    suffixes = ['UNIT', 'PARS']
    nffatype = None
    ICClass = None
    VClass = None

    def __call__(self, system, parsec, ff_args):
        '''Add contributions to the force field from a ValenceGenerator

           **Arguments:**

           system
                The System object for which a force field is being prepared

           parse
                An instance of the ParameterSection class

           ff_ars
                An instance of the FFargs class
        '''
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, self.nffatype)
        force = self.get_force()
        if len(par_table) > 0:
            self.apply(par_table, system, ff_args, force)
            return [force]


    def apply(self, par_table, system, ff_args, force):
        '''Generate terms for the system based on the par_table

           **Arguments:**

           par_table
                A dictionary with tuples of ffatypes is keys and lists of
                parameters as values.

           system
                The system for which the force field is generated.

           ff_args
                An instance of the FFArgs class.

            force
                OpenMM force object
        '''
        if system.bonds is None:
            raise ValueError('The system must have bonds in order to define valence terms.')
        part_valence = ff_args.get_part_valence(system)
        for indexes in self.iter_indexes(system):
            # We do not want terms where at least one atom index is smaller than
            # nlow, as this is (should be) an excluded interaction
            if min(indexes)<ff_args.nlow:
                # Check that this term indeed features only atoms with index<nlow
                assert max(indexes)<ff_args.nlow
                continue
            # We do not want terms where at least one atom index is higher than
            # or equal to nhigh, as this is (should be) an excluded interaction
            if ff_args.nhigh!=-1 and max(indexes)>=ff_args.nhigh:
                # Check that this term indeed features only atoms with index<nlow
                assert min(indexes)>=ff_args.nhigh
                continue
            key = tuple(system.get_ffatype(i) for i in indexes)
            par_list = par_table.get(key, [])
            for pars in par_list:
                vterm = self.get_vterm(pars, indexes)
                part_valence.add_term(vterm)
                self.add_term_to_force(force, pars, indexes)

    def get_vterm(self, pars, indexes):
        '''Return an instance of the ValenceTerm class with the proper InternalCoordinate instance

           **Arguments:**

           pars
                The parameters for the ValenceTerm class.

           indexes
                The atom indices used to define the internal coordinate
        '''
        args = pars + (self.ICClass(*indexes),)
        return self.VClass(*args)

    def iter_indexes(self, system):
        '''Iterate over all tuples of indices for the internal coordinate'''
        raise NotImplementedError

    def get_force(self):
        """Returns the appropriate force object"""
        raise NotImplementedError

    def add_term_to_force(self, force, pars, indexes):
        """Adds interaction to OpenMM force object"""
        raise NotImplementedError


class BondGenerator(ValenceGenerator):
    par_info = [('K', float), ('R0', float)]
    nffatype = 2
    ICClass = Bond
    VClass = None

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], pars

    def iter_indexes(self, system):
        return system.iter_bonds()


class BondHarmGenerator(BondGenerator):
    prefix = 'BONDHARM'
    VClass = Harmonic

    def get_force(self):
        force = mm.HarmonicBondForce()
        force.setUsesPeriodicBoundaryConditions(True)
        return force

    def add_term_to_force(self, force, pars, indexes):
        conversion = {
                'K': molmod.units.kjmol / molmod.units.angstrom ** 2 / 100,
                'R0': molmod.units.angstrom * 10,
                }
        conversion_mm = {
                'K': unit.kilojoule_per_mole / unit.nanometer ** 2,
                'R0': unit.nanometer,
                }
        K = pars[0] / conversion['K'] * conversion_mm['K']
        R0 = pars[1] / conversion['R0'] * conversion_mm['R0']
        force.addBond(int(indexes[0]), int(indexes[1]), R0, K)


class BendGenerator(ValenceGenerator):
    nffatype = 3
    ICClass = None
    VClass = Harmonic

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], pars

    def iter_indexes(self, system):
        return system.iter_angles()


class BendAngleHarmGenerator(BendGenerator):
    par_info = [('K', float), ('THETA0', float)]
    prefix = 'BENDAHARM'
    ICClass = BendAngle

    def get_force(self):
        force = mm.HarmonicAngleForce()
        force.setUsesPeriodicBoundaryConditions(True)
        return force

    def add_term_to_force(self, force, pars, indexes):
        conversion = {
                'K': molmod.units.kjmol / molmod.units.rad ** 2,
                'THETA0': molmod.units.deg,
                }
        conversion_mm = {
                'K': unit.kilojoule_per_mole / unit.radians ** 2,
                'THETA0': unit.degrees,
                }
        K = pars[0] / conversion['K'] * conversion_mm['K']
        THETA0 = pars[1] / conversion['THETA0'] * conversion_mm['THETA0']
        force.addAngle(
                int(indexes[0]),
                int(indexes[1]),
                int(indexes[2]),
                THETA0,
                K,
                )


class BendCosGenerator(BendGenerator):
    par_info = [('M', int), ('A', float), ('PHI0', float)]
    prefix = 'BENDCOS'
    ICClass = BendAngle
    VClass = Cosine

    def get_force(self):
        energy = 'A/2*(1 - cos(M*(theta - PHI0)))'
        force = mm.CustomAngleForce(energy)
        force.addPerAngleParameter('M')
        force.addPerAngleParameter('A')
        force.addPerAngleParameter('PHI0')
        force.setUsesPeriodicBoundaryConditions(True)
        return force

    def add_term_to_force(self, force, pars, indexes):
        conversion = {
                'A': molmod.units.kjmol,
                'PHI0': molmod.units.deg,
                'M': 1,
                }
        conversion_mm = {
                'A': unit.kilojoule_per_mole,
                'PHI0': unit.degrees,
                'M': unit.dimensionless,
                }
        M = pars[0] / conversion['M'] * conversion_mm['M']
        A = pars[1] / conversion['A'] * conversion_mm['A']
        PHI0 = pars[2] / conversion['PHI0'] * conversion_mm['PHI0']
        force.addAngle(
                int(indexes[0]),
                int(indexes[1]),
                int(indexes[2]),
                [M, A, PHI0],
                )


class OopDistGenerator(ValenceGenerator):
    nffatype = 4
    par_info = [('K', float), ('D0', float)]
    prefix = 'OOPDIST'
    ICClass = OopDist
    VClass = Harmonic
    allow_superposition = False

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield (key[2], key[0], key[1], key[3]), pars
        yield (key[1], key[2], key[0], key[3]), pars
        yield (key[2], key[1], key[0], key[3]), pars
        yield (key[1], key[0], key[2], key[3]), pars
        yield (key[0], key[2], key[1], key[3]), pars

    def iter_indexes(self, system):
        #Loop over all atoms; if an atom has 3 neighbors,
        #it is candidate for an OopDist term
        for atom in system.neighs1.keys():
            neighbours = list(system.neighs1[atom])
            if len(neighbours)==3:
                yield neighbours[0],neighbours[1],neighbours[2],atom

    def get_force(self):
        d0x = '(x2 - x1)'
        d0y = '(y2 - y1)'
        d0z = '(z2 - z1)'
        d1x = '(x3 - x2)'
        d1y = '(y3 - y2)'
        d1z = '(z3 - z2)'
        d2x = '(x4 - x3)'
        d2y = '(y4 - y3)'
        d2z = '(z4 - z3)'
        nx_ = '({} * {} - {} * {})'.format(d0y, d1z, d0z, d1y)
        ny_ = '({} * {} - {} * {})'.format(d0z, d1x, d0x, d1z)
        nz_ = '({} * {} - {} * {})'.format(d0x, d1y, d0y, d1x)
        norm = 'sqrt({0}^2 + {1}^2 + {2}^2)'.format(nx_, ny_, nz_)
        ndotd2 = '({0}*{3} + {1}*{4} + {2}*{5})'.format(nx_, ny_, nz_, d2x, d2y, d2z)
        pre = '(1 - delta({}))'.format(norm)
        dist = '{} * {} / {}'.format(pre, ndotd2, norm)
        energy = '0.5 * K * ({} - D0)^2'.format(dist)
        force = mm.CustomCompoundBondForce(4, energy)
        force.addPerBondParameter('K')
        force.addPerBondParameter('D0')
        force.setUsesPeriodicBoundaryConditions(True)
        return force

    def add_term_to_force(self, force, pars, indexes):
        conversion = {
                'K': molmod.units.kjmol / molmod.units.nanometer ** 2,
                'D0': molmod.units.nanometer,
                }
        conversion_mm = {
                'K': unit.kilojoule_per_mole / unit.nanometer ** 2,
                'D0': unit.nanometer,
                }
        K = pars[0] / conversion['K'] * conversion_mm['K']
        D0 = pars[1] / conversion['D0'] * conversion_mm['D0']
        force.addBond(
                [
                    int(indexes[0]),
                    int(indexes[1]),
                    int(indexes[2]),
                    int(indexes[3])],
                [K, D0],
                )

    def get_force1(self):
        energy = "0.5 * K * (distance(g1, g2) - D0)^2"
        force = mm.CustomCentroidBondForce(2, energy)
        force.addPerBondParameter('K')
        force.addPerBondParameter('D0')
        force.setUsesPeriodicBoundaryConditions(True)
        return force

    def add_term_to_force1(self, force, pars, indexes):
        group1 = [
                int(indexes[0]),
                int(indexes[1]),
                int(indexes[2]),
                ]
        group2 = [
                int(indexes[3]),
                ]
        i1 = force.addGroup(group1)
        i2 = force.addGroup(group2)
        conversion = {
                'K': molmod.units.kjmol / molmod.units.nanometer ** 2,
                'D0': molmod.units.nanometer,
                }
        conversion_mm = {
                'K': unit.kilojoule_per_mole / unit.nanometer ** 2,
                'D0': unit.nanometer,
                }
        K = pars[0] / conversion['K'] * conversion_mm['K']
        D0 = pars[1] / conversion['D0'] * conversion_mm['D0']
        force.addBond(
                [i1, i2],
                [K, D0],
                )


class TorsionGenerator(ValenceGenerator):
    nffatype = 4
    par_info = [('M', int), ('A', float), ('PHI0', float)]
    prefix = 'TORSION'
    ICClass = DihedAngle
    VClass = Cosine
    allow_superposition = True

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], pars

    def iter_indexes(self, system):
        return system.iter_dihedrals()

    def get_vterm(self, pars, indexes):
        # A torsion term with multiplicity m and rest value either 0 or pi/m
        # degrees, can be treated as a polynomial in cos(phi). The code below
        # selects the right polynomial.
        if pars[2] == 0.0 and pars[0] == 1:
            ic = DihedCos(*indexes)
            return Chebychev1(pars[1], ic, sign=-1)
        elif abs(pars[2] - np.pi/1)<1e-6 and pars[0] == 1:
            ic = DihedCos(*indexes)
            return Chebychev1(pars[1], ic, sign=1)
        elif pars[2] == 0.0 and pars[0] == 2:
            ic = DihedCos(*indexes)
            return Chebychev2(pars[1], ic, sign=-1)
        elif abs(pars[2] - np.pi/2)<1e-6 and pars[0] == 2:
            ic = DihedCos(*indexes)
            return Chebychev2(pars[1], ic, sign=1)
        elif pars[2] == 0.0 and pars[0] == 3:
            ic = DihedCos(*indexes)
            return Chebychev3(pars[1], ic, sign=-1)
        elif abs(pars[2] - np.pi/3)<1e-6 and pars[0] == 3:
            ic = DihedCos(*indexes)
            return Chebychev3(pars[1], ic, sign=1)
        elif pars[2] == 0.0 and pars[0] == 4:
            ic = DihedCos(*indexes)
            return Chebychev4(pars[1], ic, sign=-1)
        elif abs(pars[2] - np.pi/4)<1e-6 and pars[0] == 4:
            ic = DihedCos(*indexes)
            return Chebychev4(pars[1], ic, sign=1)
        elif pars[2] == 0.0 and pars[0] == 6:
            ic = DihedCos(*indexes)
            return Chebychev6(pars[1], ic, sign=-1)
        elif abs(pars[2] - np.pi/6)<1e-6 and pars[0] == 6:
            ic = DihedCos(*indexes)
            return Chebychev6(pars[1], ic, sign=1)
        else:
            return ValenceGenerator.get_vterm(self, pars, indexes)

    def get_force(self):
        #force = mm.PeriodicTorsionForce()
        #force.setUsesPeriodicBoundaryConditions(True)
        energy = 'A/2*(1 - cos(M*(theta - PHI0)))'
        force = mm.CustomTorsionForce(energy)
        force.addPerTorsionParameter('M')
        force.addPerTorsionParameter('A')
        force.addPerTorsionParameter('PHI0')
        force.setUsesPeriodicBoundaryConditions(True)
        return force

    def add_term_to_force(self, force, pars, indexes):
        conversion = {
                'A': molmod.units.kjmol,
                'PHI0': molmod.units.deg,
                'M': 1,
                }
        conversion_mm = {
                'A': unit.kilojoule_per_mole,
                'PHI0': unit.degrees,
                'M': unit.dimensionless,
                }
        M = pars[0]
        A = pars[1] / conversion['A'] * conversion_mm['A']
        PHI0 = pars[2] / conversion['PHI0'] * conversion_mm['PHI0']
        force.addTorsion(
                int(indexes[0]),
                int(indexes[1]),
                int(indexes[2]),
                int(indexes[3]),
                [M, A, PHI0],
                )


class ValenceCrossGenerator(Generator):
    '''All generators for cross valence terms derive from this class.

       **More important attributes:**

       nffatype
            The number of atoms involved in the internal coordinates. Hence
            this is also the number ffatypes in a single row in the force field
            parameter file.

       ICClass$i
            The i-th ``InternalCoordinate`` class. See ``yaff.pes.iclist``.

       VClass$i$j
            The ``ValenceTerm`` class for the cross term between IC$i and IC$j.
            See ``yaff.pes.vlist``.
    '''
    suffixes = ['UNIT', 'PARS']
    nffatype = None
    ICClass0 = None
    ICClass1 = None
    ICClass2 = None
    ICClass3 = None
    ICClass4 = None
    ICClass5 = None
    VClass01 = None
    VClass02 = None
    VClass03 = None
    VClass04 = None
    VClass05 = None
    VClass12 = None
    VClass13 = None
    VClass14 = None
    VClass15 = None
    VClass23 = None
    VClass24 = None
    VClass25 = None
    VClass34 = None
    VClass35 = None
    VClass45 = None

    def __call__(self, system, parsec, ff_args):
        '''Add contributions to the force field from a ValenceCrossGenerator

           **Arguments:**

           system
                The System object for which a force field is being prepared

           parse
                An instance of the ParameterSection class

           ff_ars
                An instance of the FFargs class
        '''
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, self.nffatype)
        if len(par_table) > 0:
            forces = self.apply(par_table, system, ff_args)
            return forces

    def apply(self, par_table, system, ff_args):
        '''Generate terms for the system based on the par_table

           **Arguments:**

           par_table
                A dictionary with tuples of ffatypes is keys and lists of
                parameters as values.

           system
                The system for which the force field is generated.

           ff_args
                An instance of the FFArgs class.
        '''
        if system.bonds is None:
            raise ValueError('The system must have bonds in order to define valence cross terms.')
        part_valence = ff_args.get_part_valence(system)
        vterms = []
        ics = []
        for i in range(6):
            for j in range(i+1,6):
                VClass = self.__class__.__dict__['VClass%i%i' %(i,j)]
                if VClass is not None:
                    vterms.append([i,j,VClass])
                    if i not in ics: ics.append(i)
                    if j not in ics: ics.append(j)
        ics = sorted(ics)
        #dict for get_indexes routines
        get_indexes = {
            0: self.get_indexes0, 1: self.get_indexes1, 2: self.get_indexes2,
            3: self.get_indexes3, 4: self.get_indexes4, 5: self.get_indexes5,
        }

        forces, conversions = self.get_forces(vterms) # dictionaries with keys (i, j)
        for indexes in self.iter_indexes(system):
            if min(indexes)<ff_args.nlow:
                # Check that this term indeed features only atoms with index<nlow
                assert max(indexes)<ff_args.nlow
                continue
            # We do not want terms where at least one atom index is higher than
            # or equal to nhigh, as this is (should be) an excluded interaction
            if ff_args.nhigh!=-1 and max(indexes)>=ff_args.nhigh:
                # Check that this term indeed features only atoms with index<nlow
                assert min(indexes)>=ff_args.nhigh
                continue
            key = tuple(system.get_ffatype(i) for i in indexes)
            par_list = par_table.get(key, [])
            for pars in par_list:
                for i, j, VClass_ij in vterms:
                    ICClass_i = self.__class__.__dict__['ICClass%i' %i]
                    assert ICClass_i is not None, 'IC%i has no ICClass defined' %i
                    ICClass_j = self.__class__.__dict__['ICClass%i' %j]
                    assert ICClass_i is not None, 'IC%i has no ICClass defined' %j
                    K_ij = pars[vterms.index([i,j,VClass_ij])]
                    rv_i = pars[len(vterms)+ics.index(i)]
                    rv_j = pars[len(vterms)+ics.index(j)]
                    args_ij = (K_ij, rv_i, rv_j, ICClass_i(*get_indexes[i](indexes)), ICClass_j(*get_indexes[j](indexes)))
                    #print('=======')
                    #print(ICClass_i.__name__, ICClass_j.__name__)
                    #print(key, indexes)
                    part_valence.add_term(VClass_ij(*args_ij))
                    self.add_term_to_forces(forces, conversions, (i, j), indexes, K_ij, rv_i, rv_j)
            #break
        #count = 0
        #for force in forces.values():
        #    count += force.getNumBonds()
        #print('Counting {} bonds in total'.format(count))
        return forces.values()

    def iter_indexes(self, system):
        '''Iterate over all tuples of indexes for the pair of internal coordinates'''
        raise NotImplementedError

    def get_indexes0(self, indexes):
        '''Get the indexes for the first internal coordinate from the whole'''
        raise NotImplementedError

    def get_indexes1(self, indexes):
        '''Get the indexes for the second internal coordinate from the whole'''
        raise NotImplementedError

    def get_indexes2(self, indexes):
        '''Get the indexes for the third internal coordinate from the whole'''
        raise NotImplementedError

    def get_indexes3(self, indexes):
        '''Get the indexes for the third internal coordinate from the whole'''
        raise NotImplementedError

    def get_indexes4(self, indexes):
        '''Get the indexes for the third internal coordinate from the whole'''
        raise NotImplementedError

    def get_indexes5(self, indexes):
        '''Get the indexes for the third internal coordinate from the whole'''
        raise NotImplementedError

    def get_forces(self, vterms):
        """Returns the appropriate force object"""
        raise NotImplementedError

    def add_term_to_force(self, forces, key, indexes, *pars):
        """Adds interaction to OpenMM force object"""
        raise NotImplementedError


class CrossGenerator(ValenceCrossGenerator):
    prefix = 'CROSS'
    par_info = [('KSS', float), ('KBS0', float), ('KBS1', float), ('R0', float), ('R1', float), ('THETA0', float)]
    nffatype = 3
    ICClass0 = Bond
    ICClass1 = Bond
    ICClass2 = BendAngle
    ICClass3 = None
    ICClass4 = None
    ICClass5 = None
    VClass01 = Cross
    VClass02 = Cross
    VClass03 = None
    VClass04 = None
    VClass05 = None
    VClass12 = Cross
    VClass13 = None
    VClass14 = None
    VClass15 = None
    VClass23 = None
    VClass24 = None
    VClass25 = None
    VClass34 = None
    VClass35 = None
    VClass45 = None

    def iter_equiv_keys_and_pars(self, key, pars):
        yield key, pars
        yield key[::-1], (pars[0], pars[2], pars[1], pars[4], pars[3], pars[5])

    def iter_indexes(self, system):
        return system.iter_angles()

    def get_indexes0(self, indexes):
        return indexes[:2]

    def get_indexes1(self, indexes):
        return indexes[1:]

    def get_indexes2(self, indexes):
        return indexes

    def get_forces(self, vterms):
        forces = {}
        conversions = {}
        for i, j, VClass_ij in vterms:
            ICClass_i = self.__class__.__dict__['ICClass%i' %i]
            ICClass_j = self.__class__.__dict__['ICClass%i' %j]
            if i == 0 and j == 1:
                energy = 'K*(distance(p1, p2) - RV0)*(distance(p2, p3) - RV1)'
                force = mm.CustomCompoundBondForce(3, energy)
                ic0_conversion = molmod.units.angstrom * 10
                ic1_conversion = molmod.units.angstrom * 10
                mm_ic0_conversion = unit.nanometer
                mm_ic1_conversion = unit.nanometer
                #key = (
                #        ('Bond', 1, 2),
                #        ('Bond', 1, 2),
                #        )
            elif i == 0 and j == 2:
                energy = 'K*(distance(p1, p2) - RV0)*(angle(p1, p2, p3) - RV1)'
                force = mm.CustomCompoundBondForce(3, energy)
                ic0_conversion = molmod.units.angstrom * 10
                ic1_conversion = 1.0
                mm_ic0_conversion = unit.nanometer
                mm_ic1_conversion = unit.radians
            elif i == 1 and j == 2:
                energy = 'K*(distance(p2, p3) - RV0)*(angle(p1, p2, p3) - RV1)'
                force = mm.CustomCompoundBondForce(3, energy)
                ic0_conversion = molmod.units.angstrom * 10
                ic1_conversion = 1.0
                mm_ic0_conversion = unit.nanometer
                mm_ic1_conversion = unit.radians
            #    key = (
            #            ('Bond', 1, 2),
            #            ('BendAngle', 1, 2, 3),
            #            )
            #    force.addPerBondParameter('K')
            #    force.addPerBondParameter('RV0')
            #    force.addPerBondParameter('RV1')
            #    force.setUsesPeriodicBoundaryConditions(True)
            #    forces[key] = force
            #    conversion = {
            #            'K': molmod.units.kjmol / (ic0_conversion * ic1_conversion),
            #            'RV0': ic0_conversion,
            #            'RV1': ic1_conversion,
            #            }
            #    conversion_mm = {
            #            'K': unit.kilojoule_per_mole / (mm_ic0_conversion * mm_ic1_conversion),
            #            'RV0': mm_ic0_conversion,
            #            'RV1': mm_ic1_conversion,
            #            }
            #    conversions[key] = (dict(conversion), dict(conversion_mm))

            #    energy = 'K*(distance(p1, p2) - RV0)*(angle(p1, p2, p3) - RV1)'
            #    force = mm.CustomCompoundBondForce(3, energy)
            #    key = (
            #            ('Bond', 2, 3),
            #            ('BendAngle', 1, 2, 3),
            #            )
            #    force.addPerBondParameter('K')
            #    force.addPerBondParameter('RV0')
            #    force.addPerBondParameter('RV1')
            #    force.setUsesPeriodicBoundaryConditions(True)
            #    forces[key] = force
            #    conversions[key] = (dict(conversion), dict(conversion_mm))
            else:
                raise NotImplementedError
            force.addPerBondParameter('K')
            force.addPerBondParameter('RV0')
            force.addPerBondParameter('RV1')
            force.setUsesPeriodicBoundaryConditions(True)
            key = (i, j)
            forces[key] = force
            conversion = {
                    'K': molmod.units.kjmol / (ic0_conversion * ic1_conversion),
                    'RV0': ic0_conversion,
                    'RV1': ic1_conversion,
                    }
            conversion_mm = {
                    'K': unit.kilojoule_per_mole / (mm_ic0_conversion * mm_ic1_conversion),
                    'RV0': mm_ic0_conversion,
                    'RV1': mm_ic1_conversion,
                    }
            conversions[key] = (dict(conversion), dict(conversion_mm))
        return forces, conversions

    def add_term_to_forces(self, forces, conversions, key, indexes, *pars):
        assert(len(pars) == 3)
        force = forces[key]
        conversion, conversion_mm = conversions[key]
        particles = [int(index) for index in indexes]
        K = pars[0] / conversion['K'] * conversion_mm['K']
        RV0 = pars[1] / conversion['RV0'] * conversion_mm['RV0']
        RV1 = pars[2] / conversion['RV1'] * conversion_mm['RV1']
        force.addBond(particles, [K, RV0, RV1])


class NonbondedGenerator(Generator):
    '''All generators for the non-bonding interactions derive from this class

       **One more important class attribute:**

       mixing_rules
            A dictionary with (par_name, rule_name): (narg, rule_id) items
    '''
    mixing_rules = None

    def process_scales(self, pardef):
        '''Process the SCALE definitions

           **Arguments:**

           pardef
                An instance of the ParameterDefinition class.

           Returns a dictionary with (numbonds, scale) items.
        '''
        result = {}
        for counter, line in pardef:
            words = line.split()
            if len(words) != 2:
                pardef.complain(counter, 'must have 2 arguments')
            try:
                num_bonds = int(words[0])
                scale = float(words[1])
            except ValueError:
                pardef.complain(counter, 'has parameters that can not be converted. The first argument must be an integer. The second argument must be a float')
            if num_bonds in result and result[num_bonds] != scale:
                pardef.complain(counter, 'contains a duplicate incompatible scale suffix')
            if scale < 0 or scale > 1:
                pardef.complain(counter, 'has a scale that is not in the range [0,1]')
            result[num_bonds] = scale
        if len(result) < 3 or len(result) > 4:
            pardef.complain(None, 'must contain three or four SCALE suffixes for each non-bonding term')
        if 1 not in result or 2 not in result or 3 not in result:
            pardef.complain(None, 'must contain a scale parameter for atoms separated by 1, 2 and 3 bonds, for each non-bonding term')
        if 4 not in result:
            result[4] = 1.0
        return result

    def process_mix(self, pardef):
        '''Process mixing rules

           **Arguments:**

           pardef
                An instance of the ParameterDefinition class.

           Returns a dictionary of (par_name, (rule_id, rule_args)) items.
        '''
        result = {}
        for counter, line in pardef:
            words = line.split()
            if len(words) < 2:
                pardef.complain(counter, 'contains a mixing rule with to few arguments. At least 2 are required')
            par_name = words[0].upper()
            rule_name = words[1].upper()
            key = par_name, rule_name
            if key not in self.mixing_rules:
                pardef.complain(counter, 'contains an unknown mixing rule')
            narg, rule_id = self.mixing_rules[key]
            if len(words) != narg+2:
                pardef.complain(counter, 'does not have the correct number of arguments. %i arguments are required' % (narg+2))
            try:
                args = tuple([float(word) for word in words[2:]])
            except ValueError:
                pardef.complain(counter, 'contains parameters that could not be converted to floating point numbers')
            result[par_name] = rule_id, args
        expected_num_rules = len(set([par_name for par_name, rule_id in self.mixing_rules]))
        if len(result) != expected_num_rules:
            pardef.complain(None, 'does not contain enough mixing rules for the generator %s' % self.prefix)
        return result


class MM3Generator(NonbondedGenerator):
    prefix = 'MM3'
    suffixes = ['UNIT', 'SCALE', 'PARS']
    par_info = [('SIGMA', float), ('EPSILON', float), ('ONLYPAULI', int)]

    def __call__(self, system, parsec, ff_args):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        par_table = self.process_pars(parsec['PARS'], conversions, 1)
        scale_table = self.process_scales(parsec['SCALE'])
        forces = self.apply(par_table, scale_table, system, ff_args)
        return forces

    def apply(self, par_table, scale_table, system, ff_args):
        # Prepare the atomic parameters
        sigmas = np.zeros(system.natom)
        epsilons = np.zeros(system.natom)
        onlypaulis = np.zeros(system.natom, np.int32)
        for i in range(system.natom):
            key = (system.get_ffatype(i),)
            par_list = par_table.get(key, [])
            if len(par_list) > 2:
                raise TypeError('Superposition should not be allowed for non-covalent terms.')
            elif len(par_list) == 1:
                sigmas[i], epsilons[i], onlypaulis[i] = par_list[0]

        for i in range(len(onlypaulis)):
            assert(onlypaulis[0] == 0)
        # Prepare the global parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        # Get the part. It should not exist yet.
        part_pair = ff_args.get_part_pair(PairPotMM3)
        if part_pair is not None:
            raise RuntimeError('Internal inconsistency: the MM3 part should not be present yet.')

        pair_pot = PairPotMM3(sigmas, epsilons, onlypaulis, ff_args.rcut, ff_args.tr)
        nlist = ff_args.get_nlist(system)
        part_pair = ForcePartPair(system, nlist, scalings, pair_pot)
        ff_args.parts.append(part_pair)

        energy = 'epsilon * (1.84 * 100000 * exp(-12 * r / sigma) - 2.25 * (sigma / r)^6);'
        energy += 'epsilon=sqrt(EPSILON1 * EPSILON2);'
        energy += 'sigma=SIGMA1 + SIGMA2;'
        force = mm.CustomNonbondedForce(energy)
        force.addPerParticleParameter('SIGMA')
        force.addPerParticleParameter('EPSILON')
        for i in range(system.pos.shape[0]):
            parameters = [
                    sigmas[i] / molmod.nanometer * unit.nanometer,
                    epsilons[i] / molmod.units.kjmol * unit.kilojoule_per_mole,
                    ]
            force.addParticle(parameters)
        force.setCutoffDistance(ff_args.rcut / molmod.units.nanometer * unit.nanometer)
        force.setNonbondedMethod(2)

        # COMPENSATE FOR EXCLUSIONS
        exclusion_force = self.get_exclusion_force(energy)
        for i in range(system.natom):
            for j in system.neighs1[i]:
                if i < j:
                    self.add_exclusion(sigmas, epsilons, i, j, exclusion_force)
            for j in system.neighs2[i]:
                if i < j:
                    self.add_exclusion(sigmas, epsilons, i, j, exclusion_force)
        #print(exclusion_force.getNumBonds())
        #raise NotImplementedError
        #bonds_tuples = []
        #for bond in system.bonds:
        #    bonds_tuples.append(tuple([int(atom) for atom in bond]))
        #scale_index = 0
        #for key, value in scale_table.items():
        #    assert(value == 0.0 or value == 1.0)
        #    if value == 0.0:
        #        scale_index += 1
        #force.createExclusionsFromBonds(
        #        bonds_tuples,
        #        scale_index,
        #        )
        #n = force.getNumExclusions()
        return [force, exclusion_force]

    @staticmethod
    def get_exclusion_force(energy):
        """Returns force object to account for exclusions"""
        force = mm.CustomBondForce('-1.0 * ' + energy)
        force.addPerBondParameter('SIGMA1')
        force.addPerBondParameter('EPSILON1')
        force.addPerBondParameter('SIGMA2')
        force.addPerBondParameter('EPSILON2')
        force.setUsesPeriodicBoundaryConditions(True)
        return force

    @staticmethod
    def add_exclusion(sigmas, epsilons, i, j, force):
        """Adds a bond between i and j"""
        SIGMA1 = sigmas[i] / (molmod.units.angstrom * 10) * unit.nanometer
        EPSILON1 = epsilons[i] / molmod.units.kcalmol * unit.kilocalories_per_mole
        SIGMA2 = sigmas[j] / (molmod.units.angstrom * 10) * unit.nanometer
        EPSILON2 = epsilons[j] / molmod.units.kcalmol * unit.kilocalories_per_mole
        parameters = [
                SIGMA1,
                EPSILON1,
                SIGMA2,
                EPSILON2,
                ]
        force.addBond(int(i), int(j), parameters)


class FixedChargeGenerator(NonbondedGenerator):
    prefix = 'FIXQ'
    suffixes = ['UNIT', 'SCALE', 'ATOM', 'BOND', 'DIELECTRIC']
    par_info = [('Q0', float), ('P', float), ('R', float)]

    def __call__(self, system, parsec, ff_args):
        self.check_suffixes(parsec)
        conversions = self.process_units(parsec['UNIT'])
        atom_table = self.process_atoms(parsec['ATOM'], conversions)
        bond_table = self.process_bonds(parsec['BOND'], conversions)
        scale_table = self.process_scales(parsec['SCALE'])
        dielectric = self.process_dielectric(parsec['DIELECTRIC'])
        forces = self.apply(atom_table, bond_table, scale_table, dielectric, system, ff_args)
        return forces

    def process_atoms(self, pardef, conversions):
        result = {}
        for counter, line in pardef:
            words = line.split()
            if len(words) != 3:
                pardef.complain(counter, 'should have 3 arguments')
            ffatype = words[0]
            if ffatype in result:
                pardef.complain(counter, 'has an atom type that was already encountered earlier')
            try:
                charge = float(words[1])*conversions['Q0']
                radius = float(words[2])*conversions['R']
            except ValueError:
                pardef.complain(counter, 'contains a parameter that can not be converted to a floating point number')
            result[ffatype] = charge, radius
        return result

    def process_bonds(self, pardef, conversions):
        result = {}
        for counter, line in pardef:
            words = line.split()
            if len(words) != 3:
                pardef.complain(counter, 'should have 3 arguments')
            key = tuple(words[:2])
            if key in result:
                pardef.complain(counter, 'has a combination of atom types that were already encountered earlier')
            try:
                charge_transfer = float(words[2])*conversions['P']
            except ValueError:
                pardef.complain(counter, 'contains a parameter that can not be converted to floating point numbers')
            result[key] = charge_transfer
            result[key[::-1]] = -charge_transfer
        return result

    def process_dielectric(self, pardef):
        result = None
        for counter, line in pardef:
            if result is not None:
                pardef.complain(counter, 'is redundant. The DIELECTRIC suffix may only occur once')
            words = line.split()
            if len(words) != 1:
                pardef.complain(counter, 'must have one argument')
            try:
                result = float(words[0])
            except ValueError:
                pardef.complain(counter, 'must have a floating point argument')
        return result

    def apply(self, atom_table, bond_table, scale_table, dielectric, system, ff_args):
        if system.charges is None:
            system.charges = np.zeros(system.natom)
        elif log.do_warning and abs(system.charges).max() != 0:
            log.warn('Overwriting charges in system.')
        system.charges[:] = 0.0
        system.radii = np.zeros(system.natom)

        # compute the charges
        for i in range(system.natom):
            pars = atom_table.get(system.get_ffatype(i))
            if pars is not None:
                charge, radius = pars
                system.charges[i] += charge
                system.radii[i] = radius
            elif log.do_warning:
                log.warn('No charge defined for atom %i with fftype %s.' % (i, system.get_ffatype(i)))
        for i0, i1 in system.iter_bonds():
            ffatype0 = system.get_ffatype(i0)
            ffatype1 = system.get_ffatype(i1)
            if ffatype0 == ffatype1:
                continue
            charge_transfer = bond_table.get((ffatype0, ffatype1))
            if charge_transfer is None:
                if log.do_warning:
                    log.warn('No charge transfer parameter for atom pair (%i,%i) with fftype (%s,%s).' % (i0, i1, system.get_ffatype(i0), system.get_ffatype(i1)))
            else:
                system.charges[i0] += charge_transfer
                system.charges[i1] -= charge_transfer

        # prepare other parameters
        scalings = Scalings(system, scale_table[1], scale_table[2], scale_table[3], scale_table[4])

        # Setup the electrostatic pars
        assert(dielectric == 1.0)
        ff_args.add_electrostatic_parts(system, scalings, dielectric)
        #part_pair = ff_args.get_part_pair(PairPotEI)
        #alpha_ = part_pair.pair_pot._get_alpha()
        #alpha_ *= molmod.units.nanometer * 1 / unit.nanometer


        force = mm.NonbondedForce()
        for i in range(system.pos.shape[0]):
            force.addParticle(
                    system.charges[i] / molmod.units.coulomb * unit.coulomb,
                    0 * unit.nanometer, # DISPERSION NOT COMPUTED IN THIS FORCE
                    0 * unit.kilocalories_per_mole, # DISPERSION NOT COMPUTED
                    )
        rcut = ff_args.rcut / (molmod.units.nanometer) * unit.nanometer
        alpha = ff_args.alpha_scale / rcut
        force.setCutoffDistance(rcut)
        force.setNonbondedMethod(4)
        delta = np.exp(-(ff_args.alpha_scale) ** 2) / 2
        delta_thres = 1e-6
        if delta < delta_thres:
            print('overriding error tolerance: delta = {}'.format(delta_thres))
            delta = delta_thres
            alpha = np.sqrt(- np.log(2 * delta_thres)) / rcut
        force.setEwaldErrorTolerance(delta)

        # ASSERT NO EXCLUSIONS
        scale_index = 0
        for key, value in scale_table.items():
            assert(value == 0.0 or value == 1.0)
            if value == 0.0:
                scale_index += 1
        assert(scale_index == 0)

        # COMPENSATE FOR GAUSSIAN CHARGES
        gaussian_force = self.get_gaussian_force(alpha)
        for i in range(system.pos.shape[0]):
            parameters = [
                    system.charges[i] / molmod.units.coulomb * unit.coulomb,
                    system.radii[i] / molmod.units.nanometer * unit.nanometer,
                    ]
            gaussian_force.addParticle(parameters)
        gaussian_force.setCutoffDistance(rcut)
        gaussian_force.setNonbondedMethod(2)
        return [force, gaussian_force]

    @staticmethod
    def get_gaussian_force(ALPHA):
        """Creates a force object that compensates for the gaussian charge distribution

            ALPHA is the 'alpha' parameter of the gaussians used for the sum in reciprocal space.
        """
        #coulomb_const = 8.9875517887 * 1e9 # in units of J * m / C2
        #coulomb_const *= 1.0e9 # in units of J * nm / C2
        #coulomb_const *= molmod.constants.avogadro / 1000 # in units of kJmol * nm / C2
        #coulomb_const /= (1 / 1.602176634e-19) ** 2
        coulomb_const = 1.38935456e2
        coulomb_const = 1.0 / molmod.units.kjmol / molmod.units.nanometer
        E_S_test = "cprod / r * erfc(ALPHA * r); "
        # SUBTRACT SHORT-RANGE CONTRIBUTION
        E_S = "- cprod / r * erfc(ALPHA * r) "
        # ADD SHORT-RANGE GAUSS CONTRIBUTION
        E_Sg = "cprod / r * (erf(A12 * r) - erf(ALPHA * r)); "
        definitions = "cprod=charge1*charge2*" + str(coulomb_const) + "; "
        definitions += "A12=1/radius1*1/radius2/sqrt(1/radius1^2 + 1/radius2^2); "
        definitions += "A1 = 1/radius1*ALPHA/sqrt(1/radius1^2 + ALPHA^2); "
        definitions += "A2 = 1/radius2*ALPHA/sqrt(1/radius2^2 + ALPHA^2); "
        energy = E_S + " + " + E_Sg + definitions
        #energy = E_Sg + definitions
        #energy = E_S_test
        #energy += "cprod=charge1*charge2*" + str(coulomb_const) + "; "
        force = mm.CustomNonbondedForce(energy)
        force.addPerParticleParameter("charge")
        force.addPerParticleParameter("radius")
        force.addGlobalParameter("ALPHA", ALPHA)
        return force


def apply_generators(system, parameters, ff_args):
    '''Populate the attributes of ff_args, prepares arguments for ForceField

       **Arguments:**

       system
            An AbstractSystem instance for which the force field object is being made

       ff_args
            An instance of the FFArgs class.

       yaml_dict
            Dictionary loaded from a .yaml file
    '''

    generators = {}
    for x in globals().values():
        if isinstance(x, type) and issubclass(x, Generator) and x.prefix is not None:
            generators[x.prefix] = x()

    # Go through all the sections of the parameter file and apply the
    # corresponding generator.
    for prefix, section in parameters.sections.items():
        generator = generators.get(prefix)
        if generator is None:
            if log.do_warning:
                log.warn('There is no generator named %s. It will be ignored.' % prefix)
        else:
            generator(system, section, ff_args)
    assert(not ff_args.tailcorrections)
    # If tail corrections are requested, go through all parts and add when necessary
    #if ff_args.tailcorrections:
    #    if system.cell.nvec==0:
    #        log.warn('Tail corrections were requested, but this makes no sense for non-periodic system. Not adding tail corrections...')
    #    elif system.cell.nvec==3:
    #        for part in ff_args.parts:
    #            # Only add tail correction to pair potentials
    #            if isinstance(part,ForcePartPair):
    #                # Don't add tail corrections to electrostatic parts whose
    #                # long-range interactions are treated using for instance Ewald
    #                if isinstance(part.pair_pot,PairPotEI) or isinstance(part.pair_pot,PairPotEIDip):
    #                    continue
    #                else:
    #                    part_tailcorrection = ForcePartTailCorrection(system, part)
    #                    ff_args.parts.append(part_tailcorrection)
    #    else:
    #        raise ValueError('Tail corrections not available for 1-D and 2-D periodic systems')

    #part_valence = ff_args.get_part(ForcePartValence)
    #if part_valence is not None and log.do_warning:
    #    # Basic check for missing terms
    #    groups = set([])
    #    nv = part_valence.vlist.nv
    #    for iv in range(nv):
    #        # Get the atoms in the energy term.
    #        atoms = part_valence.vlist.lookup_atoms(iv)
    #        # Reduce it to a set of atom indices.
    #        atoms = frozenset(sum(sum(atoms, []), []))
    #        # Keep all two- and three-body terms.
    #        if len(atoms) <= 3:
    #            groups.add(atoms)
    #    # Check if some are missing
    #    for i0, i1 in system.iter_bonds():
    #        if frozenset([i0, i1]) not in groups:
    #            log.warn('No covalent two-body term for atoms ({}, {})'.format(i0, i1))
    #    for i0, i1, i2 in system.iter_angles():
    #        if frozenset([i0, i1, i2]) not in groups:
    #            log.warn('No covalent three-body term for atoms ({}, {} {})'.format(i0, i1, i2))


def apply_generators_mm(system, parameters, ff_args, mm_system):
    """Construct an OpenMM system object that contains the forces defined in parameters"""
    #mm_system = _init_openmm_system(system)
    generators = {}
    for x in globals().values():
        if isinstance(x, type) and issubclass(x, Generator) and x.prefix is not None:
            generators[x.prefix] = x()

    # Go through all the sections of the parameter file and apply the
    # corresponding generator.
    total_forces = []
    for prefix, section in parameters.sections.items():
        generator = generators.get(prefix)
        if generator is None:
            if log.do_warning:
                log.warn('There is no generator named %s. It will be ignored.' % prefix)
        else:
            forces = generator(system, section, ff_args)
            total_forces += forces
    if total_forces is not None:
        for force in total_forces:
            mm_system.addForce(force)
    return mm_system


AVAILABLE_PREFIXES = []
#print(globals())
for x in list(globals().values()):
    if isinstance(x, type) and issubclass(x, Generator) and x.prefix is not None:
        AVAILABLE_PREFIXES.append(x.prefix)
