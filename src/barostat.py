import molmod
import h5py
import yaff

import numpy as np

from attrdict import AttrDict
from systems.systems import test_systems
from src.utils import _align


class MonteCarloBarostat(yaff.sampling.verlet.VerletHook):
    """Implements a Monte Carlo based barostat allowing full cell shape flexibility"""
    name = 'MonteCarlo'
    kind = 'stochastic'
    method = 'mcbarostat'
    INDEX_MAPPING = {
            1: tuple([0, 0]),
            2: tuple([1, 0]),
            3: tuple([1, 1]),
            4: tuple([2, 0]),
            5: tuple([2, 1]),
            6: tuple([2, 2]),
            }
    dV = 1e-3
    dh = 1e-4

    def __init__(self, temp, press, start=0, step=1, mode='full', type_prob=(5.0 / 6)):
        self.temp = temp
        self.press = press
        self.dim = 3
        self.baro_ndof = None
        self.mode = mode

        self.attempted_h = np.zeros((3, 3))
        self.accepted_h = np.zeros((3, 3))
        self.scaling_h = np.ones((3, 3)) * (1e1)
        self.scaling_h[0, 1] = 0
        self.scaling_h[0, 2] = 0
        self.scaling_h[1, 2] = 0

        self.attempted_V = 0
        self.accepted_V = 0
        self.scaling_V = 1e4

        self.type_prob = type_prob
        yaff.sampling.VerletHook.__init__(self, start, step)
        self.step = step

    def init(self, iterative):
        pass

    def pre(self, iterative, chainvel0=None):
        pass

    def post(self, iterative, chainvel0=None):
        # compute reduced coordinates
        rvecs = iterative.ff.system.cell._get_rvecs().copy()
        assert rvecs[0, 1] == 0, 'Cell shape must remain lower-triagonal for MC barostat to work correctly.'
        assert rvecs[0, 2] == 0, 'Cell shape must remain lower-triagonal for MC barostat to work correctly.'
        assert rvecs[1, 2] == 0, 'Cell shape must remain lower-triagonal for MC barostat to work correctly.'
        #vol = np.linalg.det(rvecs)
        pos = iterative.ff.system.pos
        fractional = np.transpose(np.dot(np.linalg.inv(rvecs), np.transpose(pos)))

        trial_type = self.get_trial_type()
        if trial_type == 'isotropic':
            trial, C = self._get_isotropic_trial(rvecs)
            accepted = self.apply_trial(rvecs, fractional, iterative, trial, C)
            self.attempted_V += 1
            if accepted:
                self.accepted_V += 1
        elif trial_type == 'anisotropic':
            index, trial, C = self._get_anisotropic_trial(rvecs)
            accepted = self.apply_trial(rvecs, fractional, iterative, trial, C)
            self.attempted_h[index] += 1
            if accepted:
                self.accepted_h[index] += 1

        #print('attempted_V: ', self.attempted_V)
        #print('accepted_V: ', self.accepted_V)
        #print('scaling_V: ', self.scaling_V)

        # adjust scaling to maintain approx 50% acceptance
        if self.attempted_V >= 10:
            if self.accepted_V < 0.25 * self.attempted_V:
                self.scaling_V /= 1.1
                print('adjusting scaling for volume: {}'.format(self.scaling_V))
                self.attempted_V = 0
                self.accepted_V = 0
            elif self.accepted_V > 0.75 * self.attempted_V:
                self.scaling_V *= 1.1
                print('adjusting scaling for volume: {}'.format(self.scaling_V))
                self.attempted_V = 0
                self.accepted_V = 0
        for i in range(1, 7):
            index = self.INDEX_MAPPING[i]
            if self.attempted_h[index] >= 10:
                if self.accepted_h[index] < 0.25 * self.attempted_h[index]:
                    self.scaling_h[index] /= 1.1
                    print('adjusting scaling for {}: {}'.format(index, self.scaling_h[index]))
                    self.accepted_h[index] = 0
                    self.attempted_h[index] = 0
                elif self.accepted_h[index] > 0.75 * self.attempted_h[index]:
                    self.scaling_h[index] *= 1.1
                    print('adjusting scaling for {}: {}'.format(index, self.scaling_h[index]))
                    self.accepted_h[index] = 0
                    self.attempted_h[index] = 0

    def get_trial_type(self):
        if self.mode == 'full':
            if np.random.uniform() > self.type_prob:
                trial_type = 'isotropic'
            else:
                trial_type = 'anisotropic'
        elif self.mode == 'isotropic':
            trial_type = 'isotropic'
        elif self.mode == 'constrained':
            trial_type = 'anisotropic'
        return trial_type

    def _get_isotropic_trial(self, rvecs):
        """Generates an isotropic trial"""
        dV = 2 * (np.random.uniform() - 0.5) * self.scaling_V * self.dV
        s = 1 + dV / np.linalg.det(rvecs)
        C = s ** (1.0 / 3)
        return None, C

    def _get_anisotropic_trial(self, rvecs):
        """Generates an anisotropic trial on the normalized cell tensor"""
        i = np.random.randint(1, 7)
        index = self.INDEX_MAPPING[i]
        dh = 2 * (np.random.uniform() - 0.5) * self.scaling_h[index] * self.dh
        trial = np.zeros((3, 3))
        trial[index] += dh

        # isotropic rescaling to maintain volume after trial
        rvecs__ = rvecs / np.linalg.det(rvecs) ** (1.0 / 3)
        if (i == 1) or (i == 3) or (i == 6):
            h = rvecs__[index]
            s = ((h + dh) / h) ** (-1)
        else:
            s = 1
        C = s ** (1.0 / 3)
        #print('component: {}'.format(index))
        #print(np.linalg.det(rvecs__))
        #print(np.linalg.det(rvecs__ + trial))
        #print(np.linalg.det(C * (rvecs__ + trial)))
        return index, trial, C

    def apply_trial(self, rvecs, fractional, iterative, trial, C):
        """Computes the trial positions and unit cell, and evaluates the energy"""
        def compute(pos, rvecs):
            iterative.pos[:] = pos
            iterative.gpos[:] = 0.0
            iterative.ff.update_rvecs(rvecs)
            iterative.ff.update_pos(pos)
            iterative.epot = iterative.ff.compute(iterative.gpos)
            iterative.acc = -iterative.gpos/iterative.masses.reshape(-1,1)

        natom = iterative.ff.system.natom
        epot = iterative.epot
        vol = np.linalg.det(rvecs)

        if trial is not None:
            norm = np.linalg.det(rvecs) ** (1.0 / 3)
            rvecs_ = norm * (rvecs / norm + trial)
        else:
            rvecs_ = rvecs.copy()
        rvecs_ *= C
        vol_ = np.linalg.det(rvecs_)
        assert vol_ > 0, 'Unit cell volume became negative: {}'.format(rvecs_)
        pos_ = np.transpose(np.dot(rvecs_, np.transpose(fractional)))
        compute(pos_, rvecs_)
        epot_ = iterative.epot
        beta = 1 / (molmod.constants.boltzmann * self.temp)
        c = molmod.units.kjmol

        arg = epot_ - epot
        arg += self.press * (vol_ - vol)
        arg += - beta ** (-1) * natom * np.log(vol_ / vol)
        arg += - beta ** (-1) * 1 * np.log(rvecs_[1, 1] / rvecs[1, 1] * (vol / vol_) ** (1.0 / 3))
        arg += - beta ** (-1) * 2 * np.log(rvecs_[2, 2] / rvecs[2, 2] * (vol / vol_) ** (1.0 / 3))

        # if arg < 0, then the move should always be accepted
        # else, it should be accepted with probability np.exp(-beta * arg)
        accepted = arg < 0 or np.random.uniform(0, 1) < np.exp(-beta*arg)
        if accepted:
            #print('MOVE ACCEPTED')
            # add a correction to the conserved quantity
            self.econs_correction += epot - epot_
        else:
            # revert the cell and the positions in the original state
            pos = np.transpose(np.dot(rvecs, np.transpose(fractional)))
            compute(pos, rvecs)
        return accepted


class BaroTest(object):
    """Evaluates a barostat based on a single MD run"""

    def __init__(self, name, baro, anisotropic=True, vol_constraint=False, type_prob=(5.0 / 6)):
        info = None
        for _ in test_systems:
            if _['name'] == name:
                info = AttrDict(_)
        assert(info is not None)
        self.system = yaff.System.from_file(info.path_chk)
        _align(self.system)
        self.baro = baro
        self.anisotropic = anisotropic
        self.vol_constraint = vol_constraint
        self.parameters = yaff.Parameters.from_file(info.path_pars)

    @staticmethod
    def out(f):
        """Prints evaluation metrics

        Arguments
        ---------
            f:
                a readable h5py File object.
        """
        epot = np.array(list(f['trajectory']['epot']))
        vol = np.array(list(f['trajectory']['volume']))
        press = np.array(list(f['trajectory']['press']))
        temp = np.array(list(f['trajectory']['temp']))
        ptens = np.array(list(f['trajectory']['ptens']))
        c = molmod.units.angstrom ** 3
        print('average volume: {} A ** 3 (std: {} A ** 3)'.format(np.mean(vol) / c, np.sqrt(np.var(vol)) / c))
        c = molmod.units.pascal * 1e6
        print('average pressure: {} MPa (std: {} MPa)'.format(np.mean(press) / c, np.sqrt(np.var(press)) / c))
        c = molmod.units.kelvin
        print('average temperature: {} K (std: {} K)'.format(np.mean(temp) / c, np.sqrt(np.var(temp)) / c))
        c = molmod.units.pascal * 1e6
        print('average ptens: [MPa]')
        print(np.mean(ptens, axis=0) / c)
        print('average ptens std: ')
        print(np.sqrt(np.var(ptens, axis=0)) / c)

    def __call__(self, **kwargs):
        """Performs an MD simulation and prints the results"""
        ff_args = yaff.pes.generator.FFArgs()
        yaff.pes.generator.apply_generators(self.system, self.parameters, ff_args)
        ff = yaff.ForceField(self.system, ff_args.parts, ff_args.nlist)
        f = h5py.File(kwargs['output_name'] + '.h5', 'w')
        hdf = yaff.sampling.io.HDF5Writer(f, start=kwargs['start'], step=kwargs['write'])
        xyz = yaff.sampling.io.XYZWriter(kwargs['output_name'] + '.xyz', start=kwargs['start'], step=kwargs['write'])
        vsl = yaff.sampling.verlet.VerletScreenLog(start=0, step=kwargs['write'])
        hooks = [
                hdf,
                xyz,
                vsl,
                ]
        tbc = self._get_thermo_baro(ff, **kwargs)
        hooks += tbc
        verlet = yaff.sampling.verlet.VerletIntegrator(
                ff,
                timestep=0.5 * molmod.units.femtosecond,
                hooks=hooks,
                )
        yaff.log.set_level(yaff.log.medium)
        verlet.run(kwargs['steps'])
        yaff.log.set_level(yaff.log.low)
        self.out(f)

    def _get_thermo_baro(self, ff, **kwargs):
        T = kwargs['T'] * molmod.units.kelvin
        P = kwargs['P'] * molmod.units.pascal * 1e6
        if self.baro == 'langevin':
            thermo = yaff.sampling.nvt.LangevinThermostat(T)
            baro = yaff.sampling.npt.LangevinBarostat(
                    ff,
                    T,
                    P,
                    timecon=1000.0 * molmod.units.femtosecond,
                    anisotropic=self.anisotropic,
                    vol_constraint=self.vol_constraint,
                    )
            return [yaff.sampling.npt.TBCombination(thermo, baro)]
        elif self.baro == 'mc':
            if self.anisotropic and not self.vol_constraint:
                mode = 'full'
            elif not self.anisotropic and not self.vol_constraint:
                mode = 'isotropic'
            elif self.anisotropic and self.vol_constraint:
                mode = 'constrained'
            else:
                raise NotImplementedError
            thermo = yaff.sampling.nvt.LangevinThermostat(T)
            baro = MonteCarloBarostat(
                    T,
                    P,
                    mode=mode,
                    )
            return [thermo, baro]
        elif self.baro == 'mtk':
            thermo = yaff.sampling.nvt.NHCThermostat(T, timecon=100.0 * molmod.units.femtosecond, chainlength=3) #thermostat
            baro = yaff.sampling.npt.MTKBarostat(ff, T, P, timecon=1000.0 * molmod.units.femtosecond, vol_constraint=self.vol_constraint, anisotropic=self.anisotropic)
            return [yaff.sampling.npt.TBCombination(thermo, baro)]
        else:
            raise NotImplementedError


#class BaroTest():
#    """Performs the shirts test on the MonteCarloFullBarostat"""
#    tname = 'baro'
#
#    def __init__(self):
#
#    def _internal_test(self, baro='mc', steps=1000, writer_step=100, start=5, T=None, P=None, dT=0, dP=0, name='output'):
#        self.rcut = 10 * molmod.units.angstrom
#        if baro == 'mc':
#            func_simulate = BaroTest._simulate_mc
#        elif baro == 'langevin':
#            func_simulate = BaroTest._simulate_langevin
#        else:
#            raise NotImplementedError
#        self.test_shirts(func_simulate, steps, writer_step, start, T, P, dT, dP, name)
#
#    def test_shirts(self, func_simulate, steps=1000, writer_step=100, start=0, T=None, P=None, dT=0, dP=0, name='output'):
#        epot0, vol0 = func_simulate(self, steps, writer_step, start, T, P, name=name)
#        if dT != 0:
#            assert(dP == 0)
#            epot1, vol1 = func_simulate(self, steps, writer_step, start, T + dT, P, name=name)
#        elif dP != 0:
#            assert(dT == 0)
#            epot1, vol1 = func_simulate(self, steps, writer_step, start, T, P + dP, name=name)
#        else:
#            raise ValueError('Either dT or dP should be nonzero')
#        h0 = epot0 + vol0 * P * molmod.units.pascal
#        h1 = epot1 + vol1 * P * molmod.units.pascal
#        hist1, hist0 = BaroTest._create_histograms(h0, h1)
#        beta0 = 1 / (molmod.constants.boltzmann * T)
#        beta1 = 1 / (molmod.constants.boltzmann * (T + dT))
#        print('Theoretical slope: {}'.format(beta1 - beta0))
#
#    @staticmethod
#    def _create_histograms(h0, h1):
#        hist0, bin_edges = np.histogram(h0, bins=16) # should not contain any zeros
#        hist1, _ = np.histogram(h1, bin_edges) # may contain zeros
#        l = bin_edges[-1] - bin_edges[-2]
#        x = bin_edges[1:] - l / 2
#        # REMOVE ZEROS FROM HIST1
#        i = 0
#        print(hist0)
#        print(hist1)
#        while i < len(hist1):
#            if hist1[i] < 30 or hist0[i] < 30:
#                hist1 = np.delete(hist1, i)
#                hist0 = np.delete(hist0, i)
#                x = np.delete(x, i)
#            else:
#                i += 1
#        y = np.log(hist0 / hist1)
#        plt.plot(x, y)
#        print(len(y))
#        if len(y) > 5:
#            p = np.polyfit(x, y, 1)
#        plt.plot(x, p[0] * x + p[1])
#        print(p)
#        plt.show()
#        return hist0, hist1
#
#    def _simulate_langevin(self, steps=1000, writer_step=100, start=1000, T=None, P=None, name='output'):
#        assert(T is not None)
#        assert(P is not None)
#        ff_args_ = self._get_ffargs(use_yaff=True)
#        yaff.pes.generator.apply_generators(self.system, self.parameters, ff_args_)
#        ff = yaff.ForceField(self.system, ff_args_.parts, ff_args_.nlist)
#        thermo = yaff.sampling.nvt.LangevinThermostat(T * molmod.units.kelvin)
#        baro = yaff.sampling.npt.LangevinBarostat(
#                ff,
#                T * molmod.units.kelvin,
#                P * molmod.units.pascal * 1e6,
#                timecon=1000.0 * molmod.units.femtosecond,
#                anisotropic=True,
#                vol_constraint=False,
#                )
#        tbc = yaff.sampling.npt.TBCombination(thermo, baro)
#        f = h5py.File(name + '.h5', 'w')
#        hdf = yaff.sampling.io.HDF5Writer(f, start=start, step=writer_step)
#        xyz = yaff.sampling.io.XYZWriter(name + '.xyz', start=start, step=writer_step)
#        vsl = yaff.sampling.verlet.VerletScreenLog(start=0, step=writer_step)
#        hooks = [
#                tbc,
#                hdf,
#                xyz,
#                vsl,
#                ]
#        verlet = yaff.sampling.verlet.VerletIntegrator(
#                ff,
#                timestep=0.5 * molmod.units.femtosecond,
#                hooks=hooks,
#                )
#        yaff.log.set_level(yaff.log.medium)
#        verlet.run(steps)
#        yaff.log.set_level(yaff.log.low)
#        ## return epot, vol
#        epot = np.array(list(f['trajectory']['epot']))
#        vol = np.array(list(f['trajectory']['volume']))
#        press = np.array(list(f['trajectory']['press']))
#        temp = np.array(list(f['trajectory']['temp']))
#        ptens = np.array(list(f['trajectory']['ptens']))
#        c = molmod.units.angstrom ** 3
#        print('average volume: {} A ** 3 (std: {} A ** 3)'.format(np.mean(vol) / c, np.sqrt(np.var(vol)) / c))
#        c = molmod.units.pascal * 1e6
#        print('average pressure: {} MPa (std: {} MPa)'.format(np.mean(press) / c, np.sqrt(np.var(press)) / c))
#        c = molmod.units.kelvin
#        print('average temperature: {} K (std: {} K)'.format(np.mean(temp) / c, np.sqrt(np.var(temp)) / c))
#        c = molmod.units.pascal * 1e6
#        print('average ptens: [MPa]')
#        print(np.mean(ptens, axis=0) / c)
#        print('average ptens std: ')
#        print(np.sqrt(np.var(ptens, axis=0)) / c)
#        raise ValueError
#        return epot, vol
#
#    def _simulate_mc(self, steps=1000, writer_step=100, start=1000, T=None, P=None, name='output'):
#        assert(T is not None)
#        assert(P is not None)
#        ff_args = yaff.pes.generator.FFArgs()
#        yaff.pes.generator.apply_generators(self.system, self.parameters, ff_args)
#        ff = yaff.ForceField(self.system, ff_args.parts, ff_args.nlist)
#        thermo = yaff.sampling.nvt.LangevinThermostat(T * molmod.units.kelvin)
#        baro = MonteCarloBarostat(
#                T * molmod.units.kelvin,
#                P * molmod.units.pascal * 1e6,
#                mode='full',
#                )
#        f = h5py.File(name + '.h5', 'w')
#        hdf = yaff.sampling.io.HDF5Writer(f, start=start, step=writer_step)
#        xyz = yaff.sampling.io.XYZWriter(name + '.xyz', start=start, step=writer_step)
#        vsl = yaff.sampling.verlet.VerletScreenLog(start=0, step=writer_step)
#        hooks = [
#                thermo,
#                baro,
#                hdf,
#                xyz,
#                vsl,
#                ]
#        verlet = yaff.sampling.verlet.VerletIntegrator(
#                ff,
#                timestep=0.5 * molmod.units.femtosecond,
#                hooks=hooks,
#                )
#        yaff.log.set_level(yaff.log.medium)
#        verlet.run(steps)
#        yaff.log.set_level(yaff.log.low)
#        ## return epot, vol
#        epot = np.array(list(f['trajectory']['epot']))
#        vol = np.array(list(f['trajectory']['volume']))
#        press = np.array(list(f['trajectory']['press']))
#        temp = np.array(list(f['trajectory']['temp']))
#        ptens = np.array(list(f['trajectory']['ptens']))
#        c = molmod.units.angstrom ** 3
#        print('average volume: {} A ** 3 (std: {} A ** 3)'.format(np.mean(vol) / c, np.sqrt(np.var(vol)) / c))
#        c = molmod.units.pascal * 1e6
#        print('average pressure: {} MPa (std: {} MPa)'.format(np.mean(press) / c, np.sqrt(np.var(press)) / c))
#        c = molmod.units.kelvin
#        print('average temperature: {} K (std: {} K)'.format(np.mean(temp) / c, np.sqrt(np.var(temp)) / c))
#        c = molmod.units.pascal * 1e6
#        print('average ptens: [MPa]')
#        print(np.mean(ptens, axis=0) / c)
#        print('average ptens std: ')
#        print(np.sqrt(np.var(ptens, axis=0)) / c)
#        raise ValueError
#        return epot, vol
