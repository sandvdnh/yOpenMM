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
        self.scaling_h = np.ones((3, 3)) * (1e2)
        self.scaling_h[0, 1] = 0
        self.scaling_h[0, 2] = 0
        self.scaling_h[1, 2] = 0

        self.attempted_V = 0
        self.accepted_V = 0
        self.scaling_V = 1e5

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
        elif trial_type == 'complete':
            index, trial = self._get_complete_trial()
            accepted = self.apply_trial(rvecs, fractional, iterative, trial, 1.0)
            self.attempted_h[index] += 1
            if accepted:
                self.accepted_h[index] += 1

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
        elif self.mode == 'alternative':
            trial_type = 'complete'
        else:
            raise NotImplementedError
        return trial_type

    def _get_isotropic_trial(self, rvecs):
        """Generates an isotropic trial"""
        dV = 2 * (np.random.uniform() - 0.5) * self.scaling_V * self.dV
        s = 1 + dV / np.linalg.det(rvecs)
        C = s ** (1.0 / 3)
        return None, C

    def _get_complete_trial(self):
        """Generates a complete trial move, that changes both volume and shape"""
        i = np.random.randint(1, 7)
        index = self.INDEX_MAPPING[i]
        dh = 2 * (np.random.uniform() - 0.5) * self.scaling_h[index] * self.dh
        trial = np.zeros((3, 3))
        trial[index] += dh
        return index, trial

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

        contribs = [epot_ - epot]
        contribs += [self.press * (vol_ - vol)]
        contribs += [- beta ** (-1) * natom * np.log(vol_ / vol)]
        contribs += [- beta ** (-1) * 1 * np.log(rvecs_[1, 1] / rvecs[1, 1] * (vol / vol_) ** (1.0 / 3))]
        contribs += [- beta ** (-1) * 2 * np.log(rvecs_[2, 2] / rvecs[2, 2] * (vol / vol_) ** (1.0 / 3))]
        #print('==========')
        #print('TRIAL: \t {} \t to \t {}'.format(vol, vol_))
        #print(trial)
        #print('CONTRIBS')
        #print(contribs)
        arg = sum(contribs)
        #print('TOTAL')
        #print(arg)

        # if arg < 0, then the move should always be accepted
        # else, it should be accepted with probability np.exp(-beta * arg)
        accepted = arg < 0 or np.random.uniform(0, 1) < np.exp(-beta*arg)
        if accepted:
            #print('MOVE ACCEPTED')
            # add a correction to the conserved quantity
            self.econs_correction += epot - epot_
            #print('ACCEPTED')
        else:
            # revert the cell and the positions in the original state
            pos = np.transpose(np.dot(rvecs, np.transpose(fractional)))
            compute(pos, rvecs)
        return accepted



class MonteCarloBarostat2(yaff.sampling.verlet.VerletHook):
    """Implements a Monte Carlo based barostat allowing full cell shape flexibility"""
    name = 'MonteCarlo'
    kind = 'stochastic'
    method = 'mcbarostat'
    INDEX_MAPPING = {
            1: tuple([0, 0]),
            2: tuple([1, 0]),
            3: tuple([2, 0]),
            4: tuple([1, 1]),
            5: tuple([2, 1]),
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
        self.scaling_h = np.ones((3, 3)) * (1e2)
        self.scaling_h[0, 1] = 0
        self.scaling_h[0, 2] = 0
        self.scaling_h[1, 2] = 0

        self.attempted_V = 0
        self.accepted_V = 0
        self.scaling_V = 1e5

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
        for i in range(1, 6):
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
        else:
            raise NotImplementedError
        return trial_type

    def _get_isotropic_trial(self, rvecs):
        """Generates an isotropic trial"""
        dV = 2 * (np.random.uniform() - 0.5) * self.scaling_V * self.dV
        s = 1 + dV / np.linalg.det(rvecs)
        C = s ** (1.0 / 3)
        return None, C

    def _get_anisotropic_trial(self, rvecs):
        """Generates an anisotropic trial on the normalized cell tensor"""
        i = np.random.randint(1, 6)
        index = self.INDEX_MAPPING[i]
        dh = 2 * (np.random.uniform() - 0.5) * self.scaling_h[index] * self.dh
        trial = np.zeros((3, 3))
        trial[index] += dh

        # compute change in h6 component due to possible changes in h1 or h4
        s = np.linalg.det(rvecs) ** (1.0 / 3)
        shape = rvecs / s
        if i == 1 or i == 4:
            if i == 1:
                h1 = shape[0, 0] + dh
                h4 = shape[1, 1]
            elif i == 4:
                h1 = shape[0, 0]
                h4 = shape[1, 1] + dh
            else:
                raise ValueError
            h6 = 1 / (h1 * h4)
            trial[2, 2] = h6 - shape[2, 2]
            assert np.abs(np.linalg.det(rvecs) - np.linalg.det(s * (rvecs / s + trial))) < 1e-10, 'old determinant: {} \t new determinant: {}'.format(np.linalg.det(rvecs), np.linalg.det(s * (rvecs / s + trial)))
        return index, trial, 1.0

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
            s = np.linalg.det(rvecs) ** (1.0 / 3)
            rvecs_ = s * (rvecs / s + trial)
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

        contribs = [epot_ - epot]
        contribs += [self.press * (vol_ - vol)]
        contribs += [- beta ** (-1) * natom * np.log(vol_ / vol)]
        if False: #standard scheme
            contribs += [- beta ** (-1) * (-3) * np.log(rvecs_[0, 0] / rvecs[0, 0] * (vol / vol_) ** (1.0 / 3))]
            contribs += [- beta ** (-1) * (-2) * np.log(rvecs_[1, 1] / rvecs[1, 1] * (vol / vol_) ** (1.0 / 3))]
        else: #alternative scheme
            contribs += [- beta ** (-1) * np.log(rvecs_[0, 0] / rvecs[0, 0] * (vol / vol_) ** (1.0 / 3))]

        #print('==========')
        #print('TRIAL: \t {} \t to \t {}'.format(vol, vol_))
        #print(trial, C)
        #print('CONTRIBS')
        #print(contribs)
        arg = sum(contribs)
        #print('TOTAL')
        #print(arg)

        # if arg < 0, then the move should always be accepted
        # else, it should be accepted with probability np.exp(-beta * arg)
        accepted = arg < 0 or np.random.uniform(0, 1) < np.exp(-beta*arg)
        if accepted:
            self.econs_correction += epot - epot_
            #print('ACCEPTED')
        else:
            # revert the cell and the positions in the original state
            pos = np.transpose(np.dot(rvecs, np.transpose(fractional)))
            compute(pos, rvecs)
        return accepted

