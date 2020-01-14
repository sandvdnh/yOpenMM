import molmod

import numpy as np

from yaff.sampling.npt import McDonaldBarostat
from yaff.sampling.verlet import VerletHook


class MonteCarloBarostat(VerletHook):
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

    def __init__(self, temp, press, start=0, step=10, mode='full', type_prob=0.5):
        self.temp = temp
        self.press = press
        self.dim = 3
        self.baro_ndof = None
        self.mode = mode

        self.attempted_h = np.zeros((3, 3))
        self.accepted_h = np.zeros((3, 3))
        self.scaling_h = np.ones((3, 3)) * (2e1)
        self.scaling_h[0, 1] = 0
        self.scaling_h[0, 2] = 0
        self.scaling_h[1, 2] = 0

        self.attempted_V = 0
        self.accepted_V = 0
        self.scaling_V = 1e4

        self.type_prob = type_prob
        VerletHook.__init__(self, start, step)

    def init(self, iterative):
        pass

    def pre(self, iterative, chainvel0=None):
        pass

    def post(self, iterative, chainvel0=None):
        # compute reduced coordinates
        rvecs = iterative.ff.system.cell._get_rvecs().copy()
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
        """Generates an anisotropic trial"""
        i = np.random.randint(1, 7)
        index = self.INDEX_MAPPING[i]
        dh = 2 * (np.random.uniform() - 0.5) * self.scaling_h[index] * self.dh
        trial = np.zeros((3, 3))
        trial[index] += dh

        # isotropic rescaling to maintain volume after trial
        if (i == 1) or (i == 3) or (i == 6):
            h = rvecs[index]
            s = ((h + dh) / h) ** (-1)
        else:
            s = 1
        C = s ** (1.0 / 3)
        return index, trial, s ** (1.0 / 3)

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
            rvecs_ = rvecs + trial
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
        arg += - beta ** (-1) * (1 - self.dim + natom) * np.log(vol_ / vol)
        arg += - beta ** (-1) * 1 * np.log(rvecs_[1, 1] / rvecs[1, 1])
        arg += - beta ** (-1) * 2 * np.log(rvecs_[2, 2] / rvecs[2, 2])

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
