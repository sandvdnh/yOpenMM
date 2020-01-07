from yaff.sampling.npt import McDonaldBarostat


class MonteCarloBarostat(VerletHook):
    name = 'MonteCarlo'
    kind = 'stochastic'
    method = 'barostat'

    def __init__(self, temp, press, start=0, step=1, amp=1e-3):
        self.temp = temp
        self.press = press
        self.amp = amp
        self.dim = 3
        self.numAttempted = 0
        self.numAccepted = 0
        self.scale = None

        self.baro_ndof = None

        VerletHook.__init__(self, start, step)

    def init(self, iterative):
        pass

    def pre(self, iterative, chainvel0=None):
        pass

    def post(self, iterative, chainvel0=None):
        pass

