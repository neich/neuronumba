import numpy as np

from neuronumba.basic.attr import Attr, HasAttr
from neuronumba.simulator.integrators import EulerStochastic
from neuronumba.simulator.simulator import simulate_nodelay

class FIC(HasAttr):
    dim = Attr(required=True)
    
    def compute_J(self, sc, g):
        raise NotImplementedError
    
    
class FICHerzog2022(FIC):
    alpha = Attr(default=0.75)
    
    def compute_J(self, sc, g):
        J = self.alpha * g * np.sum(sc, axis=0) + 1
        return J

class FICDeco2014(FIC):
    verbose = Attr(default=False)
    very_verbose = Attr(default=False)
    use_N_algorithm = Attr(default=True)
    model = Attr(required=True)
    obs_var = Attr(required=True)
    integrator = Attr(required=True)
    t_max = Attr(default=10000.0)
    t_warmup = Attr(default=1000.0)
    rest_rate = Attr(default=0.4032)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._min_largest_distance = None
        self._slow_factor = None

    def _update_J(self, N, tmax, delta, curr, J):  # This is the original method by Gus, from the paper...
        tmin = 1000 if (tmax > 1000) else int(tmax / 10)
        currm = np.mean(curr[tmin:tmax, :], 0)  # takes the mean of all xn values along dimension 1...
        # This is the "averaged level of the input of the local excitatory pool of each brain area,
        # i.e., I_i^{(E)}" in the text (pp 7889, right column, subsection "FIC").
        flag = 0
        if self.very_verbose: print()
        if self.very_verbose: print("[", end='')
        for n in range(N):
            if np.abs(
                    currm[n] + 0.026) > 0.005:  # if currm_i < -0.026 - 0.005 or currm_i > -0.026 + 0.005 (a tolerance)
                if currm[n] < -0.026:  # if currm_i < -0.026
                    J[n] = J[n] - delta[n]  # down-regulate
                    delta[n] = delta[n] - 0.001
                    if delta[n] < 0.001:
                        delta[n] = 0.001
                    if self.very_verbose: print("v", end='')
                else:  # if currm_i >= -0.026 (in the paper, it reads =)
                    J[n] = J[n] + delta[n]  # up-regulate
                    if self.very_verbose: print("^", end='')
            else:
                flag = flag + 1
                if self.very_verbose: print("-", end='')
        if self.very_verbose: print("]")
        return flag

    def _updateJ_N(self, N, tmax, delta, curr, J):  # 2nd version of updateJ
        tmin = 1000 if (tmax > 1000) else int(tmax / 10)
        currm = np.mean(curr[tmin:tmax, :], 0)  # takes the mean of all xn values along dimension 1...
        # This is the "averaged level of the input of the local excitatory pool of each brain area,
        # i.e., I_i^{(E)}" in the text (pp 7889, right column, subsection "FIC").
        flag = 0
        if self.very_verbose: print()
        if self.very_verbose: print("[", end='')
        # ===========================================================
        distance = np.full((N,), 10.0)
        num_above_error = 0
        largest_distance = 0
        # total_error = 0.0
        Si = np.zeros((N,))
        for i in range(N):
            # ie_100 = curr[i]  # d_raw[-100:-1, 1, i, 0]  # I_e
            # ie = currm[i]  # np.average(ie_100)
            d = currm[i] + 0.026  # ie - be_ae + 0.026
            distance[i] = d
            d_abs = abs(d)
            if largest_distance < d_abs:
                largest_distance = d_abs
            # Si[i] = np.average(d_raw[-100:-1, 2, i, 0])  # S_i
            # error_i = d*d
            # error[i] = error_i
            # total_error += error_i

        if largest_distance < self._min_largest_distance:
            self._min_largest_distance = largest_distance
        else:
            self._slow_factor *= 0.5

        for i in range(N):
            d = distance[i]  # currm[i] + 0.026
            d_abs = np.abs(d)
            if d_abs > 0.005:  # if currm_i < -0.026 - 0.005 or currm_i > -0.026 + 0.005 (a tolerance)
                num_above_error += 1
                delta_i = self._slow_factor * d_abs / 0.1  # Si[i]  # 0.003 * abs(d + 0.026) / 0.026
                if delta_i < 0.005:
                    delta_i = 0.005
                delta[i] = np.sign(d) * delta_i
            else:
                delta[i] = 0.0
            J[i] = J[i] + delta[i]
        if self.very_verbose: print("]")
        return N - num_above_error

    def compute_J(self, sc, g):
        # simulation fixed parameters:
        # ----------------------------
        dt = 0.1
        tmax = 10000

        N = sc.shape[0]
        # initialization:
        # -------------------------
        delta = 0.02 * np.ones(N)
        # A couple of initializations, needed only for updateJ_2
        self._min_largest_distance = np.inf
        self._slow_factor = 1.0

        if self.verbose:
            print("  Trials:", end=" ", flush=True)

        # ======== Balance (greedy algorithm)
        # note that we used stochastic equations to estimate the JIs
        # Doing that gives more stable solutions as the JIs for each node will be
        # a function of the variance.
        currJ = np.ones(N)
        bestJ = np.ones(N);
        bestJCount = -1;
        bestTrial = -1
        for k in range(5000):  # 5000 trials
            # integrator.resetBookkeeping()
            t_max_neuronal = int((tmax + dt))  # (tmax+dt)/dt, but with steps of 1 unit...
            self.model.configure(J=currJ)
            # recompileSignatures()
            signal = simulate_nodelay(self.model, self.integrator, sc, self.obs_var, self.t_max, self.t_warmup)

            if self.verbose:
                print(k, end='', flush=True)

            signal_d = signal - self.rest_rate
            if self.use_N_algorithm:
                flagJ = self._updateJ_N(N, tmax, delta, signal_d, currJ)  # Nacho's method... ;-)
            else:
                flagJ = self._updateJ(N, tmax, delta, signal_d, currJ)  # Gus' method, the one from [DecoEtAl2014]

            if self.verbose:
                print("({})".format(flagJ), end='', flush=True)
            if flagJ > bestJCount:
                bestJCount = flagJ
                bestJ = currJ
                bestTrial = k
                if self.verbose: print(' New min!!!', end='', flush=True)
            if flagJ == N:
                if self.verbose: print('Out !!!', flush=True)
                break
            else:
                if self.verbose: print(', ', end='', flush=True)

        if self.verbose:
            print("Final (we={}): {} trials, with {}/{} nodes solved at trial {}".format(g, k, bestJCount, N, bestTrial))
        if self.verbose:
            print('DONE!') if flagJ == N else print('FAILED!!!')
        return bestJ

        
    