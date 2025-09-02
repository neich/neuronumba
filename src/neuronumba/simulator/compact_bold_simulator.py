import time
import warnings
import math
import numpy as np

from neuronumba.basic.attr import Attr, HasAttr
from neuronumba.simulator.models import Hopf, Deco2014, Montbrio
from neuronumba.bold import BoldStephan2008
from neuronumba.simulator.monitors import RawSubSample, TemporalAverage
from neuronumba.simulator.simulator import Simulator
from neuronumba.simulator.connectivity import Connectivity
from neuronumba.simulator.history import HistoryNoDelays
from neuronumba.simulator.integrators import EulerStochastic

class CompactBoldSimulatorBase(HasAttr):
    weights = Attr(required=True, doc="Structural connectivity weights")
    use_temporal_avg_monitor = Attr(default=False, doc="Use the TemporalAverage monitor? Defaults to using the RawSubmonitor")

    def generate_bold(
        self,
        warmup_samples: int,
        simulated_samples: int
    ) -> np.ndarray:
        start_time = time.perf_counter()
        simulated_bold = self._generate_bold(warmup_samples, simulated_samples)
        elapsed_time = time.perf_counter() - start_time
        print(f"Bold simulation completed. Took: {elapsed_time:.3e}s")
        return simulated_bold

    def _generate_bold(
        self,
        warmup_samples: int,
        simulated_samples: int
    ) -> np.ndarray:
        raise NotImplementedError()

class CompactHopfSimulator(CompactBoldSimulatorBase):

    a = Attr(default=-0.5, doc="Hopf bifurcation parameter")
    omega = Attr(default=0.3, doc="Hopf frequencies in rad/s")
    g = Attr(required=True, doc="Coupling parameter")
    sigma = Attr(default=1e-03, doc="Noise amplitud")
    # sampling_period_s = Attr(default=1.0, doc="Sampling period. tr/sampling_period should be exact to not introduce misaligns")
    tr = Attr(required=True, doc="Actual TR in milliseconds")
    dt = Attr(default=0.1, doc="Delta time for the simulation in milliseconds")
    model = Attr(default=None, doc="If need to costum configure the model. Must be a Hopf model")

    # IMPORTANT: Hopf is integrated in milliseconds, but to keep API consistency, we pass the parameters (tr and dt) in milliseconds.
    # So remember to convert back to seconds before using it for the Hopf simulation 

    def _generate_bold(
        self,
        warmup_samples: int,
        simulated_samples: int
    ) -> np.ndarray:

        model = self.model
        if not model:
            model = Hopf()
        elif not isinstance(mode, Hopf):
            raise f"Model instance must be Hopf. Provided <{model.__class__.__name__}>"
        model.a = self.a
        model.omega = self.omega
        model.configure(weights=self.weights, g=self.g)

        obs_var = 'x'
        n_roi = np.shape(self.weights)[0]

        # Prepare everything
        # Remember that Hopf is integrated in seconds and not milliseconds
        integrator = EulerStochastic(dt=(self.dt/1000.0), sigmas=np.r_[self.sigma, self.sigma])
        con = Connectivity(
            weights=self.weights, 
            lengths=np.random.rand(n_roi, n_roi)*10.0 + 1.0, 
            speed=1.0
        )
        history = HistoryNoDelays()
        monitor = None
        if self.use_temporal_avg_monitor:
            monitor = TemporalAverage(
                period=(self.tr / 1000.0),
                monitor_vars=model.get_var_info([obs_var])
            )
        else:
            monitor = RawSubSample(
                period=(self.tr / 1000.0),
                monitor_vars=model.get_var_info([obs_var])
            )
        sim = Simulator(
            connectivity=con,
            model=model,
            history=history,
            integrator=integrator,
            monitors=[monitor]
        )

        # Run simulation
        sim.run(0, math.ceil((warmup_samples + simulated_samples) * (self.tr / 1000.0)))

        # Retreive simulated data and remove warmup
        sim_signal = monitor.data(obs_var)
        start_idx = int(warmup_samples)
        sim_signal = sim_signal[start_idx:, :]

        # NOTE: I don't think this is needed, we can treat sim_signal as the actual bold_signal
        # without any transformations. If we need some sub-sampling averaging, then we just can
        # use the TemporalAverage Monitor for the Compact simulator. So for now, we just comment 
        # all this code, and return the simulated signal as bold signal

        # # Now we need to convert the signal to samples of size tr
        # # 
        # # Number of samples per time bin
        # n = int((self.tr/1000.0) / self.sampling_period_s)
        # # Number of timepoints on the signal
        # l = sim_signal.shape[0]
        # # Make it multiple of n (with nan padding)
        # tmp1 = np.pad(sim_signal, ((0, n - l % n), (0, 0)),
        #                         mode='constant',
        #                         constant_values=np.nan)
        # # Reshape into blocks
        # tmp2 = tmp1.reshape(n, int(tmp1.shape[0]/n), -1)
        # # This is the final simulated time series
        # bold_signal = np.nanmean(tmp2, axis=0)

        bold_signal = sim_signal

        return bold_signal

class CompactDeco2014Simulator(CompactBoldSimulatorBase):

    g = Attr(required=True, doc="Coupling parameter")
    sigma = Attr(default=1e-03, doc="Noise amplitud")
    # sampling_period_ms = Attr(default=1.0, doc="Sampling period. tr/sampling_period should be exact to not introduce misaligns")
    tr = Attr(required=True, doc="Actual TR in milliseconds")
    dt = Attr(default=0.1, doc="Delta time for the simulation in milliseconds")
    model = Attr(default=None, doc="If need to costum configure the model. It must be a Deco2014 model")

    def _generate_bold(
        self,
        warmup_samples: int,
        simulated_samples: int
    ) -> np.ndarray:

        model = self.model
        if not model:
            model = Deco2014(auto_fic=True)
        elif not isinstance(mode, Deco2014):
            raise f"Model instance must be Deco2014. Provided <{model.__class__.__name__}>"
        model.configure(weights=self.weights, g=self.g)

        obs_var = 're'
        n_roi = np.shape(self.weights)[0]

        # Prepare everything
        integrator = EulerStochastic(dt=self.dt, sigmas=np.r_[self.sigma, self.sigma])
        con = Connectivity(
            weights=self.weights, 
            lengths=np.random.rand(n_roi, n_roi)*10.0 + 1.0,
            speed=1.0
        )
        history = HistoryNoDelays()
        monitor = None
        if self.use_temporal_avg_monitor:
            monitor = TemporalAverage(
                period=(self.tr / 1000.0),
                monitor_vars=model.get_var_info([obs_var])
            )
        else:
            monitor = RawSubSample(
                period=(self.tr / 1000.0),
                monitor_vars=model.get_var_info([obs_var])
            )
        sim = Simulator(
            connectivity=con,
            model=model,
            history=history,
            integrator=integrator,
            monitors=[monitor]
        )

        # Run simulation
        sim.run(0, math.ceil((warmup_samples + simulated_samples) * self.tr))

        # Retreive simulated data and remove warmup
        sim_signal = monitor.data(obs_var)
        start_idx = int(sim_signal.shape[0] * warmup_samples / (warmup_samples + simulated_samples))
        sim_signal = sim_signal[start_idx:, :]

        # We can proceed to convert the signal to bold
        bold_converter = BoldStephan2008(tr=self.tr)
        bold_signal = b.compute_bold(sim_signal, monitor.period)

        return bold_signal

class CompactMontbrioSimulator(CompactBoldSimulatorBase):

    g = Attr(required=True, doc="Coupling parameter")
    sigma = Attr(default=1e-03, doc="Noise amplitud")
    tr = Attr(required=True, doc="Actual TR in milliseconds")
    dt = Attr(default=0.1, doc="Delta time for the simulation in milliseconds")
    model = Attr(default=None, doc="If need to costum configure the model. It must be a Montbrio model")

    def _generate_bold(
        self,
        warmup_samples: int,
        simulated_samples: int
    ) -> np.ndarray:

        model = self.model
        if not model:
            model = Montbrio()
        elif not isinstance(mode, Montbrio):
            raise f"Model instance must be DecoMontbrio2014. Provided <{model.__class__.__name__}>"
        model.configure(weights=self.weights, g=self.g)

        obs_var = 'r_e'
        n_roi = np.shape(self.weights)[0]

        # Prepare everything
        integrator = EulerStochastic(dt=self.dt, sigmas=np.r_[self.sigma, 0.0, 0.0, 0.0, 0.0, 0.0])
        con = Connectivity(
            weights=self.weights, 
            lengths=np.random.rand(n_roi, n_roi)*10.0 + 1.0,
            speed=1.0
        )
        history = HistoryNoDelays()
        monitor = None
        if self.use_temporal_avg_monitor:
            monitor = TemporalAverage(
                period=(self.tr / 1000.0),
                monitor_vars=model.get_var_info([obs_var])
            )
        else:
            monitor = RawSubSample(
                period=(self.tr / 1000.0),
                monitor_vars=model.get_var_info([obs_var])
            )
        sim = Simulator(
            connectivity=con,
            model=model,
            history=history,
            integrator=integrator,
            monitors=[monitor]
        )

        # Run simulation
        sim.run(0, math.ceil((warmup_samples + simulated_samples) * self.tr))

        # Retreive simulated data and remove warmup
        sim_signal = monitor.data(obs_var)
        start_idx = int(sim_signal.shape[0] * warmup_samples / (warmup_samples + simulated_samples))
        sim_signal = sim_signal[start_idx:, :]

        # We can proceed to convert the signal to bold
        bold_converter = BoldStephan2008(tr=self.tr)
        bold_signal = b.compute_bold(sim_signal, monitor.period)

        return bold_signal
