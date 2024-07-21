# Neuronumba

This is an experiment to try to produce a library with the same functionality and API than
["The Virtual Brain" Project (TVB Project)](https://github.com/the-virtual-brain/tvb-root) 
but making full use of [Numba](https://numba.pydata.org/) to accelerate the computations 
and achieve performance levels similar to using C++ with dedicated optimized vector 
libraries as [Eigen](https://eigen.tuxfamily.org/).

"The Virtual Brain" Project (TVB Project) has the purpose of offering 
modern tools to the Neurosciences community, for computing, simulating
and analyzing functional and structural data of human brains, brains modeled 
at the  level of population of neurons.

Right now, neuronumba provides 3 main tools:

1. Simulation of Whole Brain Models, where the user provides a connectivity matrix, 
a model, an integrator, and a monitor to extract a signal from the simulation.

2. Compute a BOLD signal from a firing rate signal obtained from the simulation.

3. Compute a measure of the BOLD signal that can be compared with epirically obtained BOLD signals with fMRI.

A simple exemple of these three steps looks like this:

```python
import numpy as np

from neuronumba.simulator.connectivity import Connectivity
from neuronumba.simulator.models import Naskar
from neuronumba.simulator.integrators import EulerDeterministic
from neuronumba.simulator.coupling import CouplingLinearNoDelays
from neuronumba.simulator.monitors import RawSubSample
from neuronumba.simulator.simulator import Simulator
from neuronumba.bold import BoldStephan2008
from neuronumba.bold.filters import BandPassFilter
from neuronumba.observables.ph_fcd import PhFCD
from neuronumba.observables.measures import KolmogorovSmirnovStatistic


if __name__ == '__main__':
    weights = np.load('your_connectivity_matrix.npy')
    processed = np.load('your_empirical_preprocessed_measure.npy')

    n_rois = weights.shape[0]

    # We create a "fake" length matrix since we are simulating with no delays
    lengths = np.random.rand(n_rois, n_rois) * 10.0 + 1.0
    speed = 1.0

    # Initialize connectivity
    con = Connectivity(weights=weights, lengths=lengths, speed=speed)

    # Create the model
    m = Naskar()

    # Initialize the integrator
    dt = 0.1
    integ = EulerDeterministic(dt=dt)

    # Initialize the coupling, in this case linear with no delays
    coupling = CouplingLinearNoDelays(weights=weights, c_vars=np.array([0], dtype=np.int32), n_rois=n_rois)

    # Create a monitor that subsamples the signal each 1ms
    monitor = RawSubSample(period=1.0)

    # Initialize the simulator and run
    s = Simulator(connectivity=con, model=m, coupling=coupling, integrator=integ, monitors=[monitor])
    s.run(0, 100000)

    # Generate BOLD signal from monitor data
    b = BoldStephan2008()
    signal = monitor.data()[:, 0, :]
    bold = b.compute_bold(signal, monitor.period)

    # Generate measure data from BOLD signal and compare with empirical
    ph_fcd = PhFCD()
    bpf = BandPassFilter(k=2, flp=0.01, fhi=0.1, tr=2.0)
    bold_filt = bpf.filter(bold)
    simulated = ph_fcd.from_fmri(bold_filt)
    measure = KolmogorovSmirnovStatistic()
    distance = measure.distance(processed, simulated)
```


