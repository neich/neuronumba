# Neuronumba

This is an experiment to come up with a library that has the same functionality and API (or a similar as possible) than ["The Virtual Brain" Project (TVB Project)](https://github.com/the-virtual-brain/tvb-root) but making full use of [Numba](https://numba.pydata.org/) to accelerate the computations and achieve performance levels similar to C++ with dedicated optimized vector libraries as [Eigen](https://eigen.tuxfamily.org/).

"The Virtual Brain" Project (TVB Project) has the purpose of offering modern tools to the Neurosciences community, for computing, simulating and analyzing functional and structural data of human brains, brains modeled at the  level of population of neurons.

Right now, [neuronumba](https://github.com/neich/neuronumba) provides 3 main tools:

1. Simulation of Whole Brain Models, where the user provides a connectivity matrix, a model, a coupling mechanism, an integrator, and a monitor to extract a signal from the simulation.

2. Compute a BOLD signal from a firing rate signal obtained from the simulation. The idea is to simulate Functional Magnetic Resonance Imaging from synthetic data

3. Compute a measure of the BOLD signal that can be compared with empirically obtained BOLD signals with fMRI.

There are some Jupyter [notebooks](notebooks) to show how the library works.

## Installation

The package is not in the pip repository, but temporarily you can install it with:

`pip install -e "git+https://github.com/neich/neuronumba.git#egg=neuronumba&subdirectory=src"`

