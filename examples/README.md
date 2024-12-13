# How to run the examples

## `global_coupling_fitting.py`

This file contains an example to find the optimal global coupling factor tha makes the model the closest possible to the empirical data.

First f all, check the command line options:

`PYTHONPATH=../src python3 global_coupling_fitting.py --help`

And an exemple of execution would be:

`PYTHONPATH=../src python3 global_coupling_fitting.py --sc-scaling 1.0 --tmax 180 --tr 2.0 --g-range 2.0 3.0 0.1  --fmri-path ./Data_Raw/ebrains_popovych --out-path ./Data_Produced/ebrains_popovych`