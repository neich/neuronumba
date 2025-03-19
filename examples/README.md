# How to run the examples

## `global_coupling_fitting.py`

This file contains an example to find the optimal global coupling factor tha makes the model the closest possible to the empirical data.

First of all, check the command line options:

`PYTHONPATH=../src python3 global_coupling_fitting.py --help`

And an exemple of execution would be:

`PYTHONPATH=../src python3 global_coupling_fitting.py --nproc 5 --model Deco2014 --g-range 8.0 11.0 0.1 --tr 720 --tmax 600   --fmri-path ./Data_Raw/ebrains_popovych --out-path ./Data_Produced/ebrains_popovych`