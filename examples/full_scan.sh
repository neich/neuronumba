#!/bin/bash

# This script has to be executed with the corresponding virtual environment activted
# and the PYTHONPATH variable has to point to NEURONUMBA_DIR/src (if installed from repo)

python3 global_coupling_fitting.py \
    --nsubj 21 \
    --bpf 2 0.005 0.08 \
    --observables FC,KS,PS phFCD,KS,PS swFCD,KS,PS \
    --model Deco2014 \
    --obs-var re \
    --param range g 0.2 6.0 0.2 \
    --param single auto_fic True \
    --tr 720.0 \
    --tmax 600.0 \
    --fmri-path ./Data_Raw/ebrains_popovych \
    --out-path ./Data_Produced/full_scan_deco2014_fic \
    --nproc 24

python3 global_coupling_fitting.py \
    --plot-g \
    --out-path ./Data_Produced/full_scan_deco2014_no_fic


python3 global_coupling_fitting.py \
    --nsubj 21 \
    --bpf 2 0.005 0.08 \
    --observables FC,KS,PS phFCD,KS,PS swFCD,KS,PS \
    --model Montbrio \
    --obs-var r_e \
    --scale-signal 100.0 \
    --param range g 0.2 6.0 0.2 \
    --param single auto_fic False \
    --tr 720.0 \
    --tmax 600.0 \
    --fmri-path ./Data_Raw/ebrains_popovych \
    --out-path ./Data_Produced/full_scan_montbrio_no_fic \
    --nproc 24

python3 global_coupling_fitting.py \
    --plot-g \
    --out-path ./Data_Produced/full_scan_montbrio_no_fic
    
