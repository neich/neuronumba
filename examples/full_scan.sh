# This script has to be executed with the corresponding virtual environment activted
# and the PYTHONPATH variable has to point to NEURONUMBA_DIR/src (if installed from repo)


python3 global_coupling_fitting.py \
    --nsubj 21 \
    --observables FC,KS,PS \
    --observables phFCD,KS,PS \
    --observables swFCD,KS,PS \
    --model Montbrio \
    --obs-var r_e \
    --scale-signal 100.0 \
    --param range g 0.2 6.0 0.2 \
    --param single auto-fic False \
    --tr 720.0 \
    --tmax 600.0 \
    --fmri-path ./Data_Raw/ebrains_popovych \
    --out-path ./Data_Produced/full_scan_montbrio_no_fic \
    --nproc 24

