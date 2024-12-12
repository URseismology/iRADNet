#!/bin/bash

# run simulation for synthetic data
matlab -nodisplay -nosplash -nodesktop -r "addpath('./data_simulation/'); data_sim(100, 0, 5000, 0.02, 'data/Ps_RF_syn'); exit;"