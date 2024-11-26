#!/bin/bash

# run simulation for synthetic data
matlab -nodisplay -nosplash -nodesktop -r "addpath('./data_simulation/'); data_sim(10, 1000, 0.02, 'data/Sp_RF_syn1.mat'); exit;"