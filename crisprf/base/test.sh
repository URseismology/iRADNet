#!/bin/bash

# run simulation for synthetic data
matlab -nodisplay -nosplash -nodesktop -r "addpath('./crisprf/base/'); data = load('data/Ps_RF_syn1.mat'); radon3d_forward_test(data.rayP); exit;"

# figure; imagesc(m'); colorbar; saveas(gcf, 'fig/matlab.png');