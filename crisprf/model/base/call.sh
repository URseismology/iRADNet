#!/bin/bash

# run simulation for synthetic data
matlab -nodisplay -nosplash -nodesktop -r "addpath('./crisprf/model/base/'); data = load('data/Ps_RF_syn1.mat'); q = linspace(-1000, 1000, 200); m = sparse_inverse_radon_fista(data.tx', 0.02, data.rayP, q, 0, 25, 1.4, 1.0, 20); figure; imagesc(m'); colorbar; saveas(gcf, 'fig/matlab.png'); exit;"

# figure; imagesc(m'); colorbar; saveas(gcf, 'fig/matlab.png');