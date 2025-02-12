#!/bin/bash

# run simulation for synthetic data
matlab -nodisplay -nosplash -nodesktop -r "addpath('./crisprf/base/'); radon3d_forward_test(); exit;"

# figure; imagesc(m'); colorbar; saveas(gcf, 'fig/matlab.png');