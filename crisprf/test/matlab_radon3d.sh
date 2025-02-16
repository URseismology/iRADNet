#!/bin/bash

# run simulation for synthetic data
matlab -nodisplay -nosplash -nodesktop -r "addpath('./crisprf/base/'); test_lip(); exit;"

pytest crisprf/test/test_radon3d.py