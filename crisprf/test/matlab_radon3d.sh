#!/bin/bash

# run simulation for synthetic data
matlab -nodisplay -nosplash -nodesktop -r "addpath('./crisprf/base/'); lip_test(); exit;"

pytest crisprf/test/test_radon.py