#!/bin/bash

N_LAYERS=10

# fista, no training needed so eval directly
for snr in 1 2 5 10
do
    python crisprf/job/run_fista.py --snr $snr --n_layers $N_LAYERS 
done

# model training
for model in SRT_LISTA SRT_LISTA_CP SRT_AdaLISTA SRT_AdaLFISTA
do
    python crisprf/job/run_lista.py --train --n_layers $N_LAYERS --model $model
done

# model evaluation
for model in SRT_LISTA SRT_LISTA_CP SRT_AdaLISTA SRT_AdaLFISTA
do
    python crisprf/job/run_lista.py --eval --model $model
    for snr in 1 2 5 10
    do
        python crisprf/job/run_lista.py --eval --snr $snr --n_layers $N_LAYERS --model $model
    done
done