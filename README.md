## Problem Statement

Pytorch implementation of FISTA/LISTA in the CRISP-RF [^1].

[^1]: https://doi.org/10.1093/gji/ggad447

- Matlab-FISTA: [crisprf/model/base/sparse_inverse_radon_fista.m][Matlab-FISTA]
- Python-FISTA: [crisprf/model/FISTA.py][Python-FISTA]

[Python-FISTA]: crisprf/model/FISTA.py
[Matlab-FISTA]: crisprf/model/base/sparse_inverse_radon_fista.m

## TODO

- [ ] LISTA verification
- [ ] Python-LISTA training (more synthetic data)
- [ ] test Python-FISTA with more data

## Speed Comparison

||Matlab|Python|
|:-:|:-:|:-:|
|FISTA|1m2.467s|0m19.245s|

## Logs

Matlab-FISTA

```sh
$ time source crisprf/model/base/call.sh
real    1m2.467s
user    5m4.426s
sys     0m22.592s
```

Python-FISTA

```sh
$ time python crisprf/model/FISTA.py
real    0m19.245s
user    0m20.191s
sys     0m6.940s
```