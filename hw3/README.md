# HW3 - BFGS, L-BFGS

Results of all methods evaluation on [a1a](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html), [breast cancer](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) & synthetic (X ~ N(0, 1), w ~ U(-1, 1)) datasets.
Best entropy from scipy baseline optimizer: `0.2978755986559931`

Replicate report:
```
cd optimize_data | python -W ignore generate_report.py
```

## A1A 
| method                       |   entropy |   num iter |   oracle calls |   mean mem usage, MiB |   time, s |
|------------------------------|-----------|------------|----------------|-----------------------|-----------|
| Gradient Descent (armijo)    |  0.298265 |       3897 |          31377 |               105.567 |      6.63 |
| Newton (wolfe)               |  0.297876 |         10 |            157 |               108.394 |      0.07 |
| Hessian-Free Newton (armijo) |  0.297878 |         11 |            679 |               108.398 |      0.21 |
| BFGS (wolfe)                 |  0.297878 |        178 |           2207 |               109.289 |      0.55 |
| L-BFGS (wolfe)               |  0.297879 |         85 |           1189 |               109.471 |      0.31 |


## BREAST CANCER
| method                       |   entropy |   num iter |   oracle calls |   mean mem usage, MiB |   time, s |
|------------------------------|-----------|------------|----------------|-----------------------|-----------|
| Gradient Descent (armijo)    | 0.0754031 |        904 |           8136 |               109.508 |      1.02 |
| Newton (wolfe)               | 0.0753187 |          8 |             33 |               109.508 |      0.02 |
| Hessian-Free Newton (armijo) | 0.0753188 |          7 |            108 |               109.508 |      0.02 |
| BFGS (wolfe)                 | 0.0753191 |         42 |            279 |               109.512 |      0.04 |
| L-BFGS (wolfe)               | 0.0753189 |         27 |            187 |               109.512 |      0.03 |


## SYNTHETIC
| method                       |      entropy |   num iter |   oracle calls |   mean mem usage, MiB |   time, s |
|------------------------------|--------------|------------|----------------|-----------------------|-----------|
| Gradient Descent (armijo)    |  0.000368444 |        785 |           7333 |               110.348 |      0.77 |
| Newton (wolfe)               | -1.0001e-12  |          9 |             83 |               110.061 |      0.02 |
| Hessian-Free Newton (armijo) | -1.0001e-12  |          8 |           1148 |               110.035 |      0.14 |
| BFGS (wolfe)                 |  3.36889e-06 |         42 |            393 |               110.046 |      0.06 |
| L-BFGS (wolfe)               |  8.57569e-06 |         15 |            118 |               110.035 |      0.02 |


# L-BFGS History Size (on A1A)
| method              |   entropy |   num iter |   oracle calls |   mean mem usage, MiB |   time, s |
|---------------------|-----------|------------|----------------|-----------------------|-----------|
| L-BFGS (buffer=5)   |  0.297904 |        372 |           3244 |               104.811 |      0.73 |
| L-BFGS (bufffer=10) |  0.297884 |        263 |           3072 |               104.93  |      0.72 |
| L-BFGS (buffer=20)  |  0.297877 |        183 |           3361 |               104.948 |      0.77 |
| L-BFGS (buffer=50)  |  0.297879 |        103 |           1530 |               105.019 |      0.39 |
| L-BFGS (buffer=80)  |  0.297878 |         96 |           1429 |               105.122 |      0.38 |
| L-BFGS (buffer=100) |  0.297879 |         85 |           1189 |               105.148 |      0.31 |