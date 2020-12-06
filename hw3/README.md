# HW3 - BFGS, L-BFGS

Results of all methods evaluation on [a1a](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html), [breast cancer](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) & synthetic (X ~ N(0, 1), w ~ U(-1, 1)) datasets.
Best entropy from scipy baseline optimizer: `0.2978755986559931`

Replicate report:
```
cd optimize_data | python -W ignore generate_report.py
```

## A1A 
| method                       |   entropy |   num iter |   oracle calls |   time, s |
|------------------------------|-----------|------------|----------------|-----------|
| Gradient Descent (armijo)    |  0.298265 |       3897 |          31377 |      7.03 |
| Newton (wolfe)               |  0.297876 |         10 |            157 |      0.07 |
| Hessian-Free Newton (armijo) |  0.297877 |         11 |           1011 |      0.52 |
| BFGS (wolfe)                 |  0.297883 |         76 |           1810 |      0.21 |
| L-BFGS (wolfe)               |  0.297886 |         81 |           2912 |      0.3  |


## BREAST CANCER
| method                       |   entropy |   num iter |   oracle calls |   time, s |
|------------------------------|-----------|------------|----------------|-----------|
| Gradient Descent (armijo)    | 0.0754031 |        904 |           8136 |      1.04 |
| Newton (wolfe)               | 0.0753187 |          8 |             33 |      0.02 |
| Hessian-Free Newton (armijo) | 0.0753188 |          7 |            114 |      0.03 |
| BFGS (wolfe)                 | 0.0753767 |         30 |            340 |      0.03 |
| L-BFGS (wolfe)               | 0.0753189 |         27 |            527 |      0.03 |


## SYNTHETIC
| method                       |      entropy |   num iter |   oracle calls |   time, s |
|------------------------------|--------------|------------|----------------|-----------|
| Gradient Descent (armijo)    |  0.00182728  |       2978 |          29230 |      2.49 |
| Newton (wolfe)               | -1.0001e-12  |          9 |            101 |      0.01 |
| Hessian-Free Newton (armijo) | -1.0001e-12  |          6 |             90 |      0.01 |
| BFGS (wolfe)                 |  0.000235803 |         14 |            214 |      0.02 |
| L-BFGS (wolfe)               |  6.78331e-06 |         17 |            408 |      0.02 |