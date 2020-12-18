# HW3 - Proximal Gradient Descent & L1 Regularization

Results of method evaluation on [a1a](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html), [breast cancer](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) & synthetic (X ~ N(0, 1), w ~ U(-1, 1)) datasets.
Best entropy from scipy baseline optimizer: `0.2978755986559931`

Replicate report:
```
cd optimize_data | python generate_report.py
```

## A1A
| method                     |   entropy |   num iter |   oracle calls |   time, s |
|----------------------------|-----------|------------|----------------|-----------|
| ProximalGD (lambda=0.01)   |  0.369543 |        153 |            457 |  0.115764 |
| ProximalGD (lambda=0.001)  |  0.321622 |       1730 |           5188 |  1.65566  |
| ProximalGD (lambda=0.0001) |  0.300791 |      10000 |          29997 |  8.89954  |
| ProximalGD (lambda=1e-05)  |  0.298936 |      10000 |          29997 |  8.41204  |
| ProximalGD (lambda=1e-06)  |  0.298796 |      10000 |          29998 |  7.31488  |
| ProximalGD (lambda=1e-07)  |  0.298784 |      10000 |          29997 |  7.47851  |
| ProximalGD (lambda=0.0)    |  0.298782 |      10000 |          29998 |  7.29556  |

# BREAST-CANCER
| method                     |   entropy |   num iter |   oracle calls |   time, s |
|----------------------------|-----------|------------|----------------|-----------|
| ProximalGD (lambda=0.01)   | 0.102479  |         77 |            227 |  0.049993 |
| ProximalGD (lambda=0.001)  | 0.0772797 |        708 |           2118 |  0.327867 |
| ProximalGD (lambda=0.0001) | 0.0753628 |       2637 |           7907 |  1.24972  |
| ProximalGD (lambda=1e-05)  | 0.075326  |       2733 |           8193 |  1.27796  |
| ProximalGD (lambda=1e-06)  | 0.0753255 |       2815 |           8439 |  1.31048  |
| ProximalGD (lambda=1e-07)  | 0.0753259 |       2779 |           8331 |  1.30299  |
| ProximalGD (lambda=0.0)    | 0.0753257 |       2793 |           8373 |  1.38041  |


## SYNTHETIC
| method                     |     entropy |   num iter |   oracle calls |    time, s |
|----------------------------|-------------|------------|----------------|------------|
| ProximalGD (lambda=0.01)   | 0.229038    |         16 |             43 | 0.00663185 |
| ProximalGD (lambda=0.001)  | 0.0572441   |        120 |            354 | 0.0607071  |
| ProximalGD (lambda=0.0001) | 0.00885802  |        661 |           1976 | 0.276175   |
| ProximalGD (lambda=1e-05)  | 0.00094975  |       1179 |           3527 | 0.444099   |
| ProximalGD (lambda=1e-06)  | 0.00010767  |       1441 |           4311 | 0.538235   |
| ProximalGD (lambda=1e-07)  | 1.61913e-05 |       1624 |           4858 | 0.614633   |
| ProximalGD (lambda=0.0)    | 2.92638e-06 |       1811 |           5416 | 0.720773   |