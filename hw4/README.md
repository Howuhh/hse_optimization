# HW3 - Proximal Gradient Descent & L1 Regularization

Results of method evaluation on [a1a](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html), [breast cancer](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) & synthetic (X ~ N(0, 1), w ~ U(-1, 1)) datasets.
Best entropy from scipy baseline optimizer: `0.2978755986559931`

Replicate report:
```
cd optimize_data | python generate_report.py
```

## A1A
| method                     |   entropy |   num iter |   nonzero params |   oracle calls |   time, s |
|----------------------------|-----------|------------|------------------|----------------|-----------|
| ProximalGD (lambda=0.01)   |  0.369543 |        153 |               13 |            457 |  0.113495 |
| ProximalGD (lambda=0.001)  |  0.321622 |       1730 |               47 |           5188 |  1.20121  |
| ProximalGD (lambda=0.0001) |  0.300791 |      10000 |               89 |          29997 |  7.00193  |
| ProximalGD (lambda=1e-05)  |  0.298936 |      10000 |              106 |          29997 |  6.97437  |
| ProximalGD (lambda=1e-06)  |  0.298796 |      10000 |              110 |          29998 |  6.95712  |
| ProximalGD (lambda=1e-07)  |  0.298784 |      10000 |              113 |          29997 |  6.97402  |
| ProximalGD (lambda=0.0)    |  0.298782 |      10000 |              114 |          29998 |  6.95943  |

# BREAST-CANCER
| method                     |   entropy |   num iter |   nonzero params |   oracle calls |   time, s |
|----------------------------|-----------|------------|------------------|----------------|-----------|
| ProximalGD (lambda=0.01)   | 0.102479  |         77 |                8 |            227 | 0.0329738 |
| ProximalGD (lambda=0.001)  | 0.0772797 |        708 |               10 |           2118 | 0.319893  |
| ProximalGD (lambda=0.0001) | 0.0753628 |       2637 |               11 |           7907 | 1.16031   |
| ProximalGD (lambda=1e-05)  | 0.075326  |       2733 |               11 |           8193 | 1.22019   |
| ProximalGD (lambda=1e-06)  | 0.0753255 |       2815 |               11 |           8439 | 1.23171   |
| ProximalGD (lambda=1e-07)  | 0.0753259 |       2779 |               11 |           8331 | 1.22086   |
| ProximalGD (lambda=0.0)    | 0.0753257 |       2793 |               11 |           8373 | 1.21302   |


## SYNTHETIC
| method                     |     entropy |   num iter |   nonzero params |   oracle calls |    time, s |
|----------------------------|-------------|------------|------------------|----------------|------------|
| ProximalGD (lambda=0.01)   | 0.229038    |         16 |               85 |             43 | 0.00502872 |
| ProximalGD (lambda=0.001)  | 0.0572441   |        120 |               98 |            354 | 0.0408652  |
| ProximalGD (lambda=0.0001) | 0.00885802  |        661 |              101 |           1976 | 0.237762   |
| ProximalGD (lambda=1e-05)  | 0.00094975  |       1179 |              101 |           3527 | 0.409773   |
| ProximalGD (lambda=1e-06)  | 0.00010767  |       1441 |              101 |           4311 | 0.496784   |
| ProximalGD (lambda=1e-07)  | 1.61913e-05 |       1624 |              101 |           4858 | 0.570075   |
| ProximalGD (lambda=0.0)    | 2.92638e-06 |       1811 |              101 |           5416 | 0.628443   |