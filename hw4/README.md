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
| ProximalGD (lambda=0.01)   |  0.3696   |        134 |               13 |            399 | 0.0920749 |
| ProximalGD (lambda=0.001)  |  0.321902 |       1099 |               46 |           3294 | 0.79949   |
| ProximalGD (lambda=0.0001) |  0.300951 |       7919 |               89 |          23755 | 5.66443   |
| ProximalGD (lambda=1e-05)  |  0.298936 |      10000 |              106 |          29997 | 7.18926   |
| ProximalGD (lambda=1e-06)  |  0.298796 |      10000 |              110 |          29998 | 7.03433   |
| ProximalGD (lambda=1e-07)  |  0.298784 |      10000 |              113 |          29997 | 6.98379   |
| ProximalGD (lambda=0.0)    |  0.298782 |      10000 |              114 |          29998 | 7.02342   |


## BREAST-CANCER
| method                     |   entropy |   num iter |   nonzero params |   oracle calls |   time, s |
|----------------------------|-----------|------------|------------------|----------------|-----------|
| ProximalGD (lambda=0.01)   | 0.102571  |         72 |                8 |            211 |  0.038115 |
| ProximalGD (lambda=0.001)  | 0.0773355 |        666 |               11 |           1993 |  0.306254 |
| ProximalGD (lambda=0.0001) | 0.0755207 |        264 |               11 |            787 |  0.116688 |
| ProximalGD (lambda=1e-05)  | 0.0754721 |        272 |               11 |            811 |  0.123319 |
| ProximalGD (lambda=1e-06)  | 0.0754698 |        274 |               11 |            817 |  0.120735 |
| ProximalGD (lambda=1e-07)  | 0.0754689 |        277 |               11 |            826 |  0.12022  |
| ProximalGD (lambda=0.0)    | 0.0754707 |        269 |               11 |            802 |  0.118381 |


## SYNTHETIC
| method                     |    entropy |   num iter |   nonzero params |   oracle calls |    time, s |
|----------------------------|------------|------------|------------------|----------------|------------|
| ProximalGD (lambda=0.01)   | 0.229121   |         13 |               85 |             35 | 0.00393391 |
| ProximalGD (lambda=0.001)  | 0.0575409  |         85 |               98 |            250 | 0.0460989  |
| ProximalGD (lambda=0.0001) | 0.00971367 |        412 |              101 |           1230 | 0.14752    |
| ProximalGD (lambda=1e-05)  | 0.00193049 |        554 |              100 |           1654 | 0.189266   |
| ProximalGD (lambda=1e-06)  | 0.00110244 |        573 |              101 |           1710 | 0.191674   |
| ProximalGD (lambda=1e-07)  | 0.00100853 |        570 |              101 |           1701 | 0.195064   |
| ProximalGD (lambda=0.0)    | 0.00102185 |        566 |              101 |           1689 | 0.189988   |