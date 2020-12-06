# HW2 - Gradient Descent, Newton, Hessian-Free Newton

Results of all methods evaluation on [a1a](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) dataset.
Best entropy from scipy baseline optimizer: `0.2978755986559931`

Replicate report:
```
cd optimize_data | python -W ignore generate_report.py
```

## Gradien Descent 
| line search   |   entropy |   num iter |   oracle calls |   time, s |
|---------------|-----------|------------|----------------|-----------|
| golden        |  0.298463 |      10000 |         240001 |     44.95 |
| brent         |  0.298267 |       5196 |          89963 |     16.67 |
| armijo        |  0.298265 |       3897 |          31377 |      6.72 |
| wolfe         |  0.298402 |      10000 |          69433 |     17.12 |
| lipschitz     |  0.298844 |      10000 |          49999 |     10.98 |

## Newton 
| line search   |   entropy |   num iter |   oracle calls |   time, s |
|---------------|-----------|------------|----------------|-----------|
| golden        |  0.297875 |         11 |            265 |      0.09 |
| brent         |  0.297875 |          9 |            101 |      0.05 |
| armijo        |  0.297875 |         10 |             78 |      0.05 |
| wolfe         |  0.297876 |         11 |            178 |      0.07 |

## Hessian-Free Newton
| line search   |   entropy |   num iter |   oracle calls |   time, s |
|---------------|-----------|------------|----------------|-----------|
| golden        |  0.297879 |         10 |           1110 |      0.5  |
| brent         |  0.297876 |         10 |           1442 |      0.72 |
| armijo        |  0.297877 |         11 |           1011 |      0.52 |
| wolfe         |  0.297876 |         12 |           1705 |      1.01 |