# HW2 - Gradient Descent, Newton, Hessian-Free Newton

Results of all methods evaluation on [a1a](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) dataset.
Best entropy by scipy baseline optimizer: `0.2978755986559931`

## Gradien Descent 
| line search   |   entropy |   num iter |   oracle calls |   time, s |
|---------------|-----------|------------|----------------|-----------|
| golden        |  0.298463 |      10000 |         240001 |     42.96 |
| brent         |  0.298267 |       5196 |          89963 |     16.88 |
| armijo        |  0.298265 |       3897 |          31377 |      6.91 |
| wolfe         |  0.298402 |      10000 |          69433 |     17.61 |
| lipschitz     |  0.298976 |      10000 |          30001 |       8.4 |

## Newton 
| line search   |   entropy |   num iter |   oracle calls |   time, s |
|---------------|-----------|------------|----------------|-----------|
| golden        |  0.297875 |         11 |            265 |      0.1  |
| brent         |  0.297875 |          9 |            101 |      0.05 |
| armijo        |  0.297875 |         10 |             78 |      0.05 |
| wolfe         |  0.297876 |         11 |            178 |      0.09 |

## Hessian-Free Newton
| line search   |   entropy |   num iter |   oracle calls |   time, s |
|---------------|-----------|------------|----------------|-----------|
| golden        |  0.297879 |         10 |           2848 |      0.54 |
| brent         |  0.297876 |         10 |           4036 |      0.77 |
| armijo        |  0.297877 |         11 |           2865 |      0.54 |
| wolfe         |  0.297876 |         12 |           4847 |      0.87 |