# Optimization HW2

Results of all methods evaluation on [a1a](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html) dataset.
Best entropy by scipy baseline optimizer: 0.2978755986559931

## Gradien Descent 
| line search   |   entropy |   num iter |   oracle calls |   time, s |
|---------------|-----------|------------|----------------|-----------|
| golden        |  0.298463 |      10000 |         240001 |     41.31 |
| brent         |  0.298267 |       5196 |          89963 |     16.02 |
| armijo        |  0.298267 |       6626 |          49919 |     10.21 |
| wolfe         |  0.298402 |      10000 |          69433 |     16.48 |
| lipschitz     |  0.298976 |      10000 |          30001 |      7.67 |


## Newton 
| line search   |   entropy |   num iter |   oracle calls |   time, s |
|---------------|-----------|------------|----------------|-----------|
| golden        |  0.297875 |         11 |            265 |      0.08 |
| brent         |  0.297875 |          9 |            101 |      0.05 |
| armijo        |  0.297875 |         10 |             80 |      0.05 |
| wolfe         |  0.297876 |         11 |            178 |      0.08 |

## Hessian-Free Newton
| line search   |   entropy |   num iter |   oracle calls |   time, s |
|---------------|-----------|------------|----------------|-----------|
| golden        |  0.297879 |         10 |           2848 |      0.49 |
| brent         |  0.297876 |         10 |           4036 |      0.7  |
| armijo        |  0.297877 |         15 |           2153 |      0.38 |
| wolfe         |  0.297876 |         12 |           4847 |      0.85 |