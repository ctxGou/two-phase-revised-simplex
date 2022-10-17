###### Linear Programming 

$$
\text{minimize } c^Tx\\
\text{subject to } A^T x=b, x\geq0\\
x\in\mathbb{R}^n,c\in\mathbb{R}^n,A\in\mathbb{R}^{m×n},c\in\mathbb{R}^m
$$



###### Usage

`driver.py` is the driver code for quick start. `A.csv b.csv c.csv` should be in the same path.



###### Introduction

In `revisedSimplex.py` a class `revisedSimplexSolver(directory of A, directory of b, directory of c)`  is implemented.  The instancing takes three directories of the required .csv files. 

The method `bfs_initialization(self)` is the phase 1 of simplex. It returns the simplex tableau and the basic variables after a BFS of the problem is found.

The method `revised_simplex(self, digits=7, log=False)` contains a revised simplex implementation. The argument `digits` is the precision of the numerical result. The argument `log` decides whether to print the log of debugging. This method returns a string of the revised simplex result.

