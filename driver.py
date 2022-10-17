import revisedSimplex


problem = revisedSimplex.revisedSimplexSolver('A.csv', 'b.csv', 'c.csv')
print(problem.revised_simplex())