import numpy as np
import random
import revisedSimplex
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
rs = RandomState(MT19937(SeedSequence(123456789)))


for i in range(5):
    m,n = random.randint(1,100), random.randint(1,100)
    A = np.random.randint(10,size=(m,n))
    b = np.random.randint(10,size=(m,1))
    c = np.random.randint(10,size=(1,n))
    np.savetxt('A.csv',A, delimiter=',')
    np.savetxt('b.csv',b, delimiter=',')
    np.savetxt('c.csv',c, delimiter=',')

    s = revisedSimplex.revisedSimplexSolver('A.csv','b.csv','c.csv')
    print(s.revised_simplex(4))