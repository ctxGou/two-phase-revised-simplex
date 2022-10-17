from cmath import inf
import numpy as np
from copy import deepcopy
from pathlib import Path

class revisedSimplexSolver():
    # Store the coeficients of the LP.
    def __init__(self, A_dir, b_dir, c_dir):

        A, b, c = np.genfromtxt(A_dir, delimiter=','),\
            np.genfromtxt(b_dir, delimiter=',').reshape(-1,1),\
            np.genfromtxt(c_dir, delimiter=',').reshape(-1,1)

        A, b, c = np.atleast_2d(A), np.atleast_2d(b), np.atleast_2d(c)

        # numpy genfromtxt takes csv with only one column as a
        # row vector, so I transpose A if there is only one column
        # in A.

        if ',' not in Path(A_dir).read_text():
            A = A.T


        self.A = A
        self.b = b
        self.c = c

        self.num_constraints = A.shape[0]
        self.num_variables = A.shape[1]
    
    # Find the bfs of the LP with auxiliary variables.
    def bfs_initialization(self):
        
        # keep b>=0
        aux_A, aux_b = np.copy(self.A), np.copy(self.b)
        for i in range(self.num_constraints):
            if aux_b[i][0] < 0:
                aux_A[i] *= -1
                aux_b[i] *= -1

        # Build an auxiliary simplex tableau
        aux_A = np.concatenate((aux_A,np.identity(self.num_constraints)),axis=1)
        aux_simplex_tableau = np.concatenate((aux_A, aux_b),axis=1)
        aux_c = np.zeros((aux_simplex_tableau.shape[1], 1))
        
        # Basic variables, ith entry being 0 means ith var is not basis.
        # ith entry being a > 0 meas ith entry is basis and (a-1)th row in tab.
        basic_variables = np.zeros(aux_simplex_tableau.shape[1])
        

        for i in range(aux_simplex_tableau.shape[1]):
            aux_c[i][0] = 1 if i<aux_simplex_tableau.shape[1]-1 and i>=self.num_variables else 0
            basic_variables[i] = 1+i-self.num_variables if i<aux_simplex_tableau.shape[1]-1 and i>=self.num_variables else 0
        basic_variables = basic_variables[0:-1]


        aux_simplex_tableau = np.concatenate((aux_simplex_tableau, aux_c.transpose()),axis=0)
        for i in range(self.num_constraints):
            aux_simplex_tableau[-1] -= aux_simplex_tableau[i]


        # Do pivot until reduced costs turn non-negative. No auxiliary 
        # variables can be basic variables.
    
        new_basis_index = self.find_next_basis_index(aux_simplex_tableau[-1][0:-1])
        
        #print(basic_variables)
        #print(np.round(aux_simplex_tableau,3))
        
        while new_basis_index:
            
            '''
            print(basic_variables, new_basis_index)
            print(np.round(aux_simplex_tableau,3))
            '''
            

            # find kicked basis
            row = -1
            max = inf
            for i in range(self.num_constraints):
                if aux_simplex_tableau[i][new_basis_index-1] > 0 and\
                    aux_simplex_tableau[i][-1] / aux_simplex_tableau[i][new_basis_index-1] <max:
                    
                    max = aux_simplex_tableau[i][-1] / aux_simplex_tableau[i][new_basis_index-1]
                    row = i
            
            
            # starts from 0
            kicked_index = np.where(basic_variables == row + 1)[0][0]
            
            basic_variables[new_basis_index-1] = deepcopy(basic_variables[kicked_index])
            basic_variables[kicked_index] = 0
                #print(new_basis_index-1, kicked_index)
                #print(basic_variables)
            
            # elementary row transformations
            i, j = row, new_basis_index-1

            aux_simplex_tableau[i] /= aux_simplex_tableau[i][j]
            for k in range(aux_simplex_tableau.shape[0]):
                if k != i:
                    aux_simplex_tableau[k] -= aux_simplex_tableau[k][j]*aux_simplex_tableau[i]

            '''
            print(basic_variables)
            print(np.round(aux_simplex_tableau,3))
            '''
            

            new_basis_index = self.find_next_basis_index(aux_simplex_tableau[-1][0:-1])
            #print(basic_variables)
            #print(np.round(aux_simplex_tableau,3))
        
        # No BFS
        if aux_simplex_tableau[-1][-1] < 0:
            return [], None

        # Now we have a potential BFS. Any aux variable is in
        # basis should be evicted.
        while len(self.auxiliary_variable_basis_index(basic_variables)):
            evict_index = self.auxiliary_variable_basis_index(basic_variables)[0]
            row = int(basic_variables[evict_index]) - 1
            
            new_basis_index = -1 # starts from 0
            for i in range(self.num_variables):
                if aux_simplex_tableau[row][i] != 0:
                    new_basis_index = i
                    break
            

            # remove redundant constraints
            if new_basis_index == -1:
                basic_variables = np.delete(basic_variables, evict_index)
                basic_variables[basic_variables>=evict_index] -= 1
                self.num_constraints -= 1
                self.A = np.delete(self.A, row, axis=0)
                self.b = np.delete(self.b, row, axis=0)
                
                aux_simplex_tableau = np.delete(aux_simplex_tableau, row, axis=0)
                continue

            i, j = row, new_basis_index
            aux_simplex_tableau[i] /= aux_simplex_tableau[i][j]
            for k in range(aux_simplex_tableau.shape[0]):
                if k != i:
                    aux_simplex_tableau[k] -= aux_simplex_tableau[k][j]*aux_simplex_tableau[i]

            basic_variables[new_basis_index] = basic_variables[evict_index]
            basic_variables[evict_index] = 0

        #print(basic_variables)
        #print(aux_simplex_tableau)
        ret = np.delete(aux_simplex_tableau,-1, axis=0)
        ret = np.delete(ret, np.s_[self.num_variables:-1],axis=1)

        rNT = np.concatenate((self.c.T, [[0]]),axis=1)
        for i in range(len(ret)-1):
            rNT -= rNT[0,i]*ret[i]
        ret = np.concatenate((ret,rNT),axis=0)

        #print("Phase 1 result: \n", ret)

        return ret + 0.0, basic_variables


    # Do revised simplex on given BFS
    def revised_simplex(self, digits=7, log=False):
        initial_BFS, basis = self.bfs_initialization()
        
        if len(initial_BFS)==0:
            return "The LP is infeasible!" + '\n\n'

        # find initial B
        basis_indices = [0 for i in range(self.num_constraints)]

        for i in range(len(basis)):
            if basis[i]>0:
                basis_indices[int(basis[i])-1] = i

        #print(self.A)
        initial_B = self.A[:,basis_indices]
        cB = self.c[basis_indices]
        B_ = np.linalg.inv(initial_B)
        b_bar = np.dot(B_, self.b)
        
        cNT = []
        nonbasis_index = []
       
        for i in range(self.num_variables):
            if i not in basis_indices:
                nonbasis_index.append(i)

        cNT = self.c[nonbasis_index,0]
        
        
        N = self.A[:,nonbasis_index]

        # simplex multiplier
        lambdaT = np.matmul(cB.T, B_)
        """
        print(cB)
        print(initial_B)
        print(B_)
        print(b_bar)
        print(lambdaT)
        """

        nit = 0


        while True:
            if log:
                print("\nNumber of iterations:",nit)

            ## Step 1

            # Calculate reduced cost rNT
            rNT = cNT - np.matmul(lambdaT, N)
            if log:
                print('b:',basis_indices,' nb:',nonbasis_index)
                print('rNT:', rNT)
                print('cNT:', cNT)
                print('N:',N)
                print('lambdaT:',lambdaT)
                print(np.matmul(lambdaT, N))

            # If the reduced costs are non-negative, return optimal solution
            if self.array_non_negative(rNT[0]):
                # currently optimal
                x_dict = {}
                # print solutions and objective

                ret_message = 'Optimal solution found.\n'
                for i in range(len(basis_indices)):
                    x_dict[basis_indices[i]+1] = round(b_bar[i,0], digits)
                for i in range(len(nonbasis_index)):
                    x_dict[nonbasis_index[i]+1] = 0
                
                x_dict = dict(sorted(x_dict.items()))
                for x,n in x_dict.items():
                    ret_message += 'x'+str(x)+' = '+str(round(n,digits))+'\n'

                ret_message += 'The optimal objective is '+ str(round(np.matmul(lambdaT, self.b)[0][0], digits)) + '\n'

                ret_message += 'The number of iterations is ' + str(nit)+'.'

                return ret_message + '\n\n'
            
            ## Step 2

            # select q
            # q is the index in rNT[0] for which rNT[0][q] is negative and minimum
            q = np.where(rNT[0] == [min(rNT[0][rNT[0]<0])])[0][0]
            
            
            ## Step 3

            # calculate yq
            aq = self.A[:,nonbasis_index[q]].reshape(-1,1) # column vector
            yq = np.matmul(B_, aq)
            
            if log:
                print('aq:',aq,'\nyq:',yq, '\nb_bar',b_bar)
            #print(yq, '\n', b_bar, nonbasis_index[q])

            # yq <= 0 the LP is unbounded
            if self.array_non_positive(yq.T[0]):
                ret_message = 'The LP is unbounded!\n'
                ret_message += 'The number of iterations is ' + str(nit)+'.'
                return ret_message + '\n\n'

            # find p
            p = -1
            max = inf
            for i in range(self.num_constraints):
                if yq[i,0] > 0:
                    current = b_bar[i,0] / yq[i,0]
                    if current < max:
                        current = max
                        p = i
                    
            #print(basis,nonbasis_index[q])


            ## Step 4

            # update B_ lambdaT
            v = []
            for i in range(self.num_constraints):
                v.append(-yq[i,0]/yq[p,0] if i!= p else 1/yq[p,0])
            
            E_pq = np.identity(self.num_constraints)
            E_pq[:,p] = np.array(v)

            
            
            lambdaT = lambdaT + rNT[0][q]/yq[p,0] * B_[p,:]
            B_ = np.matmul(E_pq, B_)

            basis_in = nonbasis_index[q]
            basis_out = basis_indices[p]

            nonbasis_index = [basis_out if i == basis_in else i for i in nonbasis_index]
            basis_indices = [basis_in if i == basis_out else i for i in basis_indices]
            N = self.A[:,nonbasis_index]
            cNT = self.c.T[0][nonbasis_index]
            b_bar = np.matmul(B_, self.b)
            nit += 1

            if nit>5000:
                return 'Error! Too many iterations.'

    # helper functions
    def array_non_negative(self, r):
        for e in r:
            if e < 0:
                return False
        return True

    def array_non_positive(self, r):
        for e in r:
            if e > 0:
                return False
        return True

        # starts from 1 to avoid 0/False ambiguity
    def find_next_basis_index(self, r):
        for i in range(len(r)):
            if r[i] < 0:
                return i + 1
        return False
    

    # 0-index
    def auxiliary_variable_basis_index(self, basic):
        ret = []
        for i in range(self.num_variables, basic.shape[0]):
            if basic[i] > 0:
                ret.append(i)
        return ret

    def show_problem(self):
        A, b, c = self.A, self.b, self.c
        c = np.append(c, [0])
        ret = np.concatenate((A, b), axis=1)
        ret = np.concatenate((ret, np.atleast_2d(c)), axis=0)
        return "The original tableau is:\n" + str(ret)

