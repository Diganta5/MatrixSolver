import numpy as np
import scipy
import time

t_0 = time.time()



fle_list = ['fdm100','fdm1000','fdm10000']

def Dmat(A):
    n = A.shape[0]
    D = np.zeros(n)
    for i in range(n):
        D[i] = A[i][i]
    return D


def is_symm(A):
    n = len(A[0])
    for i in range(n):
        for j in range(n):
            if A[i][j]!=A[j][i]:
                return False
    return True

def spec(A):
    D = Dmat(A)
    inv_D = np.diag(1/D)
    G = np.matmul(inv_D,(np.diag(D)-A))
    eig = scipy.linalg.eigvals(G)#,right=False)
    spec_rad = np.max(np.abs(eig))
    return spec_rad

for fl in fle_list:

    A_arr = np.loadtxt(fl+'/Kmat.txt')
    bvec = np.loadtxt(fl+'/Fvec.txt')
    n = len(bvec)
    Amat = A_arr.reshape([n,n])
    
    # print(spec_rad)
    if not is_symm(Amat):
        Amat = (Amat + Amat.T)/2
    spec_rad= spec(Amat)
    omeg_opt = 2/(1+ np.sqrt(1- spec_rad**2))
    print('Omega optimum for ',fl,': ',omeg_opt)
    file_path = fl+"/opt_omega.txt"
    file = open(file_path, 'w')
    file.write(str(omeg_opt))
    file.close

t_1 = time.time()
print('Total time taken: ', t_1-t_0 )