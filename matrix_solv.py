import numpy as np
import scipy
from scipy.sparse import csc_matrix, identity, linalg

def EDF_decomp(A):
    n = A.shape[0]
    D = np.zeros([n,n])
    E = np.zeros([n,n])
    F = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if i==j:
                D[i][j] = A[i][j]
            elif i>j:
                E[i][j] = -A[i][j]

            elif j>i:
                F[i][j] = -A[i][j]
    return D,E,F
def is_symm(A):
    n = len(A[0])
    for i in range(n):
        for j in range(n):
            if A[i][j]!=A[j][i]:
                return False
    return True
            


def Gdat(A,type='j',omeg=None):
    D,E,F = EDF_decomp(A)
    if type == 'j':
        n = A.shape[0]
        G = np.matmul(scipy.linalg.inv(D),(D-A))
        eig = scipy.linalg.eig(G,right=False)
        spec_rad = np.max(np.abs(eig))
        conv_rate = -np.log(spec_rad)
        return spec_rad, conv_rate

        
    elif type == 'gs':
        M = D-E
        G = np.matmul(np.linalg.inv(M),(M-A))
        eig, _ = np.linalg.eig(G)
        spec_rad = np.max(np.abs(eig))
        conv_rate = -np.log(spec_rad)
        return spec_rad, conv_rate

    elif type == 'sor':
        L = np.linalg.inv(D-omeg*E)
        M = omeg * F + (1-omeg)*D
        G = np.matmul(L,M)
        eig, _ = np.linalg.eig(G)
        spec_rad = np.max(np.abs(eig))
        conv_rate = -np.log(spec_rad)
        return spec_rad, conv_rate


def jacobi(A,B,conv):
    n = len(B)
    x_0 = np.random.randn(n)
    x_c = np.zeros(n)
    error = 100000
    k = 0
    while error>conv:
        for l in range(n):
            sum = 0
            for m in range(n):
                if m != l:
                    sum += A[l][m]*x_0[m]
            x_c[l] = (B[l] - sum)/A[l][l]
        error = np.max(np.abs(x_c-x_0))
        x_0 = np.copy(x_c)
        k = k+1

    return x_0,error,k


def gausseid(A,B,conv):
    n = len(B)
    x_0 = np.random.randn(n)
    x_c = np.zeros(n)
    error = 100000
    k = 0
    while error>conv:
        for l in range(n):  #row step
            sum = 0
            for m in range(n):  #column step
                if m < l:
                    sum += A[l][m]*x_c[m]
                elif m > l:
                    sum += A[l][m]*x_0[m]
            x_c[l] = (B[l] - sum)/A[l][l]
        error = np.max(np.abs(x_c-x_0))
        x_0 = np.copy(x_c)
        k = k+1
    return x_0,error,k


def SOR_up(A,B,conv,omeg):
    n = len(B)
    #x_0 = np.random.randn(n)
    x_0 = np.ones(n)
    x_c = np.zeros(n)
    error = 100000
    k = 0
    while error>conv:
        for l in range(n):  #row step
            sum = 0
            for m in range(n):  #column step
                if m < l:
                    sum += A[l][m]*x_c[m]
                elif m > l:
                    sum += A[l][m]*x_0[m]
            x_c[l] = (1-omeg)*x_0[l] + omeg*(B[l] - sum)/A[l][l]
        error = np.max(np.abs(x_c-x_0))
        x_0 = np.copy(x_c)
        k = k + 1
    return x_0,error,k

def steep_des(A,B,conv):
    n = len(B)
    #x_0 = np.random.randn(n)
    x_0 = np.ones(n)
    #x_c = np.zeros(n)
    r_k = B - np.matmul(A,x_0)
    p_k = np.matmul(A,r_k)
    error = np.sqrt(np.dot(r_k,r_k))
    #error = 100000
    k = 0
    while error>conv:
        alph = np.dot(r_k,r_k)/np.dot(r_k,p_k)
        x_c = x_0 + alph*r_k
        r_k = r_k - alph*p_k
        p_k = np.matmul(A,r_k)
        k=k+1
        error = np.sqrt(np.dot(r_k,r_k))#error = np.max(np.abs(x_c-x_0))
        x_0 = np.copy(x_c)
    return x_0, error, k

def min_res(A,B,conv):
    n = len(B)
    #x_0 = np.random.randn(n)
    x_0 = np.ones(n)
    #x_c = np.zeros(n)
    r_k = B - np.matmul(A,x_0)
    p_k = np.matmul(A,r_k)
    error = np.sqrt(np.dot(r_k,r_k))
    #error = 100000
    k = 0
    while error>conv:
        alph = np.dot(p_k,r_k)/np.dot(p_k,p_k)
        x_c = x_0 + alph*r_k
        r_k = r_k - alph*p_k
        p_k = np.matmul(A,r_k)
        k=k+1
        error = np.sqrt(np.dot(r_k,r_k))#np.max(np.abs(x_c-x_0))
        x_0 = np.copy(x_c)
    return x_0, error, k

def con_grad(A,B,conv):
    n = len(B)
    #x_0 = np.random.randn(n)
    x_0 = np.ones(n)
    #x_c = np.zeros(n)
    r_k = B - np.matmul(A,x_0)
    p_k = r_k
    error = np.sqrt(np.dot(r_k,r_k))
    #error = 1000
    k = 0
    while error>conv:
        alph = np.dot(r_k,r_k)/np.dot(np.matmul(A,p_k),p_k)
        x_c = x_0 + alph*p_k
        r_k_u = r_k - alph*np.matmul(A,p_k)
        beta = np.dot(r_k_u,r_k_u)/np.dot(r_k,r_k)
        r_k = r_k_u
        p_k = r_k_u + beta*p_k
        k = k+1
        error = np.sqrt(np.dot(r_k,r_k))#np.max(np.abs(x_c-x_0))
        x_0 = np.copy(x_c)
    return x_0,error,k

def bicgstab(A,B,conv):
    n = len(B)
    #x_0 = np.random.randn(n)
    x_0 = np.ones(n)
    #x_c = np.zeros(n)
    r_k = B - np.matmul(A,x_0)
    r_k_ = np.random.randn(n)
    p_k = r_k
    error = np.sqrt(np.dot(r_k,r_k))
    #error = 100
    k = 0
    while error>conv:
        alph = np.dot(r_k,r_k_)/np.dot(np.matmul(A,p_k),r_k_)
        a_pk = np.matmul(A,p_k)
        s_k = r_k - alph*a_pk
        a_sk = np.matmul(A,s_k)
        omeg = np.dot(a_sk,s_k)/np.dot(a_sk,a_sk)
        x_c = x_0 + alph*p_k + omeg * s_k
        r_k_u = s_k - omeg*a_sk
        beta = np.dot(r_k_u,r_k_)*alph/(np.dot(r_k,r_k_u)*omeg)
        r_k = r_k_u
        p_k = r_k_u + beta* (p_k - omeg*a_pk)
        k = k+1
        error = np.sqrt(np.dot(r_k,r_k))#np.max(np.abs(x_c-x_0))
        x_0 = np.copy(x_c)

    return x_0,error,k