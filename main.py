import numpy as np
import matrix_solv as mats
import time

t_0 = time.time()

file_path = "output.txt"
file = open(file_path, 'w')

fle_list = ['fem100','fem1000','fem10000']


for fl in fle_list:

    A_arr = np.loadtxt(fl+'/Kmat.txt')
    bvec = np.loadtxt(fl+'/Fvec.txt')
    n = len(bvec)
    Amat = A_arr.reshape([n,n])
    
    # print(spec_rad)
    if not mats.is_symm(Amat):
        Amat = (Amat + Amat.T)/2
    spec_rad,_ = mats.Gdat(Amat)
    omeg_opt = 2/(1+ np.sqrt(1- spec_rad**2))
    print('Omega optimum for ',fl,': ',omeg_opt)
    file_path = fl+"/opt_omega.txt"
    file = open(file_path, 'w')
    file.write(str(omeg_opt))
    file.close


    # #SOR optimum
    # _, error, iter = mats.SOR_up(Amat,bvec,1e-6,omeg=omeg_opt)
    # SOR_data = [fl,'SOR optimum',str(iter),str(error),str(omeg_opt)]
    # for item in SOR_data:
    #     file.write(item) 
    #     file.write(' ')
    # file.write('\n') 
    # #SOR (optimum), Steepest descent, MR, CG and BICGSTAB

    # #Steepest descent
    # _, error, iter = mats.steep_des(Amat,bvec,1e-6)
    # SOR_data = [fl,'Steepest descent',str(iter),str(error)]
    # for item in SOR_data:
    #     file.write(item) 
    #     file.write(' ')
    # file.write('\n') 


    # #MR
    # _, error, iter = mats.min_res(Amat,bvec,1e-6)
    # SOR_data = [fl,'MR',str(iter),str(error)]
    # for item in SOR_data:
    #     file.write(item) 
    #     file.write(' ')
    # file.write('\n') 

    # #CG
    # _, error, iter = mats.con_grad(Amat,bvec,1e-6)
    # SOR_data = [fl,'CG',str(iter),str(error)]
    # for item in SOR_data:
    #     file.write(item) 
    #     file.write(' ')
    # file.write('\n')

    # #BICGSTAB
    # _, error, iter = mats.bicgstab(Amat,bvec,1e-6)
    # SOR_data = [fl,'BICGSTAB',str(iter),str(error)]
    # for item in SOR_data:
    #     file.write(item) 
    #     file.write(' ')
    # file.write('\n')

t_1 = time.time()
print('Total time taken: ', t_1-t_0 )

