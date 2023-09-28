#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<time.h>


void mulmat(double *mat1,double *mat2,double *mul,int n){
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			mul[i*n+j]=0;
			for(int k=0;k<n;k++){
				mul[i*n+j]+=mat1[i*n+k]*mat2[k*n+j];
			}
		}
	}	
}

void matvec(double *mat,double *vec,double *mul,int n){
	for(int i=0;i<n;i++){
		mul[i]=0;
		for(int j=0;j<n;j++){
			mul[i]+=mat[i*n+j]*vec[j];
		}
	}
}

double dot(double *vec1,double *vec2,int n){
	double pro=0;
	for(int i=0;i<n;i++){
		pro+=vec1[i]*vec2[i];
	}
	return pro;
}

double max(double *arr,int p){
    double m = arr[0];
    for (int i=1;i<p;i++)
        if (arr[i] > m)
            m = arr[i];
    return m;
}

//	Gauss Seidel
void GS(double *A,double *b,int n){
	
	clock_t begin = clock();
	
	double *x,*y,*err;
	
	x = (double *)malloc(n*sizeof(double));
	y = (double *)malloc(n*sizeof(double));
	err = (double *)malloc(n*sizeof(double));
	
	int m=0;
	for(int i=0;i<n;i++){
		x[i]=1;
		y[i]=2;
		err[i]=100;
	}

	while((max(err,n))>1e-6){
		for(int i=0;i<n;i++){
			double sum=0;
			for(int j=0;j<n;j++){
				if (j<i){
					sum+=(A[i*n+j]*x[j])/A[i*n+i];
				}
				else if (j>i){
					sum+=(A[i*n+j]*y[j])/A[i*n+i];
				}
			x[i]=b[i]/A[i*n+i]-sum;
			err[i]=fabs(x[i]-y[i]);
			}
		}
		for(int l=0;l<n;l++){
			y[l]=x[l];
		}
		m+=1;
	}
	free(x);
    free(y);
    free(err);
	
//	for(int i=0;i<n;i++){
//			printf("%lf\t",x[i]);
//	}
	
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	
	printf("\n\nNo. of Iterations using GS: %d",m);
	printf("\nThe elapsed time for GS is %lf seconds", time_spent);
}

//	Jacobi
void Jacobi(double *A,double *b,int n){
	
	clock_t begin = clock();
	
	double *x,*y,*err;
	
	x = (double *)malloc(n*sizeof(double));
	y = (double *)malloc(n*sizeof(double));
	err = (double *)malloc(n*sizeof(double));
	
	int m=0;
	for(int i=0;i<n;i++){
		x[i]=1;
		y[i]=2;
		err[i]=100;
	}

	while((max(err,n))>1e-6){
		for(int i=0;i<n;i++){
			double sum=0;
			for(int j=0;j<n;j++){
				if (j<i){
					sum+=(A[i*n+j]*y[j]);
				}
				else if (j>i){
					sum+=(A[i*n+j]*y[j]);
				}
			x[i]=(b[i]-sum)/A[i*n+i];
			err[i]=fabs(x[i]-y[i]);
			}
		}
		for(int l=0;l<n;l++){
			y[l]=x[l];
		}
		m+=1;
	}
	free(x);
    free(y);
    free(err);
	
//	for(int i=0;i<n;i++){
//			printf("%lf\t",x[i]);
//	}
		
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    
	
	printf("\n\nNo. of Iterations using Jacobi: %d",m);
	printf("\nThe elapsed time for Jacobi is %lf seconds", time_spent);
}
//SOR
void SOR(double *A,double *b,double w,int n){
	
	clock_t begin = clock();
	
	double *x,*y,*err,sum,er;
	
	x = (double *)malloc(n*sizeof(double));
	y = (double *)malloc(n*sizeof(double));
	err = (double *)malloc(n*sizeof(double));
	
	int m=0;
	for(int i=0;i<n;i++){
		x[i]=2;
		y[i]=1;
		err[i]=100;
	}
	
	while((max(err,n))>1e-6){
		for(int i=0;i<n;i++){
        	//sum=(1-w)*y[i]*A[i*n+i]+w*b[i];
			sum = 0.0;
       		for(int j=0;j<n;j++){
            	if(j<i)
	            	sum+=A[i*n+j]*y[j];
            	else if(j>i)
               		sum+=A[i*n+j]*x[j];
        	}
        	//y[i]=sum/A[i*n+i];
			er = -w*x[i] + w*(b[i]-sum)/A[i*n+i];
			y[i]=x[i] + er;
        	err[i]=fabs(er);
       }
		for(int l=0;l<n;l++){
			x[l]=y[l];
		}
		m+=1;
	}
	free(x);
    free(y);
    free(err);
//	for(int i=0;i<n;i++){
//			printf("%lf\t",x[i]);
//	}
		
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    
	
	printf("\n\nNo. of Iterations using SOR with %0.2lf: %d",w,m);
	printf("\nThe elapsed time for SOR is %lf seconds", time_spent);
}
//	Minimun Residue
void MR(double *A,double *b,int n){
	
	clock_t begin = clock();

	double *x,*r,*err,al,ar;
	double *Ax,*p;
	
	x = (double *)malloc(n*sizeof(double));
	r = (double *)malloc(n*sizeof(double));
	err = (double *)malloc(n*sizeof(double));
	Ax = (double *)malloc(n*sizeof(double));
	p = (double *)malloc(n*sizeof(double));
	
	int m=0;
	for(int i=0;i<n;i++){
		x[i]=0;
		err[i]=100;
	}
	
	matvec(A,x,Ax,n);
	for(int i=0;i<n;i++){
		r[i]=b[i]-Ax[i];
	}
	
	while((max(err,n))>1e-6){
		matvec(A,r,p,n);
		al=dot(p,r,n)/dot(p,p,n);
		for(int i=0;i<n;i++){
			ar = al*r[i];
			x[i]=x[i]+ar;
			err[i]=fabs(ar);
			r[i]=r[i]-al*p[i];
		}
		
		m+=1;
	}
	free(x);
    free(r);
    free(err);
	free(Ax);
	free(p);
//	for(int i=0;i<n;i++){
//			printf("%lf\t",x[i]);
//	}
		
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    
	
	printf("\n\nNo. of Iterations using MR: %d",m);
	printf("\nThe elapsed time for MR is %lf seconds", time_spent);
}

//	Steepest Descent
void SD(double *A,double *b,int n){
	
	clock_t begin = clock();
	
	
	double *x,*r,*err,al,ar;
	double *Ax,*p;
	
	x = (double *)malloc(n*sizeof(double));
	r = (double *)malloc(n*sizeof(double));
	err = (double *)malloc(n*sizeof(double));
	Ax = (double *)malloc(n*sizeof(double));
	p = (double *)malloc(n*sizeof(double));
	
	int m=0;
	for(int i=0;i<n;i++){
		x[i]=0;
		err[i]=100;
	}
	
	matvec(A,x,Ax,n);
	for(int i=0;i<n;i++){
		r[i]=b[i]-Ax[i];
	}
	
	while((max(err,n))>1e-6){
		matvec(A,r,p,n);
		al=dot(r,r,n)/dot(r,p,n);
		for(int i=0;i<n;i++){
			ar = al*r[i];
			x[i]=x[i] + ar;
			err[i]=fabs(ar);
			r[i]=r[i]-al*p[i];
		}
		m+=1;
	}
	free(x);
    free(r);
    free(err);
	free(Ax);
	free(p);
//	for(int i=0;i<n;i++){
//			printf("%lf\t",x[i]);
//	}
		
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	
	printf("\n\nNo. of Iterations using SD: %d",m);
	printf("\nThe elapsed time for SD is %lf seconds", time_spent);
}

//	Conjugate Gradient
void CG(double *A,double *b,int n){
	
	clock_t begin = clock();
	double *x,*r,*err,al,bt,RR,alp;
	double *Ax,*Ap,*p;
	
	x = (double *)malloc(n*sizeof(double));
	r = (double *)malloc(n*sizeof(double));
	err = (double *)malloc(n*sizeof(double));
	Ax = (double *)malloc(n*sizeof(double));
	Ap = (double *)malloc(n*sizeof(double));
	p = (double *)malloc(n*sizeof(double));
	
	int m=0;
	for(int i=0;i<n;i++){
		x[i]=0;
		err[i]=100;
	}
	
	matvec(A,x,Ax,n);
	for(int i=0;i<n;i++){
		r[i]=b[i]-Ax[i];
		p[i]=r[i];
	}
	
	while((max(err,n))>1e-6){
		matvec(A,p,Ap,n);
		RR=dot(r,r,n);
		al=RR/dot(Ap,p,n);
		for(int i=0;i<n;i++){
			alp = al*p[i];
			x[i]=x[i]+alp;
			r[i]=r[i]-al*Ap[i];
			err[i]=fabs(alp);
		}
		bt=dot(r,r,n)/RR;
		for(int i=0;i<n;i++){
			p[i]=r[i]+bt*p[i];
		}
		m+=1;
	}
	free(x);
    free(r);
    free(err);
	free(Ax);
	free(Ap);
	free(p);
//	for(int i=0;i<n;i++){
//			printf("%lf\t",x[i]);
//	}
		
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    
	
	printf("\n\nNo. of Iterations using CG: %d",m);
	printf("\nThe elapsed time for CG is %lf seconds", time_spent);
}

//	BICGSTAB
void BICGSTAB(double *A,double *b,int n){
	
	clock_t begin = clock();
	
	double *x,*r,*err,al,bt,w,Rr0,alps;
	double *Ax,*p,*Ap,*As,*s,*r0;
	
	x = (double *)malloc(n*sizeof(double));
	r = (double *)malloc(n*sizeof(double));
	err = (double *)malloc(n*sizeof(double));
	Ax = (double *)malloc(n*sizeof(double));
	p = (double *)malloc(n*sizeof(double));
	As = (double *)malloc(n*sizeof(double));
	Ap = (double *)malloc(n*sizeof(double));
	s = (double *)malloc(n*sizeof(double));
	r0 = (double *)malloc(n*sizeof(double));
	
	int m=0;
	for(int i=0;i<n;i++){
		x[i]=0;
		r0[i]=2;
		err[i]=100;
	}
	
	matvec(A,x,Ax,n);
	for(int i=0;i<n;i++){
		r[i]=b[i]-Ax[i];
		p[i]=r[i];
	}
	matvec(A,r,p,n);
	
	while((max(err,n))>1e-6){
		matvec(A,p,Ap,n);
		Rr0=dot(r,r0,n);
		al=Rr0/dot(Ap,r0,n);
		for(int i=0;i<n;i++){
			s[i]=r[i]-al*Ap[i];
		}
		matvec(A,s,As,n);
		w=dot(As,s,n)/dot(As,As,n);
		for(int i=0;i<n;i++){
			alps = al*p[i]+w*s[i];
			x[i]=x[i]+ alps;
			r[i]=s[i]-w*As[i];
			err[i]=fabs(alps);
		}
		bt=(dot(r,r0,n)/Rr0)*(al/w);
		for(int i=0;i<n;i++){
			p[i]=r[i]+bt*(p[i]-w*Ap[i]);
		}
		m+=1;
	}
	free(x);
    free(r);
    free(err);
	free(Ax);
	free(Ap);
	free(p);
	free(s);
	free(As);
	free(r0);
//	for(int i=0;i<n;i++){
//			printf("%lf\t",x[i]);
//	}
		
	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
    
	
	printf("\n\nNo. of Iterations using BICGSTAB: %d ",m);
	printf("\nThe elapsed time for BICGSTAB is %lf seconds \n", time_spent);
}

int main(){
	char *files[3];

    // Initialize the strings
    files[0] = "fdm100";
    files[1] = "fdm1000";
    files[2] = "fdm10000";
    // Print the strings
    for (int i = 0; i < 3; i++) {
        //printf("String %d: %s\n", i, strings[i]);
    
		FILE *arr1;
		FILE *arr2;
		FILE *arr3;
		FILE *arr4;
		char filepath1[256];
		char filepath2[256];
		char filepath3[256];
		char filepath4[256];
		snprintf(filepath1, sizeof(filepath1), "%s/Kmat.txt", files[i]);
		snprintf(filepath2, sizeof(filepath2), "%s/Fvec.txt", files[i]);
		snprintf(filepath3, sizeof(filepath3), "%s/kinfo.txt", files[i]);
		snprintf(filepath4, sizeof(filepath4), "%s/opt_omega.txt", files[i]);
		arr1=fopen(filepath1,"r");
		arr2=fopen(filepath2,"r");
		arr3=fopen(filepath3,"r");
		arr4=fopen(filepath4,"r");
		int n = 0;
		double omeg = 0.0;
		fscanf(arr3,"%d",&n);
		fscanf(arr4,"%lf",&omeg);
		printf("For %s \n",files[i]);
		printf("omega = %lf\n",omeg);
		double *A, *G, *b;
		A = (double *)malloc(n*n*sizeof(double));
		G = (double *)malloc(n*n*sizeof(double));
		b = (double *)malloc(n*sizeof(double));
	
	//  Input values of A[][] and b[]
		for (int i=0;i<n;i++){
			fscanf(arr2,"%lf",&b[i]);
			for (int j=0;j<n;j++){
				fscanf(arr1,"%lf",&G[i*n+j]);
			}
		}
		
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				A[i*n+j]=0.5*(G[i*n+j]+G[j*n+i]);
			}
		}
		
		
		
	//	Results

		//Jacobi(A,b,n);
		
		//GS(A,b,n);
		
		SOR(A,b,omeg,n);
		
		SD(A,b,n);
		
		MR(A,b,n);
		
		CG(A,b,n);
		
		BICGSTAB(A,b,n);

		printf("n=%d\n",n);

		free(A);
		free(G);
		free(b);
		fclose(arr1);
		fclose(arr2);
		fclose(arr3);
		fclose(arr4);
	}
}