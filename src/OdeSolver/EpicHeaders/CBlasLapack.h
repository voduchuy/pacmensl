/* 
 * ===========================================================================================
 * 
 * This file is header file for CBlasLapack.cpp
 * All function declarations are contained in this file.
 * 
 * -------------------------------------------------------------------------------------------
 * 
 * Last revision on 02/28/2016 by Ilija Jegdic
 * 
 * ===========================================================================================
 */

#ifndef __CBLASLAPACK__
#define __CBLASLAPACK__

void cdcopy(int n, const double *A, int incX, double *B, int incY);
void cdscal(int n, double c, double *A, int incX);
void cdaxpy(int n, double a, const double *X, int incX, double *Y, int incY);
void cdgemm(const char transa, const char transb, int m, int n, int k,
	    const double a, const double *A, int lda, 
	    const double *B, int ldb, double b,
	    double *C, int ldc);

double cdlange(char const normType, const int m, const int n, const double *A, const int lda, double *work);
void cdgetrf(const int m, const int n, double *A, const int lda, int *Piv, int *result);
void cdgetrs(const char trans, const int n, const int nrhs, const double *A, const int lda,
	     const int *Piv, double *B, const int ldb, int *result);

#endif
