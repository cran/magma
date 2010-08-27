/********************************************************************************
 * File:   magFactors.c
 * Author: Brian J Smith <brian-j-smith@uiowa.edu>
 *
 * Created on June 18, 2010, 11:36 AM
 *
 * This file is part of the magma R package.
 *
 * magma is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * magma is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with magma.  If not, see <http://www.gnu.org/licenses/>.
 ********************************************************************************/

#include "magUtils.h"
#include "magFactors.h"

#include <cublas.h>
#include <cuda.h>
#include <magma.h>


SEXP magChol(SEXP a)
{
   SEXP gpu = GET_SLOT(a, install("gpu")),
        b = PROTECT(NEW_OBJECT(MAKE_CLASS("magma")));
   int *DIMA = INTEGER(GET_DIM(a)), N = DIMA[0], N2 = N * N, info;
   double *B, *d_B;

   if(DIMA[1] != N) error("non-square matrix");

   b = SET_SLOT(b, install(".Data"), AS_NUMERIC(a));
   SET_SLOT(b, install("gpu"), duplicate(gpu));
   B = REAL(b);

   cublasAlloc(N2, sizeof(double), (void**)&d_B);
   checkCublasError("device memory allocation failed in 'magChol'");

   
   if(LOGICAL_VALUE(gpu)) {
      int NB = magma_get_dpotrf_nb(N);
      double *h_work;

      cudaMallocHost((void**)&h_work, NB * NB * sizeof(double));
      checkCudaError("host memory allocation failed in 'magChol'");

      // BUG 0.2: if uplo = "U" then info > 0 for large N
      // cublasSetVector(N2, sizeof(double), B, 1, d_B, 1);
      // magma_dpotrf_gpu("U", &N, d_B, &N, h_work, info);
      // cublasGetVector(N2, sizeof(double), d_B, 1, B, 1);

      cublasSetVector(N2, sizeof(double), B, 1, d_B, 1);
      magma_dpotrf_gpu("L", &N, d_B, &N, h_work, &info);
      cublasGetVector(N2, sizeof(double), d_B, 1, B, 1);

      cudaFreeHost(h_work);
   } else {
      double *h_B;

      cudaMallocHost((void**)&h_B, N2 * sizeof(double));
      checkCudaError("host memory allocation failed in 'magChol'");
      
      memcpy(h_B, B, N2 * sizeof(double));
      magma_dpotrf("L", &N, h_B, &N, d_B, &info);
      memcpy(B, h_B, N2 * sizeof(double));

      cudaFreeHost(h_B);
   }

   if(info < 0) error("illegal argument %d in 'magChol", -1 * info);
   else if(info > 0) error("leading minor of order %d is not positive definite", info);

   int i, j;
   double *ptr;
   for(j = 0; j < N; j++) {
      for(i = j + 1; i < N; i++) {
         ptr = &B[i + j * N];
         B[j + i * N] = *ptr;
         *ptr = 0.0;
      }
   }

   cublasFree(d_B);
   UNPROTECT(1);

   return b;
}


SEXP magLU(SEXP a)
{
   SEXP gpu = GET_SLOT(a, install("gpu")),
        b = PROTECT(NEW_OBJECT(MAKE_CLASS("magmaLU")));
   int *DIMA = INTEGER(GET_DIM(a)), M = DIMA[0], N = DIMA[1],
       MAXMN = (M > N ? M : N),
       K1 = (MAXMN % 32 ? (MAXMN / 32 + 1) * 32 - MAXMN : 0),
       K2 = (M % 32 ? (M / 32 + 1) * 32 - M : 0),
       NB = magma_get_dgetrf_nb(M), *ipiv, info;
   double *A = REAL(PROTECT(AS_NUMERIC(a))), *d_A, *h_work;

   b = SET_SLOT(b, install(".Data"), AS_NUMERIC(a));
   SET_SLOT(b, install("pivot"), NEW_INTEGER(M < N ? M : N));
   ipiv = INTEGER(GET_SLOT(b, install("pivot")));
   SET_SLOT(b, install("gpu"), duplicate(gpu));

   cublasAlloc((MAXMN + K1) * (MAXMN + K1) + (M + K2) * NB + 2 * NB * NB,
               sizeof(double), (void**)&d_A);
   checkCublasError("device memory allocation failed in 'magLU'");

   cudaMallocHost((void**)&h_work, N * NB * sizeof(double));
   checkCudaError("host memory allocation failed in 'magLU'");

   if(LOGICAL_VALUE(gpu)) {
      int LDA = MAXMN + K1;

      cublasSetMatrix(M, N, sizeof(double), A, M, d_A, LDA);
      magma_dgetrf_gpu(&M, &N, d_A, &LDA, ipiv, h_work, &info);
      cublasGetMatrix(M, N, sizeof(double), d_A, LDA, REAL(b), M);
   } else {
      double *h_A;

      cudaMallocHost((void**)&h_A, M * N * sizeof(double));
      checkCudaError("host memory allocation failed in 'magLU'");

      cublasAlloc((MAXMN + K1) * (MAXMN + K1) + (M * K2) * NB + 2 * NB * NB,
                  sizeof(double), (void**)&d_A);
      checkCublasError("device memory allocation failed in 'magLU'");

      memcpy(h_A, A, M * N * sizeof(double));
      magma_dgetrf(&M, &N, h_A, &M, ipiv, h_work, d_A, &info);
      memcpy(REAL(b), h_A, M * N * sizeof(double));

      cudaFreeHost(h_A);
   }

   if(info < 0) error("illegal argument %d in 'magLU'", -1 * info);
   else if(info > 0) error("factor U is singular");

   cublasFree(d_A);
   cudaFreeHost(h_work);
   UNPROTECT(2);

   return b;
}


SEXP magQR(SEXP a)
{
   SEXP gpu = GET_SLOT(a, install("gpu")),
        b = PROTECT(NEW_OBJECT(MAKE_CLASS("magmaQR")));
   int *DIMA = INTEGER(GET_DIM(a)), M = DIMA[0], N = DIMA[1],
       LENT = (M < N ? M : N), NB = magma_get_dgeqrf_nb(M), *pivot, info;
   double *A, *tau;

   A = REAL(SET_VECTOR_ELT(b, 0, AS_NUMERIC(duplicate(a))));
   SET_VECTOR_ELT(b, 1, ScalarInteger(LENT));
   tau = REAL(SET_VECTOR_ELT(b, 2, NEW_NUMERIC(LENT)));
   pivot = INTEGER(SET_VECTOR_ELT(b, 3, NEW_INTEGER(N)));

   int i;
   for(i = 1; i <= N; i++) *pivot++ = i;

   if(LOGICAL_VALUE(gpu)) {
      int LWORK = (M + N) * NB;
      double *work, *d_A, *d_work, *h_work;

      SET_SLOT(b, install("work"), NEW_NUMERIC(N * NB));
      work = REAL(GET_SLOT(b, install("work")));

      cublasAlloc(M * N, sizeof(double), (void**)&d_A);
      cublasAlloc(N * NB, sizeof(double), (void**)&d_work);
      checkCublasError("device memory allocation failed in 'magQR'");

      cudaMallocHost((void**)&h_work, LWORK * sizeof(double));
      checkCudaError("host memory allocation failed in 'magQR'");

      cublasSetVector(M * N, sizeof(double), A, 1, d_A, 1);
      magma_dgeqrf_gpu(&M, &N, d_A, &M, tau, h_work, &LWORK, d_work, &info);
      cublasGetVector(M * N, sizeof(double), d_A, 1, A, 1);
      cublasGetVector(N * NB, sizeof(double), d_work, 1, work, 1);

      cublasFree(d_A);
      cublasFree(d_work);
      cudaFreeHost(h_work);
   } else {
      int LWORK = N * NB;
      double *h_A, *h_work, *d_A;

      cudaMallocHost((void**)&h_A, M * N * sizeof(double));
      cudaMallocHost((void**)&h_work, LWORK * sizeof(double));
      checkCudaError("host memory allocation failed in 'magQR'");

      cublasAlloc(N * (M + NB), sizeof(double), (void**)&d_A);
      checkCublasError("device memory allocation failed in 'magQR'");

      memcpy(h_A, A, M * N * sizeof(double));
      magma_dgeqrf(&M, &N, h_A, &M, tau, h_work, &LWORK, d_A, &info);
      memcpy(A, h_A, M * N * sizeof(double));

      cudaFreeHost(h_A);
      cudaFreeHost(h_work);
      cublasFree(d_A);
   }

   if(info < 0) error("illegal argument %d in 'magQR'", -1 * info);

   UNPROTECT(1);

   return b;
}

