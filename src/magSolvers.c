/********************************************************************************
 * File:   magSolvers.c
 * Author: Brian J Smith <brian-j-smith@uiowa.edu>
 *
 * Created on June 20, 2010, 4:19 AM
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
#include "magSolvers.h"

#include <cublas.h>
#include <magmablas.h>
#include <magma.h>


SEXP magCholSolve(SEXP a, SEXP b)
{
   SEXP gpu = magGetGPU(a, b),
        c = PROTECT(NEW_OBJECT(MAKE_CLASS("magma")));
   int *DIMA = INTEGER(GET_DIM(a)), *DIMB = INTEGER(GET_DIM(b)),
       N = DIMA[0], NRHS = DIMB[1], info;

   if(DIMA[1] != N) error("non-square matrix");
   if(DIMB[0] != N) error("non-conformable matrices");
   
   c = SET_SLOT(c, install(".Data"), AS_NUMERIC(b));
   SET_SLOT(c, install("gpu"), duplicate(gpu));

   if(LOGICAL_VALUE(gpu)) {
      double *A = REAL(PROTECT(AS_NUMERIC(a))), *d_A, *d_B;

      cublasAlloc(N * N, sizeof(double), (void**)&d_A);
      cublasAlloc(N * NRHS, sizeof(double), (void**)&d_B);
      checkCublasError("device memory allocation failed in 'magCholSolve'");

      cublasSetVector(N * N, sizeof(double), A, 1, d_A, 1);
      cublasSetVector(N * NRHS, sizeof(double), REAL(c), 1, d_B, 1);

      magma_dpotrs_gpu("U", N, NRHS, d_A, N, d_B, N, &info);
      cublasGetVector(N * NRHS, sizeof(double), d_B, 1, REAL(c), 1);

      cublasFree(d_A);
      cublasFree(d_B);
      UNPROTECT(1);
   } else {
      double *A = REAL(PROTECT(AS_NUMERIC(a))), *d_A, *d_B;

      dpotrs_("U", &N, &NRHS, A, &N, REAL(c), &N, &info);

      UNPROTECT(1);
   }

   if(info < 0) error("Illegal argument %d in 'magCholSolve'", -1 * info);

   UNPROTECT(1);

   return c;
}


SEXP magLUSolve(SEXP a, SEXP b)
{
   SEXP gpu = magGetGPU(a, b),
        c = PROTECT(NEW_OBJECT(MAKE_CLASS("magma")));
   int *DIMA = INTEGER(GET_DIM(a)), *DIMB = INTEGER(GET_DIM(b)),
       N = DIMA[0], NRHS = DIMB[1],
       *ipiv = INTEGER(GET_SLOT(a, install("pivot"))), info;
   double *A = REAL(PROTECT(AS_NUMERIC(a)));

   if(DIMA[1] != N) error("non-square matrix");
   if(DIMB[0] != N) error("non-conformable matrices");
   
   c = SET_SLOT(c, install(".Data"), AS_NUMERIC(b));
   SET_SLOT(c, install("gpu"), duplicate(gpu));

   if(LOGICAL_VALUE(gpu)) {
      double *d_A, *d_B, *h_work;

      cublasAlloc(N * N, sizeof(double), (void**)&d_A);
      cublasAlloc(N * NRHS, sizeof(double), (void**)&d_B);
      checkCublasError("device memory allocation failed in 'magSolve'");

      cudaMallocHost((void**)&h_work, N * NRHS * sizeof(double));
      checkCudaError("host memory allocation failed in 'magSolve'");

      cublasSetVector(N * N, sizeof(double), A, 1, d_A, 1);
      cublasSetVector(N * NRHS, sizeof(double), REAL(c), 1, d_B, 1);

      magma_dgetrs_gpu("N", N, NRHS, d_A, N, ipiv, d_B, N, &info, h_work);

      cublasGetVector(N * NRHS, sizeof(double), d_B, 1, REAL(c), 1);

      cublasFree(d_A);
      cublasFree(d_B);
      cudaFreeHost(h_work);
   } else {
      dgetrs_("N", &N, &NRHS, A, &N, ipiv, REAL(c), &N, &info);
   }

   if(info < 0) error("illegal argument %d in 'magQRSolve'", -1 * info);

   UNPROTECT(2);

   return c;
}


SEXP magQRSolve(SEXP a, SEXP b)
{
   SEXP qr = VECTOR_ELT(a, 0), tau = VECTOR_ELT(a, 2),
        work = GET_SLOT(a, install("work")),
        gpu = magGetGPU(qr, b),
        c = PROTECT(NEW_OBJECT(MAKE_CLASS("magma")));
   int *DIMA = INTEGER(GET_DIM(qr)), *DIMB = INTEGER(GET_DIM(b)),
       M = DIMA[0], N = DIMA[1], NRHS = DIMB[1],
       NB = magma_get_dgeqrf_nb(M), LWORK = (M - N + NB + 2 * NRHS) * NB,
       info;
   double *h_work;

   if(M < N) error("indeterminate linear system");
   if(DIMB[0] != M) error("non-conformable matrices");

   c = SET_SLOT(c, install(".Data"), allocMatrix(REALSXP, N, NRHS));
   SET_SLOT(c, install("gpu"), duplicate(gpu));

   cudaMallocHost((void**)&h_work, LWORK * sizeof(double));
   checkCudaError("host memory allocation failed in 'magQRSolve'");

   // BUG 0.2: magma_dgeqrs_gpu returns incorrect results
   if(LOGICAL_VALUE(gpu) && 0) {
      double *A = REAL(qr), *B = REAL(PROTECT(AS_NUMERIC(b))),
             *d_A, *d_B, *d_work;

      cublasAlloc(M * N, sizeof(double), (void**)&d_A);
      cublasAlloc(M * NRHS, sizeof(double), (void**)&d_B);
      cublasAlloc(LENGTH(work), sizeof(double), (void**)&d_work);
      checkCublasError("device memory allocation failed in 'magQRSolve'");

      cublasSetVector(M * N, sizeof(double), A, 1, d_A, 1);
      cublasSetVector(M * NRHS, sizeof(double), B, 1, d_B, 1);
      cublasSetVector(LENGTH(work), sizeof(double), REAL(work), 1, d_work, 1);

      magma_dgeqrs_gpu(&M, &N, &NRHS, d_A, &M, REAL(tau), d_B, &M, h_work,
                       &LWORK, d_work, &info);

      cublasGetMatrix(N, NRHS, sizeof(double), d_B, M, REAL(c), N);

      cublasFree(d_A);
      cublasFree(d_B);
      cublasFree(d_work);
   } else {
      int i, j;      
      double *A = REAL(qr), *B = REAL(PROTECT(AS_NUMERIC(duplicate(b)))),
             ALPHA = 1.0;

      dormqr_("L", "T", &M, &NRHS, &N, A, &M, REAL(tau), B, &M,
              h_work, &LWORK, &info);
      dtrsm_("L", "U", "N", "N", &M, &NRHS, &ALPHA, A, &M, B, &M);

      magCopyMatrix(N, NRHS, REAL(c), N, B, M);
   }

   if(info < 0) error("illegal argument %d in 'magQRSolve'", -1 * info);

   cudaFreeHost(h_work);
   UNPROTECT(2);

   return c;
}


SEXP magSolve(SEXP a, SEXP b)
{
   SEXP gpu = magGetGPU(a, b),
        c = PROTECT(NEW_OBJECT(MAKE_CLASS("magma")));
   int *DIMA = INTEGER(GET_DIM(a)), *DIMB = INTEGER(GET_DIM(b)),
       N = DIMA[0], NRHS = DIMB[1], ipiv[N], info;

   if(DIMA[1] != N) error("non-square matrix");
   if(DIMB[0] != N) error("non-conformable matrices");

   c = SET_SLOT(c, install(".Data"), AS_NUMERIC(b));
   SET_SLOT(c, install("gpu"), duplicate(gpu));

   if(LOGICAL_VALUE(gpu)) {
      int K1 = (N % 32 ? (N / 32 + 1) * 32 - N : 0), LDA = N + K1,
          NB = magma_get_dgetrf_nb(N);
      double *A = REAL(PROTECT(AS_NUMERIC(a))), *d_A, *d_B, *h_work;

      cublasAlloc((N + K1) * (N + K1) + (N + K1) * NB + 2 * NB * NB,
                  sizeof(double), (void**)&d_A);
      cublasAlloc(N * NRHS, sizeof(double), (void**)&d_B);
      checkCublasError("device memory allocation failed in 'magSolve'");

      cudaMallocHost((void**)&h_work,
                     N * (NB > NRHS ? NB : NRHS) * sizeof(double));
      checkCudaError("host memory allocation failed in 'magSolve'");

      cublasSetMatrix(N, N, sizeof(double), A, N, d_A, LDA);
      cublasSetVector(N * NRHS, sizeof(double), REAL(c), 1, d_B, 1);

      magma_dgetrf_gpu(&N, &N, d_A, &LDA, ipiv, h_work, &info);
      if(info < 0) error("illegal argument %d in 'magSolve'", -1 * info);
      else if(info > 0) error("non-singular matrix");

      magma_dgetrs_gpu("N", N, NRHS, d_A, LDA, ipiv, d_B, N, &info, h_work);
      if(info < 0) error("illegal argument %d in 'magSolve'", -1 * info);

      cublasGetVector(N * NRHS, sizeof(double), d_B, 1, REAL(c), 1);

      cublasFree(d_A);
      cublasFree(d_B);
      cudaFreeHost(h_work);
      UNPROTECT(1);
   } else {
      double *A = REAL(PROTECT(AS_NUMERIC(duplicate(a))));

      dgesv_(&N, &NRHS, A, &N, ipiv, REAL(c), &N, &info);
      if(info < 0) error("illegal argument %d in 'magSolve'", -1 * info);
      else if(info > 0) error("non-singular matrix");
   
      UNPROTECT(1);
   }

   UNPROTECT(1);

   return c;
}


SEXP magTriSolve(SEXP a, SEXP b, SEXP k, SEXP uprtri, SEXP transa)
{
   SEXP gpu = magGetGPU(a, b),
        c = PROTECT(NEW_OBJECT(MAKE_CLASS("magma")));
   int *DIMA = INTEGER(GET_DIM(a)), *DIMB = INTEGER(GET_DIM(b)),
       M = DIMA[0], N = DIMB[1], K = INTEGER_VALUE(k);
   char UPLO = (LOGICAL_VALUE(uprtri) ? 'U' : 'L'),
        TRANSA = (LOGICAL_VALUE(transa) ? 'T' : 'N');
   double *A = REAL(PROTECT(AS_NUMERIC(a))), *B = REAL(PROTECT(AS_NUMERIC(b))),
          *d_A, *d_B;

   if((K <= 0) || (K > M)) error("invalid number of equations");

   c = SET_SLOT(c, install(".Data"), allocMatrix(REALSXP, K, N));
   SET_SLOT(c, install("gpu"), duplicate(gpu));

   cublasAlloc(M * M, sizeof(double), (void**)&d_A);
   cublasAlloc(M * N, sizeof(double), (void**)&d_B);
   checkCublasError("host memory allocation failed in 'magTriSolve'");

   cublasSetVector(M * M, sizeof(double), A, 1, d_A, 1);
   cublasSetVector(M * N, sizeof(double), B, 1, d_B, 1);

   if(LOGICAL_VALUE(gpu))
      magmablas_dtrsm('L', UPLO, TRANSA, 'N', K, N, 1.0, d_A, M, d_B, M);
   else
      cublasDtrsm('L', UPLO, TRANSA, 'N', K, N, 1.0, d_A, M, d_B, M);

   cublasGetMatrix(K, N, sizeof(double), d_B, M, REAL(c), K);

   cublasFree(d_A);
   cublasFree(d_B);
   UNPROTECT(3);
   
   return c;
}

