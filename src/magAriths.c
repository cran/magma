/********************************************************************************
 * File:   magAriths.c
 * Author: Brian J Smith <brian-j-smith@uiowa.edu>
 *
 * Created on June 18, 2010, 7:37 PM
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

#include "magAriths.h"
#include "magUtils.h"

#include <cublas.h>
#include <magmablas.h>


SEXP magMultmm(SEXP a, SEXP transa, SEXP b, SEXP transb)
{
   SEXP gpu = magGetGPU(a, b),
        c = PROTECT(NEW_OBJECT(MAKE_CLASS("magma")));
   int TA = LOGICAL_VALUE(transa), TB = LOGICAL_VALUE(transb),
       *DIMA = INTEGER(GET_DIM(a)), *DIMB = INTEGER(GET_DIM(b)),
       M = DIMA[TA], N = DIMB[!TB], K = DIMA[!TA],
       LDA = DIMA[0], LDB = DIMB[0], LDC = M;
   char TRANSA = (TA ? 'T' : 'N'), TRANSB = (TB ? 'T' : 'N');
   double *A = REAL(PROTECT(AS_NUMERIC(a))), *B = REAL(PROTECT(AS_NUMERIC(b))),
          *d_A, *d_B, *d_C;
 
   if(DIMB[TB] != K) error("non-conformable matrices");

   c = SET_SLOT(c, install(".Data"), allocMatrix(REALSXP, M, N));
   SET_SLOT(c, install("gpu"), duplicate(gpu));
   
   cublasAlloc(M * K, sizeof(double), (void**)&d_A);
   cublasAlloc(K * N, sizeof(double), (void**)&d_B);
   cublasAlloc(M * N, sizeof(double), (void**)&d_C);
   checkCublasError("device memory allocation failed in 'magMultmm'");

   cublasSetVector(M * K, sizeof(double), A, 1, d_A, 1);
   cublasSetVector(K * N, sizeof(double), B, 1, d_B, 1);

   if(LOGICAL_VALUE(gpu))
      magmablas_dgemm(TRANSA, TRANSB, M, N, K, 1.0, d_A, LDA, d_B, LDB, 0.0, d_C, LDC);
   else
      cublasDgemm(TRANSA, TRANSB, M, N, K, 1.0, d_A, LDA, d_B, LDB, 0.0, d_C, LDC);

   cublasGetVector(M * N, sizeof(double), d_C, 1, REAL(c), 1);

   cublasFree(d_A);
   cublasFree(d_B);
   cublasFree(d_C);

   UNPROTECT(3);

   return c;
}


SEXP magMultmv(SEXP a, SEXP transa, SEXP x, SEXP right)
{
   SEXP gpu = magGetGPU(a, x),
        y = PROTECT(NEW_OBJECT(MAKE_CLASS("magma")));
   int RHS = LOGICAL_VALUE(right), TA = (LOGICAL_VALUE(transa) ^ !RHS),
       *DIMA = INTEGER(GET_DIM(a)),       
       M = DIMA[0], N = DIMA[1], LENX = LENGTH(x), LENY = DIMA[TA], LDA = M;
   char TRANSA = (TA ? 'T' : 'N');
   double *A = REAL(PROTECT(AS_NUMERIC(a))), *X = REAL(PROTECT(AS_NUMERIC(x))),
          *d_A, *d_X, *d_Y;

   if(DIMA[!TA] != LENX) error("non-conformable matrices");

   y = SET_SLOT(y, install(".Data"),
                allocMatrix(REALSXP, (RHS ? LENY : 1), (RHS ? 1 : LENY)));
   SET_SLOT(y, install("gpu"), duplicate(gpu));

   cublasAlloc(M * N, sizeof(double), (void**)&d_A);
   cublasAlloc(LENX, sizeof(double), (void**)&d_X);
   cublasAlloc(LENY, sizeof(double), (void**)&d_Y);
   checkCublasError("device memory allocation failed in 'magMultmv'");

   cublasSetVector(M * N, sizeof(double), A, 1, d_A, 1);
   cublasSetVector(LENX, sizeof(double), X, 1, d_X, 1);

   if(LOGICAL_VALUE(gpu)) {
      if(TA)
         // BUG 0.2: Only computes first M elements of N-dimensional vector z
         // magmablas_dgemvt(M, N, 1.0, d_A, LDA, d_X, d_Y);
         cublasDgemv(TRANSA, M, N, 1.0, d_A, LDA, d_X, 1, 0.0, d_Y, 1);
      else magmablas_dgemv(M, N, d_A, LDA, d_X, d_Y);
   } else {
      cublasDgemv(TRANSA, M, N, 1.0, d_A, LDA, d_X, 1, 0.0, d_Y, 1);
   }

   cublasGetVector(LENY, sizeof(double), d_Y, 1, REAL(y), 1);

   cublasFree(d_A);
   cublasFree(d_X);
   cublasFree(d_Y);

   UNPROTECT(3);

   return y;
}

