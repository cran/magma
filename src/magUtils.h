/* 
 * File:   magUtils.h
 * Author: bjsmith
 *
 * Created on June 21, 2010, 9:19 AM
 */

#ifndef _MAGUTILS_H
#define	_MAGUTILS_H

#include <Rdefines.h>

#ifdef	__cplusplus
extern "C" {
#endif

   void magCopyMatrix(int m, int n, double *a, int lda, double *b, int ldb);
   SEXP magGetGPU(SEXP a, SEXP b);

   void magLoad();
   void magUnload();

   int checkCudaError(const char * msg);
   int checkCublasError(const char * msg);


#ifdef	__cplusplus
}
#endif

#endif	/* _MAGUTILS_H */
