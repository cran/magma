/* 
 * File:   magSolvers.h
 * Author: bjsmith
 *
 * Created on June 20, 2010, 4:19 AM
 */

#ifndef _MAGSOLVERS_H
#define	_MAGSOLVERS_H

#include <Rdefines.h>

#ifdef	__cplusplus
extern "C" {
#endif


   SEXP magCholSolve(SEXP a, SEXP b);
   SEXP magLUSolve(SEXP a, SEXP b);
   SEXP magQRSolve(SEXP a, SEXP b);
   SEXP magSolve(SEXP a, SEXP b);
   SEXP magTriSolve(SEXP a, SEXP b, SEXP k, SEXP uprtri, SEXP transa);


#ifdef	__cplusplus
}
#endif

#endif	/* _MAGSOLVERS_H */
