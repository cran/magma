/* 
 * File:   magFactors.h
 * Author: bjsmith
 *
 * Created on June 18, 2010, 11:36 AM
 */

#ifndef _MAGFACTORS_H
#define	_MAGFACTORS_H

#include <Rdefines.h>

#ifdef	__cplusplus
extern "C" {
#endif


   SEXP magChol(SEXP a);
   SEXP magLU(SEXP a);
   SEXP magQR(SEXP a);


#ifdef	__cplusplus
}
#endif

#endif	/* _MAGFACTORS_H */
