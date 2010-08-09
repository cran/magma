/* 
 * File:   magAriths.h
 * Author: Brian J Smith
 *
 * Created on June 18, 2010, 7:37 PM
 */

#ifndef _MAGARITHS_H
#define	_MAGARITHS_H

#include <Rdefines.h>

#ifdef	__cplusplus
extern "C" {
#endif


   SEXP magMultmm(SEXP a, SEXP transa, SEXP b, SEXP transb);
   SEXP magMultmv(SEXP a, SEXP transa, SEXP x, SEXP right);


#ifdef	__cplusplus
}
#endif

#endif	/* _MAGARITHS_H */

