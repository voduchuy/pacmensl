/* 
 * ===========================================================================================
 * 
 * This file is header file for EpicSundials.cpp
 * All function declarations are contained in this file.
 * 
 * -------------------------------------------------------------------------------------------
 * 
 * Last revision on 03/28/2016 by Ilija Jegdic
 * 
 * ===========================================================================================
 */

#ifndef __EpicSundials__
#define __EpicSundials__

#ifndef EPICMIN
#define EPICMIN(A, B) ((A) < (B) ? (A) : (B))
#endif

#ifndef EPICMAX
#define EPICMAX(A, B) ((A) > (B) ? (A) : (B))
#endif

#ifndef EPICSQR
#define EPICSQR(A) ((A)*(A))
#endif

realtype EPICRAbs(realtype x);
realtype EPICRSqrt(realtype x);
realtype EPICRExp(realtype x);
realtype EPICRPowerI(realtype base, int exponent);
realtype EPICRPowerR(realtype base, realtype exponent);

#endif