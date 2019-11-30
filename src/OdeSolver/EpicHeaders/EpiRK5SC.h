/* 
 * ===========================================================================================
 * 
 * This file is header file for EpiRK5SC.cpp. 
 * All constant declarations are contained in this file.
 * 
 * -------------------------------------------------------------------------------------------
 * 
 * Last revision on 02/13/2016 by Ilija Jegdic
 * 
 * ===========================================================================================
 */

#include <cvode/cvode.h>
#include <cvode/cvode_spils.h>
#include "AdaptiveKrylovPhiMultipleTimes.h"
#include "AdaptiveKrylovPhi.h"
#include "IntegratorStats.h"
#include <fstream>
#include <iostream>
#include "EpicConst.h"
#include "EpicTypes.h"

// Integrator scheme coefficients kept in a namespace since no portable way to include
// as non-integrable private static const data members of a class.
namespace EpiRK5SCNamespace
{
    
    // Constants used in EpiRK5SC
  
    static const realtype a21 = 1.0/2.0;
    static const realtype a31 = 9.0/10.0;
    static const realtype a32 = 27.0/25.0;
    static const realtype a33 = 729.0/125.0;
    
    static const realtype b1 = 1.0;
    static const realtype b2a = 18.0;
    static const realtype b2b = -250./81.0;
    static const realtype b3a = -60.0;
    static const realtype b3b = 500.0/27.0;
    
    // Inputs for Krylov iterations

    static const realtype g21 = 1.0 / 2.0;
    static const realtype g31 = 9.0 / 10.0;
    static const realtype g32 = 1.0 / 2.0;
    static const realtype g33 = 9.0 / 10.0;
    static const realtype g41 = 1.0;

    static const realtype g1Times[] = {g21, g31};
    static const realtype g2Times[] = {g32, g33};
    static const realtype g3Times[] = {g41};
    static const int g1NumTimes = 2;
    static const int g2NumTimes = 2;
    static const int g3NumTimes = 1;
    static const int Stage1NumInputVecs = 2;
    static const int Stage2NumInputVecs = 4;
    static const int Stage3NumInputVecs = 5;

}

class EpiRK5SC
{
public:
    EpiRK5SC(CVRhsFn f, CVSpilsJacTimesVecFn jtv, void *userData, int maxKrylovIters, N_Vector tmpl, const int vecLength);
    EpiRK5SC(CVRhsFn f, void *userData, int maxKrylovIters, N_Vector tmpl, const int vecLength);
    EpiRK5SC(CVRhsFn f, EPICNumJacDelta delta, void *userData, int maxKrylovIters, N_Vector tmpl, const int vecLength);
    ~EpiRK5SC();
    IntegratorStats *Integrate(realtype h, realtype t0, realtype tFinal, const int numBands, N_Vector y, realtype krylovTol, int startingBasisSizes[]);

private:
    CVRhsFn f;
    CVSpilsJacTimesVecFn jtv;
    EPICNumJacDelta delta;
    void *userData;
    AdaptiveKrylovPhiMultipleTimes *krylov1;
    AdaptiveKrylovPhiMultipleTimes *krylov2;
    AdaptiveKrylovPhi *krylov3;
    static const int NumProjectionsPerStep = 3;
    static const int MaxPhiOrder1 = 2;
    static const int MaxPhiOrder2 = 4;
    const int NEQ;
    IntegratorStats *integratorStats;

    N_Vector fy;
    N_Vector hfy;
    N_Vector hb1;
    N_Vector hb2;
    N_Vector r2;
    N_Vector r3;
    N_Vector r32;
    N_Vector r33;
    N_Vector scratchVec1;  // used as a temporary work vector by this class
    N_Vector scratchVec2;  // used as a temporary work vector by this class
    N_Vector scratchVec3;  // used as a temporary work vector by this class
    N_Vector tmpVec;  // used as a temporary work vector by the jtv method
    N_Vector zeroVec;  // zero-ed out vector, used by krylov

    // Disallow copying by "poisoning" the copy constructor and assignment operator,
    // i.e. declare private but provide no implementation.
    EpiRK5SC(const EpiRK5SC &);  // no implementation
    EpiRK5SC & operator=(const EpiRK5SC &);  // no implementation
};
