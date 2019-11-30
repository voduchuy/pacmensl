/* 
 * ===========================================================================================
 * 
 * This file is header file for EpiRK4SV.cpp. 
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
#include "AdaptiveKrylovPhi.h"
#include "AdaptiveKrylovPhiMultipleTimes.h"
#include "IntegratorStats.h"
#include "EpicTypes.h"

// Integrator scheme coefficients kept in a namespace since no portable way to include
// as non-integrable private static const data members of a class.
namespace EpiRK4SVNamespace
{
    
    // Constants used in EpiRK4SV
  
    static const realtype a21 = 1.0 / 2.0;
    static const realtype a31 = 2.0 / 3.0;
    static const realtype a21LowOrder = 3.0 / 4.0;

    static const realtype b1 = 1.0;
    static const realtype b2a = 32.0;
    static const realtype b2b = -27.0/2.0;
    static const realtype b3a = -144.0;
    static const realtype b3b = 81.0;

    static const realtype b2aLowOrder = 1.0;
    static const realtype b2bLowOrder = 92.0/9.0;

    // Inputs for Krylov iterations
    
    static const realtype g21 = 1.0 / 2.0;
    static const realtype g31 = 2.0 / 3.0;
    static const realtype g41 = 1.0;
    static const realtype g21LowOrder = 3.0 / 4.0;

    static const realtype g1Times[] = {g21, g31, g21LowOrder};
    static const realtype g3Times[] = {g41};
    static const int g1NumTimes = 3;
    static const int g3NumTimes = 1;
    static const int Stage1NumInputVecs = 2;
    static const int Stage3NumInputVecs = 5;
    static const int Stage3NumInputVecsLowOrder = 4;

    static const realtype Fac = 0.9;  // safety factor for adaptive time stepping
    static const realtype Order = -1.0 / 3.0;
}

class EpiRK4SV
{
public:
    // Constructor assumes tmpl vector is a valid vector in that it doesn't contain NAN values.
    EpiRK4SV(CVRhsFn f, CVSpilsJacTimesVecFn jtv, void *userData, int maxKrylovIters, N_Vector tmpl, const int vecLength);
    EpiRK4SV(CVRhsFn f, void *userData, int maxKrylovIters, N_Vector tmpl, const int vecLength);
    EpiRK4SV(CVRhsFn f, EPICNumJacDelta delta, void *userData, int maxKrylovIters, N_Vector tmpl, const int vecLength);
    virtual ~EpiRK4SV();
    IntegratorStats *Integrate(const realtype hStart, const realtype hMax, const realtype absTol, const realtype relTol, const realtype t0, const realtype tFinal, const int numBands, int *startingBasisSizes, N_Vector y);

 protected:
    CVRhsFn f;
    CVSpilsJacTimesVecFn jtv;
    EPICNumJacDelta delta;
    void *userData;
    AdaptiveKrylovPhiMultipleTimes *krylov;
    AdaptiveKrylovPhi *krylov2;
    static const int NumProjectionsPerStep = 3;
    static const int MaxPhiOrder1 = 2;
    static const int MaxPhiOrder2 = 4;
    const int NEQ;
    IntegratorStats *integratorStats;

    N_Vector fy;
    N_Vector hfy;
    N_Vector hb1;
    N_Vector hb2;
    N_Vector r1;
    N_Vector r2;
    N_Vector r3LowOrder;
    N_Vector scratchVec1;  // used as a temporary work vector by this class
    N_Vector scratchVec2;  // used as a temporary work vector by this class
    N_Vector scratchVec3;  // used as a temporary work vector by this class
    N_Vector scratchVec4;  // used as a temporary work vector by this class
    N_Vector scratchVec5;  // used as a temporary work vector by this class
    N_Vector scratchVec6;  // used as a temporary work vector by this class
    N_Vector tmpVec;  // used as a temporary work vector by the jtv method
    N_Vector zeroVec;  // zero-ed out vector, used by krylov

    // Disallow copying by "poisoning" the copy constructor and assignment operator,
    // i.e. declare private but provide no implementation.
    EpiRK4SV(const EpiRK4SV &);  // no implementation
    EpiRK4SV & operator=(const EpiRK4SV &);  // no implementation
};
