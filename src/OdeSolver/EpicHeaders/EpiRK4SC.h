/* 
 * ===========================================================================================
 * 
 * This file is header file for EpiRK4SC.cpp. 
 * All constant declarations are contained in this file.
 * 
 * -------------------------------------------------------------------------------------------
 * 
 * Last revision on 02/08/2016 by Ilija Jegdic
 * 
 * ===========================================================================================
 */

#include <cvode/cvode.h>
#include <cvode/cvode_spils.h>
#include "AdaptiveKrylovPhiMultipleTimes.h"
#include "AdaptiveKrylovPhi.h"
#include "IntegratorStats.h"
#include "EpicSundials.h"
#include "EpicConst.h"
#include "EpicTypes.h"

// Integrator scheme coefficients kept in a namespace since no portable way to include
// as non-integrable private static const data members of a class.
namespace EpiRK4SCNamespace
{
  
    // Constants used in EpiRK4SC
  
    static const realtype a21 = 1.0 / 2.0;
    static const realtype a31 = 2.0 / 3.0;

    static const realtype b1  = 1.0;
    static const realtype b2a = 32.0;
    static const realtype b2b = -27.0/2.0;
    static const realtype b3a = -144.0;
    static const realtype b3b = 81.0;

    static const realtype g21 = 1.0 / 2.0;
    static const realtype g31 = 2.0 / 3.0;
    static const realtype g41 = 1.0;
    
    // Inputs for Krylov iterations
    
    static const realtype g1Times[] = {g21, g31};
    static const int g1NumTimes = 2;
    static const int g3NumTimes = 1;
    static const int Stage1NumInputVecs = 2;
    static const int Stage3NumInputVecs = 5;

}

class EpiRK4SC
{
public:
    EpiRK4SC(CVRhsFn f, void *userData, int maxKrylovIters, N_Vector tmpl, const int vecLength);
    EpiRK4SC(CVRhsFn f, CVSpilsJacTimesVecFn jtv, void *userData, int maxKrylovIters, N_Vector tmpl, const int vecLength);
    EpiRK4SC(CVRhsFn f, EPICNumJacDelta delta, void *userData, int maxKrylovIters, N_Vector tmpl, const int vecLength);
    ~EpiRK4SC();
    IntegratorStats *Integrate(realtype h, realtype t0, realtype tFinal, const int numBands, N_Vector y, realtype krylovTol, int startingBasisSizes[]);

private:
    CVRhsFn f;
    CVSpilsJacTimesVecFn jtv;
    EPICNumJacDelta delta;
    void *userData;
    AdaptiveKrylovPhiMultipleTimes *krylov;
    AdaptiveKrylovPhi *krylov2;
    static const int NumProjectionsPerStep = 2;
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
    N_Vector scratchVec1;  // used as a temporary work vector by this class
    N_Vector scratchVec2;  // used as a temporary work vector by this class
    N_Vector scratchVec3;  // used as a temporary work vector by this class
    N_Vector tmpVec;       // used as a temporary work vector by the jtv method
    N_Vector zeroVec;      // zero-ed out vector, used by krylov

    // Disallow copying by "poisoning" the copy constructor and assignment operator,
    // i.e. declare private but provide no implementation.
    EpiRK4SC(const EpiRK4SC &);  // no implementation
    EpiRK4SC & operator=(const EpiRK4SC &);  // no implementation
};
