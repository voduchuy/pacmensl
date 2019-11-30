/* 
 * ===========================================================================================
 * 
 * This file is header file for EpiRK5V.cpp. 
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
namespace EpiRK5VNamespace
{
    
    // Constants used in EpiRK5V
  
    static const realtype a11 = 0.3512959269505819309205666343669703185949150765418610822442;
    static const realtype a21 = 0.8440547201165712629778740943954345394870608397760182695023;
    static const realtype a22 = 1.6905891609568963623795529686906583866077099749695686415193;

    static const realtype b1 = 1.0000000000000000000000000000413981626492407621654383185164;
    static const realtype b2 = 1.2727127317356892397444638436834144153459435577242007287745;
    static const realtype b3 = 2.2714599265422622275490331068926657550105500849448098122502;

    // Inputs for Krylov iterations
    
    static const realtype g11 = 0.3512959269505819309205666513780525924758396470877807385413;
    static const realtype g21 = 0.8440547201165712629778740927319685574987844036261402099979;
    static const realtype g22 = 0.5;
    static const realtype g31 = 1.0;
    static const realtype g32 = 0.7111109536436687035914924546923329396898123318411132877913;
    static const realtype g33 = 0.6237811195337149480885036224182618420517566408959269219908;
    
    static const realtype g33LowOrder = 1.0;

    static const realtype g1Times[] = {g11, g21, g31};
    static const realtype g2Times[] = {g22, g32};
    static const realtype g3Times[] = {g33, g33LowOrder};
    static const int g1NumTimes = 3;
    static const int g2NumTimes = 2;
    static const int g3NumTimes = 2;
    
    static const int Stage1NumInputVecs = 2;
    static const int Stage2NumInputVecs = 2;
    static const int Stage3NumInputVecs = 4;
    
    static const realtype Fac = 0.9;  // safety factor for adaptive time stepping
    static const realtype Order = -1.0 / 4.0;
}

class EpiRK5V
{
public:
    // Constructor assumes tmpl vector is a valid vector in that it doesn't contain NAN values.
    EpiRK5V(CVRhsFn f, CVSpilsJacTimesVecFn jtv, void *userData, int maxKrylovIters, N_Vector tmpl, const int vecLength);
    EpiRK5V(CVRhsFn f, void *userData, int maxKrylovIters, N_Vector tmpl, const int vecLength);
    EpiRK5V(CVRhsFn f, EPICNumJacDelta delta, void *userData, int maxKrylovIters, N_Vector tmpl, const int vecLength); 
    virtual ~EpiRK5V();
    IntegratorStats *Integrate(const realtype hStart, const realtype hMax, const realtype absTol, const realtype relTol, const realtype t0, const realtype tFinal, const int numBands, int *startingBasisSizes, N_Vector y);

protected:
    CVRhsFn f;
    CVSpilsJacTimesVecFn jtv;
    EPICNumJacDelta delta;
    void *userData;
    AdaptiveKrylovPhiMultipleTimes *krylov;
    AdaptiveKrylovPhiMultipleTimes *krylov2;
    AdaptiveKrylovPhiMultipleTimes *krylov3;
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
    N_Vector r22;
    N_Vector r3;
    N_Vector r32;
    N_Vector r33;
    N_Vector r3LowOrder;
    N_Vector r32LowOrder;
    N_Vector r33LowOrder;
    N_Vector scratchVec1;  // used as a temporary work vector by this class
    N_Vector scratchVec2;  // used as a temporary work vector by this class
    N_Vector scratchVec3;  // used as a temporary work vector by this class
    N_Vector tmpVec;  // used as a temporary work vector by the jtv method
    N_Vector zeroVec;  // zero-ed out vector, used by krylov

    // Disallow copying by "poisoning" the copy constructor and assignment operator,
    // i.e. declare private but provide no implementation.
    EpiRK5V(const EpiRK5V &);  // no implementation
    EpiRK5V & operator=(const EpiRK5V &);  // no implementation
};
