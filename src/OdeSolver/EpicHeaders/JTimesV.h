/* 
 * ===========================================================================================
 * 
 * This file is header file for JTimesV.cpp
 * All constant declarations are contained in this file.
 * 
 * -------------------------------------------------------------------------------------------
 * 
 * Last revision on 07/05/2016 by Ilija Jegdic
 * 
 * ===========================================================================================
 */

#ifndef __JTimesV__
#define __JTimesV__

#include <cvode/cvode.h>
#include <cvode/cvode_spils.h>
#include "EpicSundials.h"
#include "EpicConst.h"
#include "EpicTypes.h"
#include <math.h>

realtype DefaultDelta(N_Vector u, N_Vector v, realtype normu, realtype normv, void *userData);
realtype Delta1(N_Vector u, N_Vector v, realtype normu, realtype normv, void *userData);
realtype Delta2(N_Vector u, N_Vector v, realtype normu, realtype normv, void *userData);

class JTimesV
{
public:
    JTimesV(CVSpilsJacTimesVecFn Jtv, realtype t, N_Vector y, N_Vector fy, void *userData, N_Vector tempVector);
    JTimesV(CVSpilsJacTimesVecFn Jtv, CVRhsFn f, realtype t, N_Vector y, N_Vector fy, void *userData, N_Vector tempVector);
    JTimesV(CVSpilsJacTimesVecFn Jtv, CVRhsFn f, EPICNumJacDelta Delta, realtype t, N_Vector y, N_Vector fy, void *userData, N_Vector tempVector);
    ~JTimesV();
    void ComputeJv(N_Vector v, N_Vector Jv);

private:
    CVSpilsJacTimesVecFn Jtv;
    CVRhsFn f;
    realtype t;
    N_Vector y;
    N_Vector fy;
    EPICNumJacDelta Delta;
    void *userData;
    N_Vector tempVector;
    realtype normy;
    N_Vector scratchVec1;
    N_Vector scratchVec2;
    
};

#endif
