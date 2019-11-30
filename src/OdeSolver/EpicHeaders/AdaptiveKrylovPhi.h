/* 
 * ===========================================================================================
 * 
 * This file is header file for AdaptiveKrylovPhi.cpp. 
 * All constant declarations are contained in this file.
 * 
 * -------------------------------------------------------------------------------------------
 * 
 * Last revision on 02/28/2016 by Ilija Jegdic
 * 
 * ===========================================================================================
 */

#ifndef __AdaptiveKrylovPhi__
#define __AdaptiveKrylovPhi__

#include <sundials/sundials_nvector.h>
#include "JTimesV.h"
#include "ExpMHigham.h"
#include "IntegratorStats.h"

namespace AdaptiveKrylovPhiNamespace
{
	// Safety factor constants used for adaptivity.
	const realtype Gamma = 0.8;
	const realtype Delta = 1.2;
	const realtype Zero = RCONST(0.0);
	const realtype One = RCONST(1.0);
}

class AdaptiveKrylovPhi
{
public:
	AdaptiveKrylovPhi(int maxNumVectors, int maxIters, N_Vector templateVector, int vecLength);
	~AdaptiveKrylovPhi();
	// Assumes the times are in ascending order.
	//N_Vector Compute(int numVectors, N_Vector *inputVectors, JTimesV *jtimesv, realtype h);
	//N_Vector Compute(int numVectors, N_Vector *inputVectors, int vecLength, realtype tFinal, JTimesV *jtimesv, realtype h, realtype tol, KrylovStats *krylovStats, double *outputH);
	N_Vector Compute(const int numBands, int numVectors, N_Vector *inputVectors, N_Vector outputVector, realtype tFinal, JTimesV *jtimesv, realtype h, realtype tol, int &targetBasisSize, KrylovStats *krylovStats);

private:
	const int MaxNumVectors;
	const int MaxPhiOrder;
	const int VecLength;
	N_Vector *stateVectors;
	N_Vector scratchVector;

	// Krylov specific state.
	const int MaxKrylovIters;
	const int HMatrixLength;  // MaxIters + 1 (to accommodate last iteration)
	const int PhiMatrixLength;  // MaxIters + MaxPhiOrder + 1
	N_Vector *V;
	realtype *H;  // a matrix formatted as a single array in column-major order
	double *phiMatrix;  // matrix passed in/out of expm, of length sufficient to accommodate augmenting
	ExpMHigham *expm;

	// Convenience routines.
	double max(double a, double b) {return a > b ? a : b;}
	double min(double a, double b) {return a < b ? a : b;}

	// Routine to copy, scale by tau, and augment the H matrix into the phiMatrix.
	void SetupPhiMatrix(int krylovIterations, int p, realtype tau);

	// Disallow copying by "poisoning" copy constructor and assignment operator,
	// i.e. declare private but provide no implementation.
	AdaptiveKrylovPhi(const AdaptiveKrylovPhi &);  // no implementation
	AdaptiveKrylovPhi & operator=(const AdaptiveKrylovPhi &);  // no implementation
};

#endif