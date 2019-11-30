/* 
 * ===========================================================================================
 * 
 * This file is header file for AdaptiveKrylovPhiMultipleTimes.cpp
 * All constant declarations are contained in this file.
 * 
 * -------------------------------------------------------------------------------------------
 * 
 * Last revision on 02/28/2016 by Ilija Jegdic
 * 
 * ===========================================================================================
 */

#ifndef __AdaptiveKrylovPhiMultipleTimes__
#define __AdaptiveKrylovPhiMultipleTimes__

#include <sundials/sundials_nvector.h>
#include "JTimesV.h"
#include "ExpMHigham.h"
#include "IntegratorStats.h"

namespace AdaptiveKrylovPhiMultipleTimesNamespace
{
	// Safety factor constants used for adaptivity.
	const realtype Gamma = 0.8;
	const realtype Delta = 1.2;
	const realtype Zero = RCONST(0.0);
	const realtype One = RCONST(1.0);
}

class AdaptiveKrylovPhiMultipleTimes
{
public:
	AdaptiveKrylovPhiMultipleTimes(int maxNumVectors, int maxIters, N_Vector templateVector, int vecLength);
	~AdaptiveKrylovPhiMultipleTimes();
	// Assumes the times are in ascending order.
	//N_Vector Compute(int numVectors, N_Vector *inputVectors, JTimesV *jtimesv, realtype h);
	void Compute(const int numbands, const int numVectors, N_Vector *inputVectors, const realtype timePoints[], const int numTimePoints, N_Vector *outputVectors, JTimesV *jtimesv, realtype h, realtype tol, int &targetBasisSize, KrylovStats *krylovStats);

private:
	const int MaxNumInputVectors;
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
	double *phiMatrixSkipped;  // same as phiMatrix but for skipped time values
	ExpMHigham *expm;

	// Convenience routines.
	double themax(double a, double b) {return a > b ? a : b;}
	double themin(double a, double b) {return a < b ? a : b;}

	// Routine to copy, scale by tau, and augment the H matrix into the phiMatrix.
	void SetupPhiMatrix(double *phiMatrix, int krylovIterations, int p, realtype tau);

	// Disallow copying by "poisoning" copy constructor and assignment operator,
	// i.e. declare private but provide no implementation.
	AdaptiveKrylovPhiMultipleTimes(const AdaptiveKrylovPhiMultipleTimes &);  // no implementation
	AdaptiveKrylovPhiMultipleTimes & operator=(const AdaptiveKrylovPhiMultipleTimes &);  // no implementation
};

#endif
