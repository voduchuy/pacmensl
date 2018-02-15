static char help[] = "Formation of the two-species toggle-switch matrix.\n\n";

#include <petscvec.h>
#include <petscmat.h>
#include <petscts.h>
#include <petscviewer.h>
#include <armadillo>
#include <cmath>
#include "cme_util.hpp"
#include "fsp_matrix.hpp"
#include "KExpv.hpp"

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

/* Stoichiometric matrix of the toggle switch model */
arma::Mat<int> SM { { 1, 1, -1, 0, 0,  0 },
                    { 0, 0,  0, 1, 1, -1 } };

/* Parameters for the propensity functions */
const double ayx {6.1e-3}, axy {2.6e-3}, nyx {2.1e0}, nxy {3.0e0},
kx0 {6.8e-5}, kx {1.6e-1}, dx {0.00067}, ky0 {2.2e-3}, ky {1.7e-1}, dy {3.8e-4};


arma::Mat<double> propensity( arma::Mat<PetscInt> X);

int main(int argc, char *argv[]) {
        int ierr, myRank;

        /* CME problem sizes */
        Row<PetscInt> FSPSize({ 500, 500 }); // Size of the FSP

        /* Final time */
        PetscReal tf{100.0};

        ierr = PetscInitialize(&argc,&argv,(char*)0,help); CHKERRQ(ierr);

        MPI_Comm comm{PETSC_COMM_WORLD};
        ierr = MPI_Comm_rank( comm, &myRank );

        PetscInt nGlobal= arma::prod( FSPSize+1 );
        cout << nGlobal << endl;

        /* Create and assemble the matrix */
        Mat A;
        MatCreate(comm, &A );
        ierr = MatSetSizes( A, PETSC_DECIDE, PETSC_DECIDE, nGlobal, nGlobal );
        ierr = MatSetType( A, MATMPIAIJ );
        ierr = MatSetUp( A );

        FSPMatSetValues(comm, A, FSPSize, SM, propensity );

        PetscReal anorm;
        MatNorm( A, NORM_FROBENIUS, &anorm );
        std::cout << "Matrix generated. norm(A) = " << anorm << "\n";

        /* Create and assemble intial vector */
        Vec P0;
        VecCreate(comm, &P0);
        VecSetSizes(P0, PETSC_DECIDE, nGlobal);
        VecSetType(P0, VECMPI);

        VecSet(P0, 0.0);
        PetscInt indx[1];
        indx[0] = 0;
        VecSetValue(P0, 1, 1.0, INSERT_VALUES);
        VecAssemblyBegin(P0);
        VecAssemblyEnd(P0);

        Vec P;
        VecDuplicate(P0, &P);
        ierr = VecCopy(P0, P);

        /* Create and set the time-stepping object */
        TS ts;
        ierr = TSCreate(comm, &ts); CHKERRQ(ierr);
        ierr = TSSetSolution(ts, P); CHKERRQ(ierr);
        ierr = TSSetType(ts, TSRK); CHKERRQ(ierr);
        ierr = TSSetTime(ts, 0.0); CHKERRQ(ierr);
        ierr = TSSetMaxSteps(ts, 10000); CHKERRQ(ierr);
        ierr = TSSetMaxTime(ts, tf); CHKERRQ(ierr);
        ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);

        TSAdapt tsadapt;

        ierr = TSGetAdapt(ts, &tsadapt);
        TSAdaptSetType(tsadapt, TSADAPTBASIC);
        TSSetTolerances(ts, 1.0e-10, NULL, 1.0e-7, NULL);

        TSSetFromOptions(ts);
        /* Set the linear ODE problem */
        ierr = TSSetProblemType(ts, TS_LINEAR);
        TSSetRHSFunction(ts, NULL, TSComputeRHSFunctionLinear, NULL);
        TSSetRHSJacobian(ts, A, A, TSComputeRHSJacobianConstant, NULL);

        double tic = MPI_Wtime();
        ierr = TSSolve(ts, P);
        tic = MPI_Wtime() - tic;
        std::cout << "Solver time " << tic << std::endl;

        /* Try solving with Krylov */
        Vec w;
        VecDuplicate(P0, &w);
        VecCopy(P0, w);

        auto matvec = [A] (Vec x, Vec y)
        {
          MatMult(A, x, y);
        };

        cme::petsc::KExpv kexpv(comm, tf, matvec, w, 30, 1.0e-8, true, 2, 1.0);

        kexpv.solve();

        VecAXPY(w, -1.0, P);

        PetscReal pnorm, gap;
        VecNorm(P, NORM_1, &pnorm);
        std::cout << "|P|_1 = " << pnorm << "\n";
        VecNorm(P, NORM_2, &pnorm);
        std::cout << "|P|_2 = " << pnorm << "\n";
        VecNorm(w, NORM_2, &gap);
        std::cout << "Gap between two methods = " << gap << "\n";

        kexpv.destroy();
        ierr = TSDestroy( &ts ); CHKERRQ(ierr);
        ierr = VecDestroy( &w ); CHKERRQ(ierr);
        ierr = VecDestroy( &P0 ); CHKERRQ(ierr);
        ierr = VecDestroy( &P ); CHKERRQ(ierr);
        ierr = MatDestroy( &A ); CHKERRQ(ierr);
        ierr = PetscFinalize();
        return ierr;
}

// propensity function for toggle
arma::Mat<double> propensity( arma::Mat<PetscInt> X)
{
        arma::Mat<double> prop( X.n_cols, 6);

        for (size_t i{0}; i < X.n_cols; i++)
        {
                prop(i, 0) = kx0;
                prop(i, 1) = kx/( 1.0 + ayx*pow( double( X(1,i)), nyx ) );
                prop(i, 2) = dx*double( X(0,i) );
                prop(i, 3) = ky0;
                prop(i, 4) = ky/( 1.0 + axy*pow( double( X(0,i)), nxy) );
                prop(i, 5) = dy*double( X(1,i) );
        }

        return prop;
}
