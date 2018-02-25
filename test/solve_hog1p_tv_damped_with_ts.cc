static char help[] = "Solve the 5-species spatial hog1p model with time-varying propensities.\n\n";

#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <petscts.h>
#include <armadillo>
#include <cmath>
#include "HyperRecOp.hpp"
#include "Magnus4.hpp"
#include "hog1p_tv_damped_model.hpp"

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

void petscvec_to_file(MPI_Comm comm, Vec x, const char* filename);
void petscmat_to_file(MPI_Comm comm, Mat A, const char* filename);

typedef struct {
        cme::petsc::HyperRecOp* A = nullptr;
} Appctx;

PetscErrorCode my_jacobian(TS ts, PetscReal t, Vec u, Mat A1, Mat B1, void *appctx) {
        Appctx   *ctx = (Appctx*) appctx;
        (*ctx->A)(t).dump_to_mat(A1);
        (*ctx->A)(t).dump_to_mat(B1);
        return 0;
};

int main(int argc, char *argv[]) {
        int ierr, myRank, num_procs;
        double solver_time;

        std::string model_name = "hog1p_damped";

        using namespace hog1p_cme;

        /* CME problem sizes */
#ifdef TEST_LARGE_PROBLEM
        Row<PetscInt> FSPSize({ 3, 40, 40, 60, 60});         // Size of the FSP
        PetscReal t_final = 120;
#else
        Row<PetscInt> FSPSize({ 3, 20, 20, 10, 10});         // Size of the FSP
        PetscReal t_final = 30;
#endif

        double tic;

        ierr = PetscInitialize(&argc,&argv,(char*)0,help); CHKERRQ(ierr);

        MPI_Comm comm{PETSC_COMM_WORLD};

        MPI_Comm_size(comm, &num_procs);

        cme::petsc::HyperRecOp A( comm, FSPSize, SM, propensity, t_fun);

        /* Test the action of the operator */
        Vec P0, P;
        VecCreate(comm, &P0);
        VecCreate(comm, &P);

        VecSetSizes(P0, PETSC_DECIDE, arma::prod(FSPSize+1));
        VecSetType(P0, VECMPI);
        VecSetUp(P0);
        VecSet(P0, 0.0);
        VecSetValue(P0, 0, 1.0, INSERT_VALUES);
        VecAssemblyBegin(P0);
        VecAssemblyEnd(P0);

        VecDuplicate(P0, &P); VecCopy(P0, P);

        /* Create and set the time-stepping object */
        Appctx appctx;
        appctx.A = &A;

        Mat A1;
        MatCreate(comm, &A1);
        A.duplicate_structure(A1);

        TS ts;
        ierr = TSCreate(comm, &ts); CHKERRQ(ierr);
        ierr = TSSetFromOptions(ts); CHKERRQ(ierr);
        ierr = TSSetSolution(ts, P); CHKERRQ(ierr);
        ierr = TSSetTimeStep(ts, 1.0e-4); CHKERRQ(ierr);
        ierr = TSSetTime(ts, 0.0); CHKERRQ(ierr);
        ierr = TSSetMaxSteps(ts, 10000); CHKERRQ(ierr);
        ierr = TSSetMaxTime(ts, t_final); CHKERRQ(ierr);
        ierr = TSSetExactFinalTime(ts, TS_EXACTFINALTIME_INTERPOLATE); CHKERRQ(ierr);

        /* Set the linear ODE problem */
        ierr = TSSetProblemType(ts, TS_LINEAR);
        TSSetRHSFunction(ts, NULL, TSComputeRHSFunctionLinear, &appctx);
        TSSetRHSJacobian(ts, A1, A1, &my_jacobian, &appctx);

        tic = MPI_Wtime();
        ierr = TSSolve(ts, P);
        solver_time = MPI_Wtime() - tic;
        PetscPrintf(comm, "Solver time %4.2f \n", solver_time);

        /* Compute the marginal distributions */
        std::vector<arma::Col<PetscReal> > marginals(FSPSize.n_elem);
        for (PetscInt i{0}; i < marginals.size(); ++i )
        {
                marginals[i] = cme::petsc::marginal(P, FSPSize, i);
        }

        TSType time_scheme;
        TSGetType(ts, &time_scheme);

        MPI_Comm_rank(comm, &myRank);
        if (myRank == 0)
        {
                {
                        std::string filename = model_name + "_" + std::string(time_scheme) + "_time_" + std::to_string(num_procs) + ".dat";
                        std::ofstream file;
                        file.open(filename);
                        file << solver_time;
                        file.close();
                }
                for (PetscInt i{0}; i < marginals.size(); ++i )
                {
                        std::string filename = model_name + "_" + std::string(time_scheme) + "_marginal_" + std::to_string(i) + "_"+ std::to_string(num_procs)+ ".dat";
                        marginals[i].save(filename, arma::raw_ascii);
                }
        }

        MatDestroy(&A1);
        ierr = VecDestroy(&P); CHKERRQ(ierr);
        ierr = VecDestroy(&P0); CHKERRQ(ierr);
        A.destroy();
        ierr = PetscFinalize();
        return ierr;
}

void petscvec_to_file(MPI_Comm comm, Vec x, const char* filename)
{
        PetscViewer viewer;
        PetscViewerCreate(comm, &viewer);
        PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &viewer);
        VecView(x, viewer);
        PetscViewerDestroy(&viewer);
}

void petscmat_to_file(MPI_Comm comm, Mat A, const char* filename)
{
        PetscViewer viewer;
        PetscViewerCreate(comm, &viewer);
        PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &viewer);
        MatView(A, viewer);
        PetscViewerDestroy(&viewer);
}
