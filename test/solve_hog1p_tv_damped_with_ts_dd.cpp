static char help[] = "Solve the 5-species spatial hog1p model with time-varying propensities.\n\n";

#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <petscts.h>
#include <petscksp.h>
#include <petscsnes.h>
#include <armadillo>
#include <cmath>
#include "HyperRecOpDD.h"
#include "Magnus4.h"
#include "hog1p_tv_damped_model.h"

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

void petscvec_to_file(MPI_Comm comm, Vec x, const char* filename);
void petscmat_to_file(MPI_Comm comm, Mat A, const char* filename);

typedef struct {
        cme::petsc::HyperRecOpDD* A = nullptr;
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
        Row<PetscInt> FSPSize({ 3, 31, 31, 7, 7});         // Size of the FSP
        PetscReal t_final = 1;
#endif

        double tic;

        ierr = PetscInitialize(&argc,&argv,(char*)0,help); CHKERRQ(ierr);

        PetscBool export_full_solution {PETSC_FALSE};
        PetscOptionsGetBool(NULL, NULL, "-export_full_solution", &export_full_solution, NULL);

        MPI_Comm comm{PETSC_COMM_WORLD};

        MPI_Comm_size(comm, &num_procs);

        arma::Row<PetscInt> processor_grid(5);
        processor_grid.fill(0); processor_grid(0) = 1;

        MPI_Dims_create(num_procs, 5, processor_grid.memptr());

        std::vector<arma::Row<PetscInt>> sub_domain_dims(5);
        for (size_t i{0}; i < 5; ++i)
        {
          sub_domain_dims[i] = cme::distribute_tasks(FSPSize(i)+1, processor_grid(i));
        }

        cme::petsc::HyperRecOpDD A(comm, FSPSize, processor_grid, sub_domain_dims, SM, propensity, t_fun);

        /* Test the action of the operator */
        Vec P0, P;
        VecCreate(comm, &P0);
        VecCreate(comm, &P);

        VecSetSizes(P0, A.n_rows_here, A.n_rows_global);
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

        /* Setting the ts object and its ksp from command line options */
        TS ts;
        KSP ksp;
        SNES snes;
        PC pc;

        TSType ts_type;
        KSPType ksp_type;
        PCType pc_type;

        ierr = TSCreate(comm, &ts); CHKERRQ(ierr);
        ierr = TSSetFromOptions(ts); CHKERRQ(ierr);
        ierr = TSGetType(ts, &ts_type); CHKERRQ(ierr);

        ierr = TSGetSNES(ts, &snes); CHKERRQ(ierr);
        ierr = SNESSetFromOptions(snes); CHKERRQ(ierr);

        ierr = SNESGetKSP(snes, &ksp); CHKERRQ(ierr);
        ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);



        if (strcmp(ts_type, TSSUNDIALS) == 0)
        {
          ierr = TSSundialsGetPC(ts, &pc); CHKERRQ(ierr);
          ierr = PCSetFromOptions(pc); CHKERRQ(ierr);
        }
        else
        {
          ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
          ierr = PCSetFromOptions(pc); CHKERRQ(ierr);
        }

        PCGetType(pc, &pc_type);
        PetscPrintf(comm, "TS = %s Preconditioner = %s \n", ts_type, pc_type);

        /* Set timing and where to write the solution */
        ierr = TSSetSolution(ts, P); CHKERRQ(ierr);
        ierr = TSSetTimeStep(ts, 1.0e-6); CHKERRQ(ierr);
        ierr = TSSetTime(ts, 0.0); CHKERRQ(ierr);
        ierr = TSSetMaxSteps(ts, 10000000); CHKERRQ(ierr);
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
                marginals[i] = cme::petsc::marginal(P, FSPSize, i, A.ao);
        }

        TSType time_scheme;
        TSGetType(ts, &time_scheme);

        MPI_Comm_rank(comm, &myRank);
        if (myRank == 0)
        {
                {
                        std::string filename = model_name + "_" + std::string(time_scheme) + "_dd_time_" + std::to_string(num_procs) + ".dat";
                        std::ofstream file;
                        file.open(filename);
                        file << solver_time;
                        file.close();
                }
                for (PetscInt i{0}; i < marginals.size(); ++i )
                {
                        std::string filename = model_name + "_" + std::string(time_scheme) + "_dd_marginal_" + std::to_string(i) + "_"+ std::to_string(num_procs)+ ".dat";
                        marginals[i].save(filename, arma::raw_ascii);
                }
        }

        if (export_full_solution)
        {
          std::string filename = model_name + "_" + std::string(time_scheme) + "_dd_full_" + std::to_string(num_procs)+ ".out";
          petscvec_to_file(comm, P, filename.c_str());
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
