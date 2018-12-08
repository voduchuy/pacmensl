static char help[] = "Timing the time to solve hog1p model to time 5 min.\n\n";

#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <cme_util.h>
#include <armadillo>
#include <cmath>
#include <MatrixSet.h>
//#include <Magnus4FSP.h>
#include <FSPSolver.h>
#include <models/hog1p_tv_model.h>

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

void petscvec_to_file(MPI_Comm comm, Vec x, const char *filename);

using namespace hog1p_cme;
using namespace cme::petsc;

int main(int argc, char *argv[]) {
    int ierr, myRank, num_procs;

    std::string model_name = "hog1p";

    /* Set up initial states, probabilities, final time, FSP tolerance */
    int nSpecies = 5; // Number of species
    Row<PetscInt> FSPSize({3, 2, 2, 2, 2}); // Size of the FSP
    PetscReal t_final = 60.00*5;
    PetscReal fsp_tol = 1.0e-2;
    arma::Mat<PetscInt> X0 = {0,0,0,0,0}; X0 = X0.t();
    arma::Col<PetscReal> p0 = {1.0};

    ierr = PetscInitialize(&argc, &argv, (char *) 0, help); CHKERRQ(ierr);
    MPI_Comm comm{MPI_COMM_WORLD};
    MPI_Comm_size(comm, &num_procs);
    PetscPrintf(comm, "\n ================ \n");

    // Read options for fsp
    PartioningType fsp_par_type = Linear;
    ODESolverType fsp_odes_type = CVODE_BDF;
    char opt[100];
    PetscBool opt_set;
    ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_partitioning_type", opt, 100, &opt_set);
    if (opt_set){
        if (strcmp(opt, "ParMetis") == 0){
            fsp_par_type = ParMetis;
            PetscPrintf(PETSC_COMM_WORLD, "FSP is partitioned with ParMetis.\n");
        }
    }

    // Begin PETSC context
    {
        PetscReal tic, tic1, solver_time, total_time;

        tic = MPI_Wtime();
        arma::Row<PetscReal> expansion_factors = {0.0, 0.25,0.25,0.1,0.1};
        FSPSolver fsp(PETSC_COMM_WORLD, fsp_par_type, fsp_odes_type);
        fsp.SetInitFSPSize(FSPSize);
        fsp.SetFSPTolerance(fsp_tol);
        fsp.SetFinalTime(t_final);
        fsp.SetStoichiometry(hog1p_cme::SM);
        fsp.SetTimeFunc(hog1p_cme::t_fun);
        fsp.SetPropensity(hog1p_cme::propensity);
        fsp.SetVerbosityLevel(1);
        fsp.SetExpansionFactors(expansion_factors);
        fsp.SetInitProbabilities(X0, p0);

        tic1 = MPI_Wtime();

        fsp.SetUp();
        fsp.Solve();

        solver_time = MPI_Wtime() - tic1;
        total_time = MPI_Wtime() - tic;

        PetscPrintf(comm, "Total time (including setting up) = %.2e \n", total_time);
        PetscPrintf(comm, "Solving time = %.2e \n", solver_time);

    }
    PetscPrintf(comm, "\n ================ \n");
    //End PETSC context
    ierr = PetscFinalize();
    return ierr;
}


void petscvec_to_file(MPI_Comm comm, Vec x, const char *filename) {
    PetscViewer viewer;
    PetscViewerCreate(comm, &viewer);
    PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &viewer);
    VecView(x, viewer);
    PetscViewerDestroy(&viewer);
}
