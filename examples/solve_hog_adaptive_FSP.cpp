static char help[] = "Solve the 5-species spatial hog1p model with time-varying propensities using adaptive finite state projection.\n\n";

#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <cme_util.h>
#include <armadillo>
#include <cmath>
#include <MatrixSet.h>
//#include <Magnus4FSP.h>
#include <FSPSolver.h>
#include <models/hog1p_5d_model.h>

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
    /* CME problem sizes */
    int nSpecies = 5; // Number of species
    Row<PetscInt> FSPSize({3, 2, 2, 2, 2}); // Size of the FSP
    PetscReal t_final = 60.00*15;
    PetscReal fsp_tol = 1.0e-2;
    arma::Mat<PetscInt> X0 = {0,0,0,0,0}; X0 = X0.t();
    arma::Col<PetscReal> p0 = {1.0};

    ierr = PetscInitialize(&argc, &argv, (char *) 0, help); CHKERRQ(ierr);
    MPI_Comm comm{MPI_COMM_WORLD};
    MPI_Comm_size(comm, &num_procs);
    // Begin PETSC context
    {
        arma::Row<PetscReal> expansion_factors = {0.0, 0.25,0.25,0.05,0.05};
        FSPSolver fsp(PETSC_COMM_WORLD, Linear, CVODE_BDF);
        fsp.SetInitFSPSize(FSPSize);
        fsp.SetFSPTolerance(fsp_tol);
        fsp.SetFinalTime(t_final);
        fsp.SetStoichiometry(hog1p_cme::SM);
        fsp.SetTimeFunc(hog1p_cme::t_fun);
        fsp.SetPropensity(hog1p_cme::propensity);
        fsp.SetVerbosityLevel(2);
        fsp.SetExpansionFactors(expansion_factors);
        fsp.SetInitProbabilities(X0, p0);

        PetscReal tic, solver_time;
        tic = MPI_Wtime();
        fsp.SetUp();
        fsp.Solve();
        solver_time = MPI_Wtime() - tic;

        /* Compute the marginal distributions */
        Vec P = fsp.GetP();
        FiniteStateSubset& state_set = fsp.GetStateSubset();
        std::vector<arma::Col<PetscReal>> marginals(FSPSize.n_elem);
        for (PetscInt i{0}; i < marginals.size(); ++i) {
            marginals[i] = cme::petsc::marginal(state_set, P, i);
        }

        MPI_Comm_rank(PETSC_COMM_WORLD, &myRank);
        if (myRank == 0) {
            {
                std::string filename = model_name + "_time_" + std::to_string(num_procs) + ".dat";
                std::ofstream file;
                file.open(filename);
                file << solver_time;
                file.close();
            }
            for (PetscInt i{0}; i < marginals.size(); ++i) {
                std::string filename =
                        model_name + "_marginal_" + std::to_string(i) + "_" + std::to_string(num_procs) + ".dat";
                marginals[i].save(filename, arma::raw_ascii);
            }
        }
    }
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
