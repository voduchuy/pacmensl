static char help[] = "Formation of the two-species toggle-switch matrix.\n\n";

#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <cme_util.h>
#include <armadillo>
#include <cmath>
#include <string>
#include <fstream>
#include <MatrixSet.h>
#include <Magnus4FSP.h>
#include <FSPSolver.h>
#include "models/toggle_model.h"

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;
using namespace toggle_cme;

void petscvec_to_file(MPI_Comm comm, Vec x, const char *filename);

int main(int argc, char *argv[]) {
    int ierr, myRank;

    static char model[] = "toggle";
    /* CME problem sizes */
    int nSpecies = 2; // Number of species
    Row<PetscReal> FSPIncrement({0.25, 0.25}); // Max fraction of local_states added to each dimension when expanding the FSP
    Row<PetscInt> FSPSize({50, 50}); // Size of the initial FSP
    PetscReal t_final = 100.0;
    PetscReal fsp_tol = 1.0e-6;
    /* Specifiy initial local_states and their probabilities */
    arma::Mat<PetscInt> init_states(2,1); init_states(0,0) = 0; init_states(1,0) = 0;
    arma::Col<PetscReal> init_prob({1.0});

    ierr = PetscInitialize(&argc, &argv, (char *) 0, help); CHKERRQ(ierr);

    MPI_Comm comm = PETSC_COMM_WORLD;
    PetscMPIInt num_procs, my_rank;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &my_rank);

    /* Initialize the FSP solver object */
    cme::petsc::FSP my_fsp(comm, init_states, init_prob, SM,
            propensity, t_fun, FSPSize, FSPIncrement, t_final, fsp_tol);

    /* Solve the problem */
    PetscReal tic, exec_time;
    PetscTime(&tic);
    my_fsp.solve();
    PetscTime(&exec_time); exec_time -= tic;

    /* Export the result:
        - Final FSP dimensions.
        - Runtime.
        - Final probability vector.
     */
    std::string fname;
    if (my_rank == 0)
    {
        std::ofstream fid;
        // Export final FSP size
        fname = std::string(model) +  "_fsp_dim.txt";
        fid.open(fname, std::ios_base::app);
        fid << std::to_string(num_procs) << " " << my_fsp.get_size() << "\n";
        fid.close();
        // Export runtime
        fname = std::string(model) + "_time.txt";
        fid.open(fname, std::ios_base::app);
        fid << std::to_string(num_procs) << " " << exec_time << "\n";
        fid.close();
    }

    fname = std::string(model) + "_sol.out";
    petscvec_to_file(comm, my_fsp.get_P(), fname.c_str());

    /* Destroy FSP object and finalize */
    my_fsp.destroy();
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
