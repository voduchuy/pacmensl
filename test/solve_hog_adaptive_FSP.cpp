static char help[] = "Solve the 5-species spatial hog1p model with time-varying propensities using adaptive finite state projection.\n\n";

#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <cme_util.h>
#include <armadillo>
#include <cmath>
#include <HyperRecOp.h>
#include <Magnus4FSP.h>
#include <FSP.h>
#include <models/hog1p_tv_model.h>

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

void petscvec_to_file(MPI_Comm comm, Vec x, const char *filename);

using namespace hog1p_cme;

int main(int argc, char *argv[]) {
    int ierr, myRank;

    std::string model_name = "hog1p";

    /* CME problem sizes */
    int nSpecies = 5; // Number of species
    Row<PetscReal> FSPIncrement({0.25, 0.25, 0.25, 0.25, 0.25}); // Max fraction of states added to each dimension when expanding the FSP
    Row<PetscInt> FSPSize({3, 2, 2, 2, 2}); // Size of the FSP
    PetscReal t_final = 300.0;
    PetscReal fsp_tol = 1.0e-4;
    PetscReal mg_tol = 1.0e-6;
    arma::Mat<PetscInt> init_states(5,1); init_states.fill(0.0);
    arma::Col<PetscReal> init_prob({1.0});

    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    CHKERRQ(ierr);

    PetscBool export_full_solution {PETSC_FALSE};
    ierr = PetscOptionsGetBool(NULL, NULL, "-export_full_solution", &export_full_solution, NULL); CHKERRQ(ierr);

    MPI_Comm comm = PETSC_COMM_WORLD;
    PetscMPIInt num_procs;
    MPI_Comm_size(comm, &num_procs);

    cme::petsc::FSP my_fsp(comm, init_states, init_prob, SM,
            propensity, t_fun, FSPSize, FSPIncrement, t_final, fsp_tol, mg_tol);

    PetscReal tic = MPI_Wtime();
    my_fsp.solve();
    PetscReal solver_time = MPI_Wtime() - tic;

    /* Compute the marginal distributions */
    std::vector<arma::Col<PetscReal> > marginals(FSPSize.n_elem);
    for (PetscInt i{0}; i < marginals.size(); ++i )
    {
        marginals[i] = cme::petsc::marginal(my_fsp.get_P(), FSPSize, i);
    }

    MPI_Comm_rank(comm, &myRank);
    if (myRank == 0)
    {
        {
            std::string filename = model_name + "_time_FSP_" + std::to_string(num_procs) + ".dat";
            std::ofstream file;
            file.open(filename);
            file << solver_time;
            file.close();
        }
        for (PetscInt i{0}; i < marginals.size(); ++i )
        {
            std::string filename = model_name + "_marginal_FSP_" + std::to_string(i) + "_"+ std::to_string(num_procs)+ ".dat";
            marginals[i].save(filename, arma::raw_ascii);
        }
    }

    if (export_full_solution)
    {
        std::string filename = model_name + "_" + "FSP" + "_full_" + std::to_string(num_procs)+ ".out";
        petscvec_to_file(comm, my_fsp.get_P(), filename.c_str());
    }

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
