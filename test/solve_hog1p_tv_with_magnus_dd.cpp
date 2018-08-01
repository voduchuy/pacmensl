static char help[] = "Solve the 5-species spatial hog1p model with time-varying propensities.\n\n";

#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <petscts.h>
#include <armadillo>
#include <cmath>
#include "HyperRecOpDD.h"
#include "Magnus4.h"
#include "models/hog1p_tv_model.h"

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

void petscvec_to_file(MPI_Comm comm, Vec x, const char* filename);
void petscmat_to_file(MPI_Comm comm, Mat A, const char* filename);

int main(int argc, char *argv[]) {
        int ierr, myRank, num_procs;
        double solver_time;

        std::string model_name = "hog1p_damped";

        using namespace hog1p_cme;

        /* CME problem sizes */
#ifdef TEST_LARGE_PROBLEM
        Row<PetscInt> FSPSize({ 3, 40, 40, 60, 60}); // Size of the FSP
        PetscReal t_final = 120;
#else
        Row<PetscInt> FSPSize({ 3, 31, 31, 7, 7}); // Size of the FSP
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

        std::vector<arma::Row<PetscInt> > sub_domain_dims(5);
        for (size_t i{0}; i < 5; ++i)
        {
                sub_domain_dims[i] = cme::distribute_tasks(FSPSize(i)+1, processor_grid(i));
        }

        cme::petsc::HyperRecOpDD A(comm, FSPSize, processor_grid, sub_domain_dims, SM, propensity, t_fun);

        /* Test the action of the operator */
        Vec P0, P;
        VecCreate(comm, &P0);
        VecCreate(comm, &P);

        PetscInt i_0 = 0;
        AOApplicationToPetsc(A.ao, 1, &i_0);
        VecSetSizes(P0, A.n_rows_here, A.n_rows_global);
        VecSetFromOptions(P0);
        VecSet(P0, 0.0);
        VecSetValue(P0, i_0, 1.0, INSERT_VALUES);
        VecAssemblyBegin(P0);
        VecAssemblyEnd(P0);

        VecDuplicate(P0, &P); VecCopy(P0, P);

        auto tmatvec = [&A] (PetscReal t, Vec x, Vec y) {
                               A(t).action(x, y);
                       };
        cme::petsc::Magnus4 my_magnus(comm, t_final, tmatvec, P, 1.0e-8, 30, true, 2, 1.0e-8);
        my_magnus.tol = 1.0e-8;

        tic = MPI_Wtime();
        my_magnus.solve();
        solver_time = MPI_Wtime() - tic;

        /* Compute the marginal distributions */
        std::vector<arma::Col<PetscReal> > marginals(FSPSize.n_elem);
        for (PetscInt i{0}; i < marginals.size(); ++i )
        {
                marginals[i] = cme::petsc::marginal(P, FSPSize, i, A.ao);
        }

        MPI_Comm_rank(comm, &myRank);
        if (myRank == 0)
        {
                {
                        std::string filename = model_name + "_time_magnus_dd_" + std::to_string(num_procs) + ".dat";
                        std::ofstream file;
                        file.open(filename);
                        file << solver_time;
                        file.close();
                }
                for (PetscInt i{0}; i < marginals.size(); ++i )
                {
                        std::string filename = model_name + "_marginal_magnus_dd_" + std::to_string(i) + "_"+ std::to_string(num_procs)+ ".dat";
                        marginals[i].save(filename, arma::raw_ascii);
                }
        }

        if (export_full_solution)
        {
          std::string filename = model_name + "_" + "magnus_dd" + "_full_" + std::to_string(num_procs)+ ".out";
          petscvec_to_file(comm, P, filename.c_str());
        }

        my_magnus.destroy();
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
