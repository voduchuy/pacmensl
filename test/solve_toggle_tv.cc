static char help[] = "Formation of the two-species toggle-switch matrix.\n\n";

#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <armadillo>
#include <cmath>
#include "HyperRecOp.hpp"
#include "HyperRecOpDD.hpp"
#include "Magnus4.hpp"

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

/* Stoichiometric matrix of the toggle switch model */
arma::Mat<int> SM { { 1, 1, -1, 0, 0,  0 },
                    { 0, 0,  0, 1, 1, -1 } };

const int nReaction= 6;

/* Parameters for the propensity functions */
const double ayx {6.1e-3}, axy {2.6e-3}, nyx {2.1e0}, nxy {3.0e0},
kx0 {6.8e-5}, kx {1.6e-2}, dx {0.00067}, ky0 {2.2e-3}, ky {1.7e-2}, dy {3.8e-4};


PetscReal propensity( PetscInt *X, PetscInt k);

arma::Row<PetscReal> t_fun(PetscReal t)
{
  //return {kx0, kx, dx, ky0, ky, dy};
  return {(1.0 + std::cos(t))*kx0, kx, dx, (1.0 + std::sin(t))*ky0, ky, dy};
}

void petscvec_to_file(MPI_Comm comm, Vec x, const char* filename);

int main(int argc, char *argv[]) {
        int ierr, myRank;

        /* CME problem sizes */
        int nSpecies = 2; // Number of species
        Row<PetscInt> FSPSize({ 200, 200 }); // Size of the FSP
        PetscReal t_final = 1000.0;

        ierr = PetscInitialize(&argc,&argv,(char*)0,help); CHKERRQ(ierr);

        MPI_Comm comm{PETSC_COMM_WORLD};
        int comm_size;
        MPI_Comm_size(comm, &comm_size);


        /* Test the action of the operator */
        cme::petsc::HyperRecOp A( comm, FSPSize, SM, propensity, t_fun);
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

        auto tmatvec = [&A] (PetscReal t, Vec x, Vec y) {A(t).action(x, y); };
        cme::petsc::Magnus4 my_magnus(comm, t_final, tmatvec, P);

        double t1 = MPI_Wtime();
        my_magnus.solve();
        double t2 = MPI_Wtime() - t1;
        PetscPrintf(comm, "Solver time %2.4f \n", t2);

        // petscvec_to_file(comm, P, "toggle_tv.out");

        my_magnus.destroy();
        A.destroy();
        ierr = VecDestroy(&P); CHKERRQ(ierr);
        ierr = VecDestroy(&P0); CHKERRQ(ierr);

        /* Now do the same thing with domain-decomp ordering */
        arma::Row<PetscInt> processor_grid(2);
        MPI_Dims_create(comm_size, 2, processor_grid.memptr());

        std::vector<arma::Row<PetscInt>> sub_domain_dims(2);
        for (size_t i{0}; i < 2; ++i)
        {
          sub_domain_dims[i] = cme::distribute_tasks(FSPSize(i)+1, processor_grid(i));
        }

        cme::petsc::HyperRecOpDD A_dd(comm, FSPSize, processor_grid, sub_domain_dims, SM, propensity, t_fun);

        VecCreate(comm, &P0);
        VecCreate(comm, &P);
        VecSetSizes(P0, A_dd.n_rows_here, A_dd.n_rows_global);
        VecSetFromOptions(P0);

        PetscInt i0 {0};
        AOApplicationToPetsc(A_dd.ao, 1, &i0);

        VecSetValue(P0, i0, 1.0, INSERT_VALUES);
        VecAssemblyBegin(P0);
        VecAssemblyEnd(P0);
        VecDuplicate(P0, &P);
        VecCopy(P0, P);

        auto tmatvec2 = [&A_dd] (PetscReal t, Vec x, Vec y) {A_dd(t).action(x, y); };
        cme::petsc::Magnus4 my_magnus2(comm, t_final, tmatvec2, P);

        t1 = MPI_Wtime();
        my_magnus2.solve();
        t2 = MPI_Wtime() - t1;
        PetscPrintf(comm, "Solver time %2.4f \n", t2);

        my_magnus2.destroy();
        A_dd.destroy();
        ierr = VecDestroy(&P); CHKERRQ(ierr);
        ierr = VecDestroy(&P0); CHKERRQ(ierr);
        ierr = PetscFinalize();
        return ierr;
}

// propensity function for toggle
PetscReal propensity( PetscInt *X, PetscInt k)
{
        switch (k) {
        case 0:
                return 1.0;
        case 1:
                return 1.0/( 1.0 + ayx*pow( PetscReal( X[1]), nyx ) );
        case 2:
                return PetscReal( X[0] );
        case 3:
                return 1.0;
        case 4:
                return 1.0/( 1.0 + axy*pow( PetscReal( X[0]), nxy) );
        case 5:
                return PetscReal(X[1]);
        }
        return 0.0;
}


void petscvec_to_file(MPI_Comm comm, Vec x, const char* filename)
{
  PetscViewer viewer;
  PetscViewerCreate(comm, &viewer);
  PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &viewer);
  VecView(x, viewer);
  PetscViewerDestroy(&viewer);
}
