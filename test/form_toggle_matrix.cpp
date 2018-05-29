static char help[] = "Formation of the two-species toggle-switch matrix.\n\n";

#include <petscmat.h>
#include <petscviewer.h>
#include <armadillo>
#include "cme_util.h"
#include "fsp_matrix.h"

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


arma::Mat<double> propensity( arma::Mat<PetscInt> X);

int main(int argc, char *argv[]) {
        int ierr, myRank;


        /* CME problem sizes */
        int nSpecies = 2; // Number of species
        Row<PetscInt> FSPSize({ 50, 80 }); // Size of the FSP

        ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;

        MPI_Comm comm{PETSC_COMM_WORLD};
        ierr = MPI_Comm_rank( comm, &myRank );

        PetscInt nGlobal= arma::prod( FSPSize+1 );
        cout << nGlobal << endl;
        /* Create and assemble the matrix */
        Mat A;
        MatCreate( PETSC_COMM_WORLD, &A );
        ierr = MatSetSizes( A, PETSC_DECIDE, PETSC_DECIDE, nGlobal, nGlobal );
        ierr = MatSetType( A, MATMPIAIJ );
        ierr = MatSetUp( A );

        FSPMatSetValues(comm, A, FSPSize, SM, propensity );

        PetscReal anorm;
        MatNorm( A, NORM_FROBENIUS, &anorm ) ;
        std::cout << "Matrix generated. norm(A) = " << anorm << "\n";

        MatDestroy( &A ); CHKERRQ(ierr);

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
