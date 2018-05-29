static char help[] = "Formation of the two-species toggle-switch matrix.\n\n";

#include <petscmat.h>
#include <petscviewer.h>
#include <petscao.h>
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
        Row<PetscInt> FSPSize({ 600, 600 }); // Size of the FSP

        ierr = PetscInitialize(&argc,&argv,(char*)0,help); if (ierr) return ierr;

        MPI_Comm comm{PETSC_COMM_WORLD};
        ierr = MPI_Comm_rank( comm, &myRank );

        PetscInt nGlobal= arma::prod( FSPSize+1 );
        cout << nGlobal << endl;
        /*---------------------------------------------------------------------
            Create mat object
         */
        Mat Aadj;
        MatCreate( PETSC_COMM_WORLD, &Aadj );
        ierr = MatSetSizes( Aadj, PETSC_DECIDE, PETSC_DECIDE, nGlobal, nGlobal );
        ierr = MatSetType( Aadj, MATMPIAIJ );
        ierr = MatSetUp( Aadj );

        /*---------------------------------------------------------------------
           Setting values for Adj */

        // Get the indices of rows the current process owns, which will range from Istart to Iend-1
        PetscInt Istart, Iend;
        ierr = MatGetOwnershipRange( Aadj, &Istart, &Iend ); CHKERRQ(ierr);

        arma::Row<PetscInt> myRange(Iend-Istart);
        for ( PetscInt vi {0}; vi < myRange.n_elem; ++vi ) {
                myRange[vi]= Istart + vi;
        }

        arma::Mat<PetscInt> X= cme::ind2sub_nd( FSPSize, myRange );
        arma::dmat propVals= propensity( X );
        propVals.fill(1.0);
        PetscInt nLocal= myRange.n_elem;

        for ( size_t ir{0}; ir < nReaction; ++ir )
        {
                arma::Mat<PetscInt> RX= X + repmat( SM.col(ir), 1, nLocal);

                // locations of off-diagonal elements, out-of-bound locations are set to -1
                Row<PetscInt> rindx= cme::sub2ind_nd( FSPSize, RX );

                // off-diagonal elements
                Col<PetscScalar> prop_keep {propVals.col(ir)};

                Row<PetscInt> J1= arma::join_horiz( myRange, join_horiz( myRange, rindx ));  // column indices to enter
                Row<PetscInt> I1= arma::join_horiz( myRange, join_horiz(rindx, myRange ));  // row indices to enter

                Col<PetscScalar> vals(I1.n_elem);
                vals = join_vert( propVals.col(ir), join_vert(prop_keep, prop_keep ));


                for ( PetscInt i{0}; i < I1.n_elem; ++i )
                {
                        if ( I1(i) >= 0 && J1(i) >= 0 ) {
                                ierr = MatSetValue( Aadj, I1(i), J1(i), vals(i), INSERT_VALUES ); CHKERRQ(ierr); // note that Petsc enters values by rows
                        }
                }

        }
        ierr = MatAssemblyBegin( Aadj, MAT_FINAL_ASSEMBLY ); CHKERRQ(ierr);
        ierr = MatAssemblyEnd( Aadj, MAT_FINAL_ASSEMBLY ); CHKERRQ(ierr);
        /*---------------------------------------------------------------------
           Partitioning */
        double t1 = MPI_Wtime();

        MatPartitioning part;
        IS is;

        ierr = MatPartitioningCreate(PETSC_COMM_WORLD, &part); CHKERRQ(ierr);
        ierr = MatPartitioningSetAdjacency(part, Aadj); CHKERRQ(ierr);
        ierr = MatPartitioningSetFromOptions(part); CHKERRQ(ierr);

        ierr = MatPartitioningApply(part, &is); CHKERRQ(ierr);
        // ierr = ISView(is, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);

        ierr = PetscPrintf(PETSC_COMM_WORLD,"Partitioning time %3.2f \n",MPI_Wtime()-t1); CHKERRQ(ierr);
        /*---------------------------------------------------------------------
           Form the re-ordered matrix */
        IS isg;
        ierr = ISPartitioningToNumbering(is, &isg); CHKERRQ(ierr);
        AO ao;
        ierr = AOCreateBasicIS(isg, NULL, &ao);

        Mat A;
        ierr = MatCreate(PETSC_COMM_WORLD, &A);
        ierr = MatSetSizes( A, PETSC_DECIDE, PETSC_DECIDE, nGlobal, nGlobal ); CHKERRQ(ierr);
        ierr = MatSetType( A, MATMPIAIJ ); CHKERRQ(ierr);
        ierr = MatSetUp( A ); CHKERRQ(ierr);

        ierr = MatGetOwnershipRange( A, &Istart, &Iend ); CHKERRQ(ierr);

        myRange.resize(Iend-Istart);
        for ( PetscInt vi {0}; vi < myRange.n_elem; ++vi ) {
                myRange[vi]= Istart + vi;
        }

        AOPetscToApplication(ao, myRange.n_elem, myRange.begin());

        X= cme::ind2sub_nd( FSPSize, myRange );
        propVals= propensity( X );
        nLocal= myRange.n_elem;

        for ( size_t ir{0}; ir < nReaction; ++ir )
        {
                arma::Mat<PetscInt> RX= X + repmat( SM.col(ir), 1, nLocal);

                // locations of off-diagonal elements, out-of-bound locations are set to -1
                Row<PetscInt> rindx= cme::sub2ind_nd( FSPSize, RX );

                // off-diagonal elements
                Col<PetscScalar> prop_keep {propVals.col(ir)};

                Row<PetscInt> J1= arma::join_horiz( myRange, myRange );   // column indices to enter
                Row<PetscInt> I1= arma::join_horiz( myRange, rindx );   // row indices to enter

                Col<PetscScalar> vals(I1.n_elem);
                vals= join_vert( -1.0*propVals.col(ir), prop_keep );

                AOApplicationToPetsc(ao, I1.n_elem, I1.begin());
                AOApplicationToPetsc(ao, J1.n_elem, J1.begin());

                for ( PetscInt i{0}; i < I1.n_elem; ++i )
                {
                        if ( I1(i) >= 0 ) {
                                ierr = MatSetValue( A, I1(i), J1(i), vals(i), ADD_VALUES ); CHKERRQ(ierr); // note that Petsc enters values by rows
                        }
                }

        }
        ierr = MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY ); CHKERRQ(ierr);
        ierr = MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY ); CHKERRQ(ierr);
        /*---------------------------------------------------------------------
           Form the matrix in intial ordering */

        Mat A_old_order;
        ierr = MatCreate(PETSC_COMM_WORLD, &A_old_order);
        ierr = MatSetSizes( A_old_order, PETSC_DECIDE, PETSC_DECIDE, nGlobal, nGlobal ); CHKERRQ(ierr);
        ierr = MatSetType( A_old_order, MATMPIAIJ ); CHKERRQ(ierr);
        ierr = MatSetUp( A_old_order ); CHKERRQ(ierr);

        FSPMatSetValues(PETSC_COMM_WORLD, A_old_order, FSPSize, SM, propensity);
        /*---------------------------------------------------------------------
           Test which ordering is better for 10,000 mat-vecs
         */

        /* Create and assemble intial vector */
        Vec x, y;
        VecCreate(comm, &x);
        VecSetSizes(x, PETSC_DECIDE, nGlobal);
        VecSetType(x, VECMPI);
        VecSet(x, 1.0);
        VecAssemblyBegin(x);
        VecAssemblyEnd(x);

        VecDuplicate(x, &y);

        t1 = MPI_Wtime();
        for (size_t imv{0}; imv< 1000; ++imv)
        {
                MatMult(A, x, y);
        }

        ierr = PetscPrintf(PETSC_COMM_WORLD,"Mat-vec with re-ordered matrix %3.2f \n",MPI_Wtime()-t1); CHKERRQ(ierr);

        t1 = MPI_Wtime();
        for (size_t imv{0}; imv< 1000; ++imv)
        {
                MatMult(A_old_order, x, y);
        }
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Mat-vec with natural matrix %3.2f \n",MPI_Wtime()-t1); CHKERRQ(ierr);
        /*---------------------------------------------------------------------
           Export the matrix to binary file */
        PetscViewer viewer;
        ierr=  PetscViewerCreate( PETSC_COMM_WORLD, &viewer );
        ierr= PetscViewerBinaryOpen( PETSC_COMM_WORLD, "Adj_initial.out", FILE_MODE_WRITE, &viewer );
        ierr= MatView( Aadj, viewer ); CHKERRQ(ierr);
        ierr=  PetscViewerCreate( PETSC_COMM_WORLD, &viewer );
        ierr= PetscViewerBinaryOpen( PETSC_COMM_WORLD, "A_reordered.out", FILE_MODE_WRITE, &viewer ); CHKERRQ(ierr);
        ierr= MatView( A, viewer ); CHKERRQ(ierr);

        PetscViewerDestroy(&viewer);
        ierr = MatPartitioningDestroy( &part); CHKERRQ(ierr);
        ierr = MatDestroy(&A_old_order); CHKERRQ(ierr);
        ierr = MatDestroy( &Aadj ); CHKERRQ(ierr);
        ierr = AODestroy(&ao); CHKERRQ(ierr);
        ierr = ISDestroy(&isg); CHKERRQ(ierr);
        ierr = ISDestroy(&is); CHKERRQ(ierr);

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
