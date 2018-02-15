#include "fsp_matrix.hpp"

using std::cout;
using std::endl;

void FSPMatSetValues(MPI_Comm comm, Mat &A, arma::Row<PetscInt> &FSPSize, arma::Mat<int> SM,  PropFun propensity ){

        PetscInt ierr;
        // int myRank;
        // MPI_Comm_rank(comm, &myRank);

        PetscInt nGlobal= arma::prod( FSPSize+1 );

        /* Setting values for A */

        // Get the indices of rows the current process owns, which will range from Istart to Iend-1
        PetscInt Istart, Iend;
        ierr = MatGetOwnershipRange( A, &Istart, &Iend );

        arma::Row<PetscInt> myRange(Iend-Istart);
        for ( PetscInt vi {0}; vi < myRange.n_elem; ++vi ) {
                myRange[vi]= Istart + vi;
        }

        arma::Mat<PetscInt> X= cme::ind2sub_nd( FSPSize, myRange );
        arma::dmat propVals= propensity( X );

        PetscInt nLocal= myRange.n_elem;

        int nReaction = SM.n_cols;

        for ( size_t ir{0}; ir < nReaction; ++ir )
        {
                arma::Mat<PetscInt> RX= X + repmat( SM.col(ir), 1, nLocal);

                // locations of off-diagonal elements, out-of-bound locations are set to -1
                arma::Row<PetscInt> rindx= cme::sub2ind_nd( FSPSize, RX );

                // off-diagonal elements
                arma::Col<PetscScalar> prop_keep {propVals.col(ir)};

                arma::Row<PetscInt> J1= arma::join_horiz( myRange, myRange ); // column indices to enter
                arma::Row<PetscInt> I1= arma::join_horiz( myRange, rindx ); // row indices to enter

                arma::Col<PetscScalar> vals= join_vert( -1.0*propVals.col(ir), prop_keep );

                //PetscInt nInsert= I1.n_elem;


                // cout << "Calling MatSetValues ... " << endl
                //     << "I1.size= " << nInsert << endl;

                for ( PetscInt i{0}; i < I1.n_elem; ++i )
                {
                        if (I1(i) >= 0 )
                        {
                                ierr = MatSetValues( A, 1, &I1(i), 1, &J1(i), &vals(i), ADD_VALUES );// note that Petsc enters values by rows
                        }
                }

                // cout << " Process " << myRank << " done enter values for reaction " << ir << endl;
        }

        MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY );
        MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY );
}
