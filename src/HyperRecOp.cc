#include "HyperRecOp.hpp"
#include "cme_util.hpp"

using std::cout;
using std::endl;

namespace cme {
namespace petsc {

HyperRecOp::HyperRecOp ( MPI_Comm& new_comm, const arma::Row<Int> &new_nmax, const arma::Mat<Int> &SM, PropFun prop, TcoefFun new_t_fun) :
        comm(new_comm),
        max_num_molecules(new_nmax),
        n_reactions(SM.n_cols),
        n_rows_global(arma::prod( max_num_molecules +1 )),
        t_fun(new_t_fun)
{
        PetscInt ierr;
        PetscReal val;

        terms.resize(n_reactions+1);

        MatType mat_type;
        Int comm_size;
        MPI_Comm_size(comm, &comm_size);
        if (comm_size == 1)
        {
          mat_type = MATSEQAIJ;
        }
        else
        {
          mat_type = MATMPIAIJ;
        }

        VecCreate(comm, &work);
        VecSetFromOptions(work);
        VecSetSizes(work, PETSC_DECIDE, n_rows_global);
        VecSetUp(work);

        MatCreate(comm, &terms[n_reactions]);
        //MatSetSizes(terms[n_reactions], PETSC_DECIDE, PETSC_DECIDE, n_rows_global, n_rows_global);
        //MatSetFromOptions(terms[n_reactions]);
        /* Preallocate memory for matrix */
        if ( strcmp(mat_type, MATSEQAIJ) == 0)
        {
#ifdef HYPER_REC_OP_VERBOSE
          PetscPrintf(comm, "Allocating memory for SEQAIJ.\n");
#endif
          MatCreateSeqAIJ(comm, n_rows_global, n_rows_global, n_reactions+1, NULL, &terms[n_reactions]);
        }
        else if (strcmp(mat_type, MATMPIAIJ) == 0)
        {
#ifdef HYPER_REC_OP_VERBOSE
          PetscPrintf(comm, "Allocating memory for MPIAIJ.\n");
#endif
          MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, n_rows_global, n_rows_global, n_reactions+1, NULL, n_reactions+1, NULL, &terms[n_reactions]);
        }

        MatSetUp(terms[n_reactions]);
        // Get the indices of rows the current process owns, which will range from Istart to Iend-1
        PetscInt Istart, Iend;
        ierr = MatGetOwnershipRange( terms[n_reactions], &Istart, &Iend ); CHKERRABORT(comm, ierr);

        arma::Row<Int> my_range(Iend-Istart);
        for ( PetscInt vi {0}; vi < my_range.n_elem; ++vi )
        {
                my_range[vi]= Istart + vi;
        }

        for (PetscInt i{0}; i < my_range.n_elem; ++i)
        {
                ierr = MatSetValue( terms[n_reactions], my_range(i), my_range(i), 0.0, INSERT_VALUES );// note that Petsc enters values by rows
                CHKERRABORT(comm, ierr);
        }

        arma::Mat<Int> my_X= cme::ind2sub_nd( max_num_molecules, my_range);
        arma::Col<Int> xtmp;
        for ( Int ir{0}; ir < n_reactions; ++ir )
        {
                ierr = MatCreate(comm, &terms[ir]); CHKERRABORT(comm, ierr);
                //ierr = MatSetSizes(terms[ir], PETSC_DECIDE, PETSC_DECIDE, n_rows_global, n_rows_global); CHKERRABORT(comm, ierr);
                //ierr = MatSetFromOptions(terms[ir]); CHKERRABORT(comm, ierr);
                /* Preallocate memory for matrix */
                if ( strcmp(mat_type, MATSEQAIJ) == 0)
                {
#ifdef HYPER_REC_OP_VERBOSE
                  PetscPrintf(comm, "Allocating memory for SEQAIJ.\n");
#endif
                  MatCreateSeqAIJ(comm, n_rows_global, n_rows_global, 2, NULL, &terms[ir]);
                }
                else if (strcmp(mat_type, MATMPIAIJ) == 0)
                {
#ifdef HYPER_REC_OP_VERBOSE
                  PetscPrintf(comm, "Allocating memory for MPIAIJ.\n");
#endif                  
                  MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, n_rows_global, n_rows_global, 2, NULL, 2, NULL, &terms[ir]);
                }
                ierr = MatSetUp(terms[ir]); CHKERRABORT(comm, ierr);

                /* Set values for diagonal entries */
                for ( PetscInt i{0}; i < my_range.n_elem; ++i )
                {
                        xtmp = my_X.col(i);
                        val = prop(xtmp.begin(), ir);
                        ierr = MatSetValue( terms[ir], my_range(i), my_range(i), -1.0*val, INSERT_VALUES );
                        CHKERRABORT(comm, ierr);
                }

                /* Set values for off-diagonal entries */
                PetscInt nLocal= my_range.n_elem;
                arma::Mat<PetscInt> RX= my_X - repmat( SM.col(ir), 1, nLocal);
                arma::Row<PetscInt> rindx= cme::sub2ind_nd( max_num_molecules, RX );

                for ( PetscInt i{0}; i < my_range.n_elem; ++i )
                {
                        if (rindx(i) >= 0) {
                                xtmp = RX.col(i);
                                val = prop(xtmp.begin(), ir);
                                ierr = MatSetValue( terms[ir], my_range(i), rindx(i), val, INSERT_VALUES );// note that Petsc enters values by rows
                                CHKERRABORT(comm, ierr);
                                ierr = MatSetValue( terms[n_reactions], my_range(i), rindx(i), 0.0, INSERT_VALUES );// note that Petsc enters values by rows
                                CHKERRABORT(comm, ierr);
                        }
                }

                ierr = MatAssemblyBegin( terms[ir], MAT_FINAL_ASSEMBLY ); CHKERRABORT(comm, ierr);
        }
        for (Int ir{0}; ir < n_reactions; ++ir)
        {
                ierr = MatAssemblyEnd( terms[ir], MAT_FINAL_ASSEMBLY ); CHKERRABORT(comm, ierr);
        }
        ierr = MatAssemblyBegin( terms[n_reactions], MAT_FINAL_ASSEMBLY ); CHKERRABORT(comm, ierr);
        ierr = MatAssemblyEnd( terms[n_reactions], MAT_FINAL_ASSEMBLY ); CHKERRABORT(comm, ierr);
}

void HyperRecOp::set_time(Real t_in)
{
        t_here = t_in;
}

void HyperRecOp::duplicate_structure(Mat &A)
{
        Int ierr;
        ierr = MatDuplicate(terms[n_reactions], MAT_DO_NOT_COPY_VALUES, &A); CHKERRABORT(comm, ierr);
}

void HyperRecOp::dump_to_mat(Mat A)
{
        Int ierr;
        arma::Row<Real> coefficients = t_fun(t_here);
        for (Int ir{0}; ir < n_reactions; ++ir)
        {
                ierr = MatAXPY(A, coefficients[ir], terms[ir], SUBSET_NONZERO_PATTERN); CHKERRABORT(comm, ierr);
        }
}

void HyperRecOp::action(Vec x, Vec y)
{
        Int ierr;

        arma::Row<Real> coefficients = t_fun(t_here);

        ierr = VecSet(y, 0.0); CHKERRABORT(comm, ierr);
        for (Int ir{0}; ir < n_reactions; ++ir)
        {
                ierr = MatMult(terms[ir], x, work); CHKERRABORT(comm, ierr);

                ierr = VecAXPY(y, coefficients[ir], work); CHKERRABORT(comm, ierr);
        }
}

void HyperRecOp::print_info()
{
        PetscPrintf(comm, "This is an Op object.\n");
}
}
}
