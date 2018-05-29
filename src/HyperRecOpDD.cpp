#include "HyperRecOpDD.h"
#include "cme_util.h"

using std::cout;
using std::endl;

namespace cme {
namespace petsc {
void HyperRecOpDD::get_ordering(const arma::Row<Int> &nmax, const arma::Row<Int> &processor_grid, const std::vector<arma::Row<Int> > &sub_domain_dims)
{
        MPI_Comm cart_comm;

        size_t d = nmax.n_elem;
        size_t local_num_states;
        int rank;
        Int ierr;
        arma::Row<int> periods(d);
        arma::Row<Int> proc_coo(d);
        arma::Col<Int> x_offset(d);

        periods.fill(0);
        MPI_Cart_create(comm, d, processor_grid.memptr(), periods.memptr(), 1, &cart_comm);
        MPI_Comm_rank(cart_comm, &rank);
        MPI_Cart_coords(cart_comm, rank, d, proc_coo.memptr());

        /* Calculate the state offset, which is at the lowest corner of the sub-domain */
        for (size_t k{0}; k < d; ++k )
        {
          x_offset(k) = 0;
          for (size_t i{0}; i+1 <= proc_coo(k); ++i)
          {
            x_offset(k) += (sub_domain_dims[k]).at(i);
          }
        }

        /* Generate the state space */
        local_num_states = 1;
        arma::Row<Int> nmax_local(d);
        for (size_t k{0}; k < d; ++k)
        {
          local_num_states *= sub_domain_dims[k](proc_coo(k));
          nmax_local[k] = sub_domain_dims[k](proc_coo(k)) - 1;
        }
        arma::Row<Int> indx(local_num_states);
        for (size_t i{0}; i < local_num_states; ++i)
        {
          indx(i) = i;
        }
        local_state_space = cme::ind2sub_nd(nmax_local, indx);
        local_state_space = local_state_space + arma::repmat(x_offset, 1, local_num_states);

        /* Now generate the mapping from states to PETSc */
        indx = cme::sub2ind_nd(nmax, local_state_space);
        AOCreate(comm, &ao);
        ierr = AOCreateBasic(comm, local_num_states, indx.memptr(), NULL, &ao);
        CHKERRABORT(comm, ierr);

        n_rows_here = local_num_states;

        MPI_Comm_free(&cart_comm);

}

HyperRecOpDD::HyperRecOpDD( MPI_Comm& new_comm, const arma::Row<Int> &new_nmax, const arma::Row<Int> &processor_grid, const std::vector<arma::Row<Int> > sub_domain_dims, const arma::Mat<Int> &SM, PropFun prop, TcoefFun new_t_fun) :
        comm(new_comm),
        max_num_molecules(new_nmax),
        n_reactions(SM.n_cols),
        n_rows_global(arma::prod( max_num_molecules +1 )),
        t_fun(new_t_fun)
{
        PetscInt ierr;
        PetscReal val;

        /* Get the ordering and local state space */
        get_ordering(max_num_molecules, processor_grid, sub_domain_dims);

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
        VecSetSizes(work, n_rows_here, n_rows_global);
        VecSetUp(work);

        MatCreate(comm, &terms[n_reactions]);

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
                MatCreateAIJ(comm, n_rows_here, n_rows_here, n_rows_global, n_rows_global, n_reactions+1, NULL, n_reactions+1, NULL, &terms[n_reactions]);
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
                        MatCreateAIJ(comm, n_rows_here, n_rows_here, n_rows_global, n_rows_global, 2, NULL, 2, NULL, &terms[ir]);
                }
                ierr = MatSetUp(terms[ir]); CHKERRABORT(comm, ierr);

                /* Set values for diagonal entries */
                for ( PetscInt i{0}; i < my_range.n_elem; ++i )
                {
                        xtmp = local_state_space.col(i);
                        val = prop(xtmp.begin(), ir);
                        ierr = MatSetValue( terms[ir], my_range(i), my_range(i), -1.0*val, INSERT_VALUES );
                        CHKERRABORT(comm, ierr);
                }

                /* Set values for off-diagonal entries */
                PetscInt nLocal= my_range.n_elem;
                arma::Mat<PetscInt> RX= local_state_space - repmat( SM.col(ir), 1, nLocal);
                arma::Row<PetscInt> rindx= cme::sub2ind_nd( max_num_molecules, RX );

                AOApplicationToPetsc(ao, nLocal, rindx.memptr());

                for ( PetscInt i{0}; i < my_range.n_elem; ++i )
                {
                        xtmp = RX.col(i);
                        val = prop(xtmp.begin(), ir);
                        ierr = MatSetValue( terms[ir], my_range(i), rindx(i), val, INSERT_VALUES );        // note that Petsc enters values by rows
                        CHKERRABORT(comm, ierr);
                        ierr = MatSetValue( terms[n_reactions], my_range(i), rindx(i), 0.0, INSERT_VALUES );        // note that Petsc enters values by rows
                        CHKERRABORT(comm, ierr);
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

void HyperRecOpDD::set_time(Real t_in)
{
        t_here = t_in;
}

void HyperRecOpDD::duplicate_structure(Mat &A)
{
        Int ierr;
        ierr = MatDuplicate(terms[n_reactions], MAT_DO_NOT_COPY_VALUES, &A); CHKERRABORT(comm, ierr);
}

void HyperRecOpDD::dump_to_mat(Mat A)
{
        Int ierr;
        arma::Row<Real> coefficients = t_fun(t_here);
        for (Int ir{0}; ir < n_reactions; ++ir)
        {
                ierr = MatAXPY(A, coefficients[ir], terms[ir], SUBSET_NONZERO_PATTERN); CHKERRABORT(comm, ierr);
        }
}

void HyperRecOpDD::action(Vec x, Vec y)
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

void HyperRecOpDD::print_info()
{
        PetscPrintf(comm, "This is an Op object.\n");
}
}
}
