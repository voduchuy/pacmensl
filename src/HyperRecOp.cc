#include "HyperRecOp.hpp"
#include "cme_util.hpp"

using std::cout;
using std::endl;

namespace cme {
    namespace petsc {

        HyperRecOp::HyperRecOp(MPI_Comm &new_comm, const arma::Row<Int> &new_nmax, const arma::Mat<Int> &SM,
                               PropFun prop, TcoefFun new_t_fun) :
                comm(new_comm),
                n_reactions(SM.n_cols),
                t_fun(new_t_fun)
        {
            terms.resize(n_reactions + 1);
            GenerateMatrices(new_nmax, SM, prop);
        }

        void HyperRecOp::set_time(Real t_in) {
            t_here = t_in;
        }

        void HyperRecOp::duplicate_structure(Mat &A) {
            Int ierr;
            ierr = MatDuplicate(terms[n_reactions], MAT_DO_NOT_COPY_VALUES, &A);
            CHKERRABORT(comm, ierr);
        }

        void HyperRecOp::dump_to_mat(Mat A) {
            Int ierr;
            arma::Row<Real> coefficients = t_fun(t_here);
            for (Int ir{0}; ir < n_reactions; ++ir) {
                ierr = MatAXPY(A, coefficients[ir], terms[ir], SUBSET_NONZERO_PATTERN);
                CHKERRABORT(comm, ierr);
            }
        }

        void HyperRecOp::action(Vec x, Vec y) {
            Int ierr;

            arma::Row<Real> coefficients = t_fun(t_here);

            ierr = VecSet(y, 0.0);
            CHKERRABORT(comm, ierr);
            for (Int ir{0}; ir < n_reactions; ++ir) {
                ierr = MatMult(terms[ir], x, work);
                CHKERRABORT(comm, ierr);

                ierr = VecAXPY(y, coefficients[ir], work);
                CHKERRABORT(comm, ierr);
            }
        }

        void HyperRecOp::print_info() {
            PetscPrintf(comm, "This is an Op object.\n");
        }

        void HyperRecOp::generate_matrices(const arma::Row<Int> new_nmax, const arma::Mat<Int> &SM, PropFun prop) {
            max_num_molecules = new_nmax;
            n_rows_global = arma::prod(max_num_molecules + 1) + max_num_molecules.n_elem;
            PetscInt ierr;
            PetscReal val;

            MatType mat_type;
            Int comm_size;
            MPI_Comm_size(comm, &comm_size);
            if (comm_size == 1) {
                mat_type = MATSEQAIJ;
            } else {
                mat_type = MATMPIAIJ;
            }

            VecCreate(comm, &work);
            VecSetFromOptions(work);
            VecSetSizes(work, PETSC_DECIDE, n_rows_global);
            VecSetUp(work);

            MatCreate(comm, &terms[n_reactions]);

            /* Preallocate memory for matrix */
            if (strcmp(mat_type, MATSEQAIJ) == 0) {
#ifdef HYPER_REC_OP_VERBOSE
                PetscPrintf(comm, "Allocating memory for SEQAIJ.\n");
#endif
                MatCreateSeqAIJ(comm, n_rows_global, n_rows_global, n_reactions + 1, NULL, &terms[n_reactions]);
            } else if (strcmp(mat_type, MATMPIAIJ) == 0) {
#ifdef HYPER_REC_OP_VERBOSE
                PetscPrintf(comm, "Allocating memory for MPIAIJ.\n");
#endif
                MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, n_rows_global, n_rows_global, n_reactions + 1, NULL,
                             n_reactions + 1, NULL, &terms[n_reactions]);
            }

            ierr = MatSetOption(terms[n_reactions], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
            CHKERRABORT(comm, ierr);
            MatSetUp(terms[n_reactions]);

            PetscInt n_states = arma::prod(max_num_molecules+1);
            PetscInt Istart, Iend;
            // Get the indices of rows the current process owns, which will range from Istart to Iend-1
            ierr = MatGetOwnershipRange(terms[n_reactions], &Istart, &Iend);
            CHKERRABORT(comm, ierr);

            Iend = std::min(Iend, n_states);

            arma::Row<Int> my_range = arma::linspace<arma::Row<Int>>(Istart, Iend-1, Iend-Istart);

            arma::Mat<Int> my_X = cme::ind2sub_nd(max_num_molecules, my_range);

            arma::Col<Int> xtmp;
            for (Int ir{0}; ir < n_reactions; ++ir) {
                ierr = MatCreate(comm, &terms[ir]);
                CHKERRABORT(comm, ierr);

                /* Preallocate memory for matrix */
#ifdef HYPER_REC_OP_VERBOSE
                if (strcmp(mat_type, MATSEQAIJ) == 0) {
                    PetscPrintf(comm, "Allocating memory for SEQAIJ.\n");
                } else if (strcmp(mat_type, MATMPIAIJ) == 0) {
                    PetscPrintf(comm, "Allocating memory for MPIAIJ.\n");
                }
#endif

                if (strcmp(mat_type, MATSEQAIJ) == 0) {
                    MatCreateSeqAIJ(comm, n_rows_global, n_rows_global, 2, NULL, &terms[ir]);
                } else if (strcmp(mat_type, MATMPIAIJ) == 0) {
                    MatCreateAIJ(comm, PETSC_DECIDE, PETSC_DECIDE, n_rows_global, n_rows_global, 2, NULL, 2, NULL,
                                 &terms[ir]);
                }
                ierr = MatSetUp(terms[ir]);
                CHKERRABORT(comm, ierr);

                ierr = MatSetOption(terms[ir], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
                CHKERRABORT(comm, ierr);

                /* Set row values corresponding to the usual states, this will require no communication */
                /* Set values for diagonal entries */
                for (PetscInt i{0}; i < my_range.n_elem; ++i) {
                    xtmp = my_X.col(i);
                    val = prop(xtmp.begin(), ir);
                    ierr = MatSetValue(terms[ir], my_range(i), my_range(i), -1.0 * val, ADD_VALUES);
                    CHKERRABORT(comm, ierr);
                }

                /* Set values for off-diagonal entries */
                PetscInt nLocal = my_range.n_elem;
                arma::Mat<PetscInt> RX = my_X - repmat(SM.col(ir), 1, nLocal);
                arma::Row<PetscInt> rindx = cme::sub2ind_nd(max_num_molecules, RX);

                for (PetscInt i{0}; i < my_range.n_elem; ++i) {
                    xtmp = RX.col(i);
                    val = prop(xtmp.begin(), ir);

                    ierr = MatSetValue(terms[ir], my_range(i), rindx(i), val,
                                       ADD_VALUES);
                    CHKERRABORT(comm, ierr);
                    ierr = MatSetValue(terms[n_reactions], my_range(i), rindx(i), 0.0,
                                       INSERT_VALUES);
                    CHKERRABORT(comm, ierr);
                }

                /* Set values corresponding to the sink states, this will require communication */
                RX = my_X + repmat(SM.col(ir), 1, nLocal);
                rindx = cme::sub2ind_nd(max_num_molecules, RX);
                for (PetscInt i{0}; i < my_range.n_elem; ++i) {
                    xtmp = my_X.col(i);
                    val = prop(xtmp.begin(), ir);

                    if (rindx(i) < -1) {
                        rindx(i) = (n_states - 1) - (rindx(i)+1);
                        ierr = MatSetValue(terms[ir], rindx(i), my_range(i), val,
                                           ADD_VALUES);
                        CHKERRABORT(comm, ierr);
                        ierr = MatSetValue(terms[n_reactions], rindx(i), my_range(i), 0.0,
                                           INSERT_VALUES);
                        CHKERRABORT(comm, ierr);
                    }
                }

                ierr = MatAssemblyBegin(terms[ir], MAT_FINAL_ASSEMBLY);
                CHKERRABORT(comm, ierr);
            }
            for (Int ir{0}; ir < n_reactions; ++ir) {
                ierr = MatAssemblyEnd(terms[ir], MAT_FINAL_ASSEMBLY);
                CHKERRABORT(comm, ierr);
            }

            for (PetscInt i{0}; i < my_range.n_elem; ++i) {
                ierr = MatSetValue(terms[n_reactions], my_range(i), my_range(i), 0.0,
                                   INSERT_VALUES);// note that Petsc enters values by rows
                CHKERRABORT(comm, ierr);
            }
            ierr = MatAssemblyBegin(terms[n_reactions], MAT_FINAL_ASSEMBLY);
            CHKERRABORT(comm, ierr);
            ierr = MatAssemblyEnd(terms[n_reactions], MAT_FINAL_ASSEMBLY);
            CHKERRABORT(comm, ierr);
        }
    }
}
