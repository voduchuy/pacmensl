
#include <MatrixSet.h>

#include "MatrixSet.h"
#include "cme_util.h"

using std::cout;
using std::endl;

namespace cme {
    namespace petsc {

        MatrixSet::MatrixSet(MPI_Comm _comm) {
            MPI_Comm_dup(_comm, &comm);
        }

        void MatrixSet::DuplicateStructure(Mat &A) {
            Int ierr;
            ierr = MatDuplicate(terms[n_reactions], MAT_DO_NOT_COPY_VALUES, &A);
            CHKERRABORT(comm, ierr);
        }

        void MatrixSet::Action(PetscReal t, Vec x, Vec y) {
            Int ierr;

            arma::Row<Real> coefficients = t_fun(t);

            ierr = VecSet(y, 0.0);
            CHKERRABORT(comm, ierr);
            for (Int ir{0}; ir < n_reactions; ++ir) {
                ierr = MatMult(terms[ir], x, work);
                CHKERRABORT(comm, ierr);

                ierr = VecAXPY(y, coefficients[ir], work);
                CHKERRABORT(comm, ierr);
            }

        }

        void MatrixSet::PrintInfo() {
            PetscPrintf(comm, "This is an Op object.\n");
        }

        void MatrixSet::GenerateMatrices(FiniteStateSubset &fsp, const arma::Mat<Int> &SM, PropFun prop, TcoefFun new_t_fun) {
            PetscInt ierr;

            t_fun = new_t_fun;

            n_reactions = PetscInt(SM.n_cols);
            terms.resize(n_reactions);

            PetscInt n_local_states = fsp.GetNumLocalStates();

            fsp_size = fsp.GetFSPSize();//fsp_size;

            n_rows_local = n_local_states + fsp.GetNumSpecies();

            PetscMPIInt rank;
            MPI_Comm_rank(comm, &rank);

            // Generate matrix layout from FSP's layout
            ierr = VecCreate(comm, &work);
            CHKERRABORT(comm, ierr);
            ierr = VecSetFromOptions(work);
            CHKERRABORT(comm, ierr);
            ierr = VecSetSizes(work, n_rows_local, PETSC_DECIDE);
            CHKERRABORT(comm, ierr);
            ierr = VecSetUp(work);
            CHKERRABORT(comm, ierr);

            // Get the global and local numbers of rows
            VecGetSize(work, &n_rows_global);

            // Get local range of PETSC indices and local states

            arma::Mat<Int> my_X = fsp.GetLocalStates();
            arma::Row<Int> my_range((size_t) n_rows_local);
            ierr = VecGetOwnershipRange(work, &my_range[0], &my_range[n_rows_local - 1]);
            CHKERRABORT(comm, ierr);

            for (PetscInt i{0}; i < n_rows_local; ++i) {
                my_range[i] = my_range[0] + i;
            }

            arma::Mat<Int> Xtmp(fsp_size.n_elem, (size_t) fsp.GetNumLocalStates());
            arma::Row<PetscInt> rindx((size_t) fsp.GetNumLocalStates());
            rindx.fill(0);
            arma::Row<PetscInt> rindx_sinks((size_t) fsp.GetNumLocalStates());
            arma::Row<Int> d_nnz((size_t) n_rows_local);
            arma::Row<Int> o_nnz((size_t) n_rows_local);
            PetscReal val;
            for (Int ir{0}; ir < n_reactions; ++ir) {
                // Determine the number of nonzeros per row on the diagonal and off-diagonal portion of the matrix
                d_nnz.fill(1);
                o_nnz.fill(0);
                // nnz per row for sink states
                Xtmp = my_X + repmat(SM.col(ir), 1, fsp.GetNumLocalStates());
                rindx_sinks = sub2ind_nd(fsp_size, Xtmp);
                for (Int i{0}; i < fsp.GetNumLocalStates(); ++i) {
                    if (rindx_sinks(i) <
                        -1) {// the sub2ind return -1-d if the state exceed the bound on the d dimension
                        d_nnz((size_t) n_rows_local + (rindx_sinks(i) + 2) - 1) += 1;
                    }
                }
                // nnz per row for normal states
                Xtmp = my_X - repmat(SM.col(ir), 1, fsp.GetNumLocalStates());
                rindx = fsp.State2Petsc(Xtmp);

                for (Int i{0}; i < fsp.GetNumLocalStates(); ++i) {
                    if (rindx(i) >= my_range(0) && rindx(i) <= my_range(n_rows_local - 1)) {
                        d_nnz(i) += 1;
                    } else if (rindx(i) >= 0) {
                        o_nnz(i) += 1;
                    }
                }

                // Create matrix with preallocated memory
                ierr = MatCreateAIJ(comm, n_rows_local, n_rows_local,
                                    n_rows_global, n_rows_global, 0, &d_nnz[0], 0, &o_nnz[0],
                                    &terms[ir]);
                CHKERRABORT(comm, ierr);
                ierr = MatSetOption(terms[ir], MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
                CHKERRABORT(comm, ierr);
                ierr = MatSetUp(terms[ir]);
                CHKERRABORT(comm, ierr);

                /* Set row values corresponding to the usual local_states*/
                /* Set values for diagonal entries */
                for (size_t i{0}; i < n_local_states; ++i) {
                    val = prop(my_X.colptr(i), ir);
                    ierr = MatSetValue(terms[ir], my_range(i), my_range(i), -1.0 * val, ADD_VALUES);
                    CHKERRABORT(comm, ierr);
                }

                /* Set values for off-diagonal entries */
                for (size_t i{0}; i < n_local_states; ++i) {
                    val = prop(Xtmp.colptr(i), ir);
                    ierr = MatSetValue(terms[ir], my_range(i), rindx(i), val,
                                       ADD_VALUES);
                    CHKERRABORT(comm, ierr);
                }

                /* Set values corresponding to the sink local_states */
                for (PetscInt i{0}; i < n_local_states; ++i) {
                    if (rindx_sinks(i) < -1) {
                        val = prop(my_X.colptr(i), ir);
                        ierr = MatSetValue(terms[ir], my_range((size_t) n_rows_local - 1) + (rindx_sinks(i) + 2), my_range(i),
                                           val,
                                           ADD_VALUES);
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
        }

        MatrixSet::~MatrixSet() {
            MPI_Comm_free(&comm);
            Destroy();
        }

        void MatrixSet::Destroy() {
            for (PetscInt i{0}; i < n_reactions; ++i) {
                if (terms[i]) {
                    MatDestroy(&terms[i]);
                }
            }
            if (work) VecDestroy(&work);
        }
    }
}
