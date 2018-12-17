
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

        void MatrixSet::Action(PetscReal t, Vec x, Vec y) {
            Int ierr;
            ierr = VecGetLocalVectorRead(x, xx);
            CHKERRABORT(comm, ierr);
            ierr = VecGetLocalVectorRead(y, yy);
            CHKERRABORT(comm, ierr);
            ierr = VecGetLocalVectorRead(work, zz);
            CHKERRABORT(comm, ierr);

            arma::Row<Real> coefficients = t_fun(t);

            ierr = VecSet(y, 0.0);
            CHKERRABORT(comm, ierr);

            ierr = VecScatterBegin(action_ctx, x, lvec, INSERT_VALUES, SCATTER_FORWARD);
            CHKERRABORT(comm, ierr);
            for (Int ir{0}; ir < n_reactions; ++ir) {
                ierr = MatMult(diag_mats[ir], xx, zz);
                CHKERRABORT(comm, ierr);

                ierr = VecAXPY(yy, coefficients[ir], zz);
                CHKERRABORT(comm, ierr);
            }
            ierr = VecScatterEnd(action_ctx, x, lvec, INSERT_VALUES, SCATTER_FORWARD);
            CHKERRABORT(comm, ierr);
            for (Int ir{0}; ir < n_reactions; ++ir) {
                ierr = MatMult(offdiag_mats[ir], lvec, zz);
                CHKERRABORT(comm, ierr);

                ierr = VecAXPY(yy, coefficients[ir], zz);
                CHKERRABORT(comm, ierr);
            }
            ierr = VecRestoreLocalVectorRead(x, xx);
            CHKERRABORT(comm, ierr);
            ierr = VecRestoreLocalVectorRead(y, yy);
            CHKERRABORT(comm, ierr);
            ierr = VecRestoreLocalVectorRead(work, zz);
            CHKERRABORT(comm, ierr);
        }

        void MatrixSet::GenerateMatrices(FiniteStateSubset &fsp, const arma::Mat<Int> &SM, PropFun prop,
                                         TcoefFun new_t_fun) {
            PetscInt ierr;
            PetscMPIInt rank;
            PetscInt n_local_states = fsp.GetNumLocalStates(), own_start, own_end;
            arma::Mat<Int> my_X = fsp.GetLocalStates();
            arma::Mat<Int> can_reach_X(my_X.n_rows, my_X.n_cols), reachable_from_X(my_X.n_rows, my_X.n_cols);
            arma::Mat<Int> irnz, irnz_off, to_sinks;
            arma::Mat<Int> d_nnz, o_nnz;
            arma::Row<Int> out_indices;
            arma::Mat<PetscReal> mat_vals, mat_vals_sinks;
            ISLocalToGlobalMapping local2global_rows, local2global_lvec;

            t_fun = new_t_fun;
            n_reactions = PetscInt(SM.n_cols);
            diag_mats.resize(n_reactions);
            offdiag_mats.resize(n_reactions);
            fsp_size = fsp.GetFSPSize();//fsp_size;
            n_rows_local = n_local_states + fsp.GetNumSpecies();

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
            ierr = VecGetOwnershipRange(work, &own_start, &own_end);
            CHKERRABORT(comm, ierr);

            // Get the global and local numbers of rows
            ierr = VecGetSize(work, &n_rows_global);
            CHKERRABORT(comm, ierr);

            // Find the nnz per row of diagonal and off-diagonal matrices
            irnz.resize(n_local_states, n_reactions);
            irnz_off.resize(n_local_states, n_reactions);
            to_sinks.resize(n_local_states, n_reactions);
            out_indices.resize(n_local_states*n_reactions);
            mat_vals.resize(n_local_states, n_reactions);
            mat_vals_sinks.resize(n_local_states, n_reactions);
            d_nnz.resize(n_rows_local, n_reactions);
            o_nnz.resize(n_rows_local, n_reactions);
            d_nnz.fill(1);
            o_nnz.zeros();
            int out_count = 0;
            irnz_off.fill(-1);
            for (auto i{0}; i < n_reactions; ++i){
                can_reach_X = my_X - arma::repmat(SM.col(i), 1, my_X.n_cols);
                fsp.State2Petsc(can_reach_X, irnz.colptr(i));
                // Count nnz for rows that represent CME states
                for (auto j{0}; j < n_local_states; ++j){
                    if (irnz(j, i) >= own_start && irnz(j, i) < own_end){
                        d_nnz(j, i) += 1;
                        mat_vals(j, i) = prop(can_reach_X.colptr(j), i);
                    }
                    else if (irnz(j, i) >= 0){
                        irnz_off(j, i) = irnz(j, i);
                        irnz(j, i) = -1;
                        o_nnz(j, i) += 1;
                        out_indices(out_count) = irnz(j, i);
                        out_count++;
                        mat_vals(j, i) = prop(can_reach_X.colptr(j), i);
                    }
                }
                // Count nnz for rows that represent sink states
                reachable_from_X = my_X + arma::repmat(SM.col(i), 1, my_X.n_cols);
                fsp.State2Petsc(reachable_from_X, to_sinks.colptr(i));

                for (auto j{0}; j < n_local_states; ++j){
                    if (to_sinks(j, i) < -1){ // state j can reach a sink state
                        d_nnz(n_rows_local + (1 + to_sinks(j,i)), i) += 1;
                        to_sinks(j, i) = own_end + (1+to_sinks(j,i));
                        mat_vals_sinks(j, i) = prop(my_X.colptr(j), i);
                    }
                    else{
                        to_sinks(j, i) = -1;
                    }
                }
            }

            // Create mapping from global rows to local rows
            PetscInt *my_global_indices = new PetscInt[n_rows_local];
            for (auto i{0}; i < n_rows_local; ++i){
                my_global_indices[i] = own_start + i;
            }
            ierr = ISLocalToGlobalMappingCreate(comm, 1, n_rows_local, my_global_indices, PETSC_COPY_VALUES, &local2global_rows);
            CHKERRABORT(comm, ierr);
            ierr = ISLocalToGlobalMappingSetType(local2global_rows, ISLOCALTOGLOBALMAPPINGHASH);
            CHKERRABORT(comm, ierr);
            ierr = ISLocalToGlobalMappingSetFromOptions(local2global_rows);
            CHKERRABORT(comm, ierr);
            delete[] my_global_indices;

            // Create mapping from local ghost vec to global indices and the scatter context for matrix action
            out_indices = arma::unique(out_indices);
            lvec_length = PetscInt(out_indices.n_elem);
            ierr = VecCreateSeq(PETSC_COMM_SELF, lvec_length, &lvec);
            CHKERRABORT(comm, ierr);
            ierr = VecSetUp(lvec);
            CHKERRABORT(comm, ierr);
            ierr = ISLocalToGlobalMappingCreate(comm, 1, lvec_length, &out_indices[0], PETSC_COPY_VALUES, &local2global_lvec);
            CHKERRABORT(comm, ierr);
            ierr = ISLocalToGlobalMappingSetType(local2global_lvec, ISLOCALTOGLOBALMAPPINGHASH);
            CHKERRABORT(comm, ierr);
            ierr = ISLocalToGlobalMappingSetFromOptions(local2global_lvec);
            CHKERRABORT(comm, ierr);

            IS from_is;
            ierr = ISCreateGeneral(comm, lvec_length, &out_indices[0], PETSC_COPY_VALUES, &from_is);
            CHKERRABORT(comm, ierr);
            ierr = VecScatterCreate(work, from_is, lvec, NULL, &action_ctx);
            CHKERRABORT(comm, ierr);
            ierr = ISDestroy(&from_is);
            CHKERRABORT(comm, ierr);

            // Generate local vectors for matrix action
            VecCreateSeq(PETSC_COMM_SELF, n_rows_local, &xx);
            VecSetUp(xx);
            VecCreateSeq(PETSC_COMM_SELF, n_rows_local, &yy);
            VecSetUp(yy);
            VecCreateSeq(PETSC_COMM_SELF, n_rows_local, &zz);
            VecSetUp(zz);

            // Generate values for diagonal and off-diagonal matrices
            // Convert the global indices of nonzero entries to local indices
            ierr = ISGlobalToLocalMappingApply(local2global_rows, IS_GTOLM_MASK, n_local_states*n_reactions, irnz.memptr(), NULL, irnz.memptr());
            CHKERRABORT(comm, ierr);
            ierr = ISGlobalToLocalMappingApply(local2global_rows, IS_GTOLM_MASK, n_local_states*n_reactions, to_sinks.memptr(), NULL, to_sinks.memptr());
            CHKERRABORT(comm, ierr);
            ierr = ISGlobalToLocalMappingApply(local2global_lvec, IS_GTOLM_MASK, n_local_states*n_reactions, irnz_off.memptr(), NULL, irnz_off.memptr());
            CHKERRABORT(comm, ierr);
            ierr = ISLocalToGlobalMappingDestroy(&local2global_lvec);
            CHKERRABORT(comm, ierr);
            ierr = ISLocalToGlobalMappingDestroy(&local2global_rows);
            CHKERRABORT(comm, ierr);
            for (auto i{0}; i < n_reactions; ++i){
                ierr = MatCreate(PETSC_COMM_SELF, &diag_mats[i]);
                CHKERRABORT(comm, ierr);
                ierr = MatSetType(diag_mats[i], MATSEQAIJ);
                CHKERRABORT(comm, ierr);
                ierr = MatSetSizes(diag_mats[i], n_rows_local, n_rows_local, n_rows_local, n_rows_local);
                CHKERRABORT(comm, ierr);
                ierr = MatSetUp(diag_mats[i]);
                CHKERRABORT(comm, ierr);
                ierr = MatSeqAIJSetPreallocation(diag_mats[i], NULL, d_nnz.colptr(i));
                CHKERRABORT(comm, ierr);

                ierr = MatCreate(PETSC_COMM_SELF, &offdiag_mats[i]);
                CHKERRABORT(comm, ierr);
                ierr = MatSetType(offdiag_mats[i], MATSEQAIJ);
                CHKERRABORT(comm, ierr);
                ierr = MatSetSizes(offdiag_mats[i], n_rows_local, lvec_length, n_rows_local, lvec_length);
                CHKERRABORT(comm, ierr);
                ierr = MatSetUp(offdiag_mats[i]);
                CHKERRABORT(comm, ierr);
                ierr = MatSeqAIJSetPreallocation(offdiag_mats[i], NULL, o_nnz.colptr(i));
                CHKERRABORT(comm, ierr);

                for (auto j{0}; j < n_local_states; ++j){
                    PetscReal diag_val = prop(my_X.colptr(j), i);
                    ierr = MatSetValue(diag_mats[i], j, j, diag_val, INSERT_VALUES);
                    CHKERRABORT(comm, ierr);
                    ierr = MatSetValue(diag_mats[i], j, irnz(j, i), mat_vals(j, i), INSERT_VALUES);
                    CHKERRABORT(comm, ierr);
                    ierr = MatSetValue(diag_mats[i], to_sinks(j, i), j, mat_vals_sinks(j, i), INSERT_VALUES);
                    CHKERRABORT(comm, ierr);

                    ierr = MatSetValue(offdiag_mats[i], j, irnz_off(j, i), mat_vals(j, i), INSERT_VALUES);
                    CHKERRABORT(comm, ierr);
                }

                ierr = MatAssemblyBegin(diag_mats[i], MAT_FINAL_ASSEMBLY); CHKERRABORT(comm, ierr);
                ierr = MatAssemblyEnd(diag_mats[i], MAT_FINAL_ASSEMBLY); CHKERRABORT(comm, ierr);
                ierr = MatAssemblyBegin(offdiag_mats[i], MAT_FINAL_ASSEMBLY); CHKERRABORT(comm, ierr);
                ierr = MatAssemblyEnd(offdiag_mats[i], MAT_FINAL_ASSEMBLY); CHKERRABORT(comm, ierr);
            }
        }

        MatrixSet::~MatrixSet() {
            MPI_Comm_free(&comm);
            Destroy();
        }

        void MatrixSet::Destroy() {
            for (PetscInt i{0}; i < n_reactions; ++i) {
                if (diag_mats[i]) {
                    MatDestroy(&diag_mats[i]);
                }
                if (offdiag_mats[i]){
                    MatDestroy(&offdiag_mats[i]);
                }
            }
            if (xx) VecDestroy(&xx);
            if (yy) VecDestroy(&yy);
            if (zz) VecDestroy(&zz);
            if (work) VecDestroy(&work);
            if (lvec) VecDestroy(&lvec);
            if (action_ctx) VecScatterDestroy(&action_ctx);
        }
    }
}
