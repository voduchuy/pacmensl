
#include <Matrix/MatrixSet.h>

#include "Matrix/MatrixSet.h"
#include "util/cme_util.h"
#include "MatrixSet.h"


namespace cme {
    namespace parallel {

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
            irnz.set_size(n_local_states, n_reactions);
            irnz_off.set_size(n_local_states, n_reactions);
            to_sinks.set_size(n_local_states, n_reactions);
            out_indices.set_size(n_local_states*n_reactions);
            mat_vals.set_size(n_local_states, n_reactions);
            mat_vals_sinks.set_size(n_local_states, n_reactions);
            d_nnz.set_size(n_rows_local, n_reactions);
            o_nnz.set_size(n_rows_local, n_reactions);
            d_nnz.fill(1);
            o_nnz.zeros();
            int out_count = 0;
            irnz_off.fill(-1);
            for (auto i_reaction{0}; i_reaction < n_reactions; ++i_reaction){
                can_reach_X = my_X - arma::repmat(SM.col(i_reaction), 1, my_X.n_cols);
                fsp.State2Petsc(can_reach_X, irnz.colptr(i_reaction));
                // Count nnz for rows that represent CME states
                for (auto i_state{0}; i_state < n_local_states; ++i_state){
                    if (irnz(i_state, i_reaction) >= own_start && irnz(i_state, i_reaction) < own_end){
                        d_nnz(i_state, i_reaction) += 1;
                        mat_vals(i_state, i_reaction) = prop(can_reach_X.colptr(i_state), i_reaction);
                    }
                    else if (irnz(i_state, i_reaction) >= 0){
                        irnz_off(i_state, i_reaction) = irnz(i_state, i_reaction);
                        irnz(i_state, i_reaction) = -1;
                        mat_vals(i_state, i_reaction) = prop(can_reach_X.colptr(i_state), i_reaction);
                        o_nnz(i_state, i_reaction) += 1;
                        out_indices(out_count) = irnz_off(i_state, i_reaction);
                        out_count+=1;
                    }
                }
                // Count nnz for rows that represent sink states
                reachable_from_X = my_X + arma::repmat(SM.col(i_reaction), 1, my_X.n_cols);
                cme::sub2ind_nd(fsp.GetFSPSize(), reachable_from_X, to_sinks.colptr(i_reaction));
//                fsp.State2Petsc(reachable_from_X, to_sinks.colptr(i_reaction));

                for (auto i_state{0}; i_state < n_local_states; ++i_state){
                    if (to_sinks(i_state, i_reaction) < -1){ // state i_state can reach a sink state
                        d_nnz(n_rows_local + (1 + to_sinks(i_state,i_reaction)), i_reaction) += 1;
                        to_sinks(i_state, i_reaction) = own_end + (1+to_sinks(i_state,i_reaction));
                        mat_vals_sinks(i_state, i_reaction) = prop(my_X.colptr(i_state), i_reaction);
                    }
                    else{
                        to_sinks(i_state, i_reaction) = -1;
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
            out_indices.resize(out_count);
            arma::Row<Int> out_indices2 = arma::unique(out_indices);
            out_count = 0;
            for (auto i{0}; i < out_indices2.n_elem; ++i){
                if (out_indices2[i] < own_start || out_indices2[i] >= own_end){
                    out_indices(out_count) = out_indices2[i];
                    out_count +=1;
                }
            }

            lvec_length = PetscInt(out_count);
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
            for (auto i_reaction{0}; i_reaction < n_reactions; ++i_reaction){
                ierr = MatCreate(PETSC_COMM_SELF, &diag_mats[i_reaction]);
                CHKERRABORT(comm, ierr);
                ierr = MatSetType(diag_mats[i_reaction], MATSEQAIJ);
                CHKERRABORT(comm, ierr);
                ierr = MatSetSizes(diag_mats[i_reaction], n_rows_local, n_rows_local, n_rows_local, n_rows_local);
                CHKERRABORT(comm, ierr);
                ierr = MatSetUp(diag_mats[i_reaction]);
                CHKERRABORT(comm, ierr);
                ierr = MatSeqAIJSetPreallocation(diag_mats[i_reaction], NULL, d_nnz.colptr(i_reaction));
                CHKERRABORT(comm, ierr);

                ierr = MatCreate(PETSC_COMM_SELF, &offdiag_mats[i_reaction]);
                CHKERRABORT(comm, ierr);
                ierr = MatSetType(offdiag_mats[i_reaction], MATSEQAIJ);
                CHKERRABORT(comm, ierr);
                ierr = MatSetSizes(offdiag_mats[i_reaction], n_rows_local, lvec_length, n_rows_local, lvec_length);
                CHKERRABORT(comm, ierr);
                ierr = MatSetUp(offdiag_mats[i_reaction]);
                CHKERRABORT(comm, ierr);
                ierr = MatSeqAIJSetPreallocation(offdiag_mats[i_reaction], NULL, o_nnz.colptr(i_reaction));
                CHKERRABORT(comm, ierr);

                for (auto i_state{0}; i_state < n_local_states; ++i_state){
                    PetscReal diag_val = -1.0*prop(my_X.colptr(i_state), i_reaction);

                    ierr = MatSetValue(diag_mats[i_reaction], i_state, i_state, diag_val, INSERT_VALUES);
                    CHKERRABORT(comm, ierr);
                    ierr = MatSetValue(diag_mats[i_reaction], i_state, irnz(i_state, i_reaction), mat_vals(i_state, i_reaction), INSERT_VALUES);
                    CHKERRABORT(comm, ierr);
                    ierr = MatSetValue(diag_mats[i_reaction], to_sinks(i_state, i_reaction), i_state, mat_vals_sinks(i_state, i_reaction), INSERT_VALUES);
                    CHKERRABORT(comm, ierr);

                    ierr = MatSetValue(offdiag_mats[i_reaction], i_state, irnz_off(i_state, i_reaction), mat_vals(i_state, i_reaction), INSERT_VALUES);
                    CHKERRABORT(comm, ierr);
                }

                ierr = MatAssemblyBegin(diag_mats[i_reaction], MAT_FINAL_ASSEMBLY); CHKERRABORT(comm, ierr);
                ierr = MatAssemblyEnd(diag_mats[i_reaction], MAT_FINAL_ASSEMBLY); CHKERRABORT(comm, ierr);
                ierr = MatAssemblyBegin(offdiag_mats[i_reaction], MAT_FINAL_ASSEMBLY); CHKERRABORT(comm, ierr);
                ierr = MatAssemblyEnd(offdiag_mats[i_reaction], MAT_FINAL_ASSEMBLY); CHKERRABORT(comm, ierr);
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

        PetscInt MatrixSet::GetLocalGhostLength() {
            return lvec_length;
        }
    }
}
