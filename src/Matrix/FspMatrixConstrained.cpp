//
// Created by Huy Vo on 6/2/19.
//

#include "FspMatrixConstrained.h"

namespace pecmeal {
    FspMatrixConstrained::FspMatrixConstrained(MPI_Comm comm) : FspMatrixBase(comm) {
    }


    void FspMatrixConstrained::action(PetscReal t, Vec x, Vec y) {
        int ierr;
        FspMatrixBase::action(t, x, y);
        ierr = VecGetLocalVectorRead(x, xx);
        CHKERRABORT(comm_, ierr);
        ierr = MatMult(sinks_mat_, xx, sink_entries_);
        CHKERRABORT(comm_, ierr);
        ierr = VecScatterBegin(sink_scatter_ctx_, sink_entries_, y, ADD_VALUES, SCATTER_FORWARD);
        CHKERRABORT(comm_, ierr);
        ierr = VecScatterEnd(sink_scatter_ctx_, sink_entries_, y, ADD_VALUES, SCATTER_FORWARD);
        CHKERRABORT(comm_, ierr);
        ierr = VecRestoreLocalVectorRead(x, xx);
        CHKERRABORT(comm_, ierr);
    }

    void FspMatrixConstrained::destroy() {
        FspMatrixBase::destroy();
        if (sink_entries_ != nullptr) VecDestroy(&sink_entries_);
        if (sinks_mat_ != nullptr) MatDestroy(&sinks_mat_);
        if (sink_scatter_ctx_ != nullptr) VecScatterDestroy(&sink_scatter_ctx_);
    }

    FspMatrixConstrained::~FspMatrixConstrained() {
        destroy();
    }

    void
    FspMatrixConstrained::generate_matrices(StateSetConstrained &fsp, const arma::Mat<int> &SM,
                                            PropFun prop, TcoefFun new_t_fun) {
        PetscErrorCode ierr;

        sinks_rank_ = comm_size_ - 1; // rank of the processor that holds sink states
        num_constraints_ = fsp.GetNumConstraints();

        // Generate the entries corresponding to usual states
        FspMatrixBase::generate_values(fsp, SM, prop, new_t_fun);

        // Generate the extra blocks corresponding to sink states
        const arma::Mat<int> &state_list = fsp.GetStatesRef();
        int n_local_states = fsp.GetNumLocalStates();
        int n_constraints = fsp.GetNumConstraints();
        arma::Mat<int> can_reach_my_state;

        arma::Mat<Int> reachable_from_X(state_list.n_rows, state_list.n_cols);
        // Workspace for checking constraints
        arma::Mat<PetscInt> constraints_satisfied(n_local_states, n_constraints);
        arma::Col<PetscInt> nconstraints_satisfied;
        arma::Mat<int> d_nnz(n_constraints, n_reactions_);
        std::vector<arma::Row<int>> sink_inz(n_constraints * n_reactions_);
        std::vector<arma::Row<PetscReal>> sink_rows(n_constraints * n_reactions_);

        ierr = MatCreate(PETSC_COMM_SELF, &sinks_mat_);
        CHKERRABORT(comm_, ierr);
        ierr = MatSetType(sinks_mat_, MATSEQAIJ);
        CHKERRABORT(comm_, ierr);
        ierr = MatSetSizes(sinks_mat_, n_constraints, n_rows_local_, n_constraints, n_rows_local_);
        CHKERRABORT(comm_, ierr);
        ierr = MatSeqAIJSetPreallocation(sinks_mat_, n_constraints * n_local_states, NULL);
        CHKERRABORT(comm_, ierr);

        for (int i_reaction{0}; i_reaction < n_reactions_; ++i_reaction) {
            // Count nnz for rows that represent sink states
            can_reach_my_state = state_list + arma::repmat(SM.col(i_reaction), 1, state_list.n_cols);
            fsp.CheckConstraints(n_local_states, can_reach_my_state.colptr(0),
                                 constraints_satisfied.colptr(0));
            nconstraints_satisfied = arma::sum(constraints_satisfied, 1);

            for (int i_constr = 0; i_constr < n_constraints; ++i_constr) {
                d_nnz(i_constr, i_reaction) = n_local_states - arma::sum(constraints_satisfied.col(i_constr));
                sink_inz.at(n_constraints * i_reaction + i_constr).set_size(d_nnz(i_constr, i_reaction));
                sink_rows.at(n_constraints * i_reaction + i_constr).set_size(d_nnz(i_constr, i_reaction));
            }
            // Store the column indices and values of the nonzero entries on the sink rows
            for (int i_constr = 0; i_constr < n_constraints; ++i_constr) {
                int count = 0;
                for (int i_state = 0; i_state < n_local_states; ++i_state) {
                    if (constraints_satisfied(i_state, i_constr) == 0) {
                        sink_inz.at(n_constraints * i_reaction + i_constr).at(count) = i_state;
                        sink_rows.at(n_constraints * i_reaction + i_constr).at(count) =
                                prop(state_list.colptr(i_state), i_reaction) /
                                (PetscReal(n_constraints - nconstraints_satisfied(i_state)));
                        count += 1;
                    }
                }
            }
            for (auto i_constr{0}; i_constr < n_constraints; i_constr++) {
                ierr = MatSetValues(sinks_mat_, 1, &i_constr, d_nnz(i_constr, i_reaction),
                                    sink_inz.at(i_reaction * n_constraints + i_constr).memptr(),
                                    sink_rows.at(i_reaction * n_constraints + i_constr).memptr(), ADD_VALUES);
                CHKERRABORT(comm_, ierr);
            }
        }

        ierr = MatAssemblyBegin(sinks_mat_, MAT_FINAL_ASSEMBLY);
        CHKERRABORT(comm_, ierr);
        ierr = MatAssemblyEnd(sinks_mat_, MAT_FINAL_ASSEMBLY);
        CHKERRABORT(comm_, ierr);

        // Local vectors for computing sink entries
        ierr = VecCreateSeq(PETSC_COMM_SELF, n_constraints, &sink_entries_);
        CHKERRABORT(comm_, ierr);
        ierr = VecSetUp(sink_entries_);
        CHKERRABORT(comm_, ierr);

        // Scatter context for adding sink values
        int *sink_global_indices = new int[n_constraints];
        for (int i{0}; i < n_constraints; ++i) {
            sink_global_indices[i] = fsp.GetNumGlobalStates() + i;
        }
        IS sink_is;
        Vec tmp;
        ierr = ISCreateGeneral(comm_, n_constraints, sink_global_indices, PETSC_COPY_VALUES, &sink_is);
        CHKERRABORT(comm_, ierr);
        ierr = VecScatterCreate(sink_entries_, NULL, work_, sink_is, &sink_scatter_ctx_);
        CHKERRABORT(comm_, ierr);
        ierr = ISDestroy(&sink_is);
        CHKERRABORT(comm_, ierr);
        delete[] sink_global_indices;
    }


    void FspMatrixConstrained::determine_layout(const StateSetBase &fsp) {
        PetscErrorCode ierr;

        n_rows_local_ = fsp.GetNumLocalStates();
        if (my_rank_ == sinks_rank_) n_rows_local_ += num_constraints_;

        // Generate matrix layout from FSP's layout
        ierr = VecCreate(comm_, &work_);
        CHKERRABORT(comm_, ierr);
        ierr = VecSetFromOptions(work_);
        CHKERRABORT(comm_, ierr);
        ierr = VecSetSizes(work_, n_rows_local_, PETSC_DECIDE);
        CHKERRABORT(comm_, ierr);
        ierr = VecSetUp(work_);
        CHKERRABORT(comm_, ierr);
    }
}