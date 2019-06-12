//
// Created by Huy Vo on 12/6/18.
//
#include "OdeSolverBase.h"

namespace pecmeal {

    void OdeSolverBase::set_print_intermediate(int iprint) {
        print_intermediate = iprint;
    }

    OdeSolverBase::OdeSolverBase(MPI_Comm new_comm) {
        MPI_Comm_dup(new_comm, &comm_);
        MPI_Comm_rank(comm_, &my_rank_);
        MPI_Comm_size(comm_, &comm_size_);
    }

    void OdeSolverBase::set_final_time(PetscReal _t_final) {
        t_final_ = _t_final;
    }

    void OdeSolverBase::set_initial_solution(Vec *_sol) {
        solution_ = _sol;
    }

    void OdeSolverBase::set_rhs(std::function<void(PetscReal, Vec, Vec)> _rhs) {
        rhs_ = std::move(_rhs);
    }

    void OdeSolverBase::evaluate_rhs(PetscReal t, Vec x, Vec y) {
        rhs_(t, x, y);
    }

    void OdeSolverBase::set_current_time(PetscReal t) {
        t_now_ = t;
    }

    PetscReal OdeSolverBase::get_current_time() const {
        return t_now_;
    }

    PetscInt OdeSolverBase::solve() {
        // Make sure the necessary data has been set
        assert(solution_ != nullptr);
        assert(rhs_);
        return 0;
    }

    OdeSolverBase::~OdeSolverBase() {
        MPI_Comm_free(&comm_);
        free();
    }

    void OdeSolverBase::enable_logging() {
        logging = PETSC_TRUE;
        perf_info.n_step = 0;
        perf_info.model_time.resize(100000);
        perf_info.cpu_time.resize(100000);
        perf_info.n_eqs.resize(100000);
    }

    FiniteProblemSolverPerfInfo OdeSolverBase::get_avg_perf_info() const {
        assert(logging);

        FiniteProblemSolverPerfInfo perf_out = perf_info;

        PetscMPIInt comm_size;
        MPI_Comm_size(comm_, &comm_size);

        for (auto i{perf_out.n_step - 1}; i >= 0; --i) {
            perf_out.cpu_time[i] = perf_out.cpu_time[i] - perf_out.cpu_time[0];
            MPI_Allreduce(MPI_IN_PLACE, (void *) &perf_out.cpu_time[i], 1, MPIU_REAL, MPI_SUM, comm_);
        }

        for (auto i{0}; i < perf_out.n_step; ++i) {
            perf_out.cpu_time[i] /= PetscReal(comm_size);
        }

        return perf_out;
    }

    void
    OdeSolverBase::set_stop_condition(const std::function<int(PetscReal, Vec, void *)> &stop_check_, void *stop_data_) {
        OdeSolverBase::stop_check_ = stop_check_;
        OdeSolverBase::stop_data_ = stop_data_;
    }
}