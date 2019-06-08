//
// Created by Huy Vo on 5/29/18.
//

#include "FspSolverBase.h"

namespace cme {
    namespace parallel {
        FspSolverBase::FspSolverBase(MPI_Comm _comm, PartitioningType _part_type, ODESolverType _solve_type) {
            MPI_Comm_dup(_comm, &(FspSolverBase::comm_));
            MPI_Comm_rank(comm_, &my_rank_);
            MPI_Comm_size(comm_, &comm_size_);
            partitioning_type_ = _part_type;
            odes_type = _solve_type;
        }

        void FspSolverBase::SetInitialBounds(arma::Row<int> &_fsp_size) {
            fsp_bounds_ = _fsp_size;
        }

        void FspSolverBase::SetConstraintFunctions(fsp_constr_multi_fn *lhs_constr) {
            fsp_constr_funs_ = lhs_constr;
            have_custom_constraints_ = true;
        }

        void FspSolverBase::SetExpansionFactors(arma::Row<PetscReal> &_expansion_factors) {
            fsp_expasion_factors_ = _expansion_factors;
        }

        DiscreteDistribution FspSolverBase::Solve(PetscReal t_final, PetscReal fsp_tol) {
            Int ierr;
            if (log_fsp_events) {
                CHKERRABORT(comm_, PetscLogEventBegin(Solving, 0, 0, 0, 0));
            }

            if (verbosity_ > 1) {
                ode_solver_->set_print_intermediate(1);
            }

            fsp_tol_ = fsp_tol;
            t_final_ = t_final;
            ode_solver_->set_final_time(t_final);
            PetscInt solver_stat = 1;

            while (solver_stat) {
                if (log_fsp_events) {
                    CHKERRABORT(comm_, PetscLogEventBegin(ODESolve, 0, 0, 0, 0));
                }
                solver_stat = ode_solver_->solve();
                if (log_fsp_events) {
                    CHKERRABORT(comm_, PetscLogEventEnd(ODESolve, 0, 0, 0, 0));
                }

                // Expand the FspSolverBase if the solver halted prematurely
                if (solver_stat) {
                    for (auto i{0}; i < to_expand_.n_elem; ++i) {
                        if (to_expand_(i) == 1) {
                            fsp_bounds_(i) = (int) std::round(
                                    double(fsp_bounds_(i)) * (fsp_expasion_factors_(i) + 1.0e0) + 0.5e0);
                        }
                    }

                    if (verbosity_) {
                        PetscPrintf(comm_, "\n ------------- \n");
                        PetscPrintf(comm_, "At time t = %.2f expansion to new state_set_ size: \n",
                                    ode_solver_->get_current_time());
                        for (auto i{0}; i < fsp_bounds_.n_elem; ++i) {
                            PetscPrintf(comm_, "%d ", fsp_bounds_[i]);
                        }
                        PetscPrintf(comm_, "\n ------------- \n");
                    }
                    // Get local states corresponding to the current solution_
                    arma::Mat<PetscInt> states_old = state_set_->CopyStatesOnProc();
                    if (log_fsp_events) {
                        CHKERRABORT(comm_, PetscLogEventBegin(StateSetPartitioning, 0, 0, 0, 0));
                    }
                    ((StateSetConstrained *) state_set_)->SetShapeBounds(fsp_bounds_);
                    state_set_->Expand();
                    if (log_fsp_events) {
                        CHKERRABORT(comm_, PetscLogEventEnd(StateSetPartitioning, 0, 0, 0, 0));
                    }
                    if (verbosity_) {
                        PetscPrintf(comm_, "\n ------------- \n");
                        PetscPrintf(comm_, "New FSP number of states: %d \n", state_set_->GetNumGlobalStates());
                        PetscPrintf(comm_, "\n ------------- \n");
                    }

                    // free data of the ODE solver (they will be rebuilt at the beginning of the loop)
                    A_->destroy();
                    ode_solver_->free();
                    if (log_fsp_events) {
                        CHKERRABORT(comm_, PetscLogEventBegin(MatrixGeneration, 0, 0, 0, 0));
                    }
                    ((FspMatrixConstrained *) A_)
                            ->generate_matrices(*((StateSetConstrained *) state_set_), model_.stoichiometry_matrix_, model_.prop_, model_.t_fun_);
                    if (log_fsp_events) {
                        CHKERRABORT(comm_, PetscLogEventEnd(MatrixGeneration, 0, 0, 0, 0));
                    }

                    // Generate the expanded vector and scatter forward the current solution_
                    if (log_fsp_events) {
                        CHKERRABORT(comm_, PetscLogEventBegin(SolutionScatter, 0, 0, 0, 0));
                    }
                    Vec Pnew;
                    VecCreate(comm_, &Pnew);
                    VecSetSizes(Pnew, A_->get_num_rows_local(), PETSC_DECIDE);
                    VecSetUp(Pnew);
                    VecSet(Pnew, 0.0);

                    IS new_locations;
                    arma::Row<Int> new_states_locations = state_set_->State2Index(states_old);
                    arma::Row<Int> new_sinks_locations;

                    if (my_rank_ == comm_size_ - 1) {
                        new_sinks_locations.set_size(sinks_.n_elem);
                        Int i_end_new;
                        ierr = VecGetOwnershipRange(Pnew, NULL, &i_end_new);
                        CHKERRABORT(comm_, ierr);
                        for (auto i{0}; i < new_sinks_locations.n_elem; ++i) {
                            new_sinks_locations[i] = i_end_new - ((Int) new_sinks_locations.n_elem) + i;
                        }
                    } else {
                        new_sinks_locations.set_size(0);
                    }


                    arma::Row<Int> new_locations_vals = arma::join_horiz(new_states_locations, new_sinks_locations);
                    ierr = ISCreateGeneral(comm_, (PetscInt) new_locations_vals.n_elem, &new_locations_vals[0],
                                           PETSC_COPY_VALUES, &new_locations);
                    CHKERRABORT(comm_, ierr);

                    // Scatter from old vector to the expanded vector
                    VecScatter scatter;
                    ierr = VecScatterCreate(*p_, NULL, Pnew, new_locations, &scatter);
                    CHKERRABORT(comm_, ierr);
                    ierr = VecScatterBegin(scatter, *p_, Pnew, INSERT_VALUES, SCATTER_FORWARD);
                    CHKERRABORT(comm_, ierr);
                    ierr = VecScatterEnd(scatter, *p_, Pnew, INSERT_VALUES, SCATTER_FORWARD);
                    CHKERRABORT(comm_, ierr);

                    // Swap p_ to the expanded vector
                    ierr = VecDestroy(p_);
                    CHKERRABORT(comm_, ierr);
                    ierr = VecDuplicate(Pnew, p_);
                    CHKERRABORT(comm_, ierr);
                    ierr = VecSwap(*p_, Pnew);
                    CHKERRABORT(comm_, ierr);
                    ierr = VecScatterDestroy(&scatter);
                    CHKERRABORT(comm_, ierr);
                    ierr = VecDestroy(&Pnew);
                    CHKERRABORT(comm_, ierr);
                    if (log_fsp_events) {
                        CHKERRABORT(comm_, PetscLogEventEnd(SolutionScatter, 0, 0, 0, 0));
                    }
                }
            }

            if (log_fsp_events) {
                CHKERRABORT(comm_, PetscLogEventEnd(Solving, 0, 0, 0, 0));
            }

            return MakeOutputDistribution(0, *state_set_, *p_);
        }

        FspSolverBase::~FspSolverBase() {
            Destroy();
            if (comm_) MPI_Comm_free(&comm_);
        }

        void FspSolverBase::Destroy() {
            have_custom_constraints_ = false;
            VecDestroy(p_);
            delete A_;
            A_ = nullptr;
            delete (StateSetConstrained *) state_set_;
            state_set_ = nullptr;
            delete ode_solver_;
            ode_solver_ = nullptr;
        }

        Vec &FspSolverBase::GetP() {
            return *p_;
        }

        void FspSolverBase::SetUp() {
            // Make sure all the necessary parameters have been set
            assert(model_.t_fun_);
            assert(model_.prop_);
            assert(model_.stoichiometry_matrix_.n_elem > 0);
            assert(init_states_.n_elem > 0);
            assert(init_probs_.n_elem > 0);
            assert(fsp_bounds_.n_elem > 0);

            PetscErrorCode ierr;
            // Register events if logging is needed
            if (log_fsp_events) {
                ierr = PetscLogDefaultBegin();
                CHKERRABORT(comm_, ierr);
                ierr = PetscLogEventRegister("Finite state subset partitioning", 0, &StateSetPartitioning);
                CHKERRABORT(comm_, ierr);
                ierr = PetscLogEventRegister("Generate FSP matrices", 0, &MatrixGeneration);
                CHKERRABORT(comm_, ierr);
                ierr = PetscLogEventRegister("Solve reduced problem", 0, &ODESolve);
                CHKERRABORT(comm_, ierr);
                ierr = PetscLogEventRegister("FSP RHS evaluation", 0, &RHSEvaluation);
                CHKERRABORT(comm_, ierr);
                ierr = PetscLogEventRegister("FSP Solution scatter", 0, &SolutionScatter);
                CHKERRABORT(comm_, ierr);
                ierr = PetscLogEventRegister("FSP Set-up", 0, &SettingUp);
                CHKERRABORT(comm_, ierr);
                ierr = PetscLogEventRegister("FSP Solving total", 0, &Solving);
                CHKERRABORT(comm_, ierr);
                ierr = PetscLogEventBegin(SettingUp, 0, 0, 0, 0);
                CHKERRABORT(comm_, ierr);
            }

            state_set_ = new StateSetConstrained(comm_, model_.stoichiometry_matrix_.n_rows, partitioning_type_, repart_approach_);
            state_set_->SetStoichiometryMatrix(model_.stoichiometry_matrix_);
            if (have_custom_constraints_) {
                ((StateSetConstrained *) state_set_)->SetShape(fsp_constr_funs_, fsp_bounds_);
            } else {
                ((StateSetConstrained *) state_set_)->SetShapeBounds(fsp_bounds_);
            }
            state_set_->SetInitialStates(init_states_);
            if (log_fsp_events) {
                CHKERRABORT(comm_, PetscLogEventBegin(StateSetPartitioning, 0, 0, 0, 0));
            }
            state_set_->Expand();
            if (log_fsp_events) {
                CHKERRABORT(comm_, PetscLogEventEnd(StateSetPartitioning, 0, 0, 0, 0));
            }

            A_ = new FspMatrixConstrained(comm_);
            if (log_fsp_events) {
                CHKERRABORT(comm_, PetscLogEventBegin(MatrixGeneration, 0, 0, 0, 0));
            }
            ((FspMatrixConstrained *) A_)
                    ->generate_matrices(*((StateSetConstrained *) state_set_), model_.stoichiometry_matrix_, model_.prop_, model_.t_fun_);
            if (log_fsp_events) {
                CHKERRABORT(comm_, PetscLogEventEnd(MatrixGeneration, 0, 0, 0, 0));
            }
            if (log_fsp_events) {
                tmatvec_ = [&](Real t, Vec x, Vec y) {
                    CHKERRABORT(comm_, PetscLogEventBegin(RHSEvaluation, 0, 0, 0, 0));
                    ((FspMatrixConstrained *) A_)->action(t, x, y);
                    CHKERRABORT(comm_, PetscLogEventEnd(RHSEvaluation, 0, 0, 0, 0));
                };
            } else {
                tmatvec_ = [&](Real t, Vec x, Vec y) {
                    ((FspMatrixConstrained *) A_)->action(t, x, y);
                };
            }

            p_ = new Vec;
            arma::Row<Int> indices = state_set_->State2Index(init_states_);
            ierr = VecCreate(comm_, p_);
            CHKERRABORT(comm_, ierr);
            ierr = VecSetSizes(*p_, A_->get_num_rows_local(), PETSC_DECIDE);
            CHKERRABORT(comm_, ierr);
            ierr = VecSetFromOptions(*p_);
            CHKERRABORT(comm_, ierr);
            ierr = VecSetUp(*p_);
            CHKERRABORT(comm_, ierr);
            ierr = VecSetValues(*p_, PetscInt(init_probs_.n_elem), &indices[0], &init_probs_[0], INSERT_VALUES);
            CHKERRABORT(comm_, ierr);
            ierr = VecAssemblyBegin(*p_);
            CHKERRABORT(comm_, ierr);
            ierr = VecAssemblyEnd(*p_);
            CHKERRABORT(comm_, ierr);

            ode_solver_ = new CVODEFSP(PETSC_COMM_WORLD, CV_BDF);
            ode_solver_->set_current_time(0.0);
            ode_solver_->set_final_time(t_final_);
            ode_solver_->set_initial_solution(p_);
            ode_solver_->set_rhs(this->tmatvec_);
            auto error_checking_fp = [&](PetscReal t, Vec p, void *data) {
                return CheckFspTolerance_(t, p);
            };
            ode_solver_->set_stop_condition(error_checking_fp, nullptr);
            if (log_fsp_events) {
                ode_solver_->enable_logging();
                ierr = PetscLogEventEnd(SettingUp, 0, 0, 0, 0);
                CHKERRABORT(comm_, ierr);
            }

            sinks_.set_size(((StateSetConstrained *) state_set_)->GetNumConstraints());
            to_expand_.set_size(sinks_.n_elem);
            PetscPrintf(comm_, "Num constraints %d \n", sinks_.n_elem);
        }

        void FspSolverBase::SetVerbosity(int verbosity_level) {
            verbosity_ = verbosity_level;
        }

        void
        FspSolverBase::SetInitialDistribution(arma::Mat<Int> &_init_states, arma::Col<PetscReal> &_init_probs) {
            init_states_ = _init_states;
            init_probs_ = _init_probs;
            assert(init_probs_.n_elem == init_states_.n_cols);
        }

        const StateSetBase * FspSolverBase::GetStateSet() {
            return state_set_;
        }

        void FspSolverBase::SetLogging(PetscBool logging) {
            log_fsp_events = logging;
        }

        FspSolverComponentTiming FspSolverBase::GetAvgComponentTiming() {
            FspSolverComponentTiming timings;
            PetscMPIInt comm_size;
            MPI_Comm_size(comm_, &comm_size);

            auto get_avg_timing = [&](PetscLogEvent event) {
                PetscReal timing;
                PetscReal tmp;
                PetscEventPerfInfo info;
                int ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE, event, &info);
                CHKERRABORT(comm_, ierr);
                tmp = info.time;
                MPI_Allreduce(&tmp, &timing, 1, MPIU_REAL, MPI_SUM, comm_);
                timing /= PetscReal(comm_size);
                return timing;
            };

            timings.MatrixGenerationTime = get_avg_timing(MatrixGeneration);
            timings.StatePartitioningTime = get_avg_timing(StateSetPartitioning);
            timings.ODESolveTime = get_avg_timing(ODESolve);
            timings.RHSEvalTime = get_avg_timing(RHSEvaluation);
            timings.SolutionScatterTime = get_avg_timing(SolutionScatter);
            timings.TotalTime = get_avg_timing(SettingUp) + get_avg_timing(Solving);
            return timings;
        }

        FiniteProblemSolverPerfInfo FspSolverBase::GetSolverPerfInfo() {
            return ode_solver_->get_avg_perf_info();
        }

        void FspSolverBase::SetFromOptions() {
            PetscErrorCode ierr;
            char opt[100];
            PetscMPIInt num_procs;
            PetscBool opt_set;

            MPI_Comm_size(comm_, &num_procs);

            ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_partitioning_type", opt, 100, &opt_set);
            CHKERRABORT(comm_, ierr);
            if (opt_set) {
                partitioning_type_ = str2part(std::string(opt));
            }
            if (num_procs == 1) {
                partitioning_type_ = Graph;
            }

            ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_repart_approach", opt, 100, &opt_set);
            CHKERRABORT(comm_, ierr);
            if (opt_set) {
                repart_approach_ = str2partapproach(std::string(opt));
            }

            ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_verbosity", opt, 100, &opt_set);
            CHKERRABORT(comm_, ierr);
            if (opt_set) {
                if (strcmp(opt, "1") == 0 || strcmp(opt, "true") == 0) {
                    verbosity_ = 1;
                }
                if (strcmp(opt, "2") == 0) {
                    verbosity_ = 2;
                }
            }

            ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_log_events", opt, 100, &opt_set);
            CHKERRABORT(comm_, ierr);
            if (opt_set) {
                if (strcmp(opt, "1") == 0 || strcmp(opt, "true") == 0) {
                    log_fsp_events = PETSC_TRUE;
                }
            }

        }

        int FspSolverBase::CheckFspTolerance_(PetscReal t, Vec p) {
            to_expand_.fill(0);
            // Find the sink states
            arma::Row<PetscReal> sinks_of_p(sinks_.n_elem);
            if (my_rank_ == comm_size_ - 1) {
                const PetscReal *local_p_data;
                VecGetArrayRead(p, &local_p_data);
                int n_loc = A_->get_num_rows_local();
                for (int i{0}; i < sinks_of_p.n_elem; ++i) {
                    sinks_of_p(i) = local_p_data[n_loc - sinks_.n_elem + i];
                }
                VecRestoreArrayRead(p, &local_p_data);
            } else {
                sinks_of_p.fill(0.0);
            }
            int ierr = MPI_Allreduce(&sinks_of_p[0], &sinks_[0], sinks_of_p.n_elem, MPI_DOUBLE, MPI_SUM, comm_);
            MPICHKERRABORT(comm_, ierr);
            for (int i{0}; i < (int) sinks_.n_elem; ++i) {
                if (double(sinks_.n_elem)*sinks_(i) > (t / t_final_) * fsp_tol_) to_expand_(i) = 1;
            }
            return to_expand_.max();
        }

        void FspSolverBase::SetModel(Model &model) {
            FspSolverBase::model_ = model;
        }

        DiscreteDistribution FspSolverBase::MakeOutputDistribution(PetscReal t, const StateSetBase &state_set, Vec const &p) {
            int ierr;
            DiscreteDistribution solution;
            MPI_Comm_dup(comm_, &solution.comm);
            solution.t = t;
            solution.states = state_set.CopyStatesOnProc();
            ierr = VecDuplicate(p, &solution.p); CHKERRABORT(comm_, ierr);
            ierr = VecSetUp(solution.p); CHKERRABORT(comm_, ierr);
            ierr = VecCopy(p, solution.p); CHKERRABORT(comm_, ierr);
            return solution;
        }

        std::vector<DiscreteDistribution> FspSolverBase::Solve(const arma::Row<PetscReal> &tspan, PetscReal fsp_tol) {
            std::vector<DiscreteDistribution> outputs;
            int num_time_points = tspan.n_elem;
            outputs.reserve(num_time_points);
            for (int i = 0; i < num_time_points; ++i){
                outputs.emplace_back(FspSolverBase::Solve(tspan(i), tspan(i)*fsp_tol/tspan.max()));
            }
            return outputs;
        }
    }
}