//
// Created by Huy Vo on 5/29/18.
//

#include "FspSolverBase.h"

namespace cme {
    namespace parallel {
        FspSolverBase::FspSolverBase(MPI_Comm _comm, PartitioningType _part_type, ODESolverType _solve_type) {
            MPI_Comm_dup(_comm, &comm);
            partitioning_type = _part_type;
            odes_type = _solve_type;
        }

        void FspSolverBase::SetInitFSPBounds( arma::Row< int > &_fsp_size ) {
            fsp_bounds = _fsp_size;
        }

        void FspSolverBase::SetFSPConstraintFunctions(
                fsp_constr_multi_fn *lhs_constr) {
            fsp_constr_funs = lhs_constr;
            custom_constraints = true;
        }

        void FspSolverBase::SetExpansionFactors(arma::Row<PetscReal> &_expansion_factors) {
            fsp_expasion_factors = _expansion_factors;
        }

        void FspSolverBase::SetFSPTolerance(PetscReal _fsp_tol) {
            fsp_tol = _fsp_tol;
        }

        void FspSolverBase::SetStoichiometry(arma::Mat<Int> &stoich) {
            stoich_mat = stoich;
        }

        void FspSolverBase::SetPropensity(PropFun _prop) {
            propensity = _prop;
        }

        void FspSolverBase::SetTimeFunc(TcoefFun _t_fun) {
            t_fun = _t_fun;
        }

        void FspSolverBase::SetFinalTime(PetscReal t) {
            t_final = t;
        }

        void FspSolverBase::Solve() {
            if (log_fsp_events){
                CHKERRABORT(comm, PetscLogEventBegin(Solving, 0, 0, 0, 0));
            }

            Int ierr;
            PetscInt solver_stat = 1;
            PetscReal t = 0.0e0;

            if (verbosity > 1) {
                ode_solver->SetPrintIntermediateSteps(1);
            }

            arma::Row<PetscInt> to_expand( fsp->get_num_constraints( ));
            while (solver_stat) {
                if (log_fsp_events) {
                    CHKERRABORT(comm, PetscLogEventBegin(ODESolve, 0, 0, 0, 0));
                }
                solver_stat = ode_solver->Solve();
                if (log_fsp_events) {
                    CHKERRABORT(comm, PetscLogEventEnd(ODESolve, 0, 0, 0, 0));
                }

                // Expand the FspSolverBase if the solver halted prematurely
                if (solver_stat) {
                    to_expand = ode_solver->GetExpansionIndicator();

                    for (auto i{0}; i < to_expand.n_elem; ++i) {
                        if (to_expand(i) == 1) {
                            fsp_bounds(i) =
                                    (int) std::round(double(fsp_bounds(i)) * (fsp_expasion_factors(i) + 1.0e0) + 0.5e0);
                        }
                    }
                    if (verbosity) {
                        PetscPrintf(comm, "\n ------------- \n");
                        PetscPrintf(comm, "At time t = %.2f expansion to new fsp size: \n",
                                    ode_solver->GetCurrentTime());
                        for (auto i{0}; i < fsp_bounds.n_elem; ++i) {
                            PetscPrintf(comm, "%d ", fsp_bounds[i]);
                        }
                        PetscPrintf(comm, "\n ------------- \n");
                    }
                    // Get local states corresponding to the current solution
                    arma::Mat<PetscInt> states_old = fsp->copy_states_on_proc( );
                    if (log_fsp_events) {
                        CHKERRABORT(comm, PetscLogEventBegin(StateSetPartitioning, 0, 0, 0, 0));
                    }
                    fsp->set_shape_bounds( fsp_bounds );
                    fsp->expand( );
                    if (log_fsp_events) {
                        CHKERRABORT(comm, PetscLogEventEnd(StateSetPartitioning, 0, 0, 0, 0));
                    }
                    if (verbosity) {
                        PetscPrintf(comm, "\n ------------- \n");
                        PetscPrintf(comm, "New FSP number of states: %d \n",
                                    fsp->get_num_global_states( ));
                        PetscPrintf(comm, "\n ------------- \n");
                    }

                    // Generate the expanded vector and scatter forward the current solution
                    if (log_fsp_events) {
                        CHKERRABORT(comm, PetscLogEventBegin(SolutionScatter, 0, 0, 0, 0));
                    }
                    Vec Pnew;
                    VecCreate(comm, &Pnew);
                    VecSetSizes(Pnew, fsp->get_num_local_states( ) + fsp->get_num_constraints( ), PETSC_DECIDE);
                    VecSetUp(Pnew);
                    VecSet(Pnew, 0.0);

                    IS new_locations;
                    arma::Row<Int> new_states_locations = fsp->state2ordering( states_old, true );
                    arma::Row<Int> new_sinks_locations(to_expand.n_elem);
                    Int i_end_new;
                    ierr = VecGetOwnershipRange(Pnew, NULL, &i_end_new);
                    CHKERRABORT(comm, ierr);
                    for (auto i{0}; i < new_sinks_locations.n_elem; ++i) {
                        new_sinks_locations[i] = i_end_new - ((Int) new_sinks_locations.n_elem) + i;
                    }

                    arma::Row<Int> new_locations_vals = arma::join_horiz(new_states_locations, new_sinks_locations);
                    ierr = ISCreateGeneral(comm, (PetscInt) new_locations_vals.n_elem, &new_locations_vals[0],
                                           PETSC_COPY_VALUES, &new_locations);
                    CHKERRABORT(comm, ierr);

                    // Scatter from old vector to the expanded vector
                    VecScatter scatter;
                    ierr = VecScatterCreate(*p, NULL, Pnew, new_locations, &scatter);
                    CHKERRABORT(comm, ierr);
                    ierr = VecScatterBegin(scatter, *p, Pnew, INSERT_VALUES, SCATTER_FORWARD);
                    CHKERRABORT(comm, ierr);
                    ierr = VecScatterEnd(scatter, *p, Pnew, INSERT_VALUES, SCATTER_FORWARD);
                    CHKERRABORT(comm, ierr);

                    // Swap p to the expanded vector
                    ierr = VecDestroy(p);
                    CHKERRABORT(comm, ierr);
                    ierr = VecDuplicate(Pnew, p);
                    CHKERRABORT(comm, ierr);
                    ierr = VecSwap(*p, Pnew);
                    CHKERRABORT(comm, ierr);
                    ierr = VecScatterDestroy(&scatter);
                    CHKERRABORT(comm, ierr);
                    ierr = VecDestroy(&Pnew);
                    CHKERRABORT(comm, ierr);
                    if (log_fsp_events) {
                        CHKERRABORT(comm, PetscLogEventEnd(SolutionScatter, 0, 0, 0, 0));
                    }

                    // Free data of the ODE solver (they will be rebuilt at the beginning of the loop)
                    A->Destroy();
                    ode_solver->Free();
                    if (log_fsp_events) {
                        CHKERRABORT(comm, PetscLogEventBegin(MatrixGeneration, 0, 0, 0, 0));
                    }
                    A->GenerateMatrices(*fsp, stoich_mat, propensity, t_fun);
                    if (log_fsp_events) {
                        CHKERRABORT(comm, PetscLogEventEnd(MatrixGeneration, 0, 0, 0, 0));
                    }
                }
            }

            if (log_fsp_events){
                CHKERRABORT(comm, PetscLogEventEnd(Solving, 0, 0, 0, 0));
            }
        }

        FspSolverBase::~FspSolverBase() {
            if (comm) MPI_Comm_free(&comm);
            Destroy();
        }

        void FspSolverBase::Destroy() {
            custom_constraints = false;
            VecDestroy(p);
            delete A; A = nullptr;
            delete fsp; fsp = nullptr;
            delete ode_solver; ode_solver = nullptr;
        }

        Vec &FspSolverBase::GetP() {
            return *p;
        }

        void FspSolverBase::SetUp() {
            // Make sure all the necessary parameters have been set
            assert(t_fun);
            assert(propensity);
            assert(stoich_mat.n_elem > 0);
            assert(init_states.n_elem > 0);
            assert(init_probs.n_elem > 0);
            assert(fsp_bounds.n_elem > 0);

            PetscErrorCode ierr;
            // Register events if logging is needed
            if (log_fsp_events) {
                ierr = PetscLogDefaultBegin();
                CHKERRABORT(comm, ierr);
                ierr = PetscLogEventRegister("Finite state subset partitioning", 0, &StateSetPartitioning);
                CHKERRABORT(comm, ierr);
                ierr = PetscLogEventRegister("Generate FSP matrices", 0, &MatrixGeneration);
                CHKERRABORT(comm, ierr);
                ierr = PetscLogEventRegister("Solve reduced problem", 0, &ODESolve);
                CHKERRABORT(comm, ierr);
                ierr = PetscLogEventRegister("FSP RHS evaluation", 0, &RHSEvaluation);
                CHKERRABORT(comm, ierr);
                ierr = PetscLogEventRegister("FSP Solution scatter", 0, &SolutionScatter);
                CHKERRABORT(comm, ierr);
                ierr = PetscLogEventRegister("FSP Set-up", 0, &SettingUp);
                CHKERRABORT(comm, ierr);
                ierr = PetscLogEventRegister("FSP Solving total", 0, &Solving);
                CHKERRABORT(comm, ierr);
                ierr = PetscLogEventBegin(SettingUp, 0, 0, 0, 0);
                CHKERRABORT(comm, ierr);
            }

            fsp = new StateSetBase(comm, stoich_mat.n_rows);
            fsp->set_stoichiometry( stoich_mat );
            if (custom_constraints){
                fsp->set_shape( fsp_constr_funs, fsp_bounds );
            }
            else{
                fsp->set_shape_bounds( fsp_bounds );
            }
            fsp->set_initial_states( init_states );
            if (log_fsp_events) {
                CHKERRABORT(comm, PetscLogEventBegin(StateSetPartitioning, 0, 0, 0, 0));
            }
            fsp->set_lb_type( partitioning_type );
            fsp->set_repart_approach( repart_approach );
            fsp->expand( );
            if (log_fsp_events) {
                CHKERRABORT(comm, PetscLogEventEnd(StateSetPartitioning, 0, 0, 0, 0));
            }

            A = new FspMatrixBase(comm);
            if (log_fsp_events) {
                CHKERRABORT(comm, PetscLogEventBegin(MatrixGeneration, 0, 0, 0, 0));
            }
            A->GenerateMatrices(*fsp, stoich_mat, propensity, t_fun);
            if (log_fsp_events) {
                CHKERRABORT(comm, PetscLogEventEnd(MatrixGeneration, 0, 0, 0, 0));
            }
            if (log_fsp_events) {
                tmatvec = [&](Real t, Vec x, Vec y) {
                    CHKERRABORT(comm, PetscLogEventBegin(RHSEvaluation, 0, 0, 0, 0));
                    A->Action(t, x, y);
                    CHKERRABORT(comm, PetscLogEventEnd(RHSEvaluation, 0, 0, 0, 0));
                };
            } else {
                tmatvec = [&](Real t, Vec x, Vec y) {
                    A->Action(t, x, y);
                };
            }

            p = new Vec;
            arma::Row<Int> indices = fsp->state2ordering( init_states, true );
            ierr = VecCreate(comm, p);
            CHKERRABORT(comm, ierr);
            ierr = VecSetSizes(*p, fsp->get_num_constraints( ) + fsp->get_num_local_states( ), PETSC_DECIDE);
            CHKERRABORT(comm, ierr);
            ierr = VecSetFromOptions(*p);
            CHKERRABORT(comm, ierr);
            ierr = VecSetValues(*p, PetscInt(init_probs.n_elem), &indices[0], &init_probs[0], INSERT_VALUES);
            CHKERRABORT(comm,
                        ierr);
            ierr = VecSetUp(*p);
            CHKERRABORT(comm, ierr);
            ierr = VecAssemblyBegin(*p);
            CHKERRABORT(comm, ierr);
            ierr = VecAssemblyEnd(*p);
            CHKERRABORT(comm, ierr);

            ode_solver = new CVODEFSP( PETSC_COMM_WORLD, CV_BDF );
            ode_solver->SetFinalTime(t_final);
            ode_solver->SetFSPTolerance(fsp_tol);
            ode_solver->SetInitSolution(p);
            ode_solver->SetRHS(this->tmatvec);
            ode_solver->SetFiniteStateSubset(this->fsp);
            if (log_fsp_events) {
                ode_solver->EnableLogging();
                ierr = PetscLogEventEnd(SettingUp, 0, 0, 0, 0);
                CHKERRABORT(comm, ierr);
            }
        }

        void FspSolverBase::SetVerbosityLevel(int verbosity_level) {
            verbosity = verbosity_level;
        }

        void FspSolverBase::SetInitProbabilities(arma::Mat<Int> &_init_states, arma::Col<PetscReal> &_init_probs) {
            init_states = _init_states;
            init_probs = _init_probs;
            assert(init_probs.n_elem == init_states.n_cols);
        }

        StateSetBase* FspSolverBase::GetStateSubset() {
            return fsp;
        }

        void FspSolverBase::SetLogging(PetscBool logging) {
            log_fsp_events = logging;
        }

        FSPSolverComponentTiming FspSolverBase::GetAvgComponentTiming() {
            FSPSolverComponentTiming timings;
            PetscMPIInt comm_size;
            MPI_Comm_size(comm, &comm_size);

            auto get_avg_timing = [&](PetscLogEvent event) {
                PetscReal timing;
                PetscReal tmp;
                PetscEventPerfInfo info;
                int ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE, event, &info);
                CHKERRABORT(comm, ierr);
                tmp = info.time;
                MPI_Allreduce(&tmp, &timing, 1, MPIU_REAL, MPI_SUM, comm);
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
            return ode_solver->GetAvgPerfInfo();
        }

        void FspSolverBase::SetFromOptions() {
            PetscErrorCode ierr;
            char opt[100];
            PetscMPIInt num_procs;
            PetscBool opt_set;

            MPI_Comm_size(comm, &num_procs);

            ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_partitioning_type", opt, 100, &opt_set);
            CHKERRABORT(comm, ierr);
            if (opt_set) {
                partitioning_type = str2part(std::string(opt));
            }
            if (num_procs == 1) {
                partitioning_type = Graph;
            }

            ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_repart_approach", opt, 100, &opt_set);
            CHKERRABORT(comm, ierr);
            if (opt_set) {
                repart_approach = str2partapproach(std::string(opt));
            }

            ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_verbosity", opt, 100, &opt_set);
            CHKERRABORT(comm, ierr);
            if (opt_set) {
                if (strcmp(opt, "1") == 0 || strcmp(opt, "true") == 0) {
                    verbosity = 1;
                }
                if (strcmp(opt, "2") == 0) {
                    verbosity = 2;
                }
            }

            ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_log_events", opt, 100, &opt_set);
            CHKERRABORT(comm, ierr);
            if (opt_set) {
                if (strcmp(opt, "1") == 0 || strcmp(opt, "true") == 0) {
                    log_fsp_events = PETSC_TRUE;
                }
            }

        }


    }
}