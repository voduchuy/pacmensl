//
// Created by Huy Vo on 5/29/18.
//

#include "FspSolverMultiSinks.h"

namespace pacmensl {
FspSolverMultiSinks::FspSolverMultiSinks(MPI_Comm _comm, PartitioningType _part_type, ODESolverType _solve_type) {
  int ierr;
  ierr = MPI_Comm_dup(_comm, &(FspSolverMultiSinks::comm_));
  PACMENSLCHKERREXCEPT(ierr);
  ierr = MPI_Comm_rank(comm_, &my_rank_);
  PACMENSLCHKERREXCEPT(ierr);
  ierr = MPI_Comm_size(comm_, &comm_size_);
  PACMENSLCHKERREXCEPT(ierr);
  partitioning_type_ = _part_type;
  odes_type_ = _solve_type;
}

int FspSolverMultiSinks::SetInitialBounds(arma::Row<int> &_fsp_size) {
  fsp_bounds_ = _fsp_size;
  return 0;
}

int FspSolverMultiSinks::SetConstraintFunctions(const fsp_constr_multi_fn &lhs_constr) {
  fsp_constr_funs_ = lhs_constr;
  have_custom_constraints_ = true;
  return 0;
}

int FspSolverMultiSinks::SetExpansionFactors(arma::Row<PetscReal> &_expansion_factors) {
  fsp_expasion_factors_ = _expansion_factors;
  return 0;
}

DiscreteDistribution FspSolverMultiSinks::Advance_(PetscReal t_final, PetscReal fsp_tol) {
  PetscErrorCode ierr;
  PetscInt solver_stat;
  Vec Pnew;

  if (logging_enabled) PACMENSLCHKERREXCEPT(PetscLogEventBegin(Solving, 0, 0, 0, 0));

  if (verbosity_ > 1) ode_solver_->set_print_intermediate(1);

  fsp_tol_ = fsp_tol;
  t_final_ = t_final;
  ode_solver_->SetFinalTime(t_final);

  solver_stat = 1;
  while (solver_stat) {
    if (logging_enabled) PACMENSLCHKERREXCEPT(PetscLogEventBegin(ODESolve, 0, 0, 0, 0));
    ode_solver_->set_initial_solution(p_);
    solver_stat = ode_solver_->Solve();
    if (logging_enabled) PACMENSLCHKERREXCEPT(PetscLogEventEnd(ODESolve, 0, 0, 0, 0));

    // Expand the FspSolverBase if the solver halted prematurely
    if (solver_stat) {
      for (auto i{0}; i < to_expand_.n_elem; ++i) {
        if (to_expand_(i) == 1) {
          fsp_bounds_(i) = ( int ) std::round(
              double(fsp_bounds_(i)) * (fsp_expasion_factors_(i) + 1.0e0) + 0.5e0);
        }
      }

      if (verbosity_) {
        PetscPrintf(comm_, "\n ------------- \n");
        PetscPrintf(comm_, "At time t = %.2f expansion to new state_set_ size: \n",
                    ode_solver_->GetCurrentTime());
        for (auto i{0}; i < fsp_bounds_.n_elem; ++i) {
          PetscPrintf(comm_, "%d ", fsp_bounds_[i]);
        }
        PetscPrintf(comm_, "\n ------------- \n");
      }
      // Get local states_ corresponding to the current solution_
      arma::Mat<PetscInt> states_old = state_set_->CopyStatesOnProc();
      if (logging_enabled) {
        PACMENSLCHKERREXCEPT(PetscLogEventBegin(StateSetPartitioning, 0, 0, 0, 0));
      }
      (( StateSetConstrained * ) state_set_)->SetShapeBounds(fsp_bounds_);
      state_set_->Expand();
      if (logging_enabled) {
        PACMENSLCHKERREXCEPT(PetscLogEventEnd(StateSetPartitioning, 0, 0, 0, 0));
      }
      if (verbosity_) {
        PetscPrintf(comm_, "\n ------------- \n");
        PetscPrintf(comm_, "New FSP number of states_: %d \n", state_set_->GetNumGlobalStates());
        PetscPrintf(comm_, "\n ------------- \n");
      }

      // free data of the ODE solver (they will be rebuilt at the beginning of the loop)
      A_->Destroy();
      if (logging_enabled) {
        PACMENSLCHKERREXCEPT(PetscLogEventBegin(MatrixGeneration, 0, 0, 0, 0));
      }
      (( FspMatrixConstrained * ) A_)
          ->GenerateValues(*(( StateSetConstrained * ) state_set_), model_.stoichiometry_matrix_,
                           model_.prop_, model_.prop_args_, model_.t_fun_, model_.t_fun_args_);
      if (logging_enabled) {
        PACMENSLCHKERREXCEPT(PetscLogEventEnd(MatrixGeneration, 0, 0, 0, 0));
      }

      // Generate the expanded vector and scatter forward the current solution_
      if (logging_enabled) {
        PACMENSLCHKERREXCEPT(PetscLogEventBegin(SolutionScatter, 0, 0, 0, 0));
      }
      ierr = VecCreate(comm_, &Pnew);
      PACMENSLCHKERREXCEPT(ierr);
      ierr = VecSetSizes(Pnew, A_->get_num_rows_local(), PETSC_DECIDE);
      PACMENSLCHKERREXCEPT(ierr);
      ierr = VecSetUp(Pnew);
      PACMENSLCHKERREXCEPT(ierr);
      ierr = VecSet(Pnew, 0.0);
      PACMENSLCHKERREXCEPT(ierr);

      // Scatter from old vector to the expanded vector
      IS new_locations;
      arma::Row<Int> new_states_locations;
      try {
        new_states_locations = state_set_->State2Index(states_old);
      }
      catch (...) {
        PACMENSLCHKERREXCEPT(ierr);
      };
      arma::Row<Int> new_sinks_locations;
      if (my_rank_ == comm_size_ - 1) {
        new_sinks_locations.set_size(sinks_.n_elem);
        Int i_end_new;
        ierr = VecGetOwnershipRange(Pnew, NULL, &i_end_new);
        PACMENSLCHKERREXCEPT(ierr);
        for (auto i{0}; i < new_sinks_locations.n_elem; ++i) {
          new_sinks_locations[i] = i_end_new - (( Int ) new_sinks_locations.n_elem) + i;
        }
      } else {
        new_sinks_locations.set_size(0);
      }

      arma::Row<Int> new_locations_vals = arma::join_horiz(new_states_locations, new_sinks_locations);
      ierr = ISCreateGeneral(comm_, ( PetscInt ) new_locations_vals.n_elem, &new_locations_vals[0],
                             PETSC_COPY_VALUES, &new_locations);
      PACMENSLCHKERREXCEPT(ierr);
      VecScatter scatter;
      ierr = VecScatterCreate(*p_, NULL, Pnew, new_locations, &scatter);
      PACMENSLCHKERREXCEPT(ierr);
      ierr = VecScatterBegin(scatter, *p_, Pnew, INSERT_VALUES, SCATTER_FORWARD);
      PACMENSLCHKERREXCEPT(ierr);
      ierr = VecScatterEnd(scatter, *p_, Pnew, INSERT_VALUES, SCATTER_FORWARD);
      PACMENSLCHKERREXCEPT(ierr);
      ierr = ISDestroy(&new_locations);
      PACMENSLCHKERREXCEPT(ierr);
      ierr = VecScatterDestroy(&scatter);
      PACMENSLCHKERREXCEPT(ierr);

      // Swap p_ to the expanded vector
      ierr = VecDestroy(p_);
      PACMENSLCHKERREXCEPT(ierr);
      *p_ = Pnew;

      if (logging_enabled) {
        PACMENSLCHKERREXCEPT(PetscLogEventEnd(SolutionScatter, 0, 0, 0, 0));
      }
    }
  }

  if (logging_enabled) {
    PACMENSLCHKERREXCEPT(PetscLogEventEnd(Solving, 0, 0, 0, 0));
  }

  return DiscreteDistribution(comm_, t_final, state_set_, *p_);
}

FspSolverMultiSinks::~FspSolverMultiSinks() {
  ClearState();
  if (comm_) MPI_Comm_free(&comm_);
}

int FspSolverMultiSinks::ClearState() {
  int ierr;
  have_custom_constraints_ = false;
  if (p_ != nullptr) {
    ierr = VecDestroy(p_);
    CHKERRQ(ierr);
    p_ = nullptr;
  }
  delete ( StateSetConstrained * ) state_set_;
  state_set_ = nullptr;
  delete ( FspMatrixConstrained * ) A_;
  A_ = nullptr;
  switch (odes_type_) {
    case KRYLOV:delete ( KrylovFsp * ) ode_solver_;
      break;
    default:delete ( CvodeFsp * ) ode_solver_;
  }
  ode_solver_ = nullptr;
  return 0;
}

int FspSolverMultiSinks::SetUp() {
  int ierr{0};
  // Make sure all the necessary parameters have been set
  try {
    if (model_.t_fun_ == nullptr) {
      throw std::runtime_error("Temporal signal was not set before calling FspSolver.SetUp().");
    }
    if (model_.prop_ == nullptr) {
      throw std::runtime_error("Propensity was not set before calling FspSolver.SetUp().");
    }
    if (model_.stoichiometry_matrix_.n_elem == 0) {
      throw std::runtime_error("Empty stoichiometry matrix cannot be used for FspSolver.");
    }
    if (init_states_.n_elem == 0 || init_probs_.n_elem == 0) {
      throw std::runtime_error("Initial states and/or probabilities were not set before calling FspSolver.SetUp().");
    }
  } catch (std::runtime_error &e) {
    ierr = -1;
  }
  PACMENSLCHKERRQ(ierr);

  // Register events if logging is needed
  if (logging_enabled) {
    ierr = PetscLogDefaultBegin();
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("Finite state subset partitioning", 0, &StateSetPartitioning);
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("Generate FSP matrices", 0, &MatrixGeneration);
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("Advance reduced problem", 0, &ODESolve);
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("FSP RHS evaluation", 0, &RHSEvaluation);
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("FSP Solution scatter", 0, &SolutionScatter);
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("FSP Set-up", 0, &SettingUp);
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("FSP Solving total", 0, &Solving);
    CHKERRQ(ierr);
    ierr = PetscLogEventBegin(SettingUp, 0, 0, 0, 0);
    CHKERRQ(ierr);
  }

  state_set_ = new StateSetConstrained(comm_, model_.stoichiometry_matrix_.n_rows, partitioning_type_,
                                       repart_approach_);
  state_set_->SetStoichiometryMatrix(model_.stoichiometry_matrix_);
  if (have_custom_constraints_) {
    (( StateSetConstrained * ) state_set_)->SetShape(fsp_constr_funs_, fsp_bounds_, nullptr);
  } else {
    (( StateSetConstrained * ) state_set_)->SetShapeBounds(fsp_bounds_);
  }
  state_set_->SetInitialStates(init_states_);
  if (logging_enabled) {
    ierr = PetscLogEventBegin(StateSetPartitioning, 0, 0, 0, 0);
    PACMENSLCHKERREXCEPT(ierr);
  }
  state_set_->Expand();
  if (logging_enabled) {
    ierr = PetscLogEventEnd(StateSetPartitioning, 0, 0, 0, 0);
    PACMENSLCHKERREXCEPT(ierr);
  }

  A_ = new FspMatrixConstrained(comm_);
  if (logging_enabled) {
    CHKERRABORT(comm_, PetscLogEventBegin(MatrixGeneration, 0, 0, 0, 0));
  }
  (( FspMatrixConstrained * ) A_)
      ->GenerateValues(*(( StateSetConstrained * ) state_set_),
                       model_.stoichiometry_matrix_,
                       model_.prop_,
                       model_.prop_args_,
                       model_.t_fun_,
                       model_.t_fun_args_);
  if (logging_enabled) {
    CHKERRABORT(comm_, PetscLogEventEnd(MatrixGeneration, 0, 0, 0, 0));
  }
  if (logging_enabled) {
    tmatvec_ = [&](Real t, Vec x, Vec y) {
      CHKERRABORT(comm_, PetscLogEventBegin(RHSEvaluation, 0, 0, 0, 0));
      (( FspMatrixConstrained * ) A_)->action(t, x, y);
      CHKERRABORT(comm_, PetscLogEventEnd(RHSEvaluation, 0, 0, 0, 0));
    };
  } else {
    tmatvec_ = [&](Real t, Vec x, Vec y) {
      (( FspMatrixConstrained * ) A_)->action(t, x, y);
    };
  }

  switch (odes_type_) {
    case CVODE_BDF:ode_solver_ = new CvodeFsp(comm_, CV_BDF);
      break;
    case KRYLOV:ode_solver_ = new KrylovFsp(comm_);
      break;
    default:ode_solver_ = new CvodeFsp(comm_, CV_BDF);
  }

  ode_solver_->set_rhs(this->tmatvec_);
  auto error_checking_fp = [&](PetscReal t, Vec p, void *data) {
    return CheckFspTolerance_(t, p);
  };
  ode_solver_->SetStopCondition(error_checking_fp, nullptr);
  if (logging_enabled) {
    ode_solver_->EnableLogging();
    ierr = PetscLogEventEnd(SettingUp, 0, 0, 0, 0);
    CHKERRQ(ierr);
  }
  ode_solver_->SetUp();

  sinks_.set_size((( StateSetConstrained * ) state_set_)->GetNumConstraints());
  to_expand_.set_size(sinks_.n_elem);
  return 0;
}

int FspSolverMultiSinks::SetVerbosity(int verbosity_level) {
  verbosity_ = verbosity_level;
  return 0;
}

int FspSolverMultiSinks::SetInitialDistribution(const arma::Mat<Int> &_init_states,
                                                const arma::Col<PetscReal> &_init_probs) {
  init_states_ = _init_states;
  init_probs_ = _init_probs;
  if (init_probs_.n_elem != init_states_.n_cols) {
    return -1;
  }
  return 0;
}

const StateSetBase *FspSolverMultiSinks::GetStateSet() {
  return state_set_;
}

int FspSolverMultiSinks::SetLogging(PetscBool logging) {
  logging_enabled = logging;
}

FspSolverComponentTiming FspSolverMultiSinks::GetAvgComponentTiming() {
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

  FspSolverComponentTiming timings;
  timings.MatrixGenerationTime = get_avg_timing(MatrixGeneration);
  timings.StatePartitioningTime = get_avg_timing(StateSetPartitioning);
  timings.ODESolveTime = get_avg_timing(ODESolve);
  timings.RHSEvalTime = get_avg_timing(RHSEvaluation);
  timings.SolutionScatterTime = get_avg_timing(SolutionScatter);
  timings.TotalTime = get_avg_timing(SettingUp) + get_avg_timing(Solving);
  return timings;
}

FiniteProblemSolverPerfInfo FspSolverMultiSinks::GetSolverPerfInfo() {
  return ode_solver_->GetAvgPerfInfo();
}

int FspSolverMultiSinks::SetFromOptions() {
  PetscErrorCode ierr;
  char opt[100];
  PetscMPIInt num_procs;
  PetscBool opt_set;

  MPI_Comm_size(comm_, &num_procs);

  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_partitioning_type", opt, 100, &opt_set);
  CHKERRQ(ierr);
  if (opt_set) {
    partitioning_type_ = str2part(std::string(opt));
  }
  if (num_procs == 1) {
    partitioning_type_ = GRAPH;
  }

  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_repart_approach", opt, 100, &opt_set);
  CHKERRQ(ierr);
  if (opt_set) {
    repart_approach_ = str2partapproach(std::string(opt));
  }

  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_verbosity", opt, 100, &opt_set);
  CHKERRQ(ierr);
  if (opt_set) {
    if (strcmp(opt, "1") == 0 || strcmp(opt, "true") == 0) {
      verbosity_ = 1;
    }
    if (strcmp(opt, "2") == 0) {
      verbosity_ = 2;
    }
  }

  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_log_events", opt, 100, &opt_set);
  CHKERRQ(ierr);
  if (opt_set) {
    if (strcmp(opt, "1") == 0 || strcmp(opt, "true") == 0) {
      logging_enabled = PETSC_TRUE;
    }
  }
  return 0;
}

int FspSolverMultiSinks::CheckFspTolerance_(PetscReal t, Vec p) {
  int ierr;
  to_expand_.fill(0);
  // Find the sink states_
  arma::Row<PetscReal> sinks_of_p(sinks_.n_elem);
  if (my_rank_ == comm_size_ - 1) {
    const PetscReal *local_p_data;
    ierr = VecGetArrayRead(p, &local_p_data);
    PACMENSLCHKERREXCEPT(ierr);
    int n_loc = A_->get_num_rows_local();
    for (int i{0}; i < sinks_of_p.n_elem; ++i) {
      sinks_of_p(i) = local_p_data[n_loc - sinks_.n_elem + i];
    }
    ierr = VecRestoreArrayRead(p, &local_p_data);
    PACMENSLCHKERREXCEPT(ierr);
  } else {
    sinks_of_p.fill(0.0);
  }

  MPI_Datatype scalar_type;
  ierr = PetscDataTypeToMPIDataType(PETSC_DOUBLE, &scalar_type);
  PACMENSLCHKERREXCEPT(ierr);
  ierr = MPI_Allreduce(&sinks_of_p[0], &sinks_[0], sinks_of_p.n_elem, scalar_type, MPI_SUM, comm_);
  PACMENSLCHKERREXCEPT(ierr);
  for (int i{0}; i < ( int ) sinks_.n_elem; ++i) {
    if (sinks_(i)/fsp_tol_ > (1.0/double(sinks_.n_elem) )*(t / t_final_)) to_expand_(i) = 1;
  }
  return to_expand_.max();
}

int FspSolverMultiSinks::SetModel(Model &model) {
  FspSolverMultiSinks::model_ = model;
  return 0;
}

DiscreteDistribution FspSolverMultiSinks::Solve(PetscReal t_final, PetscReal fsp_tol) {
  PetscErrorCode ierr;
  if (p_ == nullptr) {
    p_ = new Vec;
    ierr = VecCreate(comm_, p_);
    PACMENSLCHKERREXCEPT(ierr);
    ierr = VecSetSizes(*p_, A_->get_num_rows_local(), PETSC_DECIDE);
    PACMENSLCHKERREXCEPT(ierr);
    ierr = VecSetType(*p_, VECMPI);
    PACMENSLCHKERREXCEPT(ierr);
    ierr = VecSetUp(*p_);
    PACMENSLCHKERREXCEPT(ierr);
  }
  ierr = VecSet(*p_, 0.0);
  PACMENSLCHKERREXCEPT(ierr);
  arma::Row<Int> indices = state_set_->State2Index(init_states_);
  ierr = VecSetValues(*p_, PetscInt(init_probs_.n_elem), &indices[0], &init_probs_[0], INSERT_VALUES);
  PACMENSLCHKERREXCEPT(ierr);
  ierr = VecAssemblyBegin(*p_);
  PACMENSLCHKERREXCEPT(ierr);
  ierr = VecAssemblyEnd(*p_);
  PACMENSLCHKERREXCEPT(ierr);

  ode_solver_->set_current_time(0.0);

  DiscreteDistribution solution = FspSolverMultiSinks::Advance_(t_final, fsp_tol);

  return solution;
}

std::vector<DiscreteDistribution>
FspSolverMultiSinks::SolveTspan(const std::vector<PetscReal> &tspan, PetscReal fsp_tol) {
  std::vector<DiscreteDistribution> outputs;
  PetscErrorCode ierr;
  if (p_ == nullptr) {
    p_ = new Vec;
    ierr = VecCreate(comm_, p_);
    PACMENSLCHKERREXCEPT(ierr);
    ierr = VecSetSizes(*p_, A_->get_num_rows_local(), PETSC_DECIDE);
    PACMENSLCHKERREXCEPT(ierr);
    ierr = VecSetType(*p_, VECMPI);
    PACMENSLCHKERREXCEPT(ierr);
    ierr = VecSetUp(*p_);
    PACMENSLCHKERREXCEPT(ierr);
  }
  ierr = VecSet(*p_, 0.0);
  PACMENSLCHKERREXCEPT(ierr);
  arma::Row<Int> indices = state_set_->State2Index(init_states_);
  ierr = VecSetValues(*p_, PetscInt(init_probs_.n_elem), &indices[0], &init_probs_[0], INSERT_VALUES);
  PACMENSLCHKERREXCEPT(ierr);
  ierr = VecAssemblyBegin(*p_);
  PACMENSLCHKERREXCEPT(ierr);
  ierr = VecAssemblyEnd(*p_);
  PACMENSLCHKERREXCEPT(ierr);

  int num_time_points = tspan.size();
  outputs.resize(num_time_points);

  PetscReal t_max = tspan[num_time_points - 1];

  DiscreteDistribution sol;

  ode_solver_->set_current_time(0.0);
  for (int i = 0; i < num_time_points; ++i) {
    sol = FspSolverMultiSinks::Advance_(tspan[i], tspan[i] * fsp_tol / t_max);
    outputs[i] = sol;
  }

  return outputs;
}

int FspSolverMultiSinks::SetOdesType(ODESolverType odes_type) {
  FspSolverMultiSinks::odes_type_ = odes_type;
  return 0;
}

int FspSolverMultiSinks::SetLoadBalancingMethod(PartitioningType part_type) {
  partitioning_type_ = part_type;
  return 0;
}

}