//
// Created by Huy Vo on 2019-06-28.
//

#include "SensFspSolverMultiSinks.h"

pacmensl::SensFspSolverMultiSinks::SensFspSolverMultiSinks(MPI_Comm _comm,
                                                           pacmensl::PartitioningType _part_type,
                                                           pacmensl::ODESolverType _solve_type) {
  MPI_Comm_dup(_comm, &comm_);
  MPI_Comm_rank(_comm, &my_rank_);
  MPI_Comm_size(_comm, &comm_size_);
}

PacmenslErrorCode pacmensl::SensFspSolverMultiSinks::SetConstraintFunctions(const pacmensl::fsp_constr_multi_fn &lhs_constr) {
  fsp_constr_funs_         = lhs_constr;
  have_custom_constraints_ = true;
  return 0;
}

PacmenslErrorCode pacmensl::SensFspSolverMultiSinks::SetInitialBounds(arma::Row<int> &_fsp_size) {
  fsp_bounds_ = _fsp_size;
  return 0;
}

PacmenslErrorCode pacmensl::SensFspSolverMultiSinks::SetExpansionFactors(arma::Row<PetscReal> &_expansion_factors) {
  fsp_expasion_factors_ = _expansion_factors;
  return 0;
}

PacmenslErrorCode pacmensl::SensFspSolverMultiSinks::SetModel(pacmensl::SensModel &model) {
  model_ = model;
  return 0;
}

PacmenslErrorCode pacmensl::SensFspSolverMultiSinks::SetVerbosity(int verbosity_level) {
  verbosity_ = verbosity_level;
  return 0;
}

PacmenslErrorCode pacmensl::SensFspSolverMultiSinks::SetInitialDistribution(const arma::Mat<pacmensl::Int> &_init_states,
                                                                            const arma::Col<PetscReal> &_init_probs,
                                                                            const std::vector<arma::Col<PetscReal>> &_init_sens) {
  if (init_probs_.n_elem != init_states_.n_cols) {
    return -1;
  }
  init_states_ = _init_states;
  init_probs_  = _init_probs;
  init_sens_   = _init_sens;
  return 0;
}

PacmenslErrorCode pacmensl::SensFspSolverMultiSinks::SetLoadBalancingMethod(pacmensl::PartitioningType part_type) {
  partitioning_type_ = part_type;
  return 0;
}

PacmenslErrorCode pacmensl::SensFspSolverMultiSinks::SetOdesType(ForwardSensType odes_type) {
  sens_solver_type = odes_type;
  return 0;
}

PacmenslErrorCode pacmensl::SensFspSolverMultiSinks::SetUp() {
  int ierr{0};
  // Make sure all the necessary parameters have been set
  try {
    if (model_.propensity_tfac_ == nullptr) {
      throw std::runtime_error("Temporal signal was not set before calling FspSolver.SetUp().");
    }
    if (model_.propensity_xfac_ == nullptr) {
      throw std::runtime_error("Propensity was not set before calling FspSolver.SetUp().");
    }
    if (model_.stoichiometry_matrix_.n_elem == 0) {
      throw std::runtime_error("Empty stoichiometry matrix cannot be used for FspSolver.");
    }
    if (model_.dpropensity_xfac_.empty()) {
      throw std::runtime_error("Empty sensitivity information.\n");
    }
    if (init_states_.n_elem == 0 || init_probs_.n_elem == 0) {
      throw std::runtime_error("Initial states and/or probabilities were not set before calling FspSolver.SetUp().");
    }
  } catch (std::runtime_error &e) {
    PetscPrintf(comm_, "\n %s \n", e.what());
    ierr = -1;
  } PACMENSLCHKERRQ(ierr);

  state_set_ = std::make_shared<StateSetConstrained>(comm_);
  state_set_->SetStoichiometryMatrix(model_.stoichiometry_matrix_);
  if (have_custom_constraints_) {
    state_set_->SetShape(fsp_constr_funs_, fsp_bounds_, nullptr);
  } else {
    state_set_->SetShapeBounds(fsp_bounds_);
  }
  state_set_->SetUp();
  state_set_->AddStates(init_states_);
  ierr = state_set_->Expand(); PACMENSLCHKERRQ(ierr);

  A_   = std::make_shared<SensFspMatrix<FspMatrixConstrained>>(comm_);
  ierr = A_->GenerateValues(*state_set_, model_); PACMENSLCHKERRQ(ierr);

  matvec_ = [&](Real t, Vec x, Vec y) {
    return A_->Action(t, x, y);
  };

  dmatvec_ = [&](int is, Real t, Vec x, Vec y) {
    return A_->SensAction(is, t, x, y);
  };

  auto error_checking_fp = [&](PetscReal t, Vec p, int num_pars, Vec *dp, void *data) {
    return CheckFspTolerance_(t, p);
  };

  sens_solver_ = std::make_shared<ForwardSensCvodeFsp>(comm_);
  sens_solver_->SetRhs(matvec_);
  sens_solver_->SetSensRhs(dmatvec_);
  sens_solver_->SetStopCondition(error_checking_fp, nullptr);

  sinks_.set_size(state_set_->GetNumConstraints());
  to_expand_.set_size(sinks_.n_elem);
  return ierr;
}

const pacmensl::StateSetBase *pacmensl::SensFspSolverMultiSinks::GetStateSet() {
  return state_set_.get();
}

pacmensl::SensDiscreteDistribution pacmensl::SensFspSolverMultiSinks::Solve(PetscReal t_final, PetscReal fsp_tol) {
  PetscErrorCode ierr;
  if (p_ == nullptr) {
    ierr = VecCreate(comm_, &p_); PACMENSLCHKERRTHROW(ierr);
    ierr = VecSetSizes(p_, A_->GetNumLocalRows(), PETSC_DECIDE); PACMENSLCHKERRTHROW(ierr);
    ierr = VecSetType(p_, VECMPI); PACMENSLCHKERRTHROW(ierr);
    ierr = VecSetUp(p_); PACMENSLCHKERRTHROW(ierr);
  }
  ierr = VecSet(p_, 0.0); PACMENSLCHKERRTHROW(ierr);
  arma::Row<Int> indices = state_set_->State2Index(init_states_);
  ierr = VecSetValues(p_, PetscInt(init_probs_.n_elem), &indices[0], &init_probs_[0], INSERT_VALUES); PACMENSLCHKERRTHROW(ierr);
  ierr = VecAssemblyBegin(p_); PACMENSLCHKERRTHROW(ierr);
  ierr = VecAssemblyEnd(p_); PACMENSLCHKERRTHROW(ierr);
  if (dp_.empty()) {
    dp_.resize(model_.num_parameters_);
    for (int i{0}; i < model_.num_parameters_; ++i) {
      ierr = VecCreate(comm_, &dp_[i]); PACMENSLCHKERRTHROW(ierr);
      ierr = VecSetSizes(dp_[i], A_->GetNumLocalRows(), PETSC_DECIDE); PACMENSLCHKERRTHROW(ierr);
      ierr = VecSetType(dp_[i], VECMPI); PACMENSLCHKERRTHROW(ierr);
      ierr = VecSetUp(dp_[i]); PACMENSLCHKERRTHROW(ierr);
    }
  }
  for (int       i{0}; i < model_.num_parameters_; ++i) {
    ierr = VecSet(dp_[i], 0.0); PACMENSLCHKERRTHROW(ierr);
    indices = state_set_->State2Index(init_states_);
    ierr    = VecSetValues(dp_[i], PetscInt(init_probs_.n_elem), &indices[0], init_sens_[i].memptr(), INSERT_VALUES); PACMENSLCHKERRTHROW(ierr);
    ierr = VecAssemblyBegin(dp_[i]); PACMENSLCHKERRTHROW(ierr);
    ierr = VecAssemblyEnd(dp_[i]); PACMENSLCHKERRTHROW(ierr);
  }

  t_now_   = 0.0;
  t_final_ = t_final;
  SensDiscreteDistribution solution = Advance_(t_final, fsp_tol);

  return solution;
}

std::vector<pacmensl::SensDiscreteDistribution> pacmensl::SensFspSolverMultiSinks::SolveTspan(const std::vector<
    PetscReal> &tspan,
                                                                                              PetscReal fsp_tol) {
  PetscErrorCode ierr;
  if (p_ == nullptr) {
    ierr = VecCreate(comm_, &p_); PACMENSLCHKERRTHROW(ierr);
    ierr = VecSetSizes(p_, A_->GetNumLocalRows(), PETSC_DECIDE); PACMENSLCHKERRTHROW(ierr);
    ierr = VecSetType(p_, VECMPI); PACMENSLCHKERRTHROW(ierr);
    ierr = VecSetUp(p_); PACMENSLCHKERRTHROW(ierr);
  }
  ierr = VecSet(p_, 0.0); PACMENSLCHKERRTHROW(ierr);
  arma::Row<Int> indices = state_set_->State2Index(init_states_);
  ierr = VecSetValues(p_, PetscInt(init_probs_.n_elem), &indices[0], &init_probs_[0], INSERT_VALUES); PACMENSLCHKERRTHROW(ierr);
  ierr = VecAssemblyBegin(p_); PACMENSLCHKERRTHROW(ierr);
  ierr                                                  = VecAssemblyEnd(p_); PACMENSLCHKERRTHROW(ierr);
  if (dp_.empty()) {
    dp_.resize(model_.num_parameters_);
    for (int i{0}; i < model_.num_parameters_; ++i) {
      ierr = VecCreate(comm_, &dp_[i]); PACMENSLCHKERRTHROW(ierr);
      ierr = VecSetSizes(dp_[i], A_->GetNumLocalRows(), PETSC_DECIDE); PACMENSLCHKERRTHROW(ierr);
      ierr = VecSetType(dp_[i], VECMPI); PACMENSLCHKERRTHROW(ierr);
      ierr = VecSetUp(dp_[i]); PACMENSLCHKERRTHROW(ierr);
    }
  }
  for (int       i{0}; i < model_.num_parameters_; ++i) {
    ierr = VecSet(dp_[i], 0.0); PACMENSLCHKERRTHROW(ierr);
    indices = state_set_->State2Index(init_states_);
    ierr    = VecSetValues(dp_[i], PetscInt(init_sens_[i].n_elem), &indices[0], &init_sens_[i][0], INSERT_VALUES); PACMENSLCHKERRTHROW(ierr);
    ierr = VecAssemblyBegin(dp_[i]); PACMENSLCHKERRTHROW(ierr);
    ierr = VecAssemblyEnd(dp_[i]); PACMENSLCHKERRTHROW(ierr);
  }

  std::vector<SensDiscreteDistribution> outputs;
  int                                   num_time_points = tspan.size();
  outputs.resize(num_time_points);

  PetscReal t_max = tspan[num_time_points - 1];

  t_now_   = 0.0;
  t_final_ = t_max;
  for (int i = 0; i < num_time_points; ++i) {
    outputs[i] = Advance_(tspan[i], fsp_tol);
  }

  return outputs;
}

PacmenslErrorCode pacmensl::SensFspSolverMultiSinks::ClearState() {
  PacmenslErrorCode ierr;
  ierr = VecDestroy(&p_); CHKERRQ(ierr);
  for (auto i = dp_.begin(); i != dp_.end(); ++i) {
    ierr = VecDestroy(i.base()); CHKERRQ(ierr);
  }
  state_set_.reset();
  A_.reset();
  return 0;
}

pacmensl::SensFspSolverMultiSinks::~SensFspSolverMultiSinks() {
  ClearState();
  if (comm_) MPI_Comm_free(&comm_);
}

int pacmensl::SensFspSolverMultiSinks::CheckFspTolerance_(PetscReal t, Vec p) {
  int ierr;
  to_expand_.fill(0);
  // Find the sink states_
  arma::Row<PetscReal> sinks_of_p(sinks_.n_elem);
  if (my_rank_ == comm_size_ - 1) {
    const PetscReal *local_p_data;
    ierr = VecGetArrayRead(p, &local_p_data); PACMENSLCHKERRTHROW(ierr);
    int      n_loc = A_->GetNumLocalRows();
    for (int i{0}; i < sinks_of_p.n_elem; ++i) {
      sinks_of_p(i) = local_p_data[n_loc - sinks_.n_elem + i];
    }
    ierr = VecRestoreArrayRead(p, &local_p_data); PACMENSLCHKERRTHROW(ierr);
  } else {
    sinks_of_p.fill(0.0);
  }

  MPI_Datatype scalar_type;
  ierr = PetscDataTypeToMPIDataType(PETSC_DOUBLE, &scalar_type); PACMENSLCHKERRTHROW(ierr);
  ierr = MPI_Allreduce(&sinks_of_p[0], &sinks_[0], sinks_of_p.n_elem, scalar_type, MPIU_SUM, comm_); PACMENSLCHKERRTHROW(ierr);
  for (int i{0}; i < ( int ) sinks_.n_elem; ++i) {
    if (sinks_(i) / fsp_tol_ > (1.0 / double(sinks_.n_elem)) * (t / t_final_)) to_expand_(i) = 1;
  }
  return to_expand_.max();
}

pacmensl::SensDiscreteDistribution pacmensl::SensFspSolverMultiSinks::Advance_(PetscReal t_final, PetscReal fsp_tol) {
  PetscErrorCode ierr;
  PetscInt       solver_stat;
  Vec            P_tmp;

  if (verbosity_ > 1) sens_solver_->SetStatusOutput(1);

  fsp_tol_ = fsp_tol;
  sens_solver_->SetFinalTime(t_final);

  solver_stat = 1;
  while (solver_stat) {

    ierr = sens_solver_->SetInitialSolution(p_); PACMENSLCHKERRTHROW(ierr);
    ierr = sens_solver_->SetInitialSensitivity(dp_); PACMENSLCHKERRTHROW(ierr);
    ierr = sens_solver_->SetCurrentTime(t_now_); PACMENSLCHKERRTHROW(ierr);
    ierr = sens_solver_->SetUp(); PACMENSLCHKERRTHROW(ierr);

    solver_stat = sens_solver_->Solve();
    if (solver_stat != 0 && solver_stat != 1) PACMENSLCHKERRTHROW(solver_stat);
    t_now_ = sens_solver_->GetCurrentTime();
    ierr   = sens_solver_->FreeWorkspace(); PACMENSLCHKERRTHROW(ierr);
    // Expand the FspSolverBase if the solver halted prematurely
    if (solver_stat == 1) {
      for (auto           i{0}; i < to_expand_.n_elem; ++i) {
        if (to_expand_(i) == 1) {
          fsp_bounds_(i) = ( int ) std::round(
              double(fsp_bounds_(i)) * (fsp_expasion_factors_(i) + 1.0e0) + 0.5e0);
        }
      }

      if (verbosity_ > 0) {
        PetscPrintf(comm_, "\n ------------- \n");
        PetscPrintf(comm_, "At time t = %.2f expansion to new state_set_ size: \n",
                    sens_solver_->GetCurrentTime());
        for (auto i{0}; i < fsp_bounds_.n_elem; ++i) {
          PetscPrintf(comm_, "%d ", fsp_bounds_[i]);
        }
        PetscPrintf(comm_, "\n ------------- \n");
      }
      // Get local states_ corresponding to the current solution_
      arma::Mat<PetscInt> states_old = state_set_->CopyStatesOnProc();

      state_set_->SetShapeBounds(fsp_bounds_);
      state_set_->Expand();

      if (verbosity_) {
        PetscPrintf(comm_, "\n ------------- \n");
        PetscPrintf(comm_, "New Fsp number of states_: %d \n", state_set_->GetNumGlobalStates());
        PetscPrintf(comm_, "\n ------------- \n");
      }

      // free data of the ODE solver (they will be rebuilt at the beginning of the loop)
      A_->Destroy();
      A_->GenerateValues(*state_set_, model_);

      // Generate the expanded vector and scatter forward the current solution_
      ierr = VecCreate(comm_, &P_tmp); PACMENSLCHKERRTHROW(ierr);
      ierr = VecSetSizes(P_tmp, A_->GetNumLocalRows(), PETSC_DECIDE); PACMENSLCHKERRTHROW(ierr);
      ierr = VecSetUp(P_tmp); PACMENSLCHKERRTHROW(ierr);
      ierr = VecSet(P_tmp, 0.0); PACMENSLCHKERRTHROW(ierr);

      // Scatter from old vector to the expanded vector
      IS             new_locations;
      arma::Row<Int> new_states_locations;
      try {
        new_states_locations = state_set_->State2Index(states_old);
      }
      catch (...) { PACMENSLCHKERRTHROW(ierr);
      };
      arma::Row<Int> new_sinks_locations;
      if (my_rank_ == comm_size_ - 1) {
        new_sinks_locations.set_size(sinks_.n_elem);
        Int i_end_new;
        ierr = VecGetOwnershipRange(P_tmp, NULL, &i_end_new); PACMENSLCHKERRTHROW(ierr);
        for (auto i{0}; i < new_sinks_locations.n_elem; ++i) {
          new_sinks_locations[i] = i_end_new - (( Int ) new_sinks_locations.n_elem) + i;
        }
      } else {
        new_sinks_locations.set_size(0);
      }

      arma::Row<Int> new_locations_vals = arma::join_horiz(new_states_locations, new_sinks_locations);
      ierr = ISCreateGeneral(comm_, ( PetscInt ) new_locations_vals.n_elem, &new_locations_vals[0],
                             PETSC_COPY_VALUES, &new_locations); PACMENSLCHKERRTHROW(ierr);
      VecScatter scatter;
      ierr = VecScatterCreate(p_, NULL, P_tmp, new_locations, &scatter); PACMENSLCHKERRTHROW(ierr);

      ierr = VecScatterBegin(scatter, p_, P_tmp, INSERT_VALUES, SCATTER_FORWARD); PACMENSLCHKERRTHROW(ierr);
      ierr = VecScatterEnd(scatter, p_, P_tmp, INSERT_VALUES, SCATTER_FORWARD); PACMENSLCHKERRTHROW(ierr);
      ierr = VecDestroy(&p_); PACMENSLCHKERRTHROW(ierr);
      p_ = P_tmp;
      for (int i{0}; i < model_.num_parameters_; ++i) {
        ierr = VecCreate(comm_, &P_tmp); PACMENSLCHKERRTHROW(ierr);
        ierr = VecSetSizes(P_tmp, A_->GetNumLocalRows(), PETSC_DECIDE); PACMENSLCHKERRTHROW(ierr);
        ierr = VecSetUp(P_tmp); PACMENSLCHKERRTHROW(ierr);
        ierr = VecSet(P_tmp, 0.0); PACMENSLCHKERRTHROW(ierr);
        ierr = VecScatterBegin(scatter, dp_[i], P_tmp, INSERT_VALUES, SCATTER_FORWARD); PACMENSLCHKERRTHROW(ierr);
        ierr = VecScatterEnd(scatter, dp_[i], P_tmp, INSERT_VALUES, SCATTER_FORWARD); PACMENSLCHKERRTHROW(ierr);
        ierr = VecDestroy(&dp_[i]); PACMENSLCHKERRTHROW(ierr);
        dp_[i] = P_tmp;
      }
      ierr = ISDestroy(&new_locations); PACMENSLCHKERRTHROW(ierr);
      ierr = VecScatterDestroy(&scatter); PACMENSLCHKERRTHROW(ierr);
    }
  }

  SensDiscreteDistribution out;
  ierr = MakeSensDiscreteDistribution_(out); PACMENSLCHKERRTHROW(ierr);
  return out;
}

PacmenslErrorCode pacmensl::SensFspSolverMultiSinks::MakeSensDiscreteDistribution_(pacmensl::SensDiscreteDistribution &dist) {
  PacmenslErrorCode ierr;

  ierr = MPI_Comm_dup(comm_, &dist.comm_); CHKERRMPI(ierr);
  dist.t_      = t_now_;
  dist.states_ = state_set_->CopyStatesOnProc();

  ierr = VecCreate(dist.comm_, &dist.p_); CHKERRQ(ierr);
  ierr = VecSetSizes(dist.p_, state_set_->GetNumLocalStates(), PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetType(dist.p_, VECMPI); CHKERRQ(ierr);
  ierr                = VecSetUp(dist.p_); CHKERRQ(ierr);

  // Scatter solution to dist (omitting the sink states)
  IS         src_loc;
  VecScatter scatter;
  auto       src_indx = state_set_->State2Index(state_set_->GetStatesRef());
  ierr = ISCreateGeneral(comm_, src_indx.n_elem, &src_indx[0], PETSC_USE_POINTER, &src_loc); CHKERRQ(ierr);
  ierr = VecScatterCreate(p_, src_loc, dist.p_, NULL, &scatter); CHKERRQ(ierr);

  ierr = VecScatterBegin(scatter, p_, dist.p_, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter, p_, dist.p_, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

  dist.dp_.resize(dp_.size());
  for (int i{0}; i < dp_.size(); ++i) {
    ierr = VecDuplicate(dist.p_, &dist.dp_[i]); CHKERRQ(ierr);
    ierr = VecScatterBegin(scatter, dp_[i], dist.dp_[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(scatter, dp_[i], dist.dp_[i], INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  }

  ISDestroy(&src_loc);
  VecScatterDestroy(&scatter);
  return 0;
}
