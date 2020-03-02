//
// Created by Huy Vo on 5/29/18.
//

#include <OdeSolver/TsFsp.h>
#include <OdeSolver/CvodeFsp.h>
#include "FspSolverMultiSinks.h"
//#include "OdeSolverBase.h"

namespace pacmensl {
FspSolverMultiSinks::FspSolverMultiSinks(MPI_Comm _comm, PartitioningType _part_type, ODESolverType _solve_type)
{
  int ierr;
  comm_ = _comm;
  ierr = MPI_Comm_rank(comm_, &my_rank_);
  PACMENSLCHKERRTHROW(ierr);
  ierr = MPI_Comm_size(comm_, &comm_size_);
  PACMENSLCHKERRTHROW(ierr);
  partitioning_type_ = _part_type;
  odes_type_         = _solve_type;
}

PacmenslErrorCode FspSolverMultiSinks::SetInitialBounds(arma::Row<int> &_fsp_size)
{
  fsp_bounds_ = _fsp_size;
  return 0;
}

PacmenslErrorCode FspSolverMultiSinks::SetConstraintFunctions(const fsp_constr_multi_fn &lhs_constr, void *args)
{
  fsp_constr_funs_         = lhs_constr;
  fsp_constr_args_         = args;
  have_custom_constraints_ = true;
  return 0;
}

PacmenslErrorCode FspSolverMultiSinks::SetExpansionFactors(arma::Row<PetscReal> &_expansion_factors)
{
  fsp_expasion_factors_ = _expansion_factors;
  return 0;
}

DiscreteDistribution FspSolverMultiSinks::Advance_(PetscReal t_final, PetscReal fsp_tol)
{
  PetscErrorCode ierr;
  PetscInt       solver_stat;

  if (logging_enabled) PACMENSLCHKERRTHROW(PetscLogEventBegin(Solving, 0, 0, 0, 0));

  if (verbosity_ > 1) ode_solver_->SetStatusOutput(1);

  fsp_tol_ = fsp_tol;
  ode_solver_->SetFinalTime(t_final);
  ode_solver_->SetTolerances(ode_rtol_, ode_atol_);

  ode_solver_->SetRhs(this->tmatvec_);
  if (fsp_tol_ > 0.0)
  {
    auto error_checking_fp = [&](PetscReal t, Vec p, PetscReal &te, void *data) {
      return CheckFspTolerance_(t, p, te);
    };
    ode_solver_->SetStopCondition(error_checking_fp, nullptr);
  } else
  {
    ode_solver_->SetStopCondition(nullptr, nullptr);
  }

  solver_stat = 1;
  while (solver_stat)
  {
    if (logging_enabled) PACMENSLCHKERRTHROW(PetscLogEventBegin(ODESolve, 0, 0, 0, 0));

    ierr = ode_solver_->SetInitialSolution(p_->mem());
    PACMENSLCHKERRTHROW(ierr);

    ierr = ode_solver_->SetCurrentTime(t_now_);
    PACMENSLCHKERRTHROW(ierr);

    ode_solver_->SetTolerances(ode_rtol_, ode_atol_);

    ierr = ode_solver_->SetUp();
    PACMENSLCHKERRTHROW(ierr);

    to_expand_.fill(0);

    solver_stat = ode_solver_->Solve();
    if (solver_stat != 0 && solver_stat != 1) PACMENSLCHKERRTHROW(solver_stat);

    ierr = ode_solver_->FreeWorkspace();
    PACMENSLCHKERRTHROW(ierr);
    if (logging_enabled) PACMENSLCHKERRTHROW(PetscLogEventEnd(ODESolve, 0, 0, 0, 0));


    // Expand the FspSolverBase if the solver halted prematurely
    if (solver_stat == 1)
    {
      for (auto           i{0}; i < to_expand_.n_elem; ++i)
      {
        if (to_expand_(i) == 1)
        {
          fsp_bounds_(i) = ( int ) std::round(
              double(fsp_bounds_(i)) * (fsp_expasion_factors_(i) + 1.0e0) + 0.5e0);
        }
      }

      if (verbosity_)
      {
        PetscPrintf(comm_, "\n ------------- \n");
        PetscPrintf(comm_, "At time t = %.2f expansion to new state_set_ size: \n",
                    ode_solver_->GetCurrentTime());
        for (auto i{0}; i < fsp_bounds_.n_elem; ++i)
        {
          PetscPrintf(comm_, "%d ", fsp_bounds_[i]);
        }
        PetscPrintf(comm_, "\n ------------- \n");
      }
      // Get local states_ corresponding to the current solution_
      arma::Mat<PetscInt> states_old = state_set_->CopyStatesOnProc();
      if (logging_enabled)
      {
        PACMENSLCHKERRTHROW(PetscLogEventBegin(StateSetPartitioning, 0, 0, 0, 0));
      }
      state_set_->SetShapeBounds(fsp_bounds_);
      state_set_->Expand();
      if (logging_enabled)
      {
        PACMENSLCHKERRTHROW(PetscLogEventEnd(StateSetPartitioning, 0, 0, 0, 0));
      }
      if (verbosity_)
      {
        PetscPrintf(comm_, "\n ------------- \n");
        PetscPrintf(comm_, "New Fsp number of states_: %d \n", state_set_->GetNumGlobalStates());
        PetscPrintf(comm_, "\n ------------- \n");
      }

      // free data of the ODE solver (they will be rebuilt at the beginning of the loop)
      A_->Destroy();
      if (logging_enabled)
      {
        PACMENSLCHKERRTHROW(PetscLogEventBegin(MatrixGeneration, 0, 0, 0, 0));
      }
      A_->GenerateValues(*state_set_, model_.stoichiometry_matrix_, model_.tv_reactions_, model_.prop_t_,
                         model_.prop_x_, std::vector<int>(), model_.prop_t_args_, model_.prop_x_args_);
      if (logging_enabled)
      {
        PACMENSLCHKERRTHROW(PetscLogEventEnd(MatrixGeneration, 0, 0, 0, 0));
      }

      // Generate the expanded vector and scatter forward the current solution_
      if (logging_enabled)
      {
        PACMENSLCHKERRTHROW(PetscLogEventBegin(SolutionScatter, 0, 0, 0, 0));
      }

      arma::Row<Int> new_states_locations;
      try
      {
        new_states_locations = state_set_->State2Index(states_old);
      }
      catch (...)
      {
        ierr = -1;
        PACMENSLCHKERRTHROW(ierr);
      }

      arma::Row<Int> new_sinks_locations;
      if (my_rank_ == comm_size_ - 1)
      {
        new_sinks_locations.set_size(sinks_.n_elem);
        Int i_end_new;
        i_end_new = state_set_->GetNumGlobalStates() + sinks_.n_elem;
        for (auto i{0}; i < new_sinks_locations.n_elem; ++i)
        {
          new_sinks_locations[i] = i_end_new - (( Int ) new_sinks_locations.n_elem) + i;
        }
      } else
      {
        new_sinks_locations.set_size(0);
      }

      std::vector<PetscInt> new_locations_vals = arma::conv_to<std::vector<PetscInt>>::from(arma::join_horiz(
          new_states_locations,
          new_sinks_locations));

      ierr = ExpandVec(*p_, new_locations_vals, A_->GetNumLocalRows());
      PACMENSLCHKERRTHROW(ierr);

      if (logging_enabled)
      {
        PACMENSLCHKERRTHROW(PetscLogEventEnd(SolutionScatter, 0, 0, 0, 0));
      }
    }
    t_now_ = ode_solver_->GetCurrentTime();
  }

  if (logging_enabled)
  {
    PACMENSLCHKERRTHROW(PetscLogEventEnd(Solving, 0, 0, 0, 0));
  }

  DiscreteDistribution dist;
  ierr = MakeDiscreteDistribution_(dist);
  PACMENSLCHKERRTHROW(ierr);
  return dist;
}

FspSolverMultiSinks::~FspSolverMultiSinks()
{
  ClearState();
  comm_ = nullptr;
}

PacmenslErrorCode FspSolverMultiSinks::ClearState()
{
  int ierr;

  set_up_ = false;
  ode_solver_.reset();
  p_.reset();
  A_.reset();
  state_set_.reset();
  sinks_.clear();
  to_expand_.clear();

  have_custom_constraints_ = false;
  fsp_constr_args_ = nullptr;
  fsp_constr_funs_         = nullptr;
  tmatvec_                 = nullptr;
  return 0;
}

PacmenslErrorCode FspSolverMultiSinks::SetUp()
{
  int ierr{0};
  // Make sure all the necessary parameters have been set
  try
  {
    if ( (model_.prop_t_ == nullptr) && (!model_.tv_reactions_.empty()))
    {
      throw std::runtime_error("Model has time-varying propensitites but temporal signals were not set before calling FspSolver.SetUp().");
    }
    if (model_.prop_x_ == nullptr)
    {
      throw std::runtime_error("Propensity was not set before calling FspSolver.SetUp().");
    }
    if (model_.stoichiometry_matrix_.n_elem == 0)
    {
      throw std::runtime_error("Empty stoichiometry matrix cannot be used for FspSolver.");
    }
    if (init_states_.n_elem == 0 || init_probs_.n_elem == 0)
    {
      throw std::runtime_error("Initial states and/or probabilities were not set before calling FspSolver.SetUp().");
    }
  } catch (std::runtime_error &e)
  {
    PetscPrintf(comm_, "\n %s \n", e.what());
    ierr = -1;
  }
  PACMENSLCHKERRQ(ierr);

  // Register events if logging is needed
  if (logging_enabled)
  {
    ierr = PetscLogDefaultBegin();
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("Finite state subset partitioning", 0, &StateSetPartitioning);
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("Generate Fsp matrices", 0, &MatrixGeneration);
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("Advance reduced problem", 0, &ODESolve);
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("Fsp RHS evaluation", 0, &RHSEvaluation);
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("Fsp Solution scatter", 0, &SolutionScatter);
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("Fsp Set-up", 0, &SettingUp);
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("Fsp Solving total", 0, &Solving);
    CHKERRQ(ierr);
    ierr = PetscLogEventBegin(SettingUp, 0, 0, 0, 0);
    CHKERRQ(ierr);
  }

  if (!state_set_)
  {
    state_set_ = std::make_shared<StateSetConstrained>(comm_);
    state_set_->SetStoichiometryMatrix(model_.stoichiometry_matrix_);
    if (have_custom_constraints_)
    {
      state_set_->SetShape(fsp_constr_funs_, fsp_bounds_, fsp_constr_args_);
    } else
    {
      state_set_->SetShapeBounds(fsp_bounds_);
    }
    state_set_->SetLoadBalancingScheme(partitioning_type_);
    state_set_->SetUp();
    state_set_->AddStates(init_states_);
    if (logging_enabled)
    {
      ierr = PetscLogEventBegin(StateSetPartitioning, 0, 0, 0, 0);
      PACMENSLCHKERRTHROW(ierr);
    }
    ierr       = state_set_->Expand();
    PACMENSLCHKERRQ(ierr);
    if (logging_enabled)
    {
      ierr = PetscLogEventEnd(StateSetPartitioning, 0, 0, 0, 0);
      PACMENSLCHKERRTHROW(ierr);
    }
  }

  if (!A_)
  {
    A_ = std::make_shared<FspMatrixConstrained>(comm_);
    if (logging_enabled)
    {
      CHKERRQ(PetscLogEventBegin(MatrixGeneration, 0, 0, 0, 0));
    }
    ierr = A_
        ->GenerateValues(*state_set_,
                         model_.stoichiometry_matrix_, model_.tv_reactions_,
                         model_.prop_t_,
                         model_.prop_x_, std::vector<int>(),
                         model_.prop_t_args_,
                         model_.prop_x_args_);
    PACMENSLCHKERRQ(ierr);
    if (logging_enabled)
    {
      CHKERRQ(PetscLogEventEnd(MatrixGeneration, 0, 0, 0, 0));
    }
    if (logging_enabled)
    {
      tmatvec_ = [&](Real t, Vec x, Vec y) {
        int ierr;
        CHKERRQ(PetscLogEventBegin(RHSEvaluation, 0, 0, 0, 0));
        ierr = A_->Action(t, x, y);
        CHKERRQ(PetscLogEventEnd(RHSEvaluation, 0, 0, 0, 0));
        return ierr;
      };
    } else
    {
      tmatvec_ = [&](Real t, Vec x, Vec y) {
        return A_->Action(t, x, y);
      };
    }
  }
  A_->SetTimeFun(model_.prop_t_, model_.prop_t_args_);

  if (!p_)
  {
    p_   = std::make_shared<Petsc<Vec>>();
    ierr = VecCreate(comm_, p_->mem());
    PACMENSLCHKERRTHROW(ierr);
    ierr = VecSetSizes(*p_, A_->GetNumLocalRows(), PETSC_DECIDE);
    PACMENSLCHKERRTHROW(ierr);
    ierr = VecSetType(*p_, VECMPI);
    PACMENSLCHKERRTHROW(ierr);
    ierr = VecSetUp(*p_);
    PACMENSLCHKERRTHROW(ierr);
  }

  if (!ode_solver_)
  {
    switch (odes_type_)
    {
      case CVODE:ode_solver_ = std::make_shared<CvodeFsp>(comm_);
        break;
      case KRYLOV:ode_solver_ = std::make_shared<KrylovFsp>(comm_);
        if (custom_krylov) ((KrylovFsp*) ode_solver_.get())->SetOrthLength(q_iop);
        break;
//      case EPIC:ode_solver_ = std::make_shared<EpicFsp>(comm_, model_.stoichiometry_matrix_.n_cols);
//        break;
      default:ode_solver_ = std::make_shared<TsFsp>(comm_);
        if (custom_ts_type) ((TsFsp*) ode_solver_.get())->SetTsType(ts_type_.c_str());
    }
    ode_solver_->SetFspMatPtr(A_.get());
    if (logging_enabled)
    {
      ode_solver_->EnableLogging();
      ierr = PetscLogEventEnd(SettingUp, 0, 0, 0, 0);
      CHKERRQ(ierr);
    }
  }

  sinks_.set_size(state_set_->GetNumConstraints());
  to_expand_.set_size(sinks_.n_elem);

  set_up_ = true;
  return ierr;
}

PacmenslErrorCode FspSolverMultiSinks::SetVerbosity(int verbosity_level)
{
  verbosity_ = verbosity_level;
  return 0;
}

PacmenslErrorCode FspSolverMultiSinks::SetInitialDistribution(const arma::Mat<Int> &_init_states,
                                                              const arma::Col<PetscReal> &_init_probs)
{
  init_states_ = _init_states;
  init_probs_  = _init_probs;
  if (init_probs_.n_elem != init_states_.n_cols)
  {
    return -1;
  }
  return 0;
}

PacmenslErrorCode FspSolverMultiSinks::SetInitialDistribution(DiscreteDistribution &init_dist)
{
  int ierr;
  int n_states, n_species;
  int* state_ptr;
  PetscReal* prob_ptr;

  ierr = init_dist.GetStateView(n_states, n_species, state_ptr);PACMENSLCHKERRQ(ierr);

  init_states_ = arma::Mat<PetscInt>(state_ptr, n_species, n_states);

  init_dist.GetProbView(n_states, prob_ptr);
  init_probs_ = arma::Col<PetscReal>(prob_ptr, n_states);
  init_dist.RestoreProbView(prob_ptr);

  if (init_probs_.n_elem != init_states_.n_cols)
  {
    return -1;
  }

  return 0;
}


std::shared_ptr<const StateSetBase> FspSolverMultiSinks::GetStateSet()
{
  return state_set_;
}

PacmenslErrorCode FspSolverMultiSinks::SetLogging(PetscBool logging)
{
  logging_enabled = logging;
  return 0;
}

FspSolverComponentTiming FspSolverMultiSinks::ReduceComponentTiming(char *op)
{
  PetscMPIInt comm_size;
  MPI_Comm_size(comm_, &comm_size);

  MPI_Op opt;
  if (strcmp(op, "sum")==0){
    opt = MPIU_SUM;
  }
  else if (strcmp(op, "min")==0){
    opt = MPIU_MIN;
  }
  else{
    opt = MPIU_MAX;
  }

  auto reduce_timing = [&](PetscLogEvent event) {
    PetscReal          timing;
    PetscReal          tmp;
    PetscEventPerfInfo info;
    int                ierr = PetscLogEventGetPerfInfo(PETSC_DETERMINE, event, &info);
    CHKERRABORT(comm_, ierr);
    tmp = info.time;

    MPI_Allreduce(&tmp, &timing, 1, MPIU_REAL, opt, comm_);
    return timing;
  };

  FspSolverComponentTiming timings;
  timings.MatrixGenerationTime  = reduce_timing(MatrixGeneration);
  timings.StatePartitioningTime = reduce_timing(StateSetPartitioning);
  timings.ODESolveTime          = reduce_timing(ODESolve);
  timings.RHSEvalTime           = reduce_timing(RHSEvaluation);
  timings.SolutionScatterTime   = reduce_timing(SolutionScatter);
  timings.TotalTime             = reduce_timing(SettingUp) + reduce_timing(Solving);
  return timings;
}

FiniteProblemSolverPerfInfo FspSolverMultiSinks::GetSolverPerfInfo()
{
  return ode_solver_->GetAvgPerfInfo();
}

PacmenslErrorCode FspSolverMultiSinks::SetFromOptions()
{
  PetscErrorCode ierr;
  char           opt[100];
  PetscMPIInt    num_procs;
  PetscBool      opt_set;

  MPI_Comm_size(comm_, &num_procs);

  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_partitioning_type", opt, 100, &opt_set);
  CHKERRQ(ierr);
  if (opt_set)
  {
    partitioning_type_ = str2part(std::string(opt));
  }
  if (num_procs == 1)
  {
    partitioning_type_ = PartitioningType::GRAPH;
  }

  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_repart_approach", opt, 100, &opt_set);
  CHKERRQ(ierr);
  if (opt_set)
  {
    repart_approach_ = str2partapproach(std::string(opt));
  }

  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_verbosity", opt, 100, &opt_set);
  CHKERRQ(ierr);
  if (opt_set)
  {
    if (strcmp(opt, "1") == 0 || strcmp(opt, "true") == 0)
    {
      verbosity_ = 1;
    }
    if (strcmp(opt, "2") == 0)
    {
      verbosity_ = 2;
    }
  }

  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_log_events", opt, 100, &opt_set);
  CHKERRQ(ierr);
  if (opt_set)
  {
    if (strcmp(opt, "1") == 0 || strcmp(opt, "true") == 0)
    {
      logging_enabled = PETSC_TRUE;
    }
  }
  return 0;
}

PacmenslErrorCode FspSolverMultiSinks::CheckFspTolerance_(PetscReal t, Vec p, PetscReal &tol_exceed)
{
  int ierr;
  tol_exceed = 0.0;
//  if (fsp_tol_ < 0.0) return 0;
  // Find the sink states_
  arma::Row<PetscReal> sinks_of_p(sinks_.n_elem);
  if (my_rank_ == comm_size_ - 1)
  {
    const PetscReal *local_p_data;
    ierr = VecGetArrayRead(p, &local_p_data);
    PACMENSLCHKERRTHROW(ierr);
    int n_loc;
    VecGetLocalSize(p, &n_loc);
    for (int i{0}; i < sinks_of_p.n_elem; ++i)
    {
      sinks_of_p(i) = local_p_data[n_loc - sinks_.n_elem + i];
    }
    ierr = VecRestoreArrayRead(p, &local_p_data);
    PACMENSLCHKERRTHROW(ierr);
  } else
  {
    sinks_of_p.fill(0.0);
  }
  ierr       = MPI_Allreduce(&sinks_of_p[0], &sinks_[0], sinks_of_p.n_elem, MPIU_REAL, MPIU_SUM, comm_);
  PACMENSLCHKERRTHROW(ierr);
  for (int i{0}; i < ( int ) sinks_.n_elem; ++i)
  {
    if (sinks_(i) / fsp_tol_ >= (1.0 / double(sinks_.n_elem)) * (t / t_final_))
    {
      to_expand_(i) = 1;
      tol_exceed = std::max(tol_exceed, sinks_(i) * double(sinks_.n_elem) - fsp_tol_ * (t / t_final_));
    }
  }
  return 0;
}

PacmenslErrorCode FspSolverMultiSinks::SetModel(Model &model)
{
  FspSolverMultiSinks::model_ = model;
  return 0;
}

DiscreteDistribution FspSolverMultiSinks::Solve(PetscReal t_final, PetscReal fsp_tol, PetscReal t_init)
{
  PetscErrorCode ierr;
  if (!set_up_)
  {
    ierr = SetUp();
    PACMENSLCHKERRTHROW(ierr);
  }

  ierr = VecSet(*p_, 0.0);
  PACMENSLCHKERRTHROW(ierr);
  arma::Row<Int> indices = state_set_->State2Index(init_states_);
  ierr = VecSetValues(*p_, PetscInt(init_probs_.n_elem), &indices[0], &init_probs_[0], INSERT_VALUES);
  PACMENSLCHKERRTHROW(ierr);
  ierr = VecAssemblyBegin(*p_);
  PACMENSLCHKERRTHROW(ierr);
  ierr = VecAssemblyEnd(*p_);
  PACMENSLCHKERRTHROW(ierr);

  t_now_   = t_init;
  t_final_ = t_final;
  DiscreteDistribution solution = FspSolverMultiSinks::Advance_(t_final, fsp_tol);

  return solution;
}

std::vector<DiscreteDistribution>
FspSolverMultiSinks::SolveTspan(const std::vector<PetscReal> &tspan, PetscReal fsp_tol, PetscReal t_init)
{
  PetscErrorCode ierr;
  if (!set_up_)
  {
    ierr = SetUp();
    PACMENSLCHKERRTHROW(ierr);
  }

  ierr = VecSet(*p_, 0.0);
  PACMENSLCHKERRTHROW(ierr);
  arma::Row<Int> indices = state_set_->State2Index(init_states_);
  ierr = VecSetValues(*p_, PetscInt(init_probs_.n_elem), &indices[0], &init_probs_[0], INSERT_VALUES);
  PACMENSLCHKERRTHROW(ierr);
  ierr = VecAssemblyBegin(*p_);
  PACMENSLCHKERRTHROW(ierr);
  ierr                                              = VecAssemblyEnd(*p_);
  PACMENSLCHKERRTHROW(ierr);

  std::vector<DiscreteDistribution> outputs;
  int                               num_time_points = tspan.size();
  outputs.resize(num_time_points);

  PetscReal t_max = tspan[num_time_points - 1];

  DiscreteDistribution sol;

  t_now_   = t_init;
  t_final_ = t_max;
  for (int i = 0; i < num_time_points; ++i)
  {
    sol = FspSolverMultiSinks::Advance_(tspan[i], fsp_tol);
    outputs[i] = sol;
  }

  return outputs;
}

PacmenslErrorCode FspSolverMultiSinks::SetOdesType(ODESolverType odes_type)
{
  odes_type_ = odes_type;
  return 0;
}

PacmenslErrorCode FspSolverMultiSinks::SetLoadBalancingMethod(PartitioningType part_type)
{
  partitioning_type_ = part_type;
  return 0;
}

PacmenslErrorCode FspSolverMultiSinks::SetOdeTolerances(PetscReal rel_tol, PetscReal abs_tol)
{
  ode_rtol_ = rel_tol;
  ode_atol_ = abs_tol;
  return 0;
}

PacmenslErrorCode FspSolverMultiSinks::MakeDiscreteDistribution_(DiscreteDistribution &dist)
{
  PacmenslErrorCode ierr;

  dist.comm_ = comm_;
  dist.t_      = t_now_;
  dist.states_ = state_set_->CopyStatesOnProc();

  ierr = VecCreate(dist.comm_, &dist.p_);
  CHKERRQ(ierr);
  ierr = VecSetSizes(dist.p_, state_set_->GetNumLocalStates(), PETSC_DECIDE);
  CHKERRQ(ierr);
  ierr                = VecSetUp(dist.p_);
  CHKERRQ(ierr);

  // Scatter solution to dist1 (omitting the sink states)
  IS         src_loc;
  VecScatter scatter;
  auto       src_indx = state_set_->State2Index(state_set_->GetStatesRef());
  ierr = ISCreateGeneral(comm_, src_indx.n_elem, &src_indx[0], PETSC_USE_POINTER, &src_loc);
  CHKERRQ(ierr);
  ierr = VecScatterCreate(*p_, src_loc, dist.p_, NULL, &scatter);
  CHKERRQ(ierr);

  ierr = VecScatterBegin(scatter, *p_, dist.p_, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter, *p_, dist.p_, INSERT_VALUES, SCATTER_FORWARD);
  CHKERRQ(ierr);

  ISDestroy(&src_loc);
  VecScatterDestroy(&scatter);
  return 0;
}

std::shared_ptr<OdeSolverBase> FspSolverMultiSinks::GetOdeSolver()
{
  return ode_solver_;
}

PacmenslErrorCode FspSolverMultiSinks::SetOdesPetscType(std::string ts_type)
{
  if (odes_type_ != PETSC){
    return 0;
  }
  if (ode_solver_ != nullptr){
    ((TsFsp*) ode_solver_.get())->SetTsType(ts_type);
  }
  else{
    custom_ts_type = true;
    ts_type_ = std::string(ts_type);
  }
  return 0;
}


PacmenslErrorCode FspSolverMultiSinks::SetKrylovOrthLength(int q)
{
  if (odes_type_ != KRYLOV){
    return 0;
  }
  if (ode_solver_ != nullptr){
    ((KrylovFsp*) ode_solver_.get())->SetOrthLength(q);
  }
  else{
    custom_krylov = true;
    q_iop = q;
  }
  return 0;
}

}