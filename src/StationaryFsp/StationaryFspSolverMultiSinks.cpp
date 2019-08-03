//
// Created by Huy Vo on 2019-06-25.
//

#include "StationaryFspSolverMultiSinks.h"

pacmensl::StationaryFspSolverMultiSinks::StationaryFspSolverMultiSinks(MPI_Comm comm)
{
  int ierr;
  ierr = MPI_Comm_dup(comm, &comm_); PACMENSLCHKERRTHROW(ierr);
}

PacmenslErrorCode pacmensl::StationaryFspSolverMultiSinks::SetConstraintFunctions(const fsp_constr_multi_fn &lhs_constr,
                                                                                  void *args)
{
  fsp_constr_funs_         = lhs_constr;
  fsp_constr_args_ = args;
  have_custom_constraints_ = true;
  return 0;
}

int pacmensl::StationaryFspSolverMultiSinks::SetInitialBounds(arma::Row<int> &_fsp_size)
{
  fsp_bounds_ = _fsp_size;
  return 0;
}

int pacmensl::StationaryFspSolverMultiSinks::SetExpansionFactors(arma::Row<PetscReal> &_expansion_factors)
{
  fsp_expasion_factors_ = _expansion_factors;
  return 0;
}

int pacmensl::StationaryFspSolverMultiSinks::SetModel(pacmensl::Model &model)
{
  model_ = model;
  return 0;
}

int pacmensl::StationaryFspSolverMultiSinks::SetVerbosity(int verbosity_level)
{
  verbosity_ = verbosity_level;
  return 0;
}

PacmenslErrorCode pacmensl::StationaryFspSolverMultiSinks::SetUp()
{
  if (set_up_) return 0;
  PacmenslErrorCode ierr{0};
  // Make sure all the necessary parameters have been set
  try
  {
    if (model_.prop_t_ == nullptr)
    {
      throw std::runtime_error("Temporal signal was not set before calling FspSolver.SetUp().");
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
  } PACMENSLCHKERRQ(ierr);

  state_set_ = std::unique_ptr<StateSetConstrained>(new StateSetConstrained(comm_));
  state_set_->SetStoichiometryMatrix(model_.stoichiometry_matrix_);
  if (have_custom_constraints_)
  {
    state_set_->SetShape(fsp_constr_funs_, fsp_bounds_, fsp_constr_args_);
  } else
  {
    state_set_->SetShapeBounds(fsp_bounds_);
  }
  state_set_->SetUp();
  state_set_->AddStates(init_states_);
  ierr = state_set_->Expand(); PACMENSLCHKERRQ(ierr);

  matrix_ = std::unique_ptr<StationaryFspMatrixConstrained>(new StationaryFspMatrixConstrained(comm_));
  ierr    = matrix_->GenerateValues(*state_set_,
                                    model_.stoichiometry_matrix_,
                                    model_.prop_t_,
                                    model_.prop_x_,
                                    std::vector<int>(),
                                    model_.prop_t_args_,
                                    model_.prop_x_args_); PACMENSLCHKERRQ(ierr);

  matvec_ = [&](Vec x, Vec y) {
    return matrix_->Action(0.0, x, y);
  };
  solver_ = std::unique_ptr<StationaryMCSolver>(new StationaryMCSolver(comm_));
  ierr    = solver_->SetMatVec(matvec_); PACMENSLCHKERRQ(ierr);
  ierr = solver_->SetMatDiagonal(&matrix_->diagonal_); PACMENSLCHKERRQ(ierr);

  sinks_.set_size(state_set_->GetNumConstraints());
  to_expand_.set_size(state_set_->GetNumConstraints());

  set_up_ = true;
  return 0;
}

int pacmensl::StationaryFspSolverMultiSinks::SetInitialDistribution(const arma::Mat<pacmensl::Int> &_init_states,
                                                                    const arma::Col<PetscReal> &_init_probs)
{
  init_states_ = _init_states;
  init_probs_  = _init_probs;
  return 0;
}

int pacmensl::StationaryFspSolverMultiSinks::SetLoadBalancingMethod(pacmensl::PartitioningType part_type)
{
  partitioning_type_ = part_type;
  return 0;
}

pacmensl::DiscreteDistribution pacmensl::StationaryFspSolverMultiSinks::Solve(PetscReal sfsp_tol)
{
  PetscErrorCode ierr;
  if (!set_up_) SetUp();

  if (solution_ == nullptr)
  {
    ierr = VecCreate(comm_, &solution_); PACMENSLCHKERRTHROW(ierr);
    ierr = VecSetSizes(solution_, matrix_->GetNumLocalRows(), PETSC_DECIDE); PACMENSLCHKERRTHROW(ierr);
    ierr = VecSetType(solution_, VECMPI); PACMENSLCHKERRTHROW(ierr);
    ierr = VecSetUp(solution_); PACMENSLCHKERRTHROW(ierr);
  }
  ierr = VecSet(solution_, 0.0); PACMENSLCHKERRTHROW(ierr);
  arma::Row<Int> indices = state_set_->State2Index(init_states_);
  ierr = VecSetValues(solution_, PetscInt(init_probs_.n_elem), &indices[0], &init_probs_[0], INSERT_VALUES); PACMENSLCHKERRTHROW(ierr);
  ierr = VecAssemblyBegin(solution_); PACMENSLCHKERRTHROW(ierr);
  ierr = VecAssemblyEnd(solution_); PACMENSLCHKERRTHROW(ierr);

  while (true)
  {
    ierr = solver_->SetSolutionVec(&solution_); PACMENSLCHKERRTHROW(ierr);
    ierr = solver_->SetMatDiagonal(&matrix_->diagonal_); PACMENSLCHKERRTHROW(ierr);
    ierr = solver_->SetUp(); PACMENSLCHKERRTHROW(ierr);
    ierr = solver_->Solve(); PACMENSLCHKERRTHROW(ierr);

    ierr = matrix_->EvaluateOutflows(solution_, sinks_); PACMENSLCHKERRTHROW(ierr);
    to_expand_.fill(0);
    for (int i = 0; i < to_expand_.n_elem; ++i)
    {
      if (sinks_(i) > sfsp_tol)
      {
        to_expand_(i)  = 1;
        fsp_bounds_(i) = ( int ) std::round(
            double(fsp_bounds_(i)) * (fsp_expasion_factors_(i) + 1.0e0) + 0.5e0);
      }
    }
    if (to_expand_.max() == 0)
    {
      break;
    }

    // Get local states_ corresponding to the current solution_
    arma::Mat<PetscInt> states_old = state_set_->CopyStatesOnProc();
    state_set_->SetShapeBounds(fsp_bounds_);
    state_set_->Expand();
    std::vector<PetscInt> new_states_locations;
    try
    {
      new_states_locations = arma::conv_to<std::vector<PetscInt>>::from(state_set_->State2Index(states_old));
    }
    catch (...)
    { PACMENSLCHKERRTHROW(ierr);
    }

    if (verbosity_)
    {
      PetscPrintf(comm_, "\n ------------- \n");
      PetscPrintf(comm_, "New Fsp number of states_: %d \n", state_set_->GetNumGlobalStates());
      PetscPrintf(comm_, "\n ------------- \n");
    }

    ierr = matrix_->Destroy(); PACMENSLCHKERRTHROW(ierr);
    ierr = matrix_->GenerateValues(*state_set_,
                                   model_.stoichiometry_matrix_,
                                   model_.prop_t_,
                                   model_.prop_x_,
                                   std::vector<int>(),
                                   model_.prop_t_args_,
                                   model_.prop_x_args_); PACMENSLCHKERRTHROW(ierr);

    ExpandVec(solution_, new_states_locations, matrix_->GetNumLocalRows());
  }

  pacmensl::DiscreteDistribution output;
  ierr = MakeDiscreteDistribution_(output); PACMENSLCHKERRTHROW(ierr);
  return output;
}

int pacmensl::StationaryFspSolverMultiSinks::ClearState()
{
  int ierr;
  if (solution_)
  {
    ierr = VecDestroy(&solution_); CHKERRQ(ierr);
  }
  set_up_ = false;
  return 0;
}

pacmensl::StationaryFspSolverMultiSinks::~StationaryFspSolverMultiSinks()
{
  ClearState();
  MPI_Comm_free(&comm_);
}

PacmenslErrorCode pacmensl::StationaryFspSolverMultiSinks::MakeDiscreteDistribution_(DiscreteDistribution &dist)
{
  PacmenslErrorCode ierr;

  ierr = MPI_Comm_dup(comm_, &dist.comm_); CHKERRMPI(ierr);
  dist.t_      = 0.0;
  dist.states_ = state_set_->CopyStatesOnProc();

  ierr = VecCreate(dist.comm_, &dist.p_); CHKERRQ(ierr);
  ierr = VecSetSizes(dist.p_, state_set_->GetNumLocalStates(), PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetType(dist.p_, VECMPI); CHKERRQ(ierr);
  ierr                       = VecSetUp(dist.p_); CHKERRQ(ierr);

  // Scatter solution to dist1 (omitting the sink states)
  Petsc<IS>         src_loc;
  Petsc<VecScatter> scatter;
  auto              src_indx = state_set_->State2Index(state_set_->GetStatesRef());
  ierr = ISCreateGeneral(comm_, src_indx.n_elem, &src_indx[0], PETSC_USE_POINTER, src_loc.mem()); CHKERRQ(ierr);
  ierr = VecScatterCreate(solution_, src_loc, dist.p_, NULL, scatter.mem()); CHKERRQ(ierr);

  ierr = VecScatterBegin(scatter, solution_, dist.p_, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  ierr = VecScatterEnd(scatter, solution_, dist.p_, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
  return 0;
}