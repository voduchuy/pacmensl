//
// Created by Huy Vo on 2019-06-12.
//

#include "KrylovFsp.h"

pacmensl::KrylovFsp::KrylovFsp(MPI_Comm comm) : OdeSolverBase(comm) {}

PetscInt pacmensl::KrylovFsp::Solve()
{
  // Make sure the necessary data has been set
  if (solution_ == nullptr) return -1;
  if (rhs_ == nullptr) return -1;

  PacmenslErrorCode ierr;
  PetscInt          petsc_err;

  // Copy solution_ to the temporary solution variable
  petsc_err = VecCopy(*solution_, solution_tmp_);
  CHKERRQ(petsc_err);

  // Set Krylov starting time to the current timepoint
  t_now_tmp_ = t_now_;

  // Advance the temporary solution_ until either reaching final time or Fsp error exceeding tolerance
  int       stop = 0;
  PetscReal error_excess;
  while (t_now_ < t_final_)
  {
    krylov_stat_ = AdvanceOneStep(solution_tmp_);
    PACMENSLCHKERRQ(krylov_stat_);

    // Check that the temporary solution_ satisfies Fsp tolerance
    if (stop_check_ != nullptr)
    {
      ierr = stop_check_(t_now_tmp_, solution_tmp_, error_excess, stop_data_);
      CHKERRQ(ierr);

      PetscReal t_step_tmp = t_now_tmp_ - t_now_;
      PetscInt nrej = 0;
      while (error_excess > 0.0 && nrej < 10)
      {
        stop = 1;
        nrej += 1;
        if (nrej >= 10){
          t_step_tmp = 0.0;
        }
        else{
          t_step_tmp = 0.5*t_step_tmp;
        }
        krylov_stat_ = GetDky(t_now_ + t_step_tmp, 0, solution_tmp_);
        CHKERRQ(petsc_err);
        t_now_tmp_ = t_now_ + t_step_tmp;
      }

      if (stop){
        break;
      }
    }
    t_now_       = t_now_tmp_;
    if (print_intermediate)
    {
      PetscPrintf(comm_, "t_now_ = %.2e \n", t_now_);
    }
    if (logging_enabled)
    {
      perf_info.model_time[perf_info.n_step] = t_now_;
      petsc_err = VecGetSize(*solution_, &perf_info.n_eqs[size_t(perf_info.n_step)]);
      CHKERRQ(petsc_err);
      petsc_err = PetscTime(&perf_info.cpu_time[perf_info.n_step]);
      CHKERRQ(petsc_err);
      perf_info.n_step += 1;
    }
  }
  // Copy data from temporary vector to solution_ vector
  petsc_err = VecCopy(solution_tmp_, *solution_);
  CHKERRQ(petsc_err);
  return stop;
}

int pacmensl::KrylovFsp::AdvanceOneStep(const Vec &v)
{
  if (logging_enabled) CHKERRABORT(comm_, PetscLogEventBegin(event_advance_one_step_, 0, 0, 0, 0));

  PetscErrorCode ierr;
  PetscBool      happy_breakdown, success_step, bsize_changed;
  PetscReal s, xm, err_loc, omega, omega_old, kappa, order, t_step_old, t_step_suggest;
  PetscInt ireject, m_old, m_start, m_suggest, cost_tchange, cost_mchange;


  err_loc = 0.0;
  success_step = PETSC_FALSE;
  bsize_changed = PETSC_FALSE;
  ireject = 0;

  m_start = 0;
  kappa = 2.0;
  order = m_/4;

  while (!success_step && ireject <= max_reject_){
    m_ = std::min(m_max_, std::max(m_min_, m_next_));
    ierr = GenerateBasis(v, m_start, &happy_breakdown);
    PACMENSLCHKERRQ(ierr);

    if (happy_breakdown){
      if (print_intermediate) PetscPrintf(comm_, "Happy breakdown!\n");
      t_step_ = t_final_ - t_now_tmp_;
      t_step_next_ = t_step_;
      t_step_set_ = true;
    }

    if (!t_step_set_)
    {
      PetscReal anorm;
      xm = 1.0 / double(m_);
      rhs_(0.0, v, av);
      ierr = VecNorm(av,NORM_2,&avnorm);
      CHKERRQ(ierr);
      anorm = avnorm / beta;
      double fact = pow((m_ + 1) / exp(1.0), m_ + 1) * sqrt(2 * (3.1416) * (m_ + 1));
      t_step_next_ = (1.0 / anorm) * pow((fact * abs_tol_) / (4.0 * beta * anorm), xm);
      t_step_set_  = true;
    }

    t_step_ = std::min(t_final_ - t_now_tmp_, t_step_next_);

    /* Find the norm of A*V[m_] needed for error estimation */
    if (k1 != 0)
    {
      Hm(m_ + 1, m_) = 1.0;
      rhs_(0.0, Vm[m_], av);
      ierr = VecNorm(av,NORM_2,&avnorm);
      CHKERRQ(ierr);
    }

    /* Estimate local error when stepping with stepsize t_step_ */
    mx = mb + k1;
    F  = expmat(t_step_ * Hm);
    if (k1 == 0)
    {
      err_loc = btol_;
      break;
    } else
    {
      double phi1 = std::abs(beta * F(m_, 0));
      double phi2 = std::abs(beta * F(m_ + 1, 0) * avnorm);

      if (phi1 > phi2 * 10.0)
      {
        err_loc = phi2;
      } else if (phi1 > phi2)
      {
        err_loc = (phi1 * phi2) / (phi1 - phi2);
      } else
      {
        err_loc = phi1;
      }
    }

    omega_old = omega;
    omega = err_loc/(abs_tol_* t_step_);

    /* Estimate the parameters kappa and omega needed for stepsize and dimension selection */
    if (bsize_changed && ireject > 0){
      kappa = std::max(1.1E0, std::pow(omega/omega_old, 1.0/(m_old - m_)));
    }
    else if(ireject > 0){
      order = std::max(1.0, std::log(omega/omega_old)/std::log(t_step_/t_step_old));
    }

    /* Compute the suggestions for next stepsize and next Krylov dimension*/
    t_step_suggest =  gamma_* t_step_ * pow( omega, -1.0/order);
    s       = pow(10.0, floor(log10(t_step_suggest)) - 1);
    t_step_suggest = ceil(t_step_suggest / s) * s;
    t_step_suggest = std::min(5.0*t_step_, std::max(0.2*t_step_, t_step_suggest));
    t_step_suggest = std::min(t_final_ - t_now_tmp_, t_step_suggest);

    m_suggest = m_ + std::ceil(std::log(omega/gamma_)/std::log(kappa));
    m_suggest = std::max(3*m_/4, std::min(4*m_/3+1, m_suggest));
    m_suggest = std::max(m_min_, std::min(m_max_, m_suggest));

    /* Compute the cost associated with each option: change dimension or change stepsize */
    ierr = EstimateCost_(t_step_suggest, m_, &cost_tchange); CHKERRQ(ierr);
    ierr = EstimateCost_(t_step_, m_suggest, &cost_mchange); CHKERRQ(ierr);

    if (std::ceil((t_final_ - t_now_tmp_)/t_step_suggest)*cost_tchange <= std::ceil((t_final_-t_now_tmp_)/t_step_)*cost_mchange || m_suggest == m_){
      t_step_next_ = t_step_suggest;
      m_next_ = m_;
      bsize_changed = PETSC_FALSE;
    }
    else{
      t_step_next_ = t_step_;
      m_next_ = m_suggest;
      bsize_changed = PETSC_TRUE;
    }

    /* Check that the local error per unit step is below the tolerance */
    if (omega <= delta_)
    {
      success_step = PETSC_TRUE;
    } else
    {
      if (bsize_changed){
        Hm(m_ + 1, m_) = 0.0;
      }
      if (print_intermediate){
        PetscPrintf(comm_, "t_step = %.2e m = %d t_step_next = %.2e err_loc = %.2e \n", t_step_, m_, t_step_next_, err_loc);
      }
      if (ireject == max_reject_)
      {
        // This part could be dangerous, what if one processor exits but the others continue
        PetscPrintf(comm_, "KrylovFsp: maximum number of failed steps reached\n");
        return -1;
      }
      ireject++;
      t_step_old = t_step_;
      m_old = m_;
      m_start = m_old;
    }
  }
  /* End of subspace and stepsize selection */

  mx = mb + std::max(0, ( int ) k1 - 1);
  arma::Col<double> F0(mx);
  for (size_t       ii{0}; ii < mx; ++ii)
  {
    F0(ii) = beta * F(ii, 0);
  }

  ierr = VecScale(v,0.0); CHKERRQ(ierr);
  ierr = VecMAXPY(v,mx,&F0[0],Vm.data()); CHKERRQ(ierr);

  t_now_tmp_   = t_now_tmp_ + t_step_;

  if (print_intermediate){
    PetscPrintf(comm_, "t_step = %.2e m = %d t_step_next = %.2e err_loc = %.2e \n", t_step_, m_, t_step_next_, err_loc);
  }

  if (logging_enabled) CHKERRQ(PetscLogEventEnd(event_advance_one_step_, 0, 0, 0, 0));
  return 0;
}

int pacmensl::KrylovFsp::GenerateBasis(const Vec &v,int m_start,PetscBool *happy_breakdown)
{
  if (logging_enabled) CHKERRQ(PetscLogEventBegin(event_generate_basis_, 0, 0, 0, 0));

  int       ierr, istart;
  PetscReal s;

  *happy_breakdown = PETSC_FALSE;

  if (m_start >= m_){
    return 0;
  }

  k1 = 2;
  mb = m_;

  ierr = VecNorm(v, NORM_2, &beta); CHKERRQ(ierr);


  ierr = VecCopy(v,Vm[0]); CHKERRQ(ierr);
  ierr = VecScale(Vm[0],1.0 / beta); CHKERRQ(ierr);

  istart = 0;


  if (m_start == 0){
    Hm.zeros();
  }

  /* Incomplete Orthogonalization Procedure */
  for (int             j{m_start}; j < m_; j++)
  {
    ierr = rhs_(0.0, Vm[j], Vm[j + 1]); PACMENSLCHKERRQ(ierr);
    /* Orthogonalization */
    if (q_iop > 0){
      istart = (j - q_iop + 1 >= 0) ? j - q_iop + 1 : 0;
    }

    for (int i = istart; i <= j; ++i){
      ierr = VecDot(Vm[j+1], Vm[i], &Hm(i, j)); CHKERRQ(ierr);
      ierr = VecAXPY(Vm[j+1], -1.0*Hm(i,j), Vm[i]); CHKERRQ(ierr);
    }

    ierr = VecNorm(Vm[j + 1],NORM_2,&s); CHKERRQ(ierr);
    ierr = VecScale(Vm[j + 1],1.0 / s); CHKERRQ(ierr);
    Hm(j + 1, j) = s;

    if (s < btol_)
    {
      k1 = 0;
      mb = j+1;
      *happy_breakdown = PETSC_TRUE;
      break;
    }
  }

  if (logging_enabled) CHKERRQ(PetscLogEventEnd(event_generate_basis_, 0, 0, 0, 0));
  return 0;
}

int pacmensl::KrylovFsp::SetUpWorkSpace()
{
  if (logging_enabled) CHKERRQ(PetscLogEventBegin(event_set_up_workspace_, 0, 0, 0, 0));

  if (!solution_)
  {
    PetscPrintf(comm_, "KrylovFsp error: starting solution vector is null.\n");
    return -1;
  }
  int ierr;
  Vm.resize(m_max_ + 1);
  for (int i{0}; i < m_max_ + 1; ++i)
  {
    ierr = VecDuplicate(*solution_, &Vm[i]);
    CHKERRQ(ierr);
    ierr = VecSetUp(Vm[i]);
    CHKERRQ(ierr);
  }
  ierr = VecDuplicate(*solution_, &av);
  CHKERRQ(ierr);
  ierr = VecSetUp(av);
  CHKERRQ(ierr);

  ierr = VecDuplicate(*solution_, &solution_tmp_);
  CHKERRQ(ierr);
  ierr = VecSetUp(solution_tmp_);
  CHKERRQ(ierr);

  t_step_set_ = false;

  m_next_ = m_min_;

  Hm      = arma::zeros(m_max_ + 2, m_max_ + 2);

  ierr = fspmat_->GetLocalMVFlops(&rhs_cost_loc_); CHKERRQ(ierr);

  if (logging_enabled) CHKERRQ(PetscLogEventEnd(event_set_up_workspace_, 0, 0, 0, 0));
  return 0;
}

int pacmensl::KrylovFsp::GetDky(PetscReal t, int deg, Vec p_vec)
{
  if (logging_enabled) CHKERRQ(PetscLogEventBegin(event_getdky_, 0, 0, 0, 0));

  if (t < t_now_ || t > t_now_tmp_)
  {
    PetscPrintf(comm_,
                "KrylovFsp::GetDky error: requested timepoint does not belong to the current time subinterval.\n");
    return -1;
  }

  deg = (deg < 0) ? 0 : deg;
  F   = expmat((t - t_now_) * Hm);
  mx  = mb + ( size_t ) std::max(0, ( int ) k1 - 1);
  arma::Col<double> F0(mx);
  for (size_t       ii{0}; ii < mx; ++ii)
  {
    F0(ii) = beta * F(ii, 0);
  }

  PetscErrorCode petsc_err = VecScale(p_vec, 0.0);
  CHKERRQ(petsc_err);
  petsc_err = VecMAXPY(p_vec, mx, &F0[0], Vm.data());
  CHKERRQ(petsc_err);

  if (deg > 0)
  {
    Vec vtmp;
    petsc_err = VecCreate(comm_, &vtmp);
    CHKERRQ(petsc_err);
    petsc_err = VecDuplicate(p_vec, &vtmp);
    CHKERRQ(petsc_err);
    for (int i{1}; i <= deg; ++i)
    {
      rhs_(0.0, p_vec, vtmp);
      VecSwap(p_vec, vtmp);
    }
    petsc_err = VecDestroy(&vtmp);
    CHKERRQ(petsc_err);
  }

  if (logging_enabled) CHKERRQ(PetscLogEventEnd(event_getdky_, 0, 0, 0, 0));
  return 0;
}

pacmensl::KrylovFsp::~KrylovFsp()
{
  FreeWorkspace();
}

int pacmensl::KrylovFsp::FreeWorkspace()
{
  if (logging_enabled) CHKERRQ(PetscLogEventBegin(event_free_workspace_, 0, 0, 0, 0));
  for (int i{0}; i < Vm.size(); ++i)
  {
    VecDestroy(&Vm[i]);
  }
  Vm.clear();
  if (av != nullptr) VecDestroy(&av);
  if (solution_tmp_ != nullptr) VecDestroy(&solution_tmp_);
  if (logging_enabled) CHKERRQ(PetscLogEventEnd(event_free_workspace_, 0, 0, 0, 0));
  return 0;
}

int pacmensl::KrylovFsp::SetUp()
{
  OdeSolverBase::SetUp();
  PetscErrorCode ierr;
  if (logging_enabled)
  {
    ierr = PetscLogDefaultBegin();
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("KrylovFsp SetUpWorkspace", 0, &event_set_up_workspace_);
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("KrylovFsp FreeWorkspace", 0, &event_free_workspace_);
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("KrylovFsp AdvanceOneStep", 0, &event_advance_one_step_);
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("KrylovFsp GenerateBasis", 0, &event_generate_basis_);
    CHKERRQ(ierr);
    ierr = PetscLogEventRegister("KrylovFsp GetDky", 0, &event_getdky_);
    CHKERRQ(ierr);
  }
  SetUpWorkSpace();
  return 0;
}

PacmenslErrorCode pacmensl::KrylovFsp::SetOrthLength(int q)
{
  q_iop = q;
  return 0;
}

int pacmensl::KrylovFsp::EstimateCost_(PetscReal tau_new, PetscInt m_new, PetscInt *cost)
{
  int ierr;
  PetscReal hnorm = arma::norm(Hm, "inf");
  int ns = std::ceil(hnorm*tau_new);
  PetscInt cost_local;
  PetscInt n_loc;
  ierr = VecGetLocalSize(*solution_, &n_loc); CHKERRQ(ierr);

  if (q_iop>0){
    cost_local = (m_new + 1)*rhs_cost_loc_ +
        (4*q_iop*m_new + 5*m_new + 2*q_iop - 2*q_iop*q_iop + 7)*n_loc + 2*std::ceil(25.0/3.0 + ns)*(m_new+2)*(m_new+2)*(m_new+2);
  }
  else{
    cost_local = (m_new + 1)*rhs_cost_loc_ +
        (4*m_new*m_new + 5*m_new + 2*m_new - 2*m_new*m_new + 7)*n_loc + 2*std::ceil(25.0/3.0 + ns)*(m_new+2)*(m_new+2)*(m_new+2);
  }

  ierr = MPI_Allreduce(&cost_local, cost, 1, MPIU_INT, MPIU_MAX, comm_); CHKERRQ(ierr);

  return 0;
}

