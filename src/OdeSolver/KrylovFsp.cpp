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

      if (error_excess > 0.0)
      {
        stop         = 1;
        krylov_stat_ = GetDky(t_now_, 0, solution_tmp_);
        CHKERRQ(petsc_err);
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

  PetscErrorCode petsc_err;
  int            ierr;

  PetscReal s, xm, err_loc;

  k1 = 2;
  mb = m_;

  petsc_err = VecNorm(v, NORM_2, &beta);
  CHKERRQ(petsc_err);

  if (!t_step_set_)
  {
    PetscReal anorm;
    xm = 1.0 / double(m_);
    rhs_(0.0, v, av);
    petsc_err = VecNorm(av, NORM_2, &avnorm);
    CHKERRQ(petsc_err);
    anorm = avnorm / beta;
    double fact = pow((m_ + 1) / exp(1.0), m_ + 1) * sqrt(2 * (3.1416) * (m_ + 1));
    t_step_next_ = (1.0 / anorm) * pow((fact * abs_tol_) / (4.0 * beta * anorm), xm);
    t_step_set_  = true;
  }

  t_step_ = std::min(t_final_ - t_now_tmp_, t_step_next_);
  Hm      = arma::zeros(m_ + 2, m_ + 2);

  ierr = GenerateBasis(v, m_);
  PACMENSLCHKERRQ(ierr);

  if (k1 != 0)
  {
    Hm(m_ + 1, m_) = 1.0;
    rhs_(0.0, Vm[m_], av);
    petsc_err = VecNorm(av, NORM_2, &avnorm);
    CHKERRQ(petsc_err);
  }

  int ireject{0};
  while (ireject < max_reject_)
  {
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
        xm      = 1.0 / double(m_);
      } else if (phi1 > phi2)
      {
        err_loc = (phi1 * phi2) / (phi1 - phi2);
        xm      = 1.0 / double(m_);
      } else
      {
        err_loc = phi1;
        xm      = 1.0 / double(m_ - 1);
      }
    }

    if (err_loc <= delta_ * t_step_ * abs_tol_ / t_final_)
    {
      break;
    } else
    {
      t_step_ = gamma_ * t_step_ * pow(t_step_ * abs_tol_ / (t_final_ * err_loc), xm);
      s       = pow(10.0, floor(log10(t_step_)) - 1);
      t_step_ = ceil(t_step_ / s) * s;
      if (ireject == max_reject_)
      {
        // This part could be dangerous, what if one processor exits but the others continue
        PetscPrintf(comm_, "KrylovFsp: maximum number of failed steps reached\n");
        return -1;
      }
      ireject++;
    }
  }

  mx = mb + ( size_t ) std::max(0, ( int ) k1 - 1);
  arma::Col<double> F0(mx);
  for (size_t       ii{0}; ii < mx; ++ii)
  {
    F0(ii) = beta * F(ii, 0);
  }

  petsc_err = VecScale(v, 0.0);
  CHKERRQ(petsc_err);
  petsc_err = VecMAXPY(v, mx, &F0[0], Vm.data());
  CHKERRQ(petsc_err);

  t_now_tmp_   = t_now_tmp_ + t_step_;
  t_step_next_ = gamma_ * t_step_ * pow(t_step_ * abs_tol_ / (t_final_ * err_loc), xm);
  s            = pow(10.0, floor(log10(t_step_next_)) - 1.0);
  t_step_next_ = ceil(t_step_next_ / s) * s;

  if (print_intermediate){
    PetscPrintf(comm_, "t_step = %.2e t_step_next = %.2e err_loc = %.2e \n", t_step_, t_step_next_, err_loc);
  }

  if (logging_enabled) CHKERRQ(PetscLogEventEnd(event_advance_one_step_, 0, 0, 0, 0));
  return 0;
}

int pacmensl::KrylovFsp::GenerateBasis(const Vec &v, int m)
{
  if (logging_enabled) CHKERRQ(PetscLogEventBegin(event_generate_basis_, 0, 0, 0, 0));

  int       petsc_error, ierr;
  PetscReal s;

  petsc_error = VecCopy(v, Vm[0]);
  CHKERRQ(petsc_error);
  petsc_error = VecScale(Vm[0], 1.0 / beta);
  CHKERRQ(petsc_error);

  int                  istart = 0;
  arma::Col<PetscReal> htmp(m + 2);
  /* Arnoldi loop */
  for (int             j{0}; j < m; j++)
  {
    ierr = rhs_(0.0, Vm[j], Vm[j + 1]);
    PACMENSLCHKERRQ(ierr);
    /* Orthogonalization */
    if (q_iop > 0){
      istart = (j - q_iop + 1 >= 0) ? j - q_iop + 1 : 0;
    }

    for (int iorth = 0; iorth < 1; ++iorth)
    {
      petsc_error = VecMTDot(Vm[j + 1], j - istart + 1, Vm.data() + istart, &htmp[istart]);
      CHKERRQ(petsc_error);
      for (int i{istart}; i <= j; ++i)
      {
        htmp(i) = -1.0 * htmp(i);
      }
      petsc_error = VecMAXPY(Vm[j + 1], j - istart + 1, &htmp[istart], Vm.data() + istart);
      CHKERRQ(petsc_error);
      for (int i{istart}; i <= j; ++i)
      {
        Hm(i, j) -= htmp(i);
      }
    }
    petsc_error = VecNorm(Vm[j + 1], NORM_2, &s);
    CHKERRQ(petsc_error);
    petsc_error = VecScale(Vm[j + 1], 1.0 / s);
    CHKERRQ(petsc_error);
    Hm(j + 1, j) = s;

    if (s < btol_)
    {
      k1 = 0;
      mb = j;
      if (print_intermediate) PetscPrintf(comm_, "Happy breakdown!\n");
      t_step_ = t_final_ - t_now_tmp_;
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
  Vm.resize(m_ + 1);
  for (int i{0}; i < m_ + 1; ++i)
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

