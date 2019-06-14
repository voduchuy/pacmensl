//
// Created by Huy Vo on 2019-06-12.
//

#include "KrylovFsp.h"

pecmeal::KrylovFsp::KrylovFsp(MPI_Comm comm) : OdeSolverBase(comm) {}

PetscInt pecmeal::KrylovFsp::Solve() {
  // Make sure the necessary data has been set
  assert(solution_ != nullptr);
  assert(rhs_);

  PetscInt petsc_err;

  SetUpWorkSpace();

  // Copy solution_ to the temporary solution variable
  petsc_err = VecCopy(*solution_, solution_tmp_);
  CHKERRABORT(comm_, petsc_err);

  // Set Krylov starting time to the current timepoint
  t_now_tmp_ = t_now_;

  // Advance the temporary solution_ until either reaching final time or FSP error exceeding tolerance
  int stop = 0;
  while (t_now_ < t_final_) {
    krylov_stat_ = AdvanceOneStep(solution_tmp_);
    CHKERRABORT(comm_, krylov_stat_);

    // Check that the temporary solution_ satisfies FSP tolerance
    if (stop_check_ != nullptr) stop = stop_check_(t_now_tmp_, solution_tmp_, stop_data_);
    if (stop == 1) {
      krylov_stat_ = GetDky(t_now_, 0, solution_tmp_);
      CHKERRABORT(comm_, krylov_stat_);
      break;
    } else {
      t_now_ = t_now_tmp_;
      if (print_intermediate) {
        PetscPrintf(comm_, "t_now_ = %.2e \n", t_now_);
      }
      if (logging) {
        perf_info.model_time[perf_info.n_step] = t_now_;
        petsc_err = VecGetSize(*solution_, &perf_info.n_eqs[size_t(perf_info.n_step)]);
        CHKERRABORT(comm_, petsc_err);
        petsc_err = PetscTime(&perf_info.cpu_time[perf_info.n_step]);
        CHKERRABORT(comm_, petsc_err);
        perf_info.n_step += 1;
      }
    }
  }
  // Copy data from temporary vector to solution_ vector
  CHKERRABORT(comm_, VecCopy(solution_tmp_, *solution_));

  FreeWorkspace();
  return stop;
}

int pecmeal::KrylovFsp::GetDky(PetscReal t, int deg, Vec p_vec) {
  if (t < t_now_ || t > t_now_tmp_) {
    PetscPrintf(comm_,
                "KrylovFsp::GetDky error: requested timepoint does not belong to the current time subinterval.\n");
    return -1;
  }
  deg = (deg < 0) ? 0 : deg;
  F = expmat((t - t_now_) * Hm);
  mx = mb + (size_t) std::max(0, (int) k1 - 1);
  arma::Col<double> F0(mx);
  for (size_t ii{0}; ii < mx; ++ii) {
    F0(ii) = beta * F(ii, 0);
  }

  PetscErrorCode petsc_err = VecScale(p_vec, 0.0);
  CHKERRABORT(comm_, petsc_err);
  petsc_err = VecMAXPY(p_vec, mx, &F0[0], Vm.data());
  CHKERRABORT(comm_, petsc_err);

  if (deg > 0) {
    Vec vtmp;
    petsc_err = VecCreate(comm_, &vtmp);
    CHKERRABORT(comm_, petsc_err);
    petsc_err = VecDuplicate(p_vec, &vtmp);
    CHKERRABORT(comm_, petsc_err);
    for (int i{1}; i <= deg; ++i) {
      rhs_(0.0, p_vec, vtmp);
      VecSwap(p_vec, vtmp);
    }
    petsc_err = VecDestroy(&vtmp);
    CHKERRABORT(comm_, petsc_err);
  }

  return 0;
}

int pecmeal::KrylovFsp::AdvanceOneStep(const Vec &v) {
  PetscErrorCode petsc_err;

  PetscReal s, xm, err_loc;

  mb = m_;

  petsc_err = VecNorm(v, NORM_2, &beta);
  CHKERRABORT(comm_, petsc_err);

  if (!t_step_set_) {
    PetscReal anorm;
    xm = 1.0 / double(m_);
    rhs_(0.0, v, av);
    petsc_err = VecNorm(av, NORM_2, &avnorm);
    CHKERRABORT(comm_, petsc_err);
    anorm = avnorm/beta;
    double fact = pow((m_ + 1) / exp(1.0), m_ + 1) * sqrt(2 * (3.1416) * (m_ + 1));
    t_step_next_ = (1.0 / anorm) * pow((fact * tol_) / (4.0 * beta * anorm), xm);
    t_step_set_ = true;
  }

  double t_step = std::min(t_final_ - t_now_tmp_, t_step_next_);
  Hm = arma::zeros(m_ + 2, m_ + 2);

  GenerateBasis(v, m_);

  if (k1 != 0) {
    Hm(m_ + 1, m_) = 1.0;
    rhs_(0.0, Vm[m_], av);
    petsc_err = VecNorm(av, NORM_2, &avnorm);
    CHKERRABORT(comm_, petsc_err);
  } else {
    t_step = t_final_ - t_now_;
  }

  int ireject{0};
  while (ireject < max_reject_ && k1 != 0) {
    mx = mb + k1;
    F = expmat(t_step * Hm);
    if (k1 == 0) {
      err_loc = btol_;
      break;
    } else {
      double phi1 = std::abs(beta * F(m_, 0));
      double phi2 = std::abs(beta * F(m_ + 1, 0)*avnorm);

      if (phi1 > phi2 * 10.0) {
        err_loc = phi2;
        xm = 1.0 / double(m_);
      } else if (phi1 > phi2) {
        err_loc = (phi1 * phi2) / (phi1 - phi2);
        xm = 1.0 / double(m_);
      } else {
        err_loc = phi1;
        xm = 1.0 / double(m_ - 1);
      }
    }

    if (err_loc <= delta_ * t_step * tol_) {
      break;
    } else {
      t_step = gamma_ * t_step * pow(t_step * tol_ / err_loc, xm);
      s = pow(10.0, floor(log10(t_step)) - 1);
      t_step = ceil(t_step / s) * s;
      if (ireject == max_reject_) {
        // This part could be dangerous, what if one processor exits but the others continue
        PetscPrintf(comm_, "KrylovFsp: maximum number of failed steps reached\n");
        return -1;
      }
      ireject++;
    }
  }

  mx = mb + (size_t) std::max(0, (int) k1 - 1);
  arma::Col<double> F0(mx);
  for (size_t ii{0}; ii < mx; ++ii) {
    F0(ii) = beta * F(ii, 0);
  }

  petsc_err = VecScale(v, 0.0);
  CHKERRABORT(comm_, petsc_err);


  petsc_err = VecMAXPY(v, mx, &F0[0], Vm.data());
  CHKERRABORT(comm_, petsc_err);


  t_now_tmp_ = t_now_tmp_ + t_step;
  t_step_next_ = gamma_ * t_step * pow(t_step * tol_ / err_loc, xm);
  s = pow(10.0, floor(log10(t_step_next_)) - 1.0);
  t_step_next_ = ceil(t_step_next_ / s) * s;

  return 0;
}

int pecmeal::KrylovFsp::GenerateBasis(const Vec &v, int m) {
  int petsc_error;
  PetscReal s;

  petsc_error = VecCopy(v, Vm[0]);
  CHKERRABORT(comm_, petsc_error);
  petsc_error = VecScale(Vm[0], 1.0 / beta);
  CHKERRABORT(comm_, petsc_error);

  int istart = 0;
  arma::Col<PetscReal> htmp(m+2);
  /* Arnoldi loop */
  for (int j{0}; j < m; j++) {
    rhs_(0.0, Vm[j], Vm[j + 1]);

    /* Orthogonalization */
    istart = (j - q_iop + 1 >= 0) ? j - q_iop + 1 : 0;

    for (int iorth = 0; iorth < 2; ++iorth) {
      petsc_error = VecMTDot(Vm[j + 1], j - istart + 1, Vm.data() + istart, &htmp[istart]);
      CHKERRABORT(comm_, petsc_error);
      for (int i{istart}; i <= j; ++i) {
        htmp(i) = -1.0 * htmp(i);
      }
      petsc_error = VecMAXPY(Vm[j + 1], j - istart + 1, &htmp[istart], Vm.data() + istart);
      CHKERRABORT(comm_, petsc_error);
      for (int i{istart}; i <= j; ++i) {
        Hm(i,j) -= htmp(i);
      }
    }
    petsc_error = VecNorm(Vm[j + 1], NORM_2, &s);
    CHKERRABORT(comm_, petsc_error);

    if (s < btol_) {
      k1 = 0;
      mb = j + 1;
      PetscPrintf(comm_, "Happy breakdown!\n");
      break;
    }

    Hm(j + 1, j) = s;
    petsc_error = VecScale(Vm[j + 1], 1.0 / s);
    CHKERRABORT(comm_, petsc_error);
  }
  return 0;
}

int pecmeal::KrylovFsp::SetUpWorkSpace() {
  if (!solution_) {
    PetscPrintf(comm_, "KrylovFsp error: starting solution vector is null.\n");
    return -1;
  }
  int ierr;
  Vm.resize(m_ + 1);
  for (int i{0}; i < m_ + 1; ++i) {
    ierr = VecDuplicate(*solution_, &Vm[i]);
    CHKERRABORT(comm_, ierr);
    ierr = VecSetUp(Vm[i]);
    CHKERRABORT(comm_, ierr);
  }
  ierr = VecDuplicate(*solution_, &av);
  CHKERRABORT(comm_, ierr);
  ierr = VecSetUp(av);
  CHKERRABORT(comm_, ierr);

  ierr = VecDuplicate(*solution_, &solution_tmp_);
  CHKERRABORT(comm_, ierr);
  ierr = VecSetUp(solution_tmp_);
  CHKERRABORT(comm_, ierr);
  return 0;
}

pecmeal::KrylovFsp::~KrylovFsp() {
  FreeWorkspace();
}

void pecmeal::KrylovFsp::FreeWorkspace() {
  for (int i{0}; i < Vm.size(); ++i) {
    VecDestroy(&Vm[i]);
  }
  Vm.clear();
  if (av != nullptr) VecDestroy(&av);
  if (solution_tmp_ != nullptr) VecDestroy(&solution_tmp_);
}
