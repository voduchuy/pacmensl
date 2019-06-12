//
// Created by Huy Vo on 2019-06-12.
//

#include "KrylovFsp.h"
#include "CvodeFsp.h"
pecmeal::KrylovFsp::KrylovFsp(MPI_Comm comm) : OdeSolverBase(comm) {}

PetscInt pecmeal::KrylovFsp::solve() {
  // Make sure the necessary data has been set
  assert( solution_ != nullptr );
  assert( rhs_ );

  PetscInt petsc_err;

  // Copy solution_ to the temporary solution variable
  petsc_err = VecDuplicate(*solution_, &solution_tmp_);
  CHKERRABORT(comm_, petsc_err);
  petsc_err = VecCopy(*solution_, solution_tmp_);
  CHKERRABORT(comm_, petsc_err);

  // Set Krylov starting time to the current timepoint
  t_now_tmp_ = t_now_;

  // Advance the temporary solution_ until either reaching final time or FSP error exceeding tolerance
  int stop = 0;
  while ( t_now_ < t_final_ ) {
    krylov_stat_ = GenerateBasis(solution_tmp_, m);
    CHKERRABORT(comm_, krylov_stat_);

    krylov_stat_ = AdvanceOneStep(solution_tmp_);
    CHKERRABORT(comm_, krylov_stat_);

    // Check that the temporary solution_ satisfies FSP tolerance
    if (stop_check_ != nullptr) stop = stop_check_(t_now_tmp_, solution_tmp_, stop_data_);
    if ( stop == 1) {
      krylov_stat_ = GetDky( t_now_, 0, solution_tmp_);
      break;
    } else {
      t_now_ = t_now_tmp_;
      if ( print_intermediate ) {
        PetscPrintf( comm_, "t_now_ = %.2e \n", t_now_ );
      }
      if ( logging ) {
        perf_info.model_time[ perf_info.n_step ] = t_now_;
        petsc_err = VecGetSize( *solution_, &perf_info.n_eqs[ size_t( perf_info.n_step ) ] );
        CHKERRABORT( comm_, petsc_err );
        petsc_err = PetscTime( &perf_info.cpu_time[ perf_info.n_step ] );
        CHKERRABORT( comm_, petsc_err );
        perf_info.n_step += 1;
      }
    }
  }
  // Copy data from temporary vector to solution_ vector
  CHKERRABORT( comm_, VecCopy( solution_tmp_, *solution_ ));
  return stop;
}

int pecmeal::KrylovFsp::GetDky(PetscReal t, int deg, Vec p_vec) {
  return 0;
}
