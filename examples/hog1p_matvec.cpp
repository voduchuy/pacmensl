static char help[] = "Advance_ small CMEs to benchmark intranode performance.\n\n";

#include <fstream>
#include <iomanip>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <armadillo>
#include <cmath>
#include "pacmensl_all.h"

namespace hog1p_cme {
// stoichiometric matrix of the toggle switch model
arma::Mat<PetscInt> SM{{1, -1, -1, 0, 0, 0, 0, 0, 0},
                       {0, 0, 0, 1, 0, -1, 0, 0, 0},
                       {0, 0, 0, 0, 1, 0, -1, 0, 0},
                       {0, 0, 0, 0, 0, 1, 0, -1, 0},
                       {0, 0, 0, 0, 0, 0, 1, 0, -1},};

// reaction parameters
const PetscReal k12{1.29}, k23{0.0067}, k34{0.133}, k32{0.027}, k43{0.0381}, k21{1.0e0}, kr21{0.005}, kr31{
    0.45},      kr41{0.025}, kr22{0.0116}, kr32{0.987}, kr42{0.0538}, trans{0.01}, gamma1{0.001}, gamma2{0.0049},
// parameters for the time-dependent factors
                r1{6.9e-5}, r2{7.1e-3}, eta{3.1}, Ahog{9.3e09}, Mhog{6.4e-4};

// propensity function
inline PetscReal hog_propensity(int *X, int k)
{
  switch (k)
  {
    case 0:return k12 * double(X[0] == 0) + k23 * double(X[0] == 1) + k34 * double(X[0] == 2);
    case 1:return k32 * double(X[0] == 2) + k43 * double(X[0] == 3);
    case 2:return k21 * double(X[0] == 1);
    case 3:return kr21 * double(X[0] == 1) + kr31 * double(X[0] == 2) + kr41 * double(X[0] == 3);
    case 4:return kr22 * double(X[0] == 1) + kr32 * double(X[0] == 2) + kr42 * double(X[0] == 3);
    case 5:return trans * double(X[1]);
    case 6:return trans * double(X[2]);
    case 7:return gamma1 * double(X[3]);
    case 8:return gamma2 * double(X[4]);
    default:return 0.0;
  }
}

int propensity(const int reaction,
               const int num_species,
               const int num_states,
               const int *states,
               PetscReal *outputs,
               void *args)
{
  int (*X)[5] = ( int (*)[5] ) states;
  for (int i = 0; i < num_states; ++i)
  {
    outputs[i] = hog_propensity(X[i], reaction);
  }
  return 0;
}

// function to compute the time-dependent coefficients of the propensity functions
int t_fun(double t, int num_coefs, PetscReal *outputs, void *args)
{
  if (num_coefs != 9) return -1;

  arma::Row<double> u(outputs, 9, false, true);
  u.fill(1.0);

  double h1 = (1.0 - exp(-r1 * t)) * exp(-r2 * t);

  double hog1p = pow(h1 / (1.0 + h1 / Mhog), eta) * Ahog;

  u(2) = std::max(0.0, 3200.0 - 7710.0 * (hog1p));

  return 0;
}

// Function to constraint the shape of the Fsp
int lhs_constr(PetscInt num_species, PetscInt num_constrs, PetscInt num_states, PetscInt *states, int *vals,
               void *args)
{
  if (num_species != 5)
  {
    return -1;
  }
  if (num_constrs != 7)
  {
    return -1;
  }
  for (int j{0}; j < num_states; ++j)
  {
    for (int i{0}; i < 5; ++i)
    {
      vals[num_constrs * j + i] = (states[num_species * j + i]);
    }
    vals[num_constrs * j + 5] = ((states[num_species * j + 1]) + (states[num_species * j + 3]));
    vals[num_constrs * j + 6] = ((states[num_species * j + 2]) + (states[num_species * j + 4]));
  }
  return 0;
}

arma::Row<int>    rhs_constr_hyperrec{3, 10, 10, 10, 10};
arma::Row<double> expansion_factors_hyperrec{0.0, 0.25, 0.25, 0.25, 0.25};
arma::Row<int>    rhs_constr{3, 50, 50, 50, 50, 70, 70};
arma::Row<double> expansion_factors{0.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25};
}

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

using namespace hog1p_cme;
using namespace pacmensl;

void output_marginals(MPI_Comm comm, std::string model_name, PartitioningType fsp_par_type,
                      PartitioningApproach fsp_repart_approach, std::string constraint_type,
                      DiscreteDistribution &solution, arma::Row<int> constraints);

void output_time(MPI_Comm comm,
                 std::string model_name,
                 PartitioningType fsp_par_type,
                 PartitioningApproach fsp_repart_approach,
                 std::string constraint_type,
                 FspSolverMultiSinks &fsp_solver);

void output_performance(MPI_Comm comm, std::string model_name, PartitioningType fsp_par_type,
                        PartitioningApproach fsp_repart_approach, std::string constraint_type,
                        FspSolverMultiSinks &fsp_solver);

int ParseOptions(MPI_Comm comm, PartitioningType &fsp_par_type, PartitioningApproach &fsp_repart_approach,
                 PetscBool &output_marginal, PetscBool &fsp_log_events);

int main(int argc, char *argv[])
{
  Environment my_env(&argc, &argv, help);

  PetscLogDefaultBegin();

  PetscMPIInt    ierr, my_rank, num_procs;
  PetscErrorCode petsc_err;
  MPI_Comm       comm;
  MPI_Comm_dup(PETSC_COMM_WORLD, &comm);
  MPI_Comm_size(comm, &num_procs);
  MPI_Comm_rank(comm, &my_rank);
  PetscPrintf(comm, "Solving with %d processors.\n", num_procs);

  // Register PETSc stages
  PetscLogStage stages[2];
  petsc_err = PetscLogStageRegister("Conventional mat", &stages[0]);
  CHKERRQ(petsc_err);
  petsc_err = PetscLogStageRegister("Advanced mat", &stages[1]);
  CHKERRQ(petsc_err);


  // Set up reaction network model
  std::string         model_name = "hog1p";
  Model               model(hog1p_cme::SM, hog1p_cme::t_fun, hog1p_cme::propensity, nullptr, nullptr);
  arma::Mat<PetscInt> X0         = {0, 0, 0, 0, 0};
  X0 = X0.t();
  arma::Col<PetscReal> p0 = {1.0};

  // Default options
  PartitioningType     fsp_par_type        = PartitioningType::GRAPH;
  PartitioningApproach fsp_repart_approach = PartitioningApproach::REPARTITION;

  // Generate state set
  StateSetConstrained state_set(comm);
  state_set.SetStoichiometryMatrix(model.stoichiometry_matrix_);
  state_set.SetShape(hog1p_cme::lhs_constr, hog1p_cme::rhs_constr);
  state_set.SetLoadBalancingScheme(fsp_par_type, fsp_repart_approach);
  state_set.SetUp();
  state_set.AddStates(X0);
  state_set.Expand();

  int n_states = state_set.GetNumGlobalStates();
  PetscPrintf(comm, "The state set contains %d states.\n", n_states);
  Vec x, y, z;
  VecCreate(comm, &x);
  VecSetSizes(x, state_set.GetNumLocalStates(), PETSC_DETERMINE);
  VecSetFromOptions(x);
  VecSetUp(x);
  VecSet(x, 1.0);
  VecDuplicate(x, &y);
  VecDuplicate(x, &z);

  FspMatrixBase A1(comm);
  A1.SetUseConventionalMats();
  ierr = A1.GenerateValues(state_set,
                           model.stoichiometry_matrix_,
                           model.prop_t_,
                           model.prop_x_,
                           std::vector<int>(),
                           nullptr,
                           nullptr);
  PACMENSLCHKERRQ(ierr);
  FspMatrixBase A2(comm);
  ierr      = A2.GenerateValues(state_set,
                                model.stoichiometry_matrix_,
                                model.prop_t_,
                                model.prop_x_,
                                std::vector<int>(),
                                nullptr,
                                nullptr);
  PACMENSLCHKERRQ(ierr);

  PetscEventPerfInfo info_matmult, info_scatter_begin, info_scatter_end;
  PetscLogEvent      matmult_id, scatter_begin_id, scatter_end_id;
  PetscReal          scatter_time, matmult_time;
  std::ofstream      ofs;

  //====================================================================================================================
  // Phase 1: Do 1000 matvecs with conventional matrices (no merged scatter context)
  //====================================================================================================================
  PetscLogStagePush(stages[0]);
  for (int i{0}; i < 1000; ++i)
  {
    ierr = A1.Action(0.0, x, y);
    PACMENSLCHKERRQ(ierr);
  }
  // What time did we spend in VecScatter vs MatMult?
  petsc_err = PetscLogEventGetId("MatMult", &matmult_id);
  CHKERRQ(petsc_err);
  petsc_err = PetscLogEventGetId("VecScatterEnd  ", &scatter_end_id);
  CHKERRQ(petsc_err);
  petsc_err = PetscLogEventGetId("VecScatterBegin", &scatter_begin_id);
  CHKERRQ(petsc_err);
  PetscLogEventGetPerfInfo(stages[0], matmult_id, &info_matmult);
  PetscLogEventGetPerfInfo(stages[0], scatter_begin_id, &info_scatter_begin);
  PetscLogEventGetPerfInfo(stages[0], scatter_end_id, &info_scatter_end);

  scatter_time = info_scatter_begin.time + info_scatter_end.time;
  matmult_time = info_matmult.time;

  MPI_Allreduce(MPI_IN_PLACE, &scatter_time, 1, MPIU_REAL, MPIU_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, &matmult_time, 1, MPIU_REAL, MPIU_SUM, comm);

  scatter_time = scatter_time / num_procs;
  matmult_time = matmult_time / num_procs;

  PetscPrintf(comm, "Avg scatter time %.2e \n", scatter_time);
  PetscPrintf(comm, "Avg Matmult time %.2e \n", matmult_time);

  if (my_rank == 0)
  {
    ofs.open("hog1p_mv_conventional.txt", std::ofstream::app);
    ofs << num_procs << ", " << scatter_time << ", " << matmult_time << "\n";
    ofs.close();
  }

  PetscLogStagePop();

  MPI_Barrier(comm);
  //====================================================================================================================
  // Phase 2: Do 1000 matvecs with non-conventional matrices (with merged scatter context)
  //====================================================================================================================
  PetscLogStagePush(stages[1]);
  for (int i{0}; i < 1000; ++i)
  {
    ierr = A2.Action(0.0, x, z);
    PACMENSLCHKERRQ(ierr);
  }
  // What time did we spend in VecScatter vs MatMult?
  PetscLogEventGetId("MatMult", &matmult_id);
  PetscLogEventGetId("VecScatterEnd  ", &scatter_end_id);
  PetscLogEventGetId("VecScatterBegin", &scatter_begin_id);
  PetscLogEventGetPerfInfo(stages[1], matmult_id, &info_matmult);
  PetscLogEventGetPerfInfo(stages[1], scatter_begin_id, &info_scatter_begin);
  PetscLogEventGetPerfInfo(stages[1], scatter_end_id, &info_scatter_end);

  scatter_time = info_scatter_begin.time + info_scatter_end.time;
  matmult_time = info_matmult.time + scatter_time;

  MPI_Allreduce(MPI_IN_PLACE, &scatter_time, 1, MPIU_REAL, MPIU_SUM, comm);
  MPI_Allreduce(MPI_IN_PLACE, &matmult_time, 1, MPIU_REAL, MPIU_SUM, comm);

  scatter_time = scatter_time / num_procs;
  matmult_time = matmult_time / num_procs;

  PetscPrintf(comm, "Avg scatter time %.2e \n", scatter_time);
  PetscPrintf(comm, "Avg Matmult time %.2e \n", matmult_time);

  if (my_rank == 0)
  {
    ofs.open("hog1p_mv_advanced.txt", std::ofstream::app);
    ofs << num_procs << ", " << scatter_time << ", " << matmult_time << "\n";
    ofs.close();
  }

  PetscLogStagePop();

  // Make sure matvec results are the same regardless of method
  VecAXPY(z, -1.0, y);
  PetscReal error;
  VecNorm(z, NORM_INFINITY, &error);
  PetscPrintf(comm, "|y-z| = %.2e \n", error);

  VecDestroy(&x);
  VecDestroy(&y);
  VecDestroy(&z);
  return 0;
}

