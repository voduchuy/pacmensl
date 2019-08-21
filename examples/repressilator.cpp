static char help[] = "Solve small CMEs to benchmark intranode performance.\n\n";

#include<iomanip>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <Sys.h>
#include <armadillo>
#include <cmath>
#include "pacmensl_all.h"

namespace repressilator_cme {
// stoichiometric matrix of the toggle switch model
arma::Mat<PetscInt> SM{{1, -1, 0, 0, 0, 0},
                       {0, 0, 1, -1, 0, 0},
                       {0, 0, 0, 0, 1, -1},};

// reaction parameters
const PetscReal k1{100.0}, ka{20.0}, ket{6.0}, kg{1.0};

// Function to constraint the shape of the Fsp
int lhs_constr(PetscInt num_species,
               PetscInt num_constrs,
               PetscInt num_states,
               PetscInt *states,
               int *vals,
               void *args) {
  if (num_species != 3) return -1;
  if (num_constrs != 6) return -1;
  for (int i{0}; i < num_states; ++i) {
    vals[i * num_constrs] = (states[num_species * i]);
    vals[i * num_constrs + 1] = (states[num_species * i + 1]);
    vals[i * num_constrs + 2] = (states[num_species * i + 2]);
    vals[i * num_constrs + 3] = (states[num_species * i]) * (states[num_species * i + 1]);
    vals[i * num_constrs + 4] = (states[num_species * i + 2]) * (states[num_species * i + 1]);
    vals[i * num_constrs + 5] = (states[num_species * i]) * (states[num_species * i + 2]);
  }
  return 0;
}

arma::Row<int> rhs_constr{22, 2, 2, 44, 4, 44};
arma::Row<double> expansion_factors{0.2, 0.2, 0.2, 0.2, 0.2, 0.2};
arma::Row<int> rhs_constr_hyperrec{22, 2, 2};
arma::Row<double> expansion_factors_hyperrec{0.2, 0.2, 0.2};

// propensity function
inline PetscReal propensity_rep(const PetscInt *X, const PetscInt k) {
  switch (k) {
    case 0:return 1.0 / (1.0 + ka * pow(1.0 * PetscReal(X[1]), ket));
    case 1:return 1.0 * PetscReal(X[0]);
    case 2:return 1.0 / (1.0 + ka * pow(1.0 * PetscReal(X[2]), ket));
    case 3:return 1.0 * PetscReal(X[1]);
    case 4:return 1.0 / (1.0 + ka * pow(1.0 * PetscReal(X[0]), ket));
    case 5:return 1.0 * PetscReal(X[2]);
    default:return 0.0;
  }
}

int propensity(const int reaction,
               const int num_species,
               const int num_states,
               const int *states,
               PetscReal *outputs,
               void *args) {
  int (*X)[3] = ( int (*)[3] ) states;
  for (int i = 0; i < num_states; ++i) {
    outputs[i] = propensity_rep(X[i], reaction);
  }
  return 0;
}

// function to compute the time-dependent coefficients of the propensity functions
int t_fun(double t, int num_coefs, PetscReal *outputs, void *args) {
  outputs[0] = k1;
  outputs[1] = kg;
  outputs[2] = k1;
  outputs[3] = kg;
  outputs[4] = k1;
  outputs[5] = kg;
  return 0;
}

}

using arma::dvec;
using arma::Col;
using arma::Row;
using std::cout;
using std::endl;

using namespace repressilator_cme;
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

int main(int argc, char *argv[]) {
  Environment my_env(&argc, &argv, help);

  PetscMPIInt ierr, myRank, num_procs;
  PetscErrorCode petsc_err;
  MPI_Comm comm;
  std::string part_type;
  std::string part_approach;

  MPI_Comm_dup(PETSC_COMM_WORLD, &comm);
  MPI_Comm_size(comm, &num_procs);

  // Register PETSc stages
  PetscLogStage stages[4];
  petsc_err = PetscLogStageRegister("Solve with adaptive custom state set shape", &stages[0]);
  CHKERRQ(petsc_err);
  petsc_err = PetscLogStageRegister("Solve with fixed custom state set shape", &stages[1]);
  CHKERRQ(petsc_err);
  petsc_err = PetscLogStageRegister("Solve with adaptive default state set shape", &stages[2]);
  CHKERRQ(petsc_err);
  petsc_err = PetscLogStageRegister("Solve with fixed default state set shape", &stages[3]);
  CHKERRQ(petsc_err);

  // Default problem
  PetscReal t_final = 10.0;
  PetscReal fsp_tol = 1.0e-4;
  std::string model_name = "repressilator";
  Model repressilator_model(SM, t_fun, propensity, nullptr, nullptr);
  arma::Mat<PetscInt> X0 = {21, 0, 0};
  X0 = X0.t();
  arma::Col<PetscReal> p0 = {1.0};
  arma::Mat<PetscInt> stoich_mat = SM;

  // Default options
  PartitioningType fsp_par_type = PartitioningType::GRAPH;
  PartitioningApproach fsp_repart_approach = PartitioningApproach::REPARTITION;
  ODESolverType fsp_odes_type = CVODE;
  PetscBool output_marginal = PETSC_FALSE;
  PetscBool fsp_log_events = PETSC_FALSE;

  ierr = ParseOptions(comm, fsp_par_type, fsp_repart_approach, output_marginal, fsp_log_events);
  CHKERRQ(ierr);

  FspSolverMultiSinks fsp_solver(comm, fsp_par_type, CVODE);
  fsp_solver.SetFromOptions();
  fsp_solver.SetModel(repressilator_model);
  fsp_solver.SetInitialDistribution(X0, p0);
  DiscreteDistribution solution;

  petsc_err = PetscLogStagePush(stages[0]);
  CHKERRQ(petsc_err);
  // Solve using adaptive custom constraints
  fsp_solver.SetConstraintFunctions(lhs_constr, nullptr);
  fsp_solver.SetInitialBounds(rhs_constr);
  fsp_solver.SetExpansionFactors(expansion_factors);
  fsp_solver.SetUp();
  solution = fsp_solver.Solve(t_final, fsp_tol, 0);
  std::shared_ptr<const StateSetConstrained> fss = std::static_pointer_cast<const StateSetConstrained>(fsp_solver.GetStateSet());
  arma::Row<int> final_custom_constr = fss->GetShapeBounds();
  if (fsp_log_events) {
    output_time(PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach, std::string("adaptive_custom"),
                fsp_solver);
    output_performance(PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                       std::string("adaptive_custom"), fsp_solver);
  }
  if (output_marginal) {
    output_marginals(PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                     std::string("adaptive_custom"), solution, final_custom_constr);
  }
  fsp_solver.ClearState();
  PetscPrintf(comm, "\n ================ \n");

  petsc_err = PetscLogStagePop();
  CHKERRQ(petsc_err);
  petsc_err = PetscLogStagePush(stages[1]);
  CHKERRQ(petsc_err);
  // Solve using fixed custom constraints
  fsp_solver.SetConstraintFunctions(lhs_constr, nullptr);
  fsp_solver.SetInitialBounds(final_custom_constr);
  fsp_solver.SetUp();
  solution = fsp_solver.Solve(t_final, fsp_tol, 0);
  if (fsp_log_events) {
    output_time(PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach, std::string("fixed_custom"),
                fsp_solver);
    output_performance(PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                       std::string("fixed_custom"), fsp_solver);
  }
  if (output_marginal) {
    output_marginals(PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                     std::string("fixed_custom"), solution, final_custom_constr);
  }
  fsp_solver.ClearState();
  PetscPrintf(comm, "\n ================ \n");

  petsc_err = PetscLogStagePop();
  CHKERRQ(petsc_err);
  petsc_err = PetscLogStagePush(stages[2]);
  CHKERRQ(petsc_err);
  // Solve using adaptive default constraints
  fsp_solver.SetInitialBounds(rhs_constr_hyperrec);
  fsp_solver.SetExpansionFactors(expansion_factors_hyperrec);
  fsp_solver.SetFromOptions();
  fsp_solver.SetUp();
  solution = fsp_solver.Solve(t_final, fsp_tol, 0);
  fss = std::static_pointer_cast<const StateSetConstrained>(fsp_solver.GetStateSet());
  arma::Row<int> final_hyperrec_constr = fss->GetShapeBounds();
  if (fsp_log_events) {
    output_time(PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach, std::string("adaptive_default"),
                fsp_solver);
    output_performance(PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                       std::string("adaptive_default"), fsp_solver);
  }
  if (output_marginal) {
    output_marginals(PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                     std::string("adaptive_default"), solution, final_hyperrec_constr);
  }
  fsp_solver.ClearState();
  PetscPrintf(comm, "\n ================ \n");

  petsc_err = PetscLogStagePop();
  CHKERRQ(petsc_err);
  petsc_err = PetscLogStagePush(stages[3]);
  CHKERRQ(petsc_err);
  // Solve using fixed default constraints
  fsp_solver.SetInitialBounds(final_hyperrec_constr);
  fsp_solver.SetExpansionFactors(expansion_factors_hyperrec);
  fsp_solver.SetFromOptions();
  fsp_solver.SetUp();
  solution = fsp_solver.Solve(t_final, fsp_tol, 0);
  if (fsp_log_events) {
    output_time(PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach, std::string("fixed_default"),
                fsp_solver);
    output_performance(PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                       std::string("fixed_default"), fsp_solver);
  }
  if (output_marginal) {
    output_marginals(PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                     std::string("fixed_default"), solution, final_hyperrec_constr);
  }
  fsp_solver.ClearState();
  PetscPrintf(comm, "\n ================ \n");
  return ierr;
}

int ParseOptions(MPI_Comm comm, PartitioningType &fsp_par_type, PartitioningApproach &fsp_repart_approach,
                 PetscBool &output_marginal, PetscBool &fsp_log_events) {
  std::string part_type;
  std::string part_approach;
  part_type = part2str(fsp_par_type);
  part_approach = partapproach2str(fsp_repart_approach);

  // Read options for fsp
  char opt[100];
  PetscBool opt_set;
  int ierr;
  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_partitioning_type", opt, 100, &opt_set);
  CHKERRQ(ierr);
  if (opt_set) {
    fsp_par_type = str2part(std::string(opt));
  }

  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_repart_approach", opt, 100, &opt_set);
  CHKERRQ(ierr);
  if (opt_set) {
    fsp_repart_approach = str2partapproach(std::string(opt));
  }

  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_output_marginal", opt, 100, &opt_set);
  CHKERRQ(ierr);
  if (opt_set) {
    if (strcmp(opt, "1") == 0 || strcmp(opt, "true") == 0) {
      output_marginal = PETSC_TRUE;
    }
  }

  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_log_events", opt, 100, &opt_set);
  CHKERRQ(ierr);
  if (opt_set) {
    if (strcmp(opt, "1") == 0 || strcmp(opt, "true") == 0) {
      fsp_log_events = PETSC_TRUE;
    }
  }
  PetscPrintf(comm, "Partitiniong option %s \n", part2str(fsp_par_type).c_str());
  PetscPrintf(comm, "Repartitoning option %s \n", partapproach2str(fsp_repart_approach).c_str());
  return 0;
}

void output_performance(MPI_Comm comm, std::string model_name, PartitioningType fsp_par_type,
                        PartitioningApproach fsp_repart_approach, std::string constraint_type,
                        FspSolverMultiSinks &fsp_solver) {
  int myRank, num_procs;
  MPI_Comm_rank(comm, &myRank);
  MPI_Comm_size(comm, &num_procs);

  std::string part_type;
  std::string part_approach;
  part_type = part2str(fsp_par_type);
  part_approach = partapproach2str(fsp_repart_approach);

  FspSolverComponentTiming timings = fsp_solver.GetAvgComponentTiming();
  FiniteProblemSolverPerfInfo perf_info = fsp_solver.GetSolverPerfInfo();
  double solver_time = timings.TotalTime;
  if (myRank == 0) {
    std::string filename =
        model_name + "_time_breakdown_" + std::to_string(num_procs) + "_" + part_type + "_" + part_approach +
            "_" + constraint_type + ".dat";
    std::ofstream file;
    file.open(filename);
    file << "Component, Average processor time (sec), Percentage \n";
    file << "Finite State Subset," << std::scientific << std::setprecision(2) << timings.StatePartitioningTime
         << "," << timings.StatePartitioningTime / solver_time * 100.0 << "\n" << "Matrix Generation,"
         << std::scientific << std::setprecision(2) << timings.MatrixGenerationTime << ","
         << timings.MatrixGenerationTime / solver_time * 100.0 << "\n" << "Matrix-vector multiplication,"
         << std::scientific << std::setprecision(2) << timings.RHSEvalTime << ","
         << timings.RHSEvalTime / solver_time * 100.0 << "\n" << "Others," << std::scientific
         << std::setprecision(2)
         << solver_time - timings.StatePartitioningTime - timings.MatrixGenerationTime - timings.RHSEvalTime << ","
         << (solver_time - timings.StatePartitioningTime - timings.MatrixGenerationTime - timings.RHSEvalTime) /
             solver_time * 100.0 << "\n" << "Total," << solver_time << "," << 100.0 << "\n";
    file.close();

    filename =
        model_name + "_perf_info_" + std::to_string(num_procs) + "_" + part_type + "_" + part_approach + "_" +
            constraint_type + ".dat";
    file.open(filename);
    file << "Model time, ODEs size, Average processor time (sec) \n";
    for (auto i{0}; i < perf_info.n_step; ++i) {
      file << perf_info.model_time[i] << "," << perf_info.n_eqs[i] << "," << perf_info.cpu_time[i] << "\n";
    }
    file.close();
  }
}

void output_time(MPI_Comm comm,
                 std::string model_name,
                 PartitioningType fsp_par_type,
                 PartitioningApproach fsp_repart_approach,
                 std::string constraint_type,
                 FspSolverMultiSinks &fsp_solver) {
  int myRank, num_procs;
  MPI_Comm_rank(comm, &myRank);
  MPI_Comm_size(comm, &num_procs);

  std::string part_type;
  std::string part_approach;
  part_type = part2str(fsp_par_type);
  part_approach = partapproach2str(fsp_repart_approach);

  FspSolverComponentTiming timings = fsp_solver.GetAvgComponentTiming();
  FiniteProblemSolverPerfInfo perf_info = fsp_solver.GetSolverPerfInfo();
  double solver_time = timings.TotalTime;

  if (myRank == 0) {
    {
      std::string filename =
          model_name + "_time_" + std::to_string(num_procs) + "_" + part_type + "_" + part_approach + "_" +
              constraint_type + ".dat";
      std::ofstream file;
      file.open(filename, std::ios_base::app);
      file << solver_time << "\n";
      file.close();
    }
  }
}

void output_marginals(MPI_Comm comm, std::string model_name, PartitioningType fsp_par_type,
                      PartitioningApproach fsp_repart_approach, std::string constraint_type,
                      DiscreteDistribution &solution, arma::Row<int> constraints) {
  int myRank, num_procs;
  MPI_Comm_rank(comm, &myRank);
  MPI_Comm_size(comm, &num_procs);

  std::string part_type;
  std::string part_approach;
  part_type = part2str(fsp_par_type);
  part_approach = partapproach2str(fsp_repart_approach);

  /* Compute the marginal distributions */
  std::vector<arma::Col<PetscReal>> marginals(solution.states_.n_rows);
  for (PetscInt i{0}; i < marginals.size(); ++i) {
    marginals[i] = Compute1DMarginal(solution, i);
  }

  MPI_Comm_rank(PETSC_COMM_WORLD, &myRank);
  if (myRank == 0) {
    for (PetscInt i{0}; i < marginals.size(); ++i) {
      std::string filename =
          model_name + "_marginal_" + std::to_string(i) + "_" + std::to_string(num_procs) + "_" +
              part_type + "_" + part_approach + "_" + constraint_type + ".dat";
      marginals[i].save(filename, arma::raw_ascii);
    }
    std::string filename =
        model_name + "_constraint_bounds_" + std::to_string(num_procs) + "_" + part_type + "_" + part_approach +
            "_" + constraint_type + ".dat";
    constraints.save(filename, arma::raw_ascii);
  }
}
