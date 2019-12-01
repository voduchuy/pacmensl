static char help[] = "Advance_ small CMEs to benchmark intranode performance.\n\n";

#include <fstream>
#include <iomanip>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <armadillo>
#include <cmath>
#include "pacmensl_all.h"

// stoichiometric matrix of the toggle switch model
arma::Mat<PetscInt> SM{{-1,1, 0,  0, 0, 0, 0},
                       {1,-1,-1,  1, 0, 0, 0},
                       {0, 0, 1, -1, 0, 0, 0},
                       {0, 0, 0,  0, 1, 1, -1}};


PetscReal theta[] = {3.98634865e-03, 1.00000000e-02, 8.64980068e-04, 1.02280560e-01,
3.71705981e+02, 7.88810222e-04, 2.24254520e-03, 1.75909361e-02,
2.27393435e-01, 1.35198833e-04};
// reaction parameters
const PetscReal     r1 = theta[0],
                    r2 = theta[1],
                    k01 = theta[2],
                    k12 = theta[3],
                    k10 = theta[4],
                    k21 = theta[5],
                    A10 = theta[6],
                    kr1 = theta[7],
                    kr2 = theta[8],
                    deg = theta[9],
                    T0 = 24*3600;

// propensity function
inline PetscReal tnfa_propensity(int *X, int k) {
  switch (k) {
    case 0:return double(X[0]);
    case 1:return double(X[1]);
    case 2:return double(X[1]);
    case 3:return double(X[2]);
    case 4:return double(X[1]);
    case 5:return double(X[2]);
    case 6:return double(X[3]);
    default:return 0.0;
  }
}

int propensity(const int reaction,
               const int num_species,
               const int num_states,
               const int *states,
               PetscReal *outputs,
               void *args) {
  int (*X)[4] = ( int (*)[4] ) states;
  for (int i = 0; i < num_states; ++i) {
    outputs[i] = tnfa_propensity(X[i], reaction);
  }
  return 0;
}

// function to compute the time-dependent coefficients of the propensity functions
int t_fun(double t, int num_coefs, PetscReal *out, void *args) {
  static PetscReal signal;

  if (num_coefs != 7) return -1;

  signal = std::max(0.0, exp(-r1*(t - T0))*(1.0 - exp(-r2*(t - T0))));
  out[0] = k01;
  out[1] = k10*std::max(0.0, 1.0 - A10*signal);
  out[2] = k12;
  out[3] = k21;
  out[4] = kr1;
  out[5] = kr2;
  out[6] = deg;

  return 0;
}

// Function to constraint the shape of the Fsp
int lhs_constr(PetscInt num_species, PetscInt num_constrs, PetscInt num_states, PetscInt *states, int *vals,
               void *args) {
  if (num_species != 4) {
    return -1;
  }
  if (num_constrs != 7) {
    return -1;
  }
  for (int j{0}; j < num_states; ++j) {
    for (int i{0}; i < 4; ++i) {
      vals[num_constrs * j + i] = (states[num_species * j + i]);
    }
  }
  return 0;
}

arma::Row<int> rhs_constr{2, 2, 2, 5};
arma::Row<double> expansion_factors{0.0, 0.0, 0.0, 0.2};

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

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
  MPI_Comm_dup(PETSC_COMM_WORLD, &comm);
  MPI_Comm_size(comm, &num_procs);
  PetscPrintf(comm, "Solving with %d processors.\n", num_procs);

  // Set up CME
  std::string model_name = "tnfalpha_no_drug";
  Model tnfa_model(SM, t_fun, propensity, nullptr, nullptr, std::vector<int>({0, 1, 2,3, 4, 5,6}));
  PetscReal t_final = T0 + 4*3600;

  PetscReal fsp_tol = 1.0e-6;
  arma::Mat<PetscInt> X0 = {2, 0, 0, 0};
  X0 = X0.t();
  arma::Col<PetscReal> p0 = {1.0};


  // Default options
  PartitioningType fsp_par_type = PartitioningType::GRAPH;
  PartitioningApproach fsp_repart_approach = PartitioningApproach::REPARTITION;
  PetscBool output_marginal = PETSC_FALSE;
  PetscBool fsp_log_events = PETSC_FALSE;

  ierr = ParseOptions(comm, fsp_par_type, fsp_repart_approach, output_marginal, fsp_log_events); CHKERRQ(ierr);

  FspSolverMultiSinks fsp_solver(comm, fsp_par_type, CVODE);
  fsp_solver.SetModel(tnfa_model);
  fsp_solver.SetInitialDistribution(X0, p0);
  fsp_solver.SetOdesType(PETSC);
  fsp_solver.SetFromOptions();
  DiscreteDistribution solution;

  fsp_solver.SetInitialBounds(rhs_constr);
  fsp_solver.SetExpansionFactors(expansion_factors);
  fsp_solver.SetFromOptions();
  solution = fsp_solver.Solve(t_final, fsp_tol, 0);

  if (fsp_log_events) {
    output_time(PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach, std::string("adaptive_default"),
                fsp_solver);
    output_performance(PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                       std::string("adaptive_default"), fsp_solver);
  }

  std::shared_ptr<const StateSetConstrained> fss = std::static_pointer_cast<const StateSetConstrained>(fsp_solver.GetStateSet());
  arma::Row<int> final_hyperrec_constr = fss->GetShapeBounds();
  if (output_marginal) {
    output_marginals(PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                     std::string("adaptive_default"), solution, final_hyperrec_constr);
  }
  fsp_solver.ClearState();
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
  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_partitioning_type", opt, 100, &opt_set); CHKERRQ(ierr);
  if (opt_set) {
    fsp_par_type = str2part(std::string(opt));
  }

  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_repart_approach", opt, 100, &opt_set); CHKERRQ(ierr);
  if (opt_set) {
    fsp_repart_approach = str2partapproach(std::string(opt));
  }

  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_output_marginal", opt, 100, &opt_set); CHKERRQ(ierr);
  if (opt_set) {
    if (strcmp(opt, "1") == 0 || strcmp(opt, "true") == 0) {
      output_marginal = PETSC_TRUE;
    }
  }

  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_log_events", opt, 100, &opt_set); CHKERRQ(ierr);
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
