static char help[] = "Solve small CMEs to benchmark intranode performance.\n\n";

#include<iomanip>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <Sys.h>
#include <armadillo>
#include <cmath>
#include <sys/stat.h>
#include "FspSolverMultiSinks.h"

namespace six_species_cme {
// stoichiometric matrix of the toggle switch model
arma::Mat<PetscInt> SM{{1, -1, 0, 0, 0, 0, 0, 0, -2, 2},
                       {0, 0, 0, 0, -1, 1, -1, 1, 1, -1},
                       {0, 0, 0, 0, -1, 1, 0, 0, 0, 0},
                       {0, 0, 0, 0, 1, -1, -1, 1, 0, 0},
                       {0, 0, 0, 0, 0, 0, 1, -1, 0, 0},
                       {0, 0, 1, -1, 0, 0, 0, 0, 0, 0}};

// reaction parameters
const PetscReal Avo = 6.022140857e23, c0 = 0.043, c1 = 0.0007, c2 = 0.078, c3 = 0.0039, c4 =
    0.012e09 / (Avo), c5 = 0.4791, c6 = 0.00012e09 / (Avo), c7 = 0.8765e-11, c8 =
    0.05e09 / (Avo), c9 = 0.5, avg_cell_cyc_time = 35 * 60.0;

// propensity function
inline PetscReal transreg_propensity(const PetscInt *X, const PetscInt k) {
  switch (k) {
    case 0:return c0 * PetscReal(X[5]);
    case 1:return c1 * PetscReal(X[0]);
    case 2:return c2 * PetscReal(X[3]);
    case 3:return c3 * PetscReal(X[5]);
    case 4:return PetscReal(X[1]) * PetscReal(X[2]);
    case 5:return c5 * PetscReal(X[3]);
    case 6:return PetscReal(X[3]) * PetscReal(X[1]);
    case 7:return c7 * PetscReal(X[4]);
    case 8:return 0.5 * PetscReal(X[0]) * PetscReal(X[0] - 1);
    case 9:return c9 * PetscReal(X[1]);
    default:return 0.0;
  }
}
int propensity(const int reaction,
               const int num_species,
               const int num_states,
               const int *states,
               PetscReal *outputs,
               void *args) {
  int (*X)[6] = ( int (*)[6] ) states;
  for (int i = 0; i < num_states; ++i) {
    outputs[i] = transreg_propensity(X[i], reaction);
  }
  return 0;
}

// Function to constraint the shape of the Fsp
void lhs_constr(PetscInt num_species, PetscInt num_constrs, PetscInt num_states, PetscInt *states, int *vals,
                void *args) {
  for (int j{0}; j < num_states; ++j) {
    for (int i{0}; i < 6; ++i) {
      vals[j * num_constrs + i] = states[num_species * j + i];
    }
  }
}

arma::Row<int> rhs_constr{10, 6, 1, 2, 1, 1};
arma::Row<double> expansion_factors{0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
arma::Row<int> rhs_constr_hyperrec{10, 6, 1, 2, 1, 1};
arma::Row<double> expansion_factors_hyperrec{0.2, 0.2, 0.2, 0.2, 0.2, 0.2};

// function to compute the time-dependent coefficients of the propensity functions
int t_fun(PetscReal t, int nc, double *vals, void *args) {
  arma::Row<PetscReal> u(vals, 10, false, true);
  u.fill(1.0);

  PetscReal AV = 6.022140857 * 1.0e8 * pow(2.0, t / avg_cell_cyc_time); // cell's volume
  u(4) = 0.012e09 / AV;
  u(6) = 0.00012e09 / AV;
  u(8) = 0.05e09 / AV;
  return 0;
}
}

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

using namespace six_species_cme;
using namespace pacmensl;

void output_marginals(MPI_Comm comm, std::string model_name, PartitioningType fsp_par_type,
                      PartitioningApproach fsp_repart_approach, std::string constraint_type,
                      DiscreteDistribution &solution, arma::Row<int> constraints);

void output_performance(MPI_Comm comm,std::string &model_name,PartitioningType fsp_par_type,
                        PartitioningApproach fsp_repart_approach,std::string constraint_type,
                        ODESolverType ode_type,
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

  // Register PETSc stages for profiling
  PetscLogStage stages[2];
  petsc_err = PetscLogStageRegister("Solve with adaptive default state set shape", &stages[0]);
  CHKERRQ(petsc_err);
  petsc_err = PetscLogStageRegister("Solve with fixed default state set shape", &stages[1]);
  CHKERRQ(petsc_err);

  std::string part_type;
  std::string part_approach;

  std::string model_name = "transcr_reg_6d";
  Model model(SM, t_fun, propensity, nullptr, nullptr, {4, 6, 8});

  PetscReal t_final = 60.00 * 5;
  PetscReal fsp_tol = 1.0e-4;
  arma::Mat<PetscInt> X0 = {2, 6, 0, 2, 0, 0};
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
  fsp_solver.SetInitialDistribution(X0, p0);
  fsp_solver.SetModel(model);
  DiscreteDistribution solution;

  petsc_err = PetscLogStagePush(stages[0]);
  CHKERRQ(petsc_err);
  // Solve using adaptive default constraints
  fsp_solver.SetInitialBounds(rhs_constr);
  fsp_solver.SetInitialBounds(rhs_constr_hyperrec);
  fsp_solver.SetExpansionFactors(expansion_factors_hyperrec);
  fsp_solver.SetFromOptions();
  fsp_solver.SetUp();
  solution = fsp_solver.Solve(t_final, fsp_tol, 0);
  std::shared_ptr<const StateSetConstrained> fss = std::static_pointer_cast<const StateSetConstrained>(fsp_solver.GetStateSet());
  arma::Row<int> final_hyperrec_constr = fss->GetShapeBounds();
  if (fsp_log_events) {
    output_performance(PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                       std::string("adaptive_default"), fsp_odes_type, fsp_solver);
  }
  if (output_marginal) {
    output_marginals(PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                     std::string("adaptive_default"), solution, final_hyperrec_constr);
  }
  fsp_solver.ClearState();
  PetscPrintf(comm, "\n ================ \n");

  petsc_err = PetscLogStagePop();
  CHKERRQ(petsc_err);
  petsc_err = PetscLogStagePush(stages[1]);
  CHKERRQ(petsc_err);
  // Solve using fixed default constraints
  fsp_solver.SetInitialBounds(final_hyperrec_constr);
  fsp_solver.SetUp();
  solution = fsp_solver.Solve(t_final, fsp_tol, 0);
  if (fsp_log_events) {
    output_performance(PETSC_COMM_WORLD, model_name, fsp_par_type, fsp_repart_approach,
                       std::string("fixed_default"), fsp_odes_type, fsp_solver);
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

void output_performance(MPI_Comm comm,std::string &model_name,PartitioningType fsp_par_type,
                        PartitioningApproach fsp_repart_approach,std::string constraint_type,
                        ODESolverType ode_type,
                        FspSolverMultiSinks &fsp_solver)
{
  int myRank,num_procs;
  MPI_Comm_rank(comm,&myRank);
  MPI_Comm_size(comm,&num_procs);

  std::string ode;
  if (ode_type == KRYLOV)
  {
    ode = "krylov";
  } else
  {
    ode = "cvode";
  }

  std::string part_type;
  std::string part_approach;
  part_type     = part2str(fsp_par_type);
  part_approach = partapproach2str(fsp_repart_approach);

  // Output time breakdowns
  FspSolverComponentTiming    sum_times, min_times, max_times;
  sum_times = fsp_solver.ReduceComponentTiming("sum");
  min_times = fsp_solver.ReduceComponentTiming("min");
  max_times = fsp_solver.ReduceComponentTiming("max");

  if (myRank == 0)
  {
    struct stat buffer;
    int fstat;

    std::string   filename =
                      model_name + "_time_breakdown.dat";

    fstat = stat (filename.c_str(), &buffer);

    std::ofstream file;
    file.open(filename,std::ios_base::app);

    if (fstat != 0){
      file << "ncpu, partitioner, fsp_shape, ode_solver, min_cput, max_cput, avg_cput, mat_gen_time, ode_time, state_expand_time, min_flops, max_flops, avg_flops \n";
    }

    file << num_procs << ","
         << part_type << ","
         << constraint_type << ","
         << ode << ","
         << min_times.TotalTime << ","
         << max_times.TotalTime << ","
         << sum_times.TotalTime/num_procs << ","
         << sum_times.MatrixGenerationTime/num_procs << ","
         << sum_times.ODESolveTime/num_procs << ","
         << sum_times.StatePartitioningTime/num_procs
         << min_times.TotalFlops << ","
         << max_times.TotalFlops << ","
         << sum_times.TotalFlops/num_procs << "\n"
        ;
    file.close();
  }

  FiniteProblemSolverPerfInfo perf_info   = fsp_solver.GetSolverPerfInfo();

  if (myRank == 0){
    std::string filename =
                    model_name + "_perf_info_" + std::to_string(num_procs) + "_" + part_type + "_" + part_approach + "_" +
                        constraint_type + ".dat";
    std::ofstream file;
    file.open(filename);
    file << "Model time, ODEs size, Average processor time (sec) \n";
    for (auto i{0}; i < perf_info.n_step; ++i)
    {
      file << perf_info.model_time[i] << "," << perf_info.n_eqs[i] << "," << perf_info.cpu_time[i] << "\n";
    }
    file.close();
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
