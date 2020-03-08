static char help[] = "Advance_ small CMEs to benchmark intranode performance.\n\n";

#include <fstream>
#include <iomanip>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <armadillo>
#include <cmath>
#include <sys/stat.h>
#include "pacmensl_all.h"

namespace hog1p_cme {
// stoichiometric matrix of the toggle switch model
arma::Mat<PetscInt> SM{{1,-1,-1,0,0,0,0,0,0},
                       {0,0,0,1,0,-1,0,0,0},
                       {0,0,0,0,1,0,-1,0,0},
                       {0,0,0,0,0,1,0,-1,0},
                       {0,0,0,0,0,0,1,0,-1},};

// reaction parameters
const PetscReal k12{1.29},k23{0.0067},k34{0.133},k32{0.027},k43{0.0381},k21{1.0e0},kr21{0.005},kr31{
    0.45},      kr41{0.025},kr22{0.0116},kr32{0.987},kr42{0.0538},trans{0.01},gamma1{0.001},gamma2{0.0049},
// parameters for the time-dependent factors
                r1{6.9e-5},r2{7.1e-3},eta{3.1},Ahog{9.3e09},Mhog{6.4e-4};

// propensity function
inline PetscReal hog_propensity(int *X,int k)
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
    outputs[i] = hog_propensity(X[i],reaction);
  }
  return 0;
}

// function to compute the time-dependent coefficients of the propensity functions
int t_fun(double t,int num_coefs,PetscReal *outputs,void *args)
{
  if (num_coefs != 9) return -1;

  arma::Row<double> u(outputs,9,false,true);
  u.fill(1.0);

  double h1 = (1.0 - exp(-r1 * t)) * exp(-r2 * t);

  double hog1p = pow(h1 / (1.0 + h1 / Mhog),eta) * Ahog;

  u(2) = std::max(0.0,3200.0 - 7710.0 * (hog1p));

  return 0;
}

// Function to constraint the shape of the Fsp
int lhs_constr(PetscInt num_species,PetscInt num_constrs,PetscInt num_states,PetscInt *states,int *vals,
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

arma::Row<int>    rhs_constr_hyperrec{3,10,10,10,10};
arma::Row<double> expansion_factors_hyperrec{0.0,0.25,0.25,0.25,0.25};
arma::Row<int>    rhs_constr{3,10,10,10,10,10,10};
arma::Row<double> expansion_factors{0.0,0.2,0.2,0.2,0.2,0.2,0.2};
}

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

using namespace hog1p_cme;
using namespace pacmensl;

void output_marginals(MPI_Comm comm,std::string model_name,PartitioningType fsp_par_type,
                      PartitioningApproach fsp_repart_approach,std::string constraint_type,
                      DiscreteDistribution &solution,arma::Row<int> constraints);

void output_performance(MPI_Comm comm,std::string &model_name,PartitioningType fsp_par_type,
                        PartitioningApproach fsp_repart_approach,std::string constraint_type,
                        ODESolverType ode_type,
                        FspSolverMultiSinks &fsp_solver);

int ParseOptions(MPI_Comm comm,
                 PartitioningType &fsp_par_type,
                 PartitioningApproach &fsp_repart_approach,
                 PetscBool &output_marginal,
                 PetscBool &fsp_log_events,
                 ODESolverType &ode_solver);

int main(int argc,char *argv[])
{
  Environment my_env(&argc,&argv,help);

  PetscMPIInt    ierr,myRank,num_procs;
  PetscErrorCode petsc_err;
  MPI_Comm       comm;
  MPI_Comm_dup(PETSC_COMM_WORLD,&comm);
  MPI_Comm_size(comm,&num_procs);
  PetscPrintf(comm,"Solving with %d processors.\n",num_procs);

  // Register PETSc stages
  PetscLogStage stages[2];
  petsc_err = PetscLogStageRegister("Solve with adaptive default state set shape",&stages[0]);
  CHKERRQ(petsc_err);
  petsc_err = PetscLogStageRegister("Solve with fixed default state set shape",&stages[1]);
  CHKERRQ(petsc_err);

  // Set up CME
  std::string         model_name = "hog1p";
  Model               hog1p_model(hog1p_cme::SM,hog1p_cme::t_fun,hog1p_cme::propensity,
                                  nullptr,nullptr,{2});
  PetscReal           t_final    = 60 * 3.0;
  PetscReal           fsp_tol    = 1.0e-4;
  arma::Mat<PetscInt> X0         = {0,0,0,0,0};
  X0 = X0.t();
  arma::Col<PetscReal> p0 = {1.0};


  // Default options
  PartitioningType     fsp_par_type        = PartitioningType::GRAPH;
  PartitioningApproach fsp_repart_approach = PartitioningApproach::REPARTITION;
  ODESolverType        fsp_odes_type       = CVODE;
  PetscBool            output_marginal     = PETSC_FALSE;
  PetscBool            fsp_log_events      = PETSC_FALSE;

  ierr = ParseOptions(comm,fsp_par_type,fsp_repart_approach,output_marginal,fsp_log_events,fsp_odes_type);
  CHKERRQ(ierr);

  FspSolverMultiSinks fsp_solver(comm,fsp_par_type,fsp_odes_type);
  fsp_solver.SetModel(hog1p_model);
  fsp_solver.SetInitialDistribution(X0,p0);
  fsp_solver.SetFromOptions();
  DiscreteDistribution solution;

  petsc_err = PetscLogStagePush(stages[0]);
  CHKERRQ(petsc_err);

  // Solve using adaptive default constraints
  fsp_solver.SetInitialBounds(rhs_constr_hyperrec);
  fsp_solver.SetExpansionFactors(expansion_factors_hyperrec);
  fsp_solver.SetFromOptions();
  solution = fsp_solver.Solve(t_final,fsp_tol,0);
  std::shared_ptr<const StateSetConstrained>
                 fss                   = std::static_pointer_cast<const StateSetConstrained>(fsp_solver.GetStateSet());
  arma::Row<int> final_hyperrec_constr = fss->GetShapeBounds();
  if (fsp_log_events)
  {
    output_performance(PETSC_COMM_WORLD,model_name,fsp_par_type,fsp_repart_approach,
                       std::string("adaptive_default"),fsp_odes_type,fsp_solver);
  }
  if (output_marginal)
  {
    output_marginals(PETSC_COMM_WORLD,model_name,fsp_par_type,fsp_repart_approach,
                     std::string("adaptive_default"),solution,final_hyperrec_constr);
  }
  fsp_solver.ClearState();
  PetscPrintf(comm,"\n ================ \n");

  petsc_err = PetscLogStagePop();
  CHKERRQ(petsc_err);
  petsc_err = PetscLogStagePush(stages[1]);
  CHKERRQ(petsc_err);
  // Solve using fixed default constraints
  fsp_solver.SetInitialBounds(final_hyperrec_constr);
  solution = fsp_solver.Solve(t_final,fsp_tol,0);
  if (fsp_log_events)
  {
    output_performance(PETSC_COMM_WORLD,model_name,fsp_par_type,fsp_repart_approach,
                       std::string("fixed_default"),fsp_odes_type,fsp_solver);
  }
  if (output_marginal)
  {
    output_marginals(PETSC_COMM_WORLD,model_name,fsp_par_type,fsp_repart_approach,
                     std::string("fixed_default"),solution,final_hyperrec_constr);
  }
  fsp_solver.ClearState();
  PetscPrintf(comm,"\n ================ \n");
  return ierr;
}

int ParseOptions(MPI_Comm comm,
                 PartitioningType &fsp_par_type,
                 PartitioningApproach &fsp_repart_approach,
                 PetscBool &output_marginal,
                 PetscBool &fsp_log_events,
                 ODESolverType &ode_solver)
{
  std::string part_type;
  std::string part_approach;
  part_type     = part2str(fsp_par_type);
  part_approach = partapproach2str(fsp_repart_approach);

  // Read options for fsp
  char      opt[100];
  PetscBool opt_set;
  int       ierr;
  ierr = PetscOptionsGetString(NULL,PETSC_NULL,"-fsp_partitioning_type",opt,100,&opt_set);
  CHKERRQ(ierr);
  if (opt_set)
  {
    fsp_par_type = str2part(std::string(opt));
  }

  ierr = PetscOptionsGetString(NULL,PETSC_NULL,"-fsp_repart_approach",opt,100,&opt_set);
  CHKERRQ(ierr);
  if (opt_set)
  {
    fsp_repart_approach = str2partapproach(std::string(opt));
  }

  ierr = PetscOptionsGetString(NULL,PETSC_NULL,"-fsp_output_marginal",opt,100,&opt_set);
  CHKERRQ(ierr);
  if (opt_set)
  {
    if (strcmp(opt,"1") == 0 || strcmp(opt,"true") == 0)
    {
      output_marginal = PETSC_TRUE;
    }
  }

  ierr = PetscOptionsGetString(NULL,PETSC_NULL,"-fsp_log_events",opt,100,&opt_set);
  CHKERRQ(ierr);
  if (opt_set)
  {
    if (strcmp(opt,"1") == 0 || strcmp(opt,"true") == 0)
    {
      fsp_log_events = PETSC_TRUE;
    }
  }

  ierr = PetscOptionsGetString(NULL,PETSC_NULL,"-fsp_use_solver",opt,100,&opt_set);
  CHKERRQ(ierr);
  if (opt_set)
  {
    if (strcmp(opt,"krylov") == 0)
    {
      ode_solver = KRYLOV;
    } else
    {
      ode_solver = CVODE;
    }
  }

  PetscPrintf(comm,"Partitiniong option %s \n",part2str(fsp_par_type).c_str());
  PetscPrintf(comm,"Repartitoning option %s \n",partapproach2str(fsp_repart_approach).c_str());
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
      file << "ncpu,partitioner,fsp_shape,ode_solver,min_cput,max_cput,avg_cput,mat_gen_time,ode_time,state_expand_time,min_flops,max_flops,avg_flops \n";
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
         << sum_times.StatePartitioningTime/num_procs  << ","
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

void output_marginals(MPI_Comm comm,std::string model_name,PartitioningType fsp_par_type,
                      PartitioningApproach fsp_repart_approach,std::string constraint_type,
                      DiscreteDistribution &solution,arma::Row<int> constraints)
{
  int myRank,num_procs;
  MPI_Comm_rank(comm,&myRank);
  MPI_Comm_size(comm,&num_procs);

  std::string part_type;
  std::string part_approach;
  part_type     = part2str(fsp_par_type);
  part_approach = partapproach2str(fsp_repart_approach);

  /* Compute the marginal distributions */
  std::vector<arma::Col<PetscReal>> marginals(solution.states_.n_rows);
  for (PetscInt                     i{0}; i < marginals.size(); ++i)
  {
    marginals[i] = Compute1DMarginal(solution,i);
  }

  MPI_Comm_rank(PETSC_COMM_WORLD,&myRank);
  if (myRank == 0)
  {
    for (PetscInt i{0}; i < marginals.size(); ++i)
    {
      std::string filename =
                      model_name + "_marginal_" + std::to_string(i) + "_" + std::to_string(num_procs) + "_" +
                          part_type + "_" + part_approach + "_" + constraint_type + ".dat";
      marginals[i].save(filename,arma::raw_ascii);
    }
    std::string   filename =
                      model_name + "_constraint_bounds_" + std::to_string(num_procs) + "_" + part_type + "_"
                          + part_approach +
                          "_" + constraint_type + ".dat";
    constraints.save(filename,arma::raw_ascii);
  }
}
