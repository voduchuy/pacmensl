static char help[] = "Timing the time to solve hog1p model to time 5 min.\n\n";

#include<iomanip>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <cme_util.h>
#include <armadillo>
#include <cmath>
#include <MatrixSet.h>
//#include <Magnus4FSP.h>
#include <FSPSolver.h>
#include "models/toggle_model.h"
#include "models/hog1p_5d_model.h"
#include "models/transcription_regulation_6d_model.h"
#include "models/hog1p_3d_model.h"

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

void petscvec_to_file(MPI_Comm comm, Vec x, const char *filename);

using namespace hog1p_cme;
using namespace cme::petsc;

int main(int argc, char *argv[]) {
    PetscMPIInt ierr, myRank, num_procs;

    ierr = PetscInitialize(&argc, &argv, (char *) 0, help);
    CHKERRQ(ierr);
    MPI_Comm comm;
    MPI_Comm_dup(PETSC_COMM_WORLD, &comm);
    MPI_Comm_size(comm, &num_procs);
    PetscPrintf(comm, "\n ================ \n");

    // Default options
    std::string model_name = "hog1p";
    std::string part_option = "parmetis";
    Row<PetscInt> FSPSize({3, 3, 3, 3, 3}); // Size of the FSP
    arma::Row<PetscReal> expansion_factors = {0.0, 0.5, 0.5, 0.5, 0.5};
    PetscReal t_final = 60.00 * 5;
    PetscReal fsp_tol = 1.0e-1;
    arma::Mat<PetscInt> X0 = {0, 0, 0, 0, 0};
    X0 = X0.t();
    arma::Col<PetscReal> p0 = {1.0};
    arma::Mat<PetscInt> stoich_mat = hog1p_cme::SM;
    TcoefFun t_fun = hog1p_cme::t_fun;
    PropFun propensity = hog1p_cme::propensity;
    PartioningType fsp_par_type = ParMetis;
    ODESolverType fsp_odes_type = CVODE_BDF;
    PetscBool output_marginal = PETSC_FALSE;
    PetscBool fsp_logging = PETSC_FALSE;
    PetscInt verbosity = 0;
    // Read options for fsp
    char opt[100];
    PetscBool opt_set;

    ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_model", opt, 100, &opt_set);
    CHKERRQ(ierr);
    if (opt_set) {
        if (strcmp(opt, "transcr_reg_6d") == 0) {
            model_name = "transcr_reg_6d";
            FSPSize = {10, 6, 1, 2, 1, 1}; // Size of the FSP
            expansion_factors = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
            t_final = 60.00 * 5;
            fsp_tol = 1.0e-6;
            X0 = {2, 6, 0, 2, 0, 0};
            X0 = X0.t();
            p0 = {1.0};
            stoich_mat = six_species_cme::SM;
            t_fun = six_species_cme::t_fun;
            propensity = six_species_cme::propensity;
            PetscPrintf(PETSC_COMM_WORLD, "Problem: Transcription regulation with 6 species.\n");
        } else if (strcmp(opt, "toggle") == 0) {
            model_name = "toggle";
            FSPSize = {20, 20}; // Size of the FSP
            expansion_factors = {0.5, 0.5};
            t_final = 14 * 3600;
            fsp_tol = 1.0e-6;
            X0 = {0, 0};
            X0 = X0.t();
            p0 = {1.0};
            stoich_mat = toggle_cme::SM;
            t_fun = toggle_cme::t_fun;
            propensity = toggle_cme::propensity;
            PetscPrintf(PETSC_COMM_WORLD, "Problem: Toggle switch with 2 species.\n");
        } else if (strcmp(opt, "hog3d") == 0) {
            model_name = "hog3d";
            part_option = "parmetis";
            FSPSize = {3, 10, 10}; // Size of the FSP
            expansion_factors = {0.0, 0.5, 0.5};
            t_final = 60.00 * 50;
            fsp_tol = 1.0e-6;
            X0 = {0, 0, 0};
            X0 = X0.t();
            p0 = {1.0};
            stoich_mat = hog3d_cme::SM;
            t_fun = hog3d_cme::t_fun;
            propensity = hog3d_cme::propensity;
            fsp_par_type = ParMetis;
            fsp_odes_type = CVODE_BDF;
            output_marginal = PETSC_FALSE;
            verbosity = 0;
            PetscPrintf(PETSC_COMM_WORLD, "Problem: Hog1p with 3 species.\n");
        } else {
            PetscPrintf(PETSC_COMM_WORLD, "Problem: Hog1p with 5 species.\n");
        }
    }


    ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_partitioning_type", opt, 100, &opt_set);
    CHKERRQ(ierr);
    if (opt_set) {
        if (strcmp(opt, "Linear") == 0) {
            fsp_par_type = Linear;
            part_option = "linear";
            PetscPrintf(PETSC_COMM_WORLD, "FSP is partitioned with natural ordering.\n");
        } else {
            PetscPrintf(PETSC_COMM_WORLD, "FSP is partitioned with ParMetis.\n");
        }
    }

    ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_output_marginal", opt, 100, &opt_set);
    CHKERRQ(ierr);
    if (opt_set) {
        if (strcmp(opt, "1") == 0 || strcmp(opt, "true") == 0) {
            output_marginal = PETSC_TRUE;
        }
    }
    PetscPrintf(comm, "Solving with %d processors.\n", num_procs);

    ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_verbosity", opt, 100, &opt_set);
    CHKERRQ(ierr);
    if (opt_set) {
        if (strcmp(opt, "1") == 0 || strcmp(opt, "true") == 0) {
            verbosity = 1;
        }
        if (strcmp(opt, "2") == 0) {
            verbosity = 2;
        }
    }

    ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_logging", opt, 100, &opt_set);
    CHKERRQ(ierr);
    if (opt_set) {
        if (strcmp(opt, "1") == 0 || strcmp(opt, "true") == 0) {
            fsp_logging = PETSC_TRUE;
        }
    }


    // Begin PETSC context
    {
        PetscReal tic, tic1, solver_time, total_time;

        tic = MPI_Wtime();
        FSPSolver fsp(PETSC_COMM_WORLD, fsp_par_type, fsp_odes_type);
        fsp.SetInitFSPSize(FSPSize);
        fsp.SetFSPTolerance(fsp_tol);
        fsp.SetFinalTime(t_final);
        fsp.SetStoichiometry(stoich_mat);
        fsp.SetTimeFunc(t_fun);
        fsp.SetPropensity(propensity);
        fsp.SetVerbosityLevel(verbosity);
        fsp.SetLogging(fsp_logging);
        fsp.SetExpansionFactors(expansion_factors);
        fsp.SetInitProbabilities(X0, p0);

        tic1 = MPI_Wtime();

        fsp.SetUp();
        fsp.Solve();

        solver_time = MPI_Wtime() - tic1;
        total_time = MPI_Wtime() - tic;

        PetscPrintf(comm, "Total time (including setting up) = %.2e \n", total_time);
        PetscPrintf(comm, "Solving time = %.2e \n", solver_time);

        MPI_Comm_rank(PETSC_COMM_WORLD, &myRank);
        if (myRank == 0) {
            {
                std::string filename = model_name + "_time_" + std::to_string(num_procs) + "_" + part_option + ".dat";
                std::ofstream file;
                file.open(filename);
                file << solver_time;
                file.close();
            }
        }

        if (fsp_logging){
            FSPSolverComponentTiming timings = fsp.GetAvgComponentTiming();
            if (myRank == 0){
                std::string filename = model_name + "_time_breakdown_" + std::to_string(num_procs) + "_" + part_option + ".dat";
                std::ofstream file;
                file.open(filename);
                file << "Component, Average Time (sec), Percentage \n";
                file << "State Partioning," << std::scientific << std::setprecision(2) << timings.StatePartitioningTime << "," << timings.StatePartitioningTime/solver_time*100.0 << "\n"
                     << "FSP Matrices Generation," << std::scientific << std::setprecision(2) << timings.MatrixGenerationTime << "," << timings.MatrixGenerationTime/solver_time*100.0 << "\n"
                     << "Solving truncated ODEs," << std::scientific << std::setprecision(2) << timings.ODESolveTime << "," << timings.ODESolveTime/solver_time*100.0 << "\n"
                     << "FSP Solution scattering," << std::scientific << std::setprecision(2) << timings.SolutionScatterTime << "," << timings.SolutionScatterTime/solver_time*100.0 << "\n"
                     << "Time-dependent Matrix action," << std::scientific << std::setprecision(2) << timings.RHSEvalTime << "," << timings.RHSEvalTime/solver_time*100.0 << "\n"
                     << "Total," << solver_time << "," << 100.0 << "\n";
                file.close();
            }
        }

        if (output_marginal) {
            /* Compute the marginal distributions */
            Vec P = fsp.GetP();
            FiniteStateSubset &state_set = fsp.GetStateSubset();
            std::vector<arma::Col<PetscReal>> marginals(FSPSize.n_elem);
            for (PetscInt i{0}; i < marginals.size(); ++i) {
                marginals[i] = cme::petsc::marginal(state_set, P, i);
            }

            MPI_Comm_rank(PETSC_COMM_WORLD, &myRank);
            if (myRank == 0) {
                for (PetscInt i{0}; i < marginals.size(); ++i) {
                    std::string filename =
                            model_name + "_marginal_" + std::to_string(i) + "_" + std::to_string(num_procs) + "_" +
                            part_option + ".dat";
                    marginals[i].save(filename, arma::raw_ascii);
                }
            }
        }
    }
    PetscPrintf(comm, "\n ================ \n");
    //End PETSC context
    ierr = PetscFinalize();
    return ierr;
}


void petscvec_to_file(MPI_Comm comm, Vec x, const char *filename) {
    PetscViewer viewer;
    PetscViewerCreate(comm, &viewer);
    PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &viewer);
    VecView(x, viewer);
    PetscViewerDestroy(&viewer);
}
