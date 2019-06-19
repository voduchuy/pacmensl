//
// Created by Huy Vo on 12/12/18.
//

static char help[] = "Generate Finite State Subset and output to files.\n\n";

#include<iomanip>
#include<memory.h>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>
#include <armadillo>
#include <cmath>
#include "toggle_model.h"
#include "hog1p_5d_model.h"
#include "transcription_regulation_6d_model.h"
#include "hog1p_3d_model.h"
#include "pacmensl_all.h"

using arma::dvec;
using arma::Col;
using arma::Row;

using std::cout;
using std::endl;

using namespace hog1p_cme;
using namespace pacmensl;

int main(int argc, char *argv[]) {
  //PACMENSL parallel environment object, must be created before using other PACMENSL's functionalities
  pacmensl::Environment my_env(&argc, &argv, help);

  PetscMPIInt myRank, num_procs;
  PetscErrorCode ierr;
  std::string model_name, part_option;
  fsp_constr_multi_fn *FSPConstraintFuns;
  Row<int> FSPBounds; // Size of the FSP
  arma::Mat<PetscInt> X0;
  arma::Mat<PetscInt> stoich_mat;
  PartitioningType fsp_par_type;
  std::fstream ofs;
  std::string filename;
  arma::Mat<PetscInt> local_states;

  MPI_Comm comm;
  MPI_Comm_dup(PETSC_COMM_WORLD, &comm);
  MPI_Comm_size(comm, &num_procs);
  MPI_Comm_rank(comm, &myRank);
  PetscPrintf(comm, "\n ================ \n");

  // Default options
  model_name = "toggle";
  FSPConstraintFuns = toggle_cme::lhs_constr;
  FSPBounds = toggle_cme::rhs_constr;
  stoich_mat = toggle_cme::SM;
  part_option = "graph";
  fsp_par_type = GRAPH;
  X0.set_size(2, 1);
  X0.fill(0);

  // Read options for fsp
  char opt[100];
  PetscBool opt_set;

  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_model", opt, 100, &opt_set);
  CHKERRQ(ierr);
  if (opt_set) {
    if (strcmp(opt, "transcr_reg_6d") == 0) {
      model_name = "transcr_reg_6d";
      FSPConstraintFuns = six_species_cme::lhs_constr;
      FSPBounds = six_species_cme::rhs_constr; // Size of the FSP
      X0.set_size(6, 1);
      X0.fill(0);
      stoich_mat = six_species_cme::SM;
      PetscPrintf(PETSC_COMM_WORLD, "Problem: Transcription regulation with 6 species.\n");
    } else if (strcmp(opt, "hog5d") == 0) {
      model_name = "hog1p";
      X0.set_size(5, 1);
      X0.fill(0);
      FSPConstraintFuns = hog1p_cme::lhs_constr;
      FSPBounds = hog1p_cme::rhs_constr;
      stoich_mat = hog1p_cme::SM;
      PetscPrintf(PETSC_COMM_WORLD, "Problem: Hog1p with 5 species.\n");
    } else if (strcmp(opt, "hog3d") == 0) {
      model_name = "hog3d";
      FSPConstraintFuns = hog3d_cme::lhs_constr;
      FSPBounds = hog3d_cme::rhs_constr;
      X0.set_size(3, 1);
      X0.fill(0);
      stoich_mat = hog3d_cme::SM;
      PetscPrintf(PETSC_COMM_WORLD, "Problem: Hog1p with 3 species.\n");
    } else {
      PetscPrintf(PETSC_COMM_WORLD, "Problem: Toggle-switch.\n");
    }
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "Problem: Toggle-switch.\n");
  }

  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_partitioning_type", opt, 100, &opt_set);
  CHKERRQ(ierr);
  if (opt_set) {
    fsp_par_type = str2part(opt);
    part_option = part2str(fsp_par_type);
    PetscPrintf(PETSC_COMM_WORLD, "FSP is partitionined with %s. \n", part_option.c_str());
  } else {
    PetscPrintf(PETSC_COMM_WORLD, "FSP is partitioned with Graph.\n");
  }

  PartitioningApproach fsp_repart_approach;
  ierr = PetscOptionsGetString(NULL, PETSC_NULL, "-fsp_repart_approach", opt, 100, &opt_set);
  CHKERRQ(ierr);
  if (opt_set) {
    fsp_repart_approach = str2partapproach(std::string(opt));
  }

  PetscPrintf(comm, "Distributing to %d processors.\n", num_procs);

  StateSetConstrained state_set(comm, X0.n_rows, fsp_par_type, fsp_repart_approach);
  state_set.SetStoichiometryMatrix(stoich_mat);
  state_set.SetInitialStates(X0);
    state_set.SetShape( FSPConstraintFuns, FSPBounds, nullptr );

  state_set.Expand();

  local_states = state_set.CopyStatesOnProc();
  local_states = local_states.t();

  filename =
      model_name + "_local_states_" + std::to_string(myRank) + "_of_" + std::to_string(num_procs) + "_" + part_option
          + ".dat";
  local_states.save(filename, arma::raw_ascii);

  PetscPrintf(comm, "\n ================ \n");

  return ierr;
}

