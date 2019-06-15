//
// Created by Huy Vo on 12/3/18.
//
static char help[] = "Test the generation of the distributed Finite State Subset for the toggle model.\n\n";

#include<petsc.h>
#include<petscvec.h>
#include<petscmat.h>
#include<petscao.h>
#include<armadillo>
#include"cme_util.h"
#include"StateSetConstrained.h"

using namespace pecmeal;

arma::Mat<PetscInt> SM{
    {1, -1, 0, 0},
    {0, 0, 1, -1}
};

int main(int argc, char *argv[]) {
  //PECMEAL parallel environment object, must be created before using other PECMEAL's functionalities
  pecmeal::Environment my_env(&argc, &argv, help);

  int ierr;

  MPI_Comm comm;
  ierr = MPI_Comm_dup(MPI_COMM_WORLD, &comm);
  CHKERRQ(ierr);
  int my_rank;
  MPI_Comm_rank(comm, &my_rank);
  arma::Mat<PetscInt> X0(2, 2);
  X0.col(0).fill(0);
  X0.col(1).fill(my_rank + 1);

  StateSetConstrained fsp(comm, 2, Graph, Repartition);
  fsp.SetStoichiometryMatrix(SM);
  fsp.SetInitialStates(X0);

  // Generate a small FSP
  arma::Row<int> fsp_size = {3, 3};
  fsp.SetShapeBounds(fsp_size);
  PetscPrintf(comm, "Initial states:\n");
  PetscPrintf(comm, "State | Petsc ordering \n");
  auto print_states = [](void *data) -> void {
    auto mat_data = (arma::Mat<PetscInt> *) data;
    std::cout << *mat_data;
    return;
  };
  arma::Mat<PetscInt> local_states = fsp.CopyStatesOnProc();
  arma::Row<PetscInt> petsc_indices = fsp.State2Index(local_states);
  arma::Mat<PetscInt> local_table = arma::join_horiz(local_states.t(), petsc_indices.t());
  pecmeal::sequential_action(comm, print_states, (void *) &local_table);
  fsp.Expand();
  local_states = fsp.CopyStatesOnProc();
  petsc_indices = fsp.State2Index(local_states);
  local_table = arma::join_horiz(local_states.t(), petsc_indices.t());
  pecmeal::sequential_action(comm, print_states, (void *) &local_table);
  CHKERRQ(ierr);
  MPI_Comm_free(&comm);
}
