#include <petscmat.h>
#include <petscvec.h>
#include <petsctime.h>
#include <fstream>
#include <iostream>

char help[] = "Testing the efficiency of the system through perfectly parallel matrix-vector products.";

int main(int argc, char *argv[]) {
  PetscInt ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help); CHKERRQ(ierr);

  MPI_Comm comm = PETSC_COMM_WORLD;
  PetscInt n_rows_global = 1000;
  PetscInt i_start, i_end, num_procs, rank_proc;

  MPI_Comm_size(comm, &num_procs);
  MPI_Comm_rank(comm, &rank_proc);

  /* Generate a complete diagonal parallel matrix, this means zero communication between processes */

  Mat A;
  MatCreate(comm, &A);
  MatSetFromOptions(A);
  MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n_rows_global, n_rows_global );
  MatSetUp(A);
  MatGetOwnershipRange(A, &i_start, &i_end);

  for (PetscInt i_row{i_start}; i_row < i_end; ++i_row)
  {
    MatSetValue(A, i_row, i_row, 1.0, INSERT_VALUES);
  }
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  /* Generate vectors for matmult */
  Vec x,y;
  VecCreate(comm, &x);
  VecSetFromOptions(x);
  VecSetSizes(x, PETSC_DECIDE, n_rows_global);
  VecSetUp(x);
  VecSet(x, 1.0);

  VecDuplicate(x, &y);

  /* Now do 1000 matrix-vector products */
  PetscReal tic;
  PetscTime(& tic);
  for (PetscInt i{0}; i < 1000; i++) {
    MatMult(A, x, y);
  }
  PetscReal matvec_time;
  PetscTime(& matvec_time);
  matvec_time -= tic;

  if (rank_proc == 0) {
    std::ofstream f;
    f.open("matvec_test.dat", std::ios_base::app);
    f << std::to_string(num_procs) << " " << matvec_time << "\n";
    f.close();
  }

  MatDestroy(&A);
  ierr = PetscFinalize();
  return 0;
}
