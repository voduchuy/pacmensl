
#include "cme_util.h"

namespace pecmeal {

int PecmealInit(int *argc, char ***argv, const char *help) {
  PetscErrorCode ierr;
  int mpi_initialized;
  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) MPI_Init(argc, argv);
  ierr = PetscInitialize(argc, argv, (char *) 0, help);
  CHKERRQ(ierr);
  float ver;
  ierr = Zoltan_Initialize(*argc, *argv, &ver);
  CHKERRQ(ierr);
  return 0;
}

int PecmealFinalize() {
  PetscErrorCode ierr;
  ierr = PetscFinalize();
  CHKERRQ(ierr);

  int mpi_finalized;
  MPI_Finalized(&mpi_finalized);
  if (!mpi_finalized) MPI_Finalize();
  return ierr;
  return 0;
}

void sequential_action(MPI_Comm comm, std::function<void(void *)> action, void *data) {
  int my_rank, comm_size;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_size);

  if (comm_size == 1) {
    action(data);
    return;
  }

  int print;
  MPI_Status status;
  if (my_rank == 0) {
    std::cout << "Processor " << my_rank << "\n";
    action(data);
    MPI_Send(&print, 1, MPI_INT, my_rank + 1, 1, comm);
  } else {
    MPI_Recv(&print, 1, MPI_INT, my_rank - 1, 1, comm, &status);
    std::cout << "Processor " << my_rank << "\n";
    action(data);
    if (my_rank < comm_size - 1) {
      MPI_Send(&print, 1, MPI_INT, my_rank + 1, 1, comm);
    }
  }
  MPI_Barrier(comm);
}

double round2digit(double x) {
  if (x == 0.0e0) return x;
  double p1 = std::pow(10.0e0, round(log10(x) - SQR1) - 1.0e0);
  return trunc(x / p1 + 0.55e0) * p1;
}
}
