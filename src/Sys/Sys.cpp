
#include "Sys.h"

namespace pacmensl {

static bool _PACMENSL_INIT_MPI   = false;
static bool _PACMENSL_INIT_PETSC = false;

int PACMENSLInit(int *argc, char ***argv, const char *help)
{
  PetscErrorCode ierr;

  int            mpi_initialized;
  PetscBool petsc_initialized;

  MPI_Initialized(&mpi_initialized);
  if (mpi_initialized == 0) {
    MPI_Init(argc, argv);
    _PACMENSL_INIT_MPI = true;
  }

  PetscInitialized(&petsc_initialized);
  if (petsc_initialized == PETSC_FALSE)
  {
    ierr = PetscInitialize(argc, argv, ( char * ) 0, help);
    CHKERRQ(ierr);
    _PACMENSL_INIT_PETSC = true;
  }

  float ver;
  if (argc)
  {
    ierr = Zoltan_Initialize(*argc, *argv, &ver);
  } else
  {
    ierr = Zoltan_Initialize(0, nullptr, &ver);
  }
  CHKERRQ(ierr);

  return 0;
}

int PACMENSLFinalize()
{
  PetscErrorCode ierr = 0;
  if (_PACMENSL_INIT_PETSC)
  {
    ierr = PetscFinalize();
    CHKERRQ(ierr);
  }

  if (_PACMENSL_INIT_MPI)
  {
    int mpi_finalized;
    MPI_Finalized(&mpi_finalized);
    if (!mpi_finalized) MPI_Finalize();
  }
  return ierr;
}

void sequential_action(MPI_Comm comm, std::function<void(void *)> action, void *data)
{
  int my_rank, comm_size;
  MPI_Comm_rank(comm, &my_rank);
  MPI_Comm_size(comm, &comm_size);

  if (comm_size == 1)
  {
    action(data);
    return;
  }

  int        print;
  MPI_Status status;
  if (my_rank == 0)
  {
    std::cout << "Processor " << my_rank << "\n";
    action(data);
    MPI_Send(&print, 1, MPI_INT, my_rank + 1, 1, comm);
  } else
  {
    MPI_Recv(&print, 1, MPI_INT, my_rank - 1, 1, comm, &status);
    std::cout << "Processor " << my_rank << "\n";
    action(data);
    if (my_rank < comm_size - 1)
    {
      MPI_Send(&print, 1, MPI_INT, my_rank + 1, 1, comm);
    }
  }
  MPI_Barrier(comm);
}

double round2digit(double x)
{
  if (x == 0.0e0) return x;
  double p1 = std::pow(10.0e0, round(log10(x) - SQR1) - 1.0e0);
  return trunc(x / p1 + 0.55e0) * p1;
}

Environment::Environment()
{
  if (~initialized)
  {
    PetscErrorCode ierr;
    int            mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized == 0)
    {
      MPI_Init(nullptr, nullptr);
      init_mpi = true;
    }
    PetscBool petsc_initialized;
    PetscInitialized(&petsc_initialized);
    if (petsc_initialized == PETSC_FALSE)
    {
      ierr       = PetscInitialize(nullptr, nullptr, ( char * ) 0, nullptr);
      init_petsc = true;
    }
    float ver;
    ierr        = Zoltan_Initialize(0, nullptr, &ver);
    initialized = true;
  }
}

Environment::Environment(int *argc, char ***argv, const char *help)
{
  if (~initialized)
  {
    PetscErrorCode ierr;
    int            mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (mpi_initialized == 0)
    {
      MPI_Init(argc, argv);
      init_mpi = true;
    }
    PetscBool petsc_initialized;
    PetscInitialized(&petsc_initialized);
    if (petsc_initialized == PETSC_FALSE)
    {
      ierr = PetscInitialize(argc, argv, ( char * ) 0, help);
      CHKERRABORT(MPI_COMM_WORLD, ierr);
      init_petsc = true;
    }
    float ver;
    if (argc)
    {
      ierr = Zoltan_Initialize(*argc, *argv, &ver);
    } else
    {
      ierr = Zoltan_Initialize(0, nullptr, &ver);
    }
    CHKERRABORT(MPI_COMM_WORLD, ierr);
    initialized = true;
  }
}

Environment::~Environment()
{
  if (initialized)
  {
    if (init_petsc)
    {
      PetscErrorCode ierr;
      ierr = PetscFinalize();
    }

    if (init_mpi)
    {
      int mpi_finalized;
      MPI_Finalized(&mpi_finalized);
      if (!mpi_finalized) MPI_Finalize();
    }
  }
}
}
