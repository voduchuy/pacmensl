//
// Created by Huy Vo on 2019-06-25.
//

#include "StationaryFspSolverMultiSinks.h"

pacmensl::StationaryFspSolverMultiSinks::StationaryFspSolverMultiSinks(MPI_Comm comm) {
  int ierr;
  ierr = MPI_Comm_dup(comm, &comm_);
  PACMENSLCHKERREXCEPT(ierr);
}
