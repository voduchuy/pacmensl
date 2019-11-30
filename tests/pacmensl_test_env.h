//
// Created by Huy Vo on 2019-06-21.
//

#ifndef PACMENSL_MY_TEST_ENV_H
#define PACMENSL_MY_TEST_ENV_H

#include "Sys.h"
#include "gtest_mpi_listener.h"

namespace pacmensl {
namespace test {
class PACMENSLEnvironment : public ::testing::Environment {
 public:
  PACMENSLEnvironment(int argc, char* argv[]){
    argc_ = argc;
    argv_ = argv;
    int  err  = PACMENSLInit(&argc_, &argv_, ( char * ) 0);
//    ASSERT_FALSE(err);
  }

  void SetUp() override {

  }

  void TearDown() override {

  }

  ~PACMENSLEnvironment() override {
    int err = PACMENSLFinalize();
//    ASSERT_FALSE(err);
  }

  int argc_;
  char **argv_;
};
}
}

int main(int argc, char *argv[]) {
  // Initialize MPI
  MPI_Init(&argc, &argv);
  PetscInitialize(&argc, &argv, NULL, NULL);

  ::testing::InitGoogleTest(&argc, argv);

  ::testing::AddGlobalTestEnvironment(new pacmensl::test::PACMENSLEnvironment(argc, argv));

  // Get the event listener list.
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();

  // Remove default listener
  delete listeners.Release(listeners.default_result_printer());

  // Adds MPI listener; Google Test owns this pointer
  listeners.Append(new MPIMinimalistPrinter);

  int ierr = RUN_ALL_TESTS();

  if (ierr == 0 ){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) printf("SUCCESS!\n");
  }

  PetscFinalize();
  MPI_Finalize();
  return ierr;
}

#endif //PACMENSL_MY_TEST_ENV_H
