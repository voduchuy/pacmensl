//
// Created by Huy Vo on 2019-06-21.
//

#ifndef PACMENSL_MY_TEST_ENV_H
#define PACMENSL_MY_TEST_ENV_H

#include "gtest_mpi_listener.h"

namespace pacmensl { namespace test {
class PACMENSLEnvironment : public ::testing::Environment {
 public:
  void SetUp() override {
    char **argv;
    int argc = 0;
    int err = PACMENSLInit(&argc, &argv, ( char * ) 0);
    ASSERT_FALSE(err);
  }

  void TearDown() override {
    int err = PACMENSLFinalize();
    ASSERT_FALSE(err);
  }

  ~PACMENSLEnvironment() override {}
};
}}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);

  // Initialize MPI
  MPI_Init(&argc, &argv);

  ::testing::AddGlobalTestEnvironment(new pacmensl::test::PACMENSLEnvironment);

  // Get the event listener list.
  ::testing::TestEventListeners &listeners = ::testing::UnitTest::GetInstance()->listeners();

  // Remove default listener
  delete listeners.Release(listeners.default_result_printer());

  // Adds MPI listener; Google Test owns this pointer
  listeners.Append(new MPIMinimalistPrinter);

  return RUN_ALL_TESTS();
}

#endif //PACMENSL_MY_TEST_ENV_H
