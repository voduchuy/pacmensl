/*
MIT License

Copyright (c) 2020 Huy Vo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/**
 * \file StatePartitionerBase.h
 */
 
#ifndef PACMENSL_STATESETPARTITIONERBASE_H
#define PACMENSL_STATESETPARTITIONERBASE_H

#include <zoltan.h>
#include <armadillo>
#include <mpi.h>
#include "Sys.h"
#include "string.h"

 
namespace pacmensl {
/**
 * @enum pacmensl::PartitioningType
 * @brief enum class to represents the Zoltan method to use in load-balancing routines.
 */
enum class PartitioningType {
  BLOCK, ///< Use block method. Only object weights are considered. No communication cost is considered.
  GRAPH, ///< Balance object weights while trying to minimize communication. This model is symmetric, if A -> B then B -> A.
  HYPERGRAPH, ///< Balance object weights while trying to minimize communication. Communications could be one-way in hypergraph models.
  HIERARCHICAL ///< Hierarchical/hybrid method. Not yet supported.
};

/**
 * @enum pacmensl::PartitioningApproach
 * @brief enum class to represents the partitioning approach to use in load-balancing routines.
 */
enum class PartitioningApproach {
  FROMSCRATCH,  ///< Assume no redistribution cost.
  REPARTITION, ///< Minimize a trade-off of communication and data redistribution
  REFINE ///< Refine from an existing partition. Fast but not as high-quality as the other two options.
};

/**
 * @brief Base class for state space partitioning.
 * @details This class defines the public interface and implements the common methods and attributes of all partitioner objects.
 * @details It also implements a simple partitioning approach that assigns states equally to the processes without considering their topology.
 */
class StatePartitionerBase {
 public:
  /**
   * @brief Constructor.
   * @param _comm (in) Communicator context.
   */
  explicit StatePartitionerBase(MPI_Comm _comm);

  /**
   * @brief Set the load-balancing approach.
   * @param _approach (in) load-balancing approach.
   */
  void set_lb_approach(PartitioningApproach _approach) { approach = _approach; };

  /**
   * @brief Partition a distributed set of states.
   * @details This method is __collective__, meaning that it must be called by all owning processes.
   * @param states (in/out) Set of states owned by the calling process, arranged column-wise. On return gives the re-distributed states.
   * @param state_directory (in/out) Zoltan distributed directory of the states.
   * @param stoich_mat (in) Stoichiometry matrix.
   * @param layout (in/out) Array of number of states owned by each process.
   * @return Error code: 0 if success, -1 otherwise.
   */
  int partition(arma::Mat<int> &states, Zoltan_DD_Struct *state_directory, arma::Mat<int> &stoich_mat,
                int *layout);

  virtual ~StatePartitionerBase();

 protected:
  MPI_Comm         comm_;
  int              my_rank_;
  int              comm_size_;
  Zoltan_Struct    *zoltan_lb_      = nullptr;
  arma::Mat<int>   *state_ptr_      = nullptr;
  arma::Mat<int>   *stoich_mat_ptr_ = nullptr;
  Zoltan_DD_Struct *state_dir_ptr_  = nullptr;
  int              *ind_starts      = nullptr;

  PartitioningType     type     = PartitioningType::BLOCK;
  PartitioningApproach approach = PartitioningApproach::REPARTITION;

  int num_species_      = 0;
  int num_local_states_ = 0; ///< Number of local states held by the processor in the current partitioning
  int *layout_ = nullptr;
  int *states_indices_; ///< one-dimensional indices of the states
  float
      *states_weights_; ///< Computational weights associated with each state, here we assign to these weights the number of FLOPs needed

  virtual void set_zoltan_parameters();

  virtual void generate_data();

  virtual void free_data();

  void state2ordering(arma::Mat<PetscInt> &state, PetscInt *indx);

  /* Zoltan interface functions */
  static int zoltan_num_obj(void *data, int *ierr);

  static void zoltan_obj_list(void *data, int num_gid_entries, int num_lid_entries,
                              ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts,
                              int *ierr);

  static int zoltan_obj_size(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                             ZOLTAN_ID_PTR local_id, int *ierr);

  static void zoltan_pack_states(void *data, int num_gid_entries, int num_lid_entries, int num_ids,
                                 ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int *dest,
                                 int *sizes, int *idx, char *buf, int *ierr);

  static void zoltan_mid_migrate_pp(void *data, int num_gid_entries, int num_lid_entries, int num_import,
                                    ZOLTAN_ID_PTR import_global_ids, ZOLTAN_ID_PTR import_local_ids,
                                    int *import_procs, int *import_to_part,
                                    int num_export, ZOLTAN_ID_PTR export_global_ids,
                                    ZOLTAN_ID_PTR export_local_ids,
                                    int *export_procs, int *export_to_part, int *ierr);

  static void zoltan_unpack_states(void *data, int num_gid_entries, int num_ids, ZOLTAN_ID_PTR global_ids,
                                   int *sizes, int *idx, char *buf, int *ierr);
};

/* Utitlity functions to convert a string description of the partitioning method or approach to its corresponding enum class value */
std::string part2str(PartitioningType part);

PartitioningType str2part(std::string str);

std::string partapproach2str(PartitioningApproach part_approach);

PartitioningApproach str2partapproach(std::string str);
}

#endif //PACMENSL_STATESETPARTITIONERBASE_H
