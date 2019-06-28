//
// Created by Huy Vo on 12/4/18.
//

#ifndef PACMENSL_FINITESTATESUBSET_H
#define PACMENSL_FINITESTATESUBSET_H

#include <zoltan.h>
#include <petscis.h>

#include "StatePartitioner.h"
#include "Sys.h"
#include "pacmenMath.h"

namespace pacmensl {

struct FiniteStateSubsetLogger {
  /// For event logging
  PetscLogEvent state_exploration_event;
  PetscLogEvent check_constraints_event;
  PetscLogEvent add_states_event;
  PetscLogEvent call_partitioner_event;
  PetscLogEvent zoltan_dd_stuff_event;
  PetscLogEvent total_update_dd_event;
  PetscLogEvent distribute_frontiers_event;

  void register_all(MPI_Comm comm);

  void event_begin(PetscLogEvent event);

  void event_end(PetscLogEvent event);
};


/**
 * @brief Base class for the Finite State Subset object.
 * @details The Finite State Subset object contains the data and methods related to the storage, management, and
 * parallel
 * distribution of the states included by the Finite State Projection algorithm. Our current implementation relies on
 * Zoltan's dynamic load-balancing tools for the parallel distribution of these states into the processors.
 * */
class StateSetBase {
 public: NOT_COPYABLE_NOT_MOVABLE(StateSetBase);

  explicit StateSetBase(MPI_Comm new_comm);
  PacmenslErrorCode SetNumSpecies(int num_species);
  int SetStoichiometryMatrix(const arma::Mat<int> &SM);
  int AddStates(const arma::Mat<int> &X);
  int SetLoadBalancingScheme(PartitioningType type, PartitioningApproach approach = PartitioningApproach::REPARTITION);

  virtual PacmenslErrorCode SetUp();

  virtual PacmenslErrorCode Expand() { return 0; };

  arma::Row<PetscInt> State2Index(arma::Mat<PetscInt> &state) const;
  void State2Index(arma::Mat<PetscInt> &state, int *indx) const;
  void State2Index(int num_states, const int *state, int *indx) const;

  MPI_Comm GetComm() const;
  int GetNumLocalStates() const;
  int GetNumGlobalStates() const;
  int GetNumSpecies() const;
  int GetNumReactions() const;
  const arma::Mat<int> &GetStatesRef() const;
  arma::Mat<int> CopyStatesOnProc() const;
  void CopyStatesOnProc(int num_local_states, int *state_array) const;
  std::tuple<int, int> GetOrderingStartEnd() const;

  int Clear();
  virtual ~StateSetBase();

 protected:

  static const int hash_table_length_ = 1000000;

  MPI_Comm comm_ = nullptr;

  int set_up_ = 0;
  int stoich_set_ = 0;

  int comm_size_;
  int my_rank_;

  arma::Mat<int> stoichiometry_matrix_; ///< Stoichiometry matrix. Initialized via SetStoichiometryMatrix().
  int num_species_ = 0; ///< Number of species

  int num_reactions_ = 0; ///< Number of reactions. Initialized via SetStoichiometryMatrix().
  int num_global_states_ = 0; ///< Total number of states on all owning processors. This is determined via \ref
  // update_state_layout_.
  int num_local_states_ = 0; ///< Number of states own by this processor.

  double lb_threshold_ = 0.2; ///< Threshold for calling load-balancing after state set expansion. Only call
  // load-balancing if the new state set has more than lb_threshold_*100 percent of the old state set.
  int num_global_states_old_ = 0; ///< Number of global states before a call to Expand(). This helps determine
  // whether load balancing is needed.

  FiniteStateSubsetLogger logger_; ///< Data structure for logging.

  PartitioningType lb_type_ = PartitioningType::GRAPH; ///< Type of partitioning in the load-balancing algorithm. See also PartitioningType.
  PartitioningApproach lb_approach_ = PartitioningApproach::REPARTITION; ///< Approach to partitioning. See PartitioningApproach.
  StatePartitioner partitioner_; ///< Object to do load-balancing for the owning StateSetBase object.

  /// Armadillo array to store states owned by this processor
  /**
   * States are stored as column vectors of integers.
   */
  arma::Mat<int> local_states_;

  /**
   * @brief Status of local states.
   * @details Status(i) = 1 if state i (in local ordering) is active, -1 if inactive.
   */
  arma::Row<char> local_states_status_;

  /**
   * @brief Parallel layout of the state set.
   * @details state_layout_(i) = number of states owned by processor i.
   */
  arma::Row<int> state_layout_;

  /**
   * @brief Starting global indices of local state subsets.
   * @details processor i owns states with global indices from ind_starts_(i) to ind_starts(i+1)-1.
   */
  arma::Row<int> ind_starts_;

  /**
   * @brief local indexing of frontier states on this processor.
   */
  arma::uvec frontier_lids_;

  /**
   * @brief global ids of frontier states on this processor. Each global id is a multi-dimensional integral vector.
   */
  arma::Mat<int> local_frontier_gids_;

  /**
   * @brief The actual frontier states assigned for this processor. These frontiers need not be owned by the
   * processor (contrary to those in local_states_).
   */
  arma::Mat<int> frontiers_;


  /// Directory of states
  /**
   * This is essentially a parallel hash table. We use it to store existing states for fast lookup.
   * The GID/key field of each entry is the multi-dimensional state vector.
   * The LID field is the local index of the state in its owning processor.
   * The data field is the state's status. 0: inactive, 1: active. Active states are used to generate new states during
   * state space exploration.
   *
   * __See also__
   * Zoltan's manual page: http://www.cs.sandia.gov/Zoltan/ug_html/ug_util_dd.html
   */
  Zoltan_DD_Struct *state_directory_ = nullptr;

  /// Zoltan struct for load-balancing the state space search
  Zoltan_Struct *zoltan_explore_ = nullptr;

  void init_zoltan_parameters();

  void distribute_frontiers();

  void load_balance();

  int update_layout();

  void update_state_indices();
  void update_state_status(arma::Mat<PetscInt> states, arma::Row<char> status);
  void update_state_indices_status(arma::Mat<PetscInt> states, arma::Row<PetscInt> local_ids,
                                   arma::Row<char> status);
  void retrieve_state_status();

  static int zoltan_num_frontier(void *data, int *ierr);
  static void zoltan_frontier_list(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                                   ZOLTAN_ID_PTR local_ids, int wgt_dim, float *obj_wgts, int *ierr);
  static int zoltan_frontier_size(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id,
                                  ZOLTAN_ID_PTR local_id, int *ierr);
  static void
  pack_frontiers(void *data, int num_gid_entries, int num_lid_entries, int num_ids, ZOLTAN_ID_PTR global_ids,
                 ZOLTAN_ID_PTR local_ids, int *dest, int *sizes, int *idx, char *buf, int *ierr);
  static void
  unpack_frontiers(void *data, int num_gid_entries, int num_ids, ZOLTAN_ID_PTR global_ids, int *sizes, int *idx,
                   char *buf, int *ierr);
  static void mid_frontier_migration(void *data, int num_gid_entries, int num_lid_entries, int num_import,
                                     ZOLTAN_ID_PTR import_global_ids, ZOLTAN_ID_PTR import_local_ids,
                                     int *import_procs, int *import_to_part, int num_export,
                                     ZOLTAN_ID_PTR export_global_ids, ZOLTAN_ID_PTR export_local_ids,
                                     int *export_procs, int *export_to_part, int *ierr);
};
}

#endif //PACMENSL_FINITESTATESUBSET_H
