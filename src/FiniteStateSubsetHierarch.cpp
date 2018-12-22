//
// Created by Huy Vo on 12/4/18.
//

#include <FiniteStateSubsetGraph.h>
#include <FiniteStateSubsetHierarch.h>


#include "FiniteStateSubsetGraph.h"

namespace cme {
    namespace parallel {

        FiniteStateSubsetHierarch::FiniteStateSubsetHierarch(MPI_Comm new_comm) : FiniteStateSubset(
                new_comm) {
            partitioning_type = Hierarch;

            Zoltan_Set_Param(zoltan, "LB_METHOD", "HIER");
            Zoltan_Set_Param(zoltan, "HIER_DEBUG_LEVEL", "0");
            Zoltan_Set_Param(zoltan, "RETURN_LISTS", "PARTS");
            Zoltan_Set_Param(zoltan, "DEBUG_LEVEL", "0");

            Zoltan_Set_Hier_Num_Levels_Fn(zoltan, &zoltan_hier_num_levels, (void*) this);
            Zoltan_Set_Hier_Method_Fn(zoltan, &zoltan_hier_method, (void*) this);
            Zoltan_Set_Hier_Part_Fn(zoltan, &zoltan_hier_part, (void*) this);

            //
            // Initialize hierarchical data
            //
            // Split processors based on shared memory
            MPI_Info mpi_info;
            MPI_Info_create(&mpi_info);
            MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, mpi_info, &my_node);

            int my_node_rank, my_intranode_rank;
            MPI_Comm node_leader;
            int is_leader;

            // My rank within the node
            MPI_Comm_rank(my_node, &my_intranode_rank);

            // Am I the leader of my node?
            is_leader = my_intranode_rank == 0? 1 : 0;

            // Distinguish node leaders and others
            MPI_Comm_split(comm, is_leader, 0, &node_leader);

            // If I am the leader, my rank in the leader group is my node rank
            MPI_Comm_rank(node_leader, &my_node_rank);

            // Broadcast my node rank to the others in my node group
            MPI_Bcast(&my_node_rank, 1, MPI_INT, 0, my_node);

            std::cout << "My node has rank " << my_node_rank << "\n";

            my_part[0] = my_node_rank;
            my_part[1] = my_intranode_rank;

            repart = PETSC_FALSE;
            MPI_Comm_free(&node_leader);
            MPI_Info_free(&mpi_info);
        }

        FiniteStateSubsetHierarch::~FiniteStateSubsetHierarch() {
            MPI_Comm_free(&my_node);
        }

        void FiniteStateSubsetHierarch::GenerateStatesAndOrdering() {
            PetscErrorCode ierr;
            // This can only be done after the stoichiometry has been set
            if (stoich_set == 0) {
                throw std::runtime_error("FiniteStateSubset: stoichiometry is required for Graph partitioning type.");
            }

            Zoltan_Set_Param(zoltan, "LB_APPROACH", "PARTITION");

            //
            // Initial temporary partitioning based on lexicographic ordering
            //
            arma::Mat<PetscInt> local_states_tmp = get_my_naive_local_states();

            //
            // Generate Graph data
            //
            generate_graph_data(local_states_tmp);
            generate_geometric_data(fsp_size, local_states_tmp);

            //
            // Use Zoltan to create partitioning, then wrap with Petsc's IS
            //
            call_zoltan_partitioner();

            //
            // Convert Zoltan's output to Petsc ordering and layout
            //
            compute_petsc_ordering_from_zoltan();

            // Generate local states
            get_local_states_from_ao();

            free_graph_data();
            free_geometric_data();
            free_zoltan_part_variables();
        }

        void FiniteStateSubsetHierarch::ExpandToNewFSPSize(arma::Row<PetscInt> new_fsp_size) {
            PetscErrorCode ierr;
            assert(new_fsp_size.n_elem == fsp_size.n_elem);
            for (auto i{0}; i < fsp_size.n_elem; ++i) {
                assert(new_fsp_size(i) >= fsp_size(i));
            }
            if (local_states.n_cols == 0) {
                PetscPrintf(PETSC_COMM_SELF,
                            "FiniteStateSubsetGraph: found empty local states array, probably because GenerateStatesAndOrdering was never called.\n");
                MPI_Abort(comm, -1);
            }
            arma::Row<PetscInt> fsp_size_old = fsp_size;
            SetSize(new_fsp_size);

            if (lex2petsc) AODestroy(&lex2petsc);
            if (vec_layout) PetscLayoutDestroy(&vec_layout);

            //
            // Switch Zoltan's mode to "repartition"
            //
            repart = PETSC_TRUE;
            //
            // Explore for new states that satisfy the new FSP bounds
            //
            PetscInt n_new_states;
            arma::Mat<PetscInt> new_candidates, local_states_tmp;
            arma::Row<PetscInt> is_new_states;

            new_candidates = get_my_naive_local_states();
            is_new_states.set_size(new_candidates.n_cols);

            n_new_states = 0;
            for (auto i{0}; i < new_candidates.n_cols; ++i) {
                is_new_states(i) = 0;
                for (auto j{0}; j < n_species; ++j) {
                    if (new_candidates(j, i) > fsp_size_old(j)) {
                        is_new_states(i) = 1;
                        n_new_states += 1;
                        break;
                    }
                }
            }
            PetscInt n_local_tmp = local_states.n_cols + n_new_states;
            local_states_tmp.set_size(fsp_size.n_elem, n_local_tmp);
            for (auto i{0}; i < local_states.n_cols; ++i) {
                local_states_tmp.col(i) = local_states.col(i);
            }
            for (auto i{0}, k{(PetscInt) local_states.n_cols}; i < new_candidates.n_cols; ++i) {
                if (is_new_states(i)) {
                    local_states_tmp.col(k) = new_candidates.col(i);
                    k++;
                }
            }

            //
            // Create the graph data
            //
            generate_graph_data(local_states_tmp);
            generate_geometric_data(new_fsp_size, local_states_tmp);

            //
            // Use Zoltan to create partitioning, then wrap with processor_id, then proceed as usual
            //
            call_zoltan_partitioner();

            //
            // Convert Zoltan's output to Petsc ordering and layout
            //
            compute_petsc_ordering_from_zoltan();

            //
            // Generate local states
            //
            get_local_states_from_ao();

            free_graph_data();
            free_geometric_data();
            free_zoltan_part_variables();
        }

        int zoltan_hier_num_levels(void *data, int *ierr) {
            auto fss_data = (FiniteStateSubsetHierarch *) data;
            *ierr = ZOLTAN_OK;
            return fss_data->num_levels;
        }

        int zoltan_hier_part(void *data, int level, int *ierr) {
            auto fss_data = (FiniteStateSubsetHierarch *) data;
            if (level >= fss_data->num_levels) {
                *ierr = ZOLTAN_FATAL;
                std::cout
                        << "Zoltan_hier_part requests higher level than the max number of levels in the hierarchical partitioning.\n";
                return -1;
            }
            *ierr = ZOLTAN_OK;
            return ((int) fss_data->my_part[level]);
        }

        void zoltan_hier_method(void *data, int level, struct Zoltan_Struct *zz, int *ierr) {
            auto fss_data = (FiniteStateSubsetHierarch *) data;
            if (level >= fss_data->num_levels) {
                *ierr = ZOLTAN_FATAL;
                std::cout
                        << "Zoltan_hier_part requests higher level than the max number of levels in the hierarchical partitioning.\n";
                return;
            }
            fss_data->set_zoltan_parameters(level, zz);
            *ierr = ZOLTAN_OK;
        }

        void FiniteStateSubsetHierarch::set_zoltan_parameters(int level, Zoltan_Struct *zz) {
            Zoltan_Set_Num_Obj_Fn(zz, &zoltan_num_obj, (void *) &this->adj_data);
            Zoltan_Set_Obj_List_Fn(zz, &zoltan_obj_list, (void *) &this->adj_data);
            Zoltan_Set_Num_Geom_Fn(zz, &zoltan_num_geom, (void *) this);
            Zoltan_Set_Geom_Multi_Fn(zz, &zoltan_geom_multi, (void *) this);
            Zoltan_Set_Num_Edges_Fn(zz, &zoltan_num_edges, (void *) &this->adj_data);
            Zoltan_Set_Edge_List_Fn(zz, &zoltan_edge_list, (void *) &this->adj_data);
            Zoltan_Set_HG_Size_CS_Fn(zz, &zoltan_get_hypergraph_size, (void *) &this->adj_data);
            Zoltan_Set_HG_CS_Fn(zz, &zoltan_get_hypergraph, (void *) &this->adj_data);
            switch (level) {
                case 0: // Hypergraph partitioning for inter-node level
                    Zoltan_Set_Param(zz, "LB_METHOD", "GRAPH");
                    Zoltan_Set_Param(zz, "GRAPH_PACKAGE", "Parmetis");
                    Zoltan_Set_Param(zz, "PARMETIS_METHOD", "PartGeomKway");
                    Zoltan_Set_Param(zz, "RETURN_LISTS", "PARTS");
                    Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0");
                    Zoltan_Set_Param(zz, "IMBALANCE_TOL", "1.01");
                    Zoltan_Set_Param(zz, "OBJ_WEIGHT_DIM", "0"); // use Zoltan default vertex weights
                    Zoltan_Set_Param(zz, "EDGE_WEIGHT_DIM", "0");// use Zoltan default hyperedge weights
                    Zoltan_Set_Param(zz, "CHECK_GRAPH", "0");
                    Zoltan_Set_Param(zz, "GRAPH_SYMMETRIZE", "TRANSPOSE");
                    if (repart) {
                        Zoltan_Set_Param(zz, "PARMETIS_METHOD", "AdaptiveRepart");
                        Zoltan_Set_Param(zz, "LB_APPROACH", "REPARTITION");
                    } else {
                    Zoltan_Set_Param(zz, "LB_APPROACH", "PARTITION");
                    }
                    break;
                case 1: // RCB for intra-node level
                    Zoltan_Set_Param(zz, "LB_METHOD", "RCB");
                    Zoltan_Set_Param(zz, "IMBALANCE_TOL", "1.01");
                    Zoltan_Set_Param(zz, "RETURN_LISTS", "PARTS");
                    Zoltan_Set_Param(zz, "DEBUG_LEVEL", "0");
                    break;
                default:
                    break;
            }
        }
    }
}
