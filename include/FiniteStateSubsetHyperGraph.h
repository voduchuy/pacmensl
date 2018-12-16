//
// Created by Huy Vo on 12/15/18.
//

#ifndef PARALLEL_FSP_FINITESTATESUBSETZOLTAN_H
#define PARALLEL_FSP_FINITESTATESUBSETZOLTAN_H

#include "FiniteStateSubset.h"

namespace cme{
    namespace petsc{
        class FiniteStateSubsetHyperGraph: public FiniteStateSubset{
            struct HyperGraphData{
                int num_local_vertices;
                int num_local_pins;
                ZOLTAN_ID_PTR vtx_gid;
                int *vtx_edge_ptr;
                ZOLTAN_ID_PTR pin_gid;
            } hg_data;
        public:
            explicit FiniteStateSubsetHyperGraph(MPI_Comm new_comm);;
            void GenerateStatesAndOrdering() override;
            ~FiniteStateSubsetHyperGraph();

            friend void zoltan_get_hypergraph_size(void *data, int *num_lists, int *num_pins, int *format, int *ierr);
            friend void zoltan_get_hypergraph(void *data, int num_gid_entries, int num_vtx_edge, int num_pins,
                    int format, ZOLTAN_ID_PTR vtx_edge_gid, int *vtx_edge_ptr, ZOLTAN_ID_PTR pin_gid, int *ierr);
        };
        void zoltan_get_hypergraph_size(void *data, int *num_lists, int *num_pins, int *format, int *ierr);
        void zoltan_get_hypergraph(void *data, int num_gid_entries, int num_vtx_edge, int num_pins,
                                          int format, ZOLTAN_ID_PTR vtx_edge_gid, int *vtx_edge_ptr, ZOLTAN_ID_PTR pin_gid, int *ierr);
    }
}


#endif //PARALLEL_FSP_FINITESTATESUBSETZOLTAN_H
