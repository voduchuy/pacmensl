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

#ifndef PACMENSL_STATEPARTITIONERHYPERGRAPH_H
#define PACMENSL_STATEPARTITIONERHYPERGRAPH_H

#include "StatePartitionerBase.h"


namespace pacmensl {
  /**
   * @brief State space partitioner using the hyper-graph load-balancing model.
   */
    class StatePartitionerHyperGraph : public StatePartitionerBase {
    protected:
        int *num_edges; ///< Number of states that share information with each local states
        int num_reachable_states; ///< Number of nz entries on the rows of the FSP matrix corresponding to local states
        int *reachable_states; ///< Global indices of nz entries on the rows corresponding to local states
        float *edge_weights; ///< For storing the edge weights in graph model
        int *edge_ptr; ///< reachable_states[edge_ptr[i] to ege_ptr[i+1]-1] contains the ids of states connected to local state i

        void set_zoltan_parameters() override;

        void generate_data() override;

        void free_data() override;

        /* Zoltan interface */
        static void zoltan_hg_size(void *data, int *num_lists, int *num_pins, int *format, int *ierr);

        static void zoltan_hg_edge_list(void *data, int num_gid_entries, int num_vertices, int num_pins, int format,
                                        ZOLTAN_ID_PTR vtx_gid, int *vtx_edge_ptr, ZOLTAN_ID_PTR pin_gid, int *ierr);

        static void zoltan_hg_size_eweights(void *data, int *num_edges, int *ierr);

        static void zoltan_hg_edge_weights(void *data, int num_gid_entries, int num_lid_entries, int num_edges,
                                           int edge_weight_dim, ZOLTAN_ID_PTR edge_GID,
                                           ZOLTAN_ID_PTR edge_LID, float *edge_weight, int *ierr);

    public:
        StatePartitionerHyperGraph(MPI_Comm _comm) : StatePartitionerBase(_comm) {};

        ~StatePartitionerHyperGraph() override {};
    };
}
#endif //PACMENSL_FINITESTATESUBSETHYPERGRAPH_H
