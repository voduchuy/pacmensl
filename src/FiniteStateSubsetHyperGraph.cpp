//
// Created by Huy Vo on 12/15/18.
//

#include "FiniteStateSubsetHyperGraph.h"


namespace cme {
    namespace parallel {
        void FiniteStateSubsetHyperGraph::GenerateStatesAndOrdering() {
            // This can only be done after the stoichiometry has been set
            if (stoich_set == 0) {
                throw std::runtime_error(
                        "FiniteStateSubset: stoichiometry is required for HyperGraph partioning type.");
            }

            PetscErrorCode ierr;

            PetscLogEventBegin(generate_hg_event, 0, 0, 0, 0);
            // Create the hypergraph data
            PetscInt local_start_tmp, local_end_tmp, n_local_tmp, nnz, i_here;
            arma::Col<PetscInt> x((size_t) n_species);
            arma::Mat<PetscInt> brx((size_t) n_species, (size_t) stoichiometry.n_cols);
            arma::Row<PetscInt> irx((size_t) stoichiometry.n_cols);
            ZOLTAN_ID_PTR vtx_gid;
            int *vtx_edge_ptr;
            ZOLTAN_ID_PTR pin_gid;

            ierr = PetscLayoutCreate(comm, &vec_layout);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetSize(vec_layout, n_states_global);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetUp(vec_layout);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutGetRange(vec_layout, &local_start_tmp, &local_end_tmp);
            CHKERRABORT(comm, ierr);
            n_local_tmp = local_end_tmp - local_start_tmp;

            vtx_gid = (ZOLTAN_ID_PTR) Zoltan_Malloc(n_local_tmp * sizeof(ZOLTAN_ID_TYPE), __FILE__, __LINE__);
            vtx_edge_ptr = new int[n_local_tmp];
            pin_gid = (ZOLTAN_ID_PTR) Zoltan_Malloc((1 + stoichiometry.n_cols) * n_local_tmp * sizeof(ZOLTAN_ID_TYPE),
                                                    __FILE__, __LINE__);

            nnz = 0;
            n_local_states = n_local_tmp;
            for (PetscInt i = 0; i < n_local_tmp; ++i) {
                i_here = i + local_start_tmp;
                x = ind2sub_nd(fsp_size, i_here);
                // Find indices of states connected to x
                brx = arma::repmat(x, 1, stoichiometry.n_cols) - stoichiometry;
                irx = sub2ind_nd(fsp_size, brx);

                vtx_gid[i] = (ZOLTAN_ID_TYPE) i_here;
                vtx_edge_ptr[i] = (int) nnz;
                pin_gid[nnz] = (ZOLTAN_ID_TYPE) i_here;
                nnz++;
                for (PetscInt j = 0; j < stoichiometry.n_cols; ++j) {
                    if (irx.at(j) > 0) {
                        pin_gid[nnz] = (ZOLTAN_ID_TYPE) irx.at(j);
                        nnz++;
                    }
                }
            }
            adj_data.num_local_states = n_local_tmp;
            adj_data.num_reachable_states_rows = nnz;
            adj_data.rows_edge_ptr = vtx_edge_ptr;
            adj_data.reachable_states_rows_gid = pin_gid;
            adj_data.states_gid = vtx_gid;
            PetscLogEventEnd(generate_hg_event, 0, 0, 0, 0);

            PetscLogEventBegin(call_zoltan_event, 0, 0, 0, 0);
            // Use Zoltan to create partitioning, then wrap with processor_id, then proceed as usual
            IS processor_id, global_numbering;
            int zoltan_err;
            int changes, num_gid_entries, num_lid_entries, num_import, num_export;
            ZOLTAN_ID_PTR import_global_ids, import_local_ids, export_global_ids, export_local_ids;
            int *import_procs, *import_to_part, *export_procs, *export_to_part;

            zoltan_err = Zoltan_LB_Partition(
                    zoltan,
                    &changes,
                    &num_gid_entries,
                    &num_lid_entries,
                    &num_import,
                    &import_global_ids,
                    &import_local_ids,
                    &import_procs,
                    &import_to_part,
                    &num_export,
                    &export_global_ids,
                    &export_local_ids,
                    &export_procs,
                    &export_to_part);

            // Copy Zoltan's result to Petsc IS
            ierr = ISCreateGeneral(comm, (PetscInt) num_export, export_procs, PETSC_COPY_VALUES, &processor_id);
            CHKERRABORT(comm, ierr);
            ierr = ISPartitioningToNumbering(processor_id, &global_numbering);
            CHKERRABORT(comm, ierr);

            Zoltan_Free((void **) &vtx_gid, __FILE__, __LINE__);
            Zoltan_Free((void **) &pin_gid, __FILE__, __LINE__);
            delete[] vtx_edge_ptr;
            Zoltan_LB_Free_Part(&import_global_ids, &import_local_ids, &import_procs, &import_to_part);
            Zoltan_LB_Free_Part(&export_global_ids, &export_local_ids, &export_procs, &export_to_part);
            ierr = PetscLayoutDestroy(&vec_layout);
            CHKERRABORT(comm, ierr);

            PetscLogEventEnd(call_zoltan_event, 0, 0, 0, 0);

            PetscLogEventBegin(generate_ao_event, 0, 0, 0, 0);
            // Figure out how many states each processor own by reducing the proc_id_array
            PetscMPIInt my_rank, comm_size;
            MPI_Comm_size(comm, &comm_size);
            MPI_Comm_rank(comm, &my_rank);
            std::vector<PetscInt> n_own((size_t) comm_size);
            ierr = ISPartitioningCount(processor_id, comm_size, &n_own[0]);
            CHKERRABORT(comm, ierr);
            n_local_states = n_own[my_rank];

            // Generate layout
            PetscInt layout_start, layout_end;
            ierr = PetscLayoutCreate(comm, &vec_layout);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetLocalSize(vec_layout, n_local_states + n_species);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetSize(vec_layout, PETSC_DECIDE);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutSetUp(vec_layout);
            CHKERRABORT(comm, ierr);
            ierr = PetscLayoutGetRange(vec_layout, &layout_start, &layout_end);
            CHKERRABORT(comm, ierr);

            // Adjust global numbering to add sink states
            const PetscInt *proc_id_array, *global_numbering_array;
            CHKERRABORT(comm, ISGetIndices(processor_id, &proc_id_array));
            CHKERRABORT(comm, ISGetIndices(global_numbering, &global_numbering_array));
            std::vector<PetscInt> corrected_global_numbering(n_local_tmp + n_species);
            for (PetscInt i{0}; i < local_end_tmp - local_start_tmp; ++i) {
                corrected_global_numbering.at(i) =
                        global_numbering_array[i] + proc_id_array[i] * ((PetscInt) fsp_size.n_elem);
            }

            std::vector<PetscInt> fsp_indices(n_local_tmp + n_species);
            for (PetscInt i = 0; i < local_end_tmp - local_start_tmp; ++i) {
                fsp_indices.at(i) = local_start_tmp + i;
            }

            for (PetscInt i = 0; i < n_species; ++i) {
                corrected_global_numbering[local_end_tmp - local_start_tmp + n_species - i - 1] = layout_end - 1 - i;
                fsp_indices[local_end_tmp - local_start_tmp + n_species - i - 1] =
                        n_states_global + my_rank * n_species + i;
            }
            CHKERRABORT(comm, ISRestoreIndices(processor_id, &proc_id_array));
            CHKERRABORT(comm, ISRestoreIndices(global_numbering, &global_numbering_array));
            CHKERRABORT(comm, ISDestroy(&processor_id));
            CHKERRABORT(comm, ISDestroy(&global_numbering));

            // Create the AO object that maps from lexicographic ordering to Petsc Vec ordering
            IS fsp_is, petsc_is;
            ierr = ISCreateGeneral(comm, n_local_tmp + n_species, &fsp_indices[0],
                                   PETSC_COPY_VALUES, &fsp_is);
            CHKERRABORT(comm, ierr);
            ierr = ISCreateGeneral(comm, n_local_tmp + n_species, &corrected_global_numbering[0],
                                   PETSC_COPY_VALUES, &petsc_is);
            CHKERRABORT(comm, ierr);
            ierr = AOCreate(comm, &lex2petsc);
            CHKERRABORT(comm, ierr);
            ierr = AOSetIS(lex2petsc, fsp_is, petsc_is);
            CHKERRABORT(comm, ierr);
            ierr = AOSetType(lex2petsc, AOMEMORYSCALABLE);
            CHKERRABORT(comm, ierr);
            ierr = AOSetFromOptions(lex2petsc);
            CHKERRABORT(comm, ierr);
            ierr = ISDestroy(&fsp_is);
            CHKERRABORT(comm, ierr);
            ierr = ISDestroy(&petsc_is);
            CHKERRABORT(comm, ierr);
            PetscLogEventEnd(generate_ao_event, 0, 0, 0, 0);

            // Generate local states
            std::vector<PetscInt> petsc_local_indices((size_t) n_local_states);
            ierr = PetscLayoutGetRange(vec_layout, &petsc_local_indices[0], NULL);
            CHKERRABORT(comm, ierr);
            for (PetscInt i{0}; i < n_local_states; ++i) {
                petsc_local_indices[i] = petsc_local_indices[0] + i;
            }
            ierr = AOPetscToApplication(lex2petsc, n_local_states, &petsc_local_indices[0]);
            CHKERRABORT(comm, ierr);
            local_states = ind2sub_nd<arma::Mat<PetscInt>>(fsp_size, petsc_local_indices);
        }

        FiniteStateSubsetHyperGraph::FiniteStateSubsetHyperGraph(MPI_Comm new_comm) : FiniteStateSubset(new_comm) {
            partitioning_type = HyperGraph;
            Zoltan_Set_Param(zoltan, "LB_METHOD", "HYPERGRAPH");
            Zoltan_Set_Param(zoltan, "HYPERGRAPH_PACKAGE", "PHG");
            Zoltan_Set_Param(zoltan, "LB_APPROACH", "PARTITION");
            Zoltan_Set_Param(zoltan, "RETURN_LISTS", "PARTS");
            Zoltan_Set_Param(zoltan, "DEBUG_LEVEL", "0");
            Zoltan_Set_Param(zoltan, "OBJ_WEIGHT_DIM", "0"); // use Zoltan default vertex weights
            Zoltan_Set_Param(zoltan, "EDGE_WEIGHT_DIM", "0");// use Zoltan default hyperedge weights

            PetscLogEventRegister("Generate Hypergraph data", 0, &generate_hg_event);
            PetscLogEventRegister("Call Zoltan_LB", 0, &call_zoltan_event);
            PetscLogEventRegister("Generate AO", 0, &generate_ao_event);
        }

        FiniteStateSubsetHyperGraph::~FiniteStateSubsetHyperGraph() {
            Zoltan_Destroy(&zoltan);
        }
    }
}