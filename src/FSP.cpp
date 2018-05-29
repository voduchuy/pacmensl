//
// Created by Huy Vo on 5/29/18.
//

#include "FSP.h"

namespace cme{
    namespace petsc{
        void FSP::solve()
        {
            Int ierr;

            PetscReal t = 0.0e0;
            while (t < t_final) {
                my_magnus.solve();
                t = my_magnus.t_now;
                // Expand the FSP if the solver halted prematurely
                if (t < t_final) {
                    arma::Row<PetscInt> FSPSize_old {FSPSize};
                    for (size_t i{0}; i < my_magnus.to_expand.n_elem; ++i)
                    {
                        if (my_magnus.to_expand(i) == 1)
                        {
                            FSPSize(i) = (PetscInt) std::ceil(((double) FSPSize(i)) * (FSPIncrement(i) + 1.0e0));
                        }
                    }
                    std::cout << FSPSize << std::endl;

                    // Generate the expanded matrices
                    A.destroy();
                    A.generate_matrices(FSPSize, stoich_mat, propensity);

                    // Generate the expanded vector and scatter forward the current solution
                    Vec Pnew;
                    VecCreate(comm, &Pnew);
                    VecSetSizes(Pnew, PETSC_DECIDE, arma::prod(FSPSize + 1) + FSPSize.n_elem);
                    VecSetType(Pnew, VECMPI);
                    VecSetUp(Pnew);
                    VecSet(Pnew, 0.0);

                    IS is_old;
                    
                    PetscInt istart, iend, n_old, n_new;
                    ierr = VecGetSize(Pnew, &n_new); CHKERRABORT(comm, ierr);
                    ierr = VecGetSize(P, &n_old); CHKERRABORT(comm, ierr);

                    ierr = VecGetOwnershipRange(P, &istart, &iend); CHKERRABORT(comm, ierr); 
                    arma::Row<PetscInt> indices_sinks; indices_sinks.set_size(0);
                    if (iend >= n_old)
                    {
                        indices_sinks.set_size(my_magnus.n_sinks);

                        for (PetscInt i {0}; i < my_magnus.n_sinks; ++i)
                        {
                            indices_sinks(i) = n_new + (i-my_magnus.n_sinks);
                        }
                    }
                    iend = std::min(iend, arma::prod(FSPSize_old+1));
                    arma::Row<PetscInt> indices_new = arma::linspace<arma::Row<PetscInt>>(istart, iend - 1, iend - istart); CHKERRABORT(comm, ierr);
                    arma::Mat<PetscInt> my_X = cme::ind2sub_nd(FSPSize_old, indices_new);
                    indices_new = cme::sub2ind_nd(FSPSize, my_X);
                    indices_new = arma::join_horiz(indices_new, indices_sinks);

                    ierr = ISCreateGeneral(comm, (PetscInt) indices_new.n_elem, indices_new.begin(), PETSC_COPY_VALUES, &is_old); CHKERRABORT(comm, ierr);

                    VecScatter scatter;
                    ierr = VecScatterCreate(P, NULL, Pnew, is_old, &scatter); CHKERRABORT(comm, ierr);
                    ierr = VecScatterBegin(scatter, P, Pnew, INSERT_VALUES, SCATTER_FORWARD); CHKERRABORT(comm, ierr);
                    ierr = VecScatterEnd(scatter, P, Pnew, INSERT_VALUES, SCATTER_FORWARD); CHKERRABORT(comm, ierr);

                    ierr = VecDestroy(&P); CHKERRABORT(comm, ierr);
                    ierr = VecDuplicate(Pnew, &P); CHKERRABORT(comm, ierr);
                    ierr = VecSwap(P, Pnew); CHKERRABORT(comm, ierr);
                    ierr = VecScatterDestroy(&scatter); CHKERRABORT(comm, ierr);
                    ierr = VecDestroy(&Pnew); CHKERRABORT(comm, ierr);
                    std::cout << "Update Magnus4 object \n";
                    // Reset the Magnus4 object
                    my_magnus.destroy();
                    my_magnus.update_vector(P, tmatvec);
                }
            }


            std::cout << FSPSize << std::endl;
        }
    }
}