//
// Created by Huy Vo on 5/29/18.
//

#ifndef PARALLEL_FSP_FSP_H
#define PARALLEL_FSP_FSP_H

#include<algorithm>
#include<cme_util.h>
#include<cstdlib>
#include<cmath>
#include<Magnus4FSP.h>
#include<HyperRecOp.h>
#include<petsc.h>

namespace cme{
    namespace petsc{
        class FSP {
            using Real = PetscReal;
            using Int = PetscInt;
        private:
            MPI_Comm comm = NULL;
            Vec P;
            HyperRecOp A;
            arma::Row<Int> FSPSize;
            arma::Row<Real> FSPIncrement;
            Real t_final;
            Real fsp_tol;
            Magnus4FSP my_magnus;

            arma::Mat<Int> stoich_mat;
            PropFun propensity;
            TcoefFun t_fun;
            std::function<void (PetscReal, Vec, Vec)> tmatvec;
        public:

            FSP(MPI_Comm _comm, arma::Mat<Int> init_states, arma::Col<Real> init_prob, arma::Mat<Int> _stoich_mat,
                PropFun _propensity, TcoefFun _t_fun, arma::Row<Int> _FSPSize, arma::Row<Real> _FSPIncrement, Real _t_final, Real _fsp_tol,
            Real _mg_tol = 1.0e-8, Real _kry_tol = 1.0e-8):
                    comm(_comm),
                    t_final(_t_final),
                    fsp_tol(_fsp_tol),
                    propensity(_propensity),
                    stoich_mat(_stoich_mat),
                    t_fun(_t_fun),
                    FSPSize(_FSPSize),
                    FSPIncrement(_FSPIncrement),
                    A(_comm, _FSPSize, _stoich_mat, _propensity, _t_fun),
                    my_magnus(comm, 0.0, _t_final, (Int) _FSPSize.n_elem, _fsp_tol, _mg_tol, _kry_tol)
                    {
                        // Set up the initial vector
                        VecCreate(comm, &P);
                        VecSetSizes(P, PETSC_DECIDE, arma::prod(FSPSize + 1) + FSPSize.n_elem);
                        VecSetType(P, VECMPI);
                        VecSetUp(P);
                        VecSet(P, 0.0);

                        arma::Row<Int> init_indices = sub2ind_nd(FSPSize, init_states);
                        Int i1, i2;
                        VecGetOwnershipRange(P, &i1, &i2);
                        for (size_t i{0}; i < init_indices.n_elem; ++i)
                        {
                            if (i1 <= init_indices(i) && init_indices(i) < i2){
                                VecSetValue(P, init_indices(i), init_prob(i), INSERT_VALUES);
                            }
                        }
                        VecAssemblyBegin(P);
                        VecAssemblyEnd(P);

                        // Update the Magnus object with the initial vector information
                        tmatvec = [this] (PetscReal t, Vec x, Vec y){A(t).action(x,y);};
                        my_magnus.update_vector(P, tmatvec);
            }

            Vec get_P()
            {
                return P;
            }

            void solve();

            void destroy()
            {
                my_magnus.destroy();
                VecDestroy(&P);
            }

        };
    }
}


#endif //PARALLEL_FSP_FSP_H
