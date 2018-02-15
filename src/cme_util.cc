#include "cme_util.hpp"

namespace cme{
  namespace petsc{
      arma::Col<PetscReal> marginal(Vec P, arma::Row<PetscInt> &nmax, PetscInt species)
      {
        PetscInt ierr;

        MPI_Comm comm; PetscObjectGetComm((PetscObject) P, &comm);
        PetscInt n_local; VecGetLocalSize(P, &n_local);
        PetscReal *local_data; VecGetArray(P, &local_data);

        arma::Col<PetscReal> p_local(local_data, n_local, false, true);
        arma::Col<PetscReal> v(nmax(species)+1); v.fill(0.0);

        PetscInt Istart, Iend;
        ierr = VecGetOwnershipRange( P, &Istart, &Iend ); CHKERRABORT(comm, ierr);

        arma::Row<PetscInt> my_range(Iend-Istart);
        for ( PetscInt vi {0}; vi < my_range.n_elem; ++vi )
        {
                my_range[vi]= Istart + vi;
        }

        arma::Mat<PetscInt> my_X= cme::ind2sub_nd( nmax, my_range);

        for (PetscInt i{0}; i < n_local; ++i)
        {
          v(my_X(species, i)) += p_local(i);
        }

        MPI_Barrier( comm );

        arma::Col<PetscReal> w(nmax(species)+1); w.fill(0.0);
        MPI_Allreduce( &v[0], &w[0], v.size(), MPI_DOUBLE, MPI_SUM, comm );

        VecRestoreArray(P, &local_data);

        return w;
      }
  }
}
