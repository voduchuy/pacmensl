//
// Created by Huy Vo on 6/4/19.
//

#include "DiscreteDistribution.h"

cme::parallel::DiscreteDistribution::~DiscreteDistribution() {
    VecDestroy(&p);
}

cme::parallel::DiscreteDistribution::DiscreteDistribution(const cme::parallel::DiscreteDistribution &dist) {
    MPI_Comm_dup(dist.comm, &comm);
    t = dist.t;
    VecDuplicate(dist.p, &p);
    VecCopy(dist.p, p);
    states = dist.states;
}

cme::parallel::DiscreteDistribution::DiscreteDistribution() {
    comm = nullptr;
}

cme::parallel::DiscreteDistribution::DiscreteDistribution(cme::parallel::DiscreteDistribution &&dist) {
    comm = dist.comm;
    t = dist.t;
    states = std::move(dist.states);
    p = dist.p;

    dist.p = nullptr;
    dist.comm =nullptr;
    PetscPrintf(comm, "Move constructor is called.\n");
}

cme::parallel::DiscreteDistribution &
cme::parallel::DiscreteDistribution::operator=(const cme::parallel::DiscreteDistribution &dist) {
    MPI_Comm_dup(dist.comm, &comm);
    t = dist.t;
    VecDuplicate(dist.p, &p);
    VecCopy(dist.p, p);
    states = dist.states;
    return *this;
}

cme::parallel::DiscreteDistribution &
cme::parallel::DiscreteDistribution::operator=(cme::parallel::DiscreteDistribution &&dist) noexcept {
    if (comm) MPI_Comm_free(&comm);
    if (p) VecDestroy(&p);
    states.set_size(0, 0);

    comm = dist.comm;
    t = dist.t;
    states = std::move(dist.states);
    p = dist.p;

    dist.comm = nullptr;
    dist.p = nullptr;

    return *this;
}

arma::Col<PetscReal> cme::parallel::Compute1DMarginal(const cme::parallel::DiscreteDistribution dist, int species) {
    arma::Col<PetscReal> md_on_proc;
    // Find the max molecular count
    int num_species = dist.states.n_rows;
    arma::Col<int> max_molecular_counts_on_proc(num_species);
    arma::Col<int> max_molecular_counts(num_species);
    max_molecular_counts_on_proc = arma::max(dist.states, 1);
    int ierr = MPI_Allreduce(&max_molecular_counts_on_proc[0], &max_molecular_counts[0], num_species, MPI_INT, MPI_MAX, dist.comm);
    MPICHKERRABORT(dist.comm, ierr);
    md_on_proc.resize(max_molecular_counts(species)+1);
    md_on_proc.fill(0.0);
    PetscReal *p_dat;
    VecGetArray(dist.p, &p_dat);
    for (int i{0}; i < dist.states.n_cols; ++i){
        md_on_proc(dist.states(species, i)) += p_dat[i];
    }
    VecRestoreArray(dist.p, &p_dat);
    arma::Col<PetscReal> md(md_on_proc.n_elem);
    MPI_Allreduce((void*) md_on_proc.memptr(), (void*) md.memptr(), md_on_proc.n_elem, MPI_DOUBLE, MPI_SUM, dist.comm);
    return md;
}
