//
// Created by Huy Vo on 4/23/19.
//

#include<zoltan.h>
#include<armadillo>

int main(int argc, char *argv[]){
    float ver;
    Zoltan_Initialize(argc, argv, &ver);

    MPI_Comm comm = MPI_COMM_WORLD;
    int my_rank;
    MPI_Comm_rank(comm, &my_rank);

    Zoltan_DD_Struct *zoltan_dd;

    arma::Mat<ZOLTAN_ID_TYPE> X0 = {{0,0}, {0, 1}, {0, 2},{0, 3}}; X0 = X0.t();
    arma::Mat<ZOLTAN_ID_TYPE> X1 = {{0,0}, {1, 0}, {2, 0},{3, 0}}; X1 = X1.t();
    arma::Row<ZOLTAN_ID_TYPE> lid = {0, 1, 2, 3};
    arma::Row<char> status = {0, 1, -2, -1};

    Zoltan_DD_Create(&zoltan_dd, comm, 2, 1, 1, 100000, 1);


    arma::Mat<ZOLTAN_ID_TYPE> gid;
    gid = (my_rank == 0)? X0 : X1;

    Zoltan_DD_Update(zoltan_dd, gid.memptr(), lid.memptr(), nullptr, nullptr, gid.n_cols);

    Zoltan_DD_Print(zoltan_dd);

    gid = (my_rank == 0)? X1 : X0;
    Zoltan_DD_Update(zoltan_dd, gid.memptr(), nullptr, status.memptr(), nullptr, gid.n_cols);

    Zoltan_DD_Print(zoltan_dd);


    arma::Row<char> retrieved_status(status.n_elem);
    gid = (my_rank == 0)? X0 : X1;
    status.set_size(0);

    Zoltan_DD_Find(zoltan_dd, gid.memptr(), nullptr, retrieved_status.memptr(), nullptr, gid.n_cols, nullptr);

    std::cout << retrieved_status;

    Zoltan_DD_Destroy(&zoltan_dd);
    MPI_Finalize();
}