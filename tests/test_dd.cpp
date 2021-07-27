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