# PACMENSL

PACMENSL (pek-meal) : Parallel extensible Chemical master equation Analysis Library.

This is a part of the SSIT project at Munsky Group.

## Prerequisites
Compilation and build tools:
* CMake (3.10 or higher)
* C, CXX compilers.

An MPI implementation (e.g., OpenMPI, MPICH) on your system.

PACMENSL requires the following libraries to be installed on your system:

* Armadillo
* Parmetis
* Zoltan
* PETSc
* Sundials

In addition, PETSc and Sundials must be built with double-precision scalar types. Sundials must be enabled with PETSc support.

## Semi-automatic installation of the prerequisites

We have interactive Python scripts to download, configure, build and install the required libraries above. In order to download and install third-party libraries with our scripts, follow these steps:

1. Create three separate directories for storing downloaded source files (e.g. 'src'), for writing configuration and build files (e.g. 'build'), and for installation (e.g. 'install').
1. cd to the 'ext' directory within PACMENSL's folder.
1. Type 'python get_ext_libraries.py' if you want to download and install all of the libraries. Otherwise, type 'python ext_<library>.py' to install the individual libraries. Replace 'python' with your preferred python binary.
1. After installation, make sure to add the paths to the installed headers and library files to your environment variables.

## Installing PACMENSL
