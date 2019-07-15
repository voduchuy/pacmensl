# PACMENSL

PACMENSL (pak-men) : Parallel extensible Chemical master equation Analysis Library.

This is a part of the SSIT project at Munsky Group.

## Prerequisites
Compilation and build tools:
* CMake (3.10 or higher) (https://cmake.org/download/)
* C, CXX compilers.

An MPI implementation (OpenMPI, MPICH) already installed on your system.

Python:
* Python 3.6 or higher
* wget (for using the dowload scripts)


Additional requirements:

* Armadillo (http://arma.sourceforge.net/download.html)
* Metis (http://glaros.dtc.umn.edu/gkhome/metis/metis/download)
* Parmetis (http://glaros.dtc.umn.edu/gkhome/metis/parmetis/download)
* Zoltan (https://github.com/trilinos/Trilinos/tree/master/packages/zoltan)
* PETSc (https://www.mcs.anl.gov/petsc/download/)
* Sundials (https://computation.llnl.gov/projects/sundials/sundials-software)

In addition, PETSc and Sundials must be built with double-precision scalar types. Sundials must be enabled with PETSc support.

## Semi-automatic installation of the prerequisites

You can run our interactive Python scripts in the folder 'ext' that automatically download, configure, build and install the required libraries above into a custom folder. In order to download and install third-party libraries with our scripts, follow these steps:

1. Create three separate directories for storing downloaded source files (e.g. 'src'), for writing configuration and build files (e.g. 'build'), and for installation (e.g. 'install').
1. cd to the 'ext' directory within PACMENSL's folder.
1. Type 'python get_ext_libraries.py' if you want to download and install all of the libraries. Otherwise, type 'python ext_<library>.py' to install the individual libraries. Replace 'python' with your preferred python binary.
1. After installation, make sure to add the paths to the installed headers and library files to your environment variables. In Linux/MacOS you will need to set the following environment variables:
  ```
  export LD_LIBRARY_PATH=<path_to_your_install_dir>/lib:${LD_LIBRARY_PATH}
  
  export LIBRARY_PATH=<path_to_your_install_dir>/lib:${LIBRARY_PATH}
  
  export CPATH=<path_to_your_install_dir>/include:${CPATH}
  
  export PATH=<path_to_your_install_dir>/bin:${PATH}
```
## Installing PACMENSL
