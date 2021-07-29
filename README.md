# PACMENSL

PACMENSL: PArallel Chemical Master EquatioN Solver Library. The library implements the Finite State Projection algorithm to numerically solve the CME on high performance computing nodes using MPI. There are also objects that might be useful for developing parallel implementations for other flavors of the FSP for sensitivity analysis and computing the stationary distribution.

## Reference

Vo, H. D. and Munsky, B. E. "A Parallel Implementation of the Finite State Projection Algorithm for the Solution of the Chemical Master Equation". https://www.biorxiv.org/content/10.1101/2020.06.30.180273v2

## Contact

Huy Vo: huydvo@colostate.edu.

## Dependencies

Required:
* [CMake](https://cmake.org/download/) (3.10 or higher) 
* C, C++ compilers.
* An MPI implementation (OpenMPI, MPICH) already installed on your system. On MacOS you can install OpenMPI via
 Homebrew:
```
brew update
brew install openmpi
```
* [Armadillo](http://arma.sourceforge.net/download.html)
* [Zoltan](https://github.com/trilinos/Trilinos/tree/master/packages/zoltan)
* [PETSc](https://www.mcs.anl.gov/petsc/download/)
* [SUNDIALS](https://computation.llnl.gov/projects/sundials/sundials-software)

Optionally, if you want to use graph-partitioning methods for load-balancing:
* [Metis](http://glaros.dtc.umn.edu/gkhome/metis/metis/download)
* [Parmetis](http://glaros.dtc.umn.edu/gkhome/metis/parmetis/download)

If you want to build the unit tests:
* GoogleTest.

To generate documentation:
* [Doxygen](https://www.doxygen.nl/index.html).

In addition, PETSc and Sundials must be built with double-precision scalar types. Sundials must be enabled with PETSc support.

## Installation

PACMENSL can be installed using CMake. You can add the following options to the ```cmake``` command to customize the
 build:
 
 * ```-DBUILD_EXAMPLES```
    * Whether to build example programs (```ON``) or not (```OFF```).
 * ```-DBUILD_TESTS```
    * Whether to build unit tests (```ON``) or not (```OFF```). You must have GoogleTest framework installed on your
     system. Otherwise, this option will be turned off.
 * ```-DBUILD_SHARED_LIBS```
    * Build shared library (```ON```) or static library (```OFF```).
 * ```-DENABLE_PARMETIS```
    * To enable interface to ParMetis (```ON``) or not (```OFF```).
 * ```-DEXAMPLES_INSTALL_DIR```
    * Where to install the compiled examples.
 * ```-DTESTS_INSTALL_DIR```
    * Where to install the compiled tests.
 * ```-DCMAKE_INSTALL_PREFIX```
    * Path to installing the compiled library.

For a minimal build:
* Create a folder, say ```build```. Make that the current working directory.
* Run the command 
    ``` 
   cmake -DBUILD_EXAMPLES=ON -DBUILD_TESTS=OFF -DBUILD_SHARED_LIBS=ON -DENABLE_PARMETIS=OFF $(path_to_PACMENSL
   root_folder)
    ```        
  Here, ```$(path_to_PACMENSL_root_folder)``` is the path to PACMENSL source directory.
* To build the library and examples:
    ```make -j4```
* Install the library ```make install```. On a Linux or MacOS system, this will install the compiled library to the
 default folder ```/usr/local/lib```.

## Documentation
Documentation is provided as comments in the source code. It is also possible to generate seperate documentations using Doxygen. The subfolder 'docs' contains the configuration for Doxygen to generate documentations in HTML and LaTex formats. To generate the documentation, follow these steps:
1. Change working directory to `docs`.
2. Run Doxygen with the configuration file provided: 
```
doxygen doc_config
```
3. Doxygen will output documentations in HTML and LaTex formats. 

## Usage 
To use the library, simply add the link flag ```-lpacmensl``` when compiling your program. See the in-source
 documentations and the example source codes in the folder ```examples``` for the syntax.

## Python wrapper
Python wrapper is available at https://github.com/voduchuy/pypacmensl. 
 

