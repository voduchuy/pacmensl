# StMcmcCme

Explore the application of ST-MCMC for Bayesian inference and reduced order/surrogate models to CME models. This repository contains the Python modules and scripts to reproduce the results in the paper `Catanach et al. Bayesian Inference of Stochastic Reaction Networks using Multifidelity Sequential Tempered Markov Chain Monte Carlo (https://arxiv.org/abs/2001.01373)`.

# Prerequisites

To execute the scripts in this repository, the following software packages need to be installed on your system:

* An MPI implementation, such as OpenMPI or MPICH.
* MPI for Python (https://mpi4py.readthedocs.io/en/stable/).
* Python 3.6+.
* NumPy.
* The parallel CME solver package PACMENSL (included in the subfolder ```cme/pacmensl``` of this repository) and its Python interface (included in the subfolder ```cme/pypacmensl```). See the instructions in those packages for installation guide and additional requirements.

# Running the examples

The codes to perform Bayesian inference on the three example models in the paper are provided in the subfolder ```examples```. 

For the ```repressilator``` and ```compartmental_gene_expression``` examples, you may simulate a new set of smFISH data by running the scripts with name pattern ```<example_name>_model.py``` (please only use a single core). After that step, an .npz file of the name ```<example_name>_data.npz``` will be created in the same folder. For the Inflammation Response example, the data (a subset of the real experimental data collected by Kalb et al. (https://journals.plos.org/plosone/article/comments?id=10.1371/journal.pone.0215602)) has been included.

To run the Bayesian inference from the generated or real datasets, use the syntax
```
mpirun -np <number_of_processors> <example_name>_inference.py <sampler_type>
```
where the options for ```<sampler_type>``` are:
* ```full```: use the original ST-MCMC sampler with the maximum fidelity model.
* ```it```: use the Multifidelity ST-MCMC sampler with Information Theoretic criteria.
* ```ess```: use the Multifidelity ST-MCMC sampler with ESS-based briding.
* ```it_tuned```: use the Multifidelity ST-MCMC sampler with Tuned Information Theoretic criteria.

Note that these examples may take a few days on several nodes of a HPC cluster to run all the sampler variants. We have included the test results when ran on Sandia's HPC system, as well as the scripts to analyze them, in the subfolder ```results```.

For a quick test of these methods, the ```poisson``` example is included to demonstrate the sampling methods on a simple problem of infering the parameter of a Poisson distribution. This example can be executed in a few minutes on a laptop.

# Contacts

Thomas Catanach (tacatan@sandia.gov)
Huy Vo (huydvo@colostate.edu)
