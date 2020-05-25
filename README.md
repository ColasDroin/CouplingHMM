# Code repository: Low-dimensional Dynamics of Two Coupled Biological Oscillators

This project aims to study the influence of the cell-cycle on the circadian clock using a Hidden Markov Model for inference, and a Expectation-Maximization method for parameters optimization. Most of the figures on the paper *Low-dimensional Dynamics of Two Coupled Biological Oscillators* published in _Nature Physics_ can be recreated from this code. The most important computations are made in the main scripts located in the folder "Scripts", but some supplementary analysis are done in the folder "SupplementaryAnalysis", as well as "RawDataAnalysis".

**NB**: for tracibility purposes, the data is currently not available in this repository, but will be provided on request. The code is therefore unusable as it is.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for reproducibility purposes.

### Prerequisites

We recommend you use Anaconda as a Python distribution in order to meet the required most of the Python packages used in this project. The following Python packages are used in the code:
* scipy
* shutil
* copy
* matplotlib
* numpy
* os
* pickle
* random
* seaborn
* subprocess
* sys

## Running the tests

The pipeline is composed of 10 main scripts, as well as complementary analysis scripts. The main scripts are the following:
- 1_wrap_initial_parameters.py: Estimate parameters from the raw data and wrap them into a pickle.
- 2_optimize_parameters_non_dividing.py: Using the estimated parameters as initial conditions, optimize the parameters on non-dividing traces.
- 3_CV_smoothing.py: Make a cross-validation on the dividing traces to find the optimal smoothing parameter for the coupling function. This can be extremely long to run.
- 4_optimize_parameters_dividing.py: Optimize the coupling function on the dividing traces. Better results are obtained if the coupling bias (from script 7) has been computed before.
- 5_validate_inference_non_dividing.py: Validate the value of the parameters estimated from the raw non-dividing traces, by generating traces in silico and compare the estimations to the theoretical parameters.
- 6_validate_estimation_non_dividing.py: Validate the value of the parameters optimized on the raw non-dividing traces. Same method as previously but seeing this time how the optimized parameters compare to the theoretical ones. The estimated parameters are used as initial condition.
- 7_validate_inference_dividing.py: Validate the optimized coupling function, by generating dividing traces in silico and compare the optimized one to the theoretical one. This script is also generated to generate the coupling used to correct the inference bias.
- 8_compute_final_fits_and_attractor.py: Compute final fits and phase-space density.
- 9_study_deterministic_system.py: Study the deterministic system with the inferred optimal parameters.
- 10_study_stochastic_system.py: Study the stochastic system with the inferred optimal parameters.

Using the script main.py will execute these scripts in order and output parameter files, as well as important figures of the paper. Once the parameter files have been generated, the complementary scripts can be executed as well. The complementary scripts are located in the the folder "SupplementaryAnalysis", and their description is given as the header of the main function of the script.


## Question

In case of question about the code, please contact colas.droin [at] epfl.ch. In case of question about the study, please contact felix.naef [at] epfl.ch

## Authors

* **Colas Droin** - *Code*
* **Eric Paquet** - *Biological experiments, R code (not in this repository)*
* **Felix Naef** - *Supervision*


## License

This project is licensed under the EPFL License.

## Acknowledgments

This project is funded by the FNS and the EPFL.
