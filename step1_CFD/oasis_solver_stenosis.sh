#!/bin/bash
#-----------------------------------------------------------------------------------------------------------------------
# oasis_solver_stenosis.sh
# SLURM job wrapper to run the Oasis CFD solver inside an Apptainer container.
#
# __author__ = Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__   = 2025-09
#
# PURPOSE:
#   - Read case parameters (exported by the submitter script) and properly launch Oasis solver under MPI.
#   - To properly run Oasis solver with the specified problem.
#
# REQUIREMENTS:
#   - Apptainer container image providing legacy FEniCS + Oasis (e.g., ~/containers/fenics-legacy/fenics-oasis.sif).
#   - BSLSolver sources available locally (used in PYTHONPATH): https://github.com/Biomedical-Simulation-Lab/BSLSolver.git
#
# EXECUTION:
#   - This script is called by the case-specific bash script launcher (e.g., "run_oasis_case_casename.sh").
#   - Note: This script cannot be submitted to SLURM directly.
#
# IMPORATANT NOTE:
#   - Change the paths and directories below according to your file system.
#
# Adapted from solver-v2.sh written by 2022 Anna Haley (ahaley@mie.utoronto.ca). 
# Copyright (C) 2025 University of Toronto, Biomedical Simulation Lab.
#-----------------------------------------------------------------------------------------------------------------------


#SBATCH --output=hpclog/%x_%j.txt
#SBATCH --export=ALL

set -euo pipefail


#---------------------------------------- Basic Setup -----------------------------------------------
# Navigate to the script directory
cd $SLURM_SUBMIT_DIR

#export OMP_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1

# Slurm tasks (fallback to num_cores if exported, else 1)
NP="${SLURM_NTASKS:-1}"


# Ensure consistency between file names

#this is where the naming.py script is run
# all it does is creates a naming structure that will be standard and used throughout, including in the post-processing
# then we just set a variable to use as a shorter of the casename
casename_full="${casename}_ts${timesteps_per_cycle}_cy${cycles}_saveFreq${save_frequency}" #"PTSeg028_base_0p64_ts10000"

results_folder="./results/$casename_full"

#echo "Finished naming"


# Check for restarts and set log file name

# Now we need to check and see if we are restarting or running for the first time
#simdone="0"
#restart_no=0 #Note that after the first restart, the restart number will always be 1 because we are using Oasis and not Mehdi's io.py

# Variable for the current log file
log_file="./logs/${casename_full}_${SLURM_JOB_ID}"


#---------------------------------------- Define Paths ----------------------------------------

# This is the location to all case-specific mesh data (contains ./data) --> should be the same as where you submit sbatch 'run_oasis_case.sh' from
CASE_DIR="$SLURM_SUBMIT_DIR"

# This is location of CFD Oasis scripts (oasis_problem.py and oasis_solver.sh) in your directory
PATH_OASIS_SOLVER="/scratch/ranbar/My_Projects/Study2_stenosis/scripts/step1_CFD"

# This is the location of the BSLSolver in your directory
PATH_BSLSOLVER="/home/$scinet_user/BSLSolver"       # path to your BSLSolver

# This is the location of fenics-oasis container image file in your directory
PATH_CONTAINER="/home/$scinet_user/containers/fenics-legacy/fenics-oasis.sif"  

# This is the location of shim file for aliasing ufl-legacy as ufl (only required for fenics-legacy)
PATH_UFL_SHIM="/scratch/$scinet_user/pyshims"                                                     


# Setup JIT cache directory
PATH_JIT_CACHE="/scratch/$scinet_user/.cache"
mkdir -p "$PATH_JIT_CACHE"


#------------------------------------- Create Binds and python paths --------------------------------------

# Create and export python paths for the container: BSLSolver + shim + oasis solver directory
PATH_PYTHON="${PATH_UFL_SHIM}:${PATH_BSLSOLVER}:${PATH_OASIS_SOLVER}"


export APPTAINERENV_PYTHONPATH="$PATH_PYTHON${PYTHONPATH:+:$PYTHONPATH}"
export APPTAINERENV_DIJITSO_CACHE_DIR="$PATH_JIT_CACHE"


# Bind what you need for the container; add more binds if your data/code live elsewhere
BIND_OPTS="--bind /scratch:/scratch \
--bind ${CASE_DIR}:${CASE_DIR} \
--bind ${PATH_OASIS_SOLVER}:${PATH_OASIS_SOLVER} \
--bind ${PATH_JIT_CACHE}:${PATH_JIT_CACHE} \
--pwd ${CASE_DIR}"



#---------------------------------- Write to log ---------------------------------------------

echo "--------------------------------------------------------------------------------"
echo "Pulsatile Tinnitus - CFD Solver Oasis - Biomedical Simulation Lab - University of Toronto — $(date)"
echo "Case name:                      ${casename}"
echo "Case name (full):               ${casename_full}"
echo "Results folder:                 ${results_folder}"
echo "Log File:                       ${log_file}"
echo "Number of cores:                ${NP}"
echo "Number of cycles:               ${cycles}"
echo "Timesteps per cycle:            ${timesteps_per_cycle}"
echo "Inlet BC type:                  ${inlet_BC_type}"
echo "Save frequency:                 ${save_frequency}"
echo "Save first cycle:               ${save_first_cycle}"
echo "--------------------------------------------------------------------------------"

#echo "Restart Number: " \$(( \$restart_no+1 ))
#echo \$(date) > \$log_file

# Sanity check of paths
#echo "PY_PATH= $PATH_PYTHON"
#echo "BIND_OPTS= $BIND_OPTS"
#echo "SLURM_SUBMIT_DIR= $SLURM_SUBMIT_DIR"



#---------------------------------- Launch Solver ---------------------------------------------

# Note: HYDRA_LAUNCHER=fork helps some MPI-in-container setups

#APPTAINERENV_PYTHONPATH="$PATH_PYTHON${PYTHONPATH:+:$PYTHONPATH}" \
apptainer exec \
  --env HYDRA_LAUNCHER=fork \
  $BIND_OPTS ~/containers/fenics-legacy/fenics-oasis.sif \
  mpirun -n $NP oasis NSfracStep problem=oasis_problem_stenosis \
  mesh_name=$casename \
  cycles=$cycles \
  period=$period \
  timesteps=$timesteps_per_cycle \
  viscosity_mu_Pas=$viscosity_mu_Pas \
  density_kgm3=$density_kgm3 \
  uOrder=$uOrder \
  inlet_BC_type=$inlet_BC_type \
  save_frequency=$save_frequency \
  save_first_cycle=$save_first_cycle \
  checkpoint=$checkpoint \
  &>> $log_file



#------------------------------------ Write to log ---------------------------------------------------
echo $(date) >> $log_file
echo "hpclog/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.txt" >> "$log_file"
sleep 30

