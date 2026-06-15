#!/bin/bash
#-----------------------------------------------------------------------------------------------------------------------
# run_oasis_PT.sh
# Case-specific launcher for Oasis CFD jobs on SLURM (Trillium style clusters).
#
# __author__ = Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__   = 2025-09
#
# PURPOSE:
#   - Define all case parameters for CFD in one place and submit oasis_solver.sh via sbatch.
#   - Optional flags let you override key settings without editing the file.
#
# REQUIREMENTS:
#   - oasis_solver.sh (the job script this wrapper submits)
#   - oasis_problem.py (required in oasis_solver.sh)
#
# EXECUTION:
#   - Run this script from terminal by:
#     <./run_oasis_PT.sh>
#
# IMPORATNT NOTES:
#   - This script should be ran from the PT case-specific directory containing the mesh data (under /data folder).
#   - "PATH_OASIS_SOLVER" which is the path to "oasis_solver.sh" should be modified for each user.
#   - No need to copy oasis_solver.sh and oasis_problem.py for each case (just have these 2 scripts in one directory and call it using "PATH_OASIS_SOLVER" variable).
#
# Adapted from solver-v2.sh written by 2022 Anna Haley (ahaley@mie.utoronto.ca) and solver.sh written by 2018 Mehdi Najafi (mnuoft@gmail.com). 
# Copyright (C) 2025 University of Toronto, Biomedical Simulation Lab.
#-----------------------------------------------------------------------------------------------------------------------

set -euo pipefail


# Define path to your CFD Oasis solver (change accordingly)
PATH_OASIS_SOLVER="/scratch/ranbar/My_Projects/Study2_srenosis/scripts/step1_CFD/oasis_solver_stenosis.sh"


# Safety check for if file exists
if [[ ! -f "$PATH_OASIS_SOLVER" ]]; then
    echo "ERROR: Oasis slurm job script 'oasis_solver_stenosis.sh' not found:"
    echo "       $OASIS_SLURM"
    exit 1
fi


#--------------------------------------- Input parameters -------------------------------------------------
# Define all the input parameters to run the CFD solver Oasis job script.

# Anything listed between set -a and set +a can be used as variables in any subsequent commands.
# This is used as an alternative to exporting each variable separately.
set -a 

scinet_user=ranbar                # your cluster username (need to specify this or this won't work at all)
group_name=def-steinman           # group allocation to run the job under
debug=on                          # job partition -> choose between 'on'/'off' (Whether or not you are using debug node)
num_cores=100                     # number of cores to use per node (everything runs on a single node) 
required_time="00:19:59"          # amount of time cluster will need to run the case (max 24 hours)
post_processing_time_minutes=180  # amount of time needed to post-process the case (this is run on a single proc)

casename="eccentric_stenosis"          # What your case will be called in the output files & on cluster -- should be the same name as this script without the sh
cycles=1                          # number of cycles to run, determines total simulation time (default: 2)
period=915.0                      # waveform period [ms] (default: 915 ms)
timesteps_per_cycle=1000         # number of timesteps for each cycle (default: 2000)
viscosity_mu_Pas=0.0037           # dynamic viscosity [Pa.s] (default: 0.0037 Pa.s)
density_kgm3=1057                 # blood density [kg/m3] (default: 1057 kg/m3)
uOrder=1                          # velocity FE order (default: 1)
inlet_BC_type="ramp"              # type of the inlet boundary condition --> options: {'pulsatile', 'ramp', 'custom'} (default: 'pulsatile')

save_first_cycle=True             # flag to save first cycle or not (default: False)
save_frequency=10                  # write solution every N steps (default: 5)
checkpoint=500                    # write restart every N steps

set +a



#------------------------------------ Set parameters ----------------------------------------

# Assign the partition on cluster
if [ $debug == "on" ]; then 
  partition="debug"   # max 1 hour
else
  partition="compute" # max 24 hours
fi


# Create a jobname for SLURM
if [[ ! -z jobname ]]; then
  jobname="${casename}_cy${cycles}_ts${timesteps_per_cycle}"
fi

# Create log directories
mkdir -p ./logs ./hpclog


#------------------------------------------- Submit the job ----------------------------------------------------
# All config is exported via environment (sbatch --export=ALL).
# oasis_solver.sh should read these variables (e.g., $cycles, $save_frequency, …).

# Submit the job with defined parameters
sbatch --export=ALL \
       --account=$group_name \
       --time=$required_time \
       --job-name=$jobname \
       --nodes=1 \
       --ntasks-per-node=$num_cores \
       --partition=$partition \
       "$PATH_OASIS_SOLVER" "$@"

# Note:
# in <"$OASIS_SOLVER" "$@"> "$@" just includes any additional keyword arguments passed when you run this script.
# eg. the command "in_default.sh hello" would pass the variable hello to this script,
# and it would be then passed to oasis_solver.sh file subsequently, and could be accessed with $1.
