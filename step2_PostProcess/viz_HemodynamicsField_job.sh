#!/bin/bash
#-----------------------------------------------------------------------------------------------------------------------
# viz_HemodynamicsField_job.sh
# SLURM wrapper to run viz_HemodynamicsField.py for a specific case on Trillium-style clusters.
#
# __author__ = Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__   = 2026-05
#
# EXECUTION:
#   sbatch viz_HemodynamicsField_job.sh
#
# Copyright (C) 2026 University of Toronto, Biomedical Simulation Lab.
#-----------------------------------------------------------------------------------------------------------------------

#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=100
#SBATCH --time=00:20:00
#SBATCH --job-name PT_HemodynamicsField
#SBATCH --output=PT_HemodynamicsField_%j.txt

set -euo pipefail
echo "Job started: $(date)"

# ---------------------------------- Define Paths -----------------------------------------------------------------------
CASE=PTSeg043_noLabbe_base
BASE_DIR=$SCRATCH/My_Projects/Study1_PTRamp/cases/$CASE
MESH_FOLDER="$BASE_DIR/step1_CFD/data"
INPUT="$BASE_DIR/step1_CFD/results/${CASE}_ts10000_cy6_saveFreq5"
OUTPUT="$BASE_DIR/step2_PostProcess/Hemodynamics/Field"
CONFIG="$BASE_DIR/step2_PostProcess/configs/${CASE}_viz_config.yaml"

SCRIPT="/scratch/ranbar/My_Projects/Study1_PTRamp/scripts/step2_PostProcess/visualization/viz_HemodynmicsField.py"

# --------------------------------- Load Modules ------------------------------------------------------------------------
module load StdEnv/2023 gcc/12.3 python/3.12.4
source $HOME/virtual_envs/pyvista36/bin/activate
module load vtk/9.3.0

# ------------------------------ Export Directories --------------------------------------------------------------------
mkdir -p "$OUTPUT"
mkdir -p "$SCRATCH/.config/mpl"
export MPLCONFIGDIR=$SCRATCH/.config/mpl


# ------------------------------ Run Script ----------------------------------------------------------------------------
python "$SCRIPT"                            \
    --case_name         "$CASE"             \
    --input_folder      "$INPUT"            \
    --mesh_folder       "$MESH_FOLDER"      \
    --output_folder     "$OUTPUT"           \
    --config_file       "$CONFIG"           \
    --target_flowrate   2.00 7.70 8.00  \
    --save_freq         5                   \
    --velocity_isovalue 0.5                 \
    --qcri_isovalue     50000                \
    --frame_spacing     10


#---------------------- For running directly from commandline use below ---------------------------
# Note1: You HAVE to load the modules first from terminal then run below
# Note2: You HAVE to comment this part if submitting this file through sbatch

python viz_HemodynmicsField.py                          \
    --case_name         "PTSeg043_noLabbe_base"            \
    --input_folder      "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg043_noLabbe_base/step1_CFD/results/PTSeg043_noLabbe_base_ts10000_cy6_saveFreq5/" \
    --mesh_folder       "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg043_noLabbe_base/step1_CFD/data"   \
    --output_folder     "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg043_noLabbe_base/step2_PostProcess/Hemodynamics/Field" \
    --config_file       "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg043_noLabbe_base/step2_PostProcess/configs/PTSeg043_noLabbe_base_viz_config.yaml" \
    --target_flowrate   8                            \
    --save_freq         5                                \
    --velocity_isovalue 0.5                              \
    --qcri_isovalue     50000                            \
    --frame_spacing     10                               \
    --cam_parallel_scale 8                              \
#    --cam_position      349.3 47.96 -694.79             \
#    --cam_focal_point   18.9 -19.1 -593.1                    \
#    --cam_view_up       0.24 0.23 0.94



echo "Job finished: $(date)"

