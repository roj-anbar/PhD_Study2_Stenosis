#!/bin/bash
#-----------------------------------------------------------------------------------------------------------------------
# compute_Spectrogram_job.sh
# SLURM wrapper to run compute_Spectrogram.py for a specific case on Trillium style clusters.
#
# __author__ = Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__   = 2025-11
#
# PURPOSE:
#   - Define all case parameters for performing post-processing on CFD results.
#   - Optional flags let you override key settings without editing the file.
#
# REQUIREMENTS:
#   - compute_Spectrogram.py (in the same directory as this bash file)
#   - A virtual environment including pyvista
#
# EXECUTION:
#   - Run this script from terminal by:
#     <sbatch compute_Spectrogram_job.sh>
#
# Copyright (C) 2025 University of Toronto, Biomedical Simulation Lab.
#-----------------------------------------------------------------------------------------------------------------------

#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=192
#SBATCH --time=00:30:00
#SBATCH --job-name PT_Spectrogram
#SBATCH --output=PT_Spectrogram_%j.txt


set -euo pipefail
echo "Job started: $(date)"

# ---------------------------------- Define Paths -------------------------------------------------------------------------------
CASE=PTSeg106_base_0p64                                              # Case name
BASE_DIR=$SCRATCH/My_Projects/Study1_PTRamp/cases/$CASE                 # Parent directory of the case
MESH_FOLDER="$BASE_DIR/step1_CFD/data"                                  # Path to mesh data folder containing the h5 mesh
CENTERLINE="$MESH_FOLDER/${CASE}_centerline_points.csv"                 # Path to centerline csv file used to construct ROIs
INPUT="$BASE_DIR/step1_CFD/results/${CASE}_ts10000_cy6_saveFreq1"       # Path to CFD results folder containing timeseries HDF5 files
OUTPUT="$BASE_DIR/step2_PostProcess"                                    # Path to saving spectrogram files
SPECTROGRAM_REGIONS="$OUTPUT/configs/${CASE}_spectrogram_regions.csv"   # Path to spectrogram regions csv file used to generate regional specs

SCRIPT="/scratch/ranbar/My_Projects/Study1_PTRamp/scripts/step2_PostProcess/compute_Spectrogram.py"


# --------------------------------- Load Modules -------------------------------------------------------------------------------
module load StdEnv/2023 gcc/12.3 python/3.12.4
source $HOME/virtual_envs/pyvista36/bin/activate
module load vtk/9.3.0


# ------------------------------ Export Directories ----------------------------------------------------------------------------
mkdir -p "$OUTPUT"
mkdir -p "$SCRATCH/.config/mpl"
#mkdir -p "$SCRATCH/.local/share/pyvista" "$SCRATCH/.local/temp"

export MPLCONFIGDIR=$SCRATCH/.config/mpl                    #matplotlib config
export PYVISTA_OFF_SCREEN=true                              #tells pyvista to use off-screen rendering

#export TMPDIR=$SCRATCH/.local/temp                          #temporary directory used by python
#export XDG_RUNTIME_DIR=$SCRATCH/.local/temp                 #runtime directory used by GUI stuff
#export PYVISTA_USERDATA_PATH=$SCRATCH/.local/share/pyvista  #pyvista userdata
#export PYVISTA_USE_PANEL=true
#export OMP_NUM_THREADS=1


# ------------------------------ Run Scripts ------------------------------------------------------------------------------------

# Run the script once for all anatomical regions (i.e. stenosis, sigmoid sinus, ...).
# H5 files are read a single time and reused for every region defined in SPECTROGRAM_REGIONS.
# Region-specific parameters (start/end IDs, stride, radius) are loaded from that CSV; everything else is controlled by the flags below.
# You can obtain ROI parameters (e.g. ROI_start_center_id, ROI_end_center_id, ...) from the spreadsheet below:
# https://utoronto-my.sharepoint.com/:x:/g/personal/rojin_anbarafshan_mail_utoronto_ca/IQAE4WxcfxZtTa3r6agUO6xUAUsh_3MP7hhnwm9_BzdpsV0?e=WVALzt

python "$SCRIPT" \
    --case_name             "$CASE" \
    --input_folder          "$INPUT" \
    --mesh_folder           "$MESH_FOLDER" \
    --output_folder         "$OUTPUT" \
    --ROI_center_csv        "$CENTERLINE" \
    --spec_regions_csv      "$SPECTROGRAM_REGIONS" \
    --spec_quantity         "wallpressure" \
    --window_length         2732 \
    --ROI_type              "cylinder" \
    --flag_multi_ROI        \
    --flag_save_ROI
#    --timesteps_per_cyc 10000 



# ------------ For running each ROI separately (no ROI_regions_csv file)
#python "$SCRIPT" \
#    --case_name             "$CASE" \
#    --input_folder          "$INPUT" \
#    --mesh_folder           "$MESH_FOLDER" \
#    --output_folder         "$OUTPUT" \
#    --centerline_csv        "$CENTERLINE" \
#    --spec_quantity         "wallpressure" \
#    --window_length         2732 \
#    --ROI_type              "cylinder" \
#    --ROI_radius            10 \
#    --ROI_stride            2 \
#    --ROI_start_center_id   1108 \
#    --ROI_end_center_id     1188 \
#    --flag_multi_ROI        
#    --flag_save_ROI
#    --timesteps_per_cyc 10000 


#---------------------- For running directly from commandline use below ---------------------------
# Note1: You HAVE to load the modules first from terminal then run below
# Note2: You HAVE to comment this part if submitting this file through sbatch

# python compute_Spectrogram.py \
#     --case_name             "PTSeg106_base_0p64" \
#     --input_folder          "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg106_base_0p64/step1_CFD/results/PTSeg106_base_0p64_ts10000_cy6_saveFreq1/" \
#     --mesh_folder           "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg106_base_0p64/step1_CFD/data" \
#     --output_folder         "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg106_base_0p64/step2_PostProcess" \
#     --ROI_center_csv        "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg106_base_0p64/step1_CFD/data/PTSeg106_base_0p64_centerline_points.csv" \
#     --spec_quantity         "wallpressure" \
#     --window_length         2732 \
#     --ROI_type              "cylinder" \
#     --ROI_start_center_id   1110 \
#     --ROI_end_center_id     1112 \
#     --ROI_radius            10 \
#     --ROI_stride            4 \
#     --flag_multi_ROI        
#     --flag_save_ROI

#    --spec_regions_csv      "$SCRATCH/My_Projects/Study1_PTRamp/cases/PTSeg028_base_0p64/step2_PostProcess/configs/PTSeg028_base_0p64_spectrogram_regions.csv" \


wait
