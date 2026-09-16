#!/bin/bash
#-----------------------------------------------------------------------------------------------------------------------
# extract_circumferential_pressure_job.sh
# SLURM wrapper to run extract_circumferential_pressure.py for a specific case on Trillium-style clusters.
#
# __author__ = Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__   = 2026-09
#
# PURPOSE:
#   - Step 1: Sample N evenly-spaced circumferential wall nodes at a given axial slice of an
#             idealized stenosis geometry, and save them as a .vtp file for ParaView inspection.
#   - Step 2: Read CFD pressure snapshots for those nodes and save a .npz time-series file.
#
# EXECUTION:
#   sbatch extract_circumferential_pressure_job.sh
#
# Copyright (C) 2026 University of Toronto, Biomedical Simulation Lab.
#-----------------------------------------------------------------------------------------------------------------------

#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:10:00
#SBATCH --job-name PT_CircNodes
#SBATCH --output=PT_CircNodes_%j.txt


set -euo pipefail
echo "Job started: $(date)"

# ---------------------------------- Define Paths -----------------------------------------------------------------------
CASE=eccStenosis                                                                    # Case name
BASE_DIR=$SCRATCH/My_Projects/Study2_stenosis/cases/case0_eccStenosis/modelOwais    # Parent directory of the case
MESH_FOLDER="$BASE_DIR/step1_CFD/data"                                           # Folder containing mesh .h5 or .xml.gz
INPUT="$BASE_DIR/step1_CFD/results/rampoffset2mLs/${CASE}_clean_ts12000_cy6_saveFreq1"          # Path to CFD results folder containing timeseries HDF5 files
OUTPUT="$BASE_DIR/step2_PostProcess/ModalAnalysis"                        # Output folder for .vtp files

SCRIPT="$SCRATCH/My_Projects/Study2_stenosis/scripts/step2_PostProcess/compute_SpatialModes_wallpressure.py"

# ---------------------------------- Step 1: Slice Parameters -----------------------------------------------------------
SLICE_XCOORD_D="8.0 10.0 12.0"  # Space-separated list of axial slice positions [D units]; one figure per slice
N_POINTS=32               # Number of evenly-spaced circumferential sample points
PIPE_AXIS=0               # Axis along which the pipe runs: 0=X, 1=Y, 2=Z
#PIPE_DIAMETER=           # Uncomment and set if you want to override the bounding-box estimate

# ---------------------------------- Step 2: CFD / Pressure Parameters --------------------------------------------------
DENSITY=1057              # Blood density [kg/m³]  (Oasis stores p/rho; multiplied to get Pa)
PERIOD_S=1.0              # Flow period [s]
# TIMESTEPS_PER_CYC and SAVE_FREQ are parsed automatically from the INPUT folder name
# (expects '_ts<int>' and '_saveFreq<int>' patterns).  Uncomment to override:
#TIMESTEPS_PER_CYC=12000
#SAVE_FREQ=1


# --------------------------------- Load Modules ------------------------------------------------------------------------
module load StdEnv/2023 gcc/12.3 python/3.12.4
source $HOME/virtual_envs/pyvista36/bin/activate
module load vtk/9.3.0


# --------------------------------- Export Directories -----------------------------------------------------------------
mkdir -p "$OUTPUT"
mkdir -p "$SCRATCH/.config/mpl"

export MPLCONFIGDIR=$SCRATCH/.config/mpl
export PYVISTA_OFF_SCREEN=true


# --------------------------------- Run Script -------------------------------------------------------------------------
python "$SCRIPT" \
    --case_name         "$CASE"          \
    --mesh_folder       "$MESH_FOLDER"   \
    --input_folder      "$INPUT"         \
    --output_folder     "$OUTPUT"        \
    --slice_xcoord_D    $SLICE_XCOORD_D  \
    --n_wallNodes       $N_POINTS        \
    --pipe_axis         $PIPE_AXIS       \
    --density           $DENSITY         \
    --period_seconds    $PERIOD_S
#   --pipe_diameter     6.35             # uncomment to override diameter estimate
#   --timesteps_per_cyc $TIMESTEPS_PER_CYC  # uncomment to override folder-name parse
#   --save_freq         $SAVE_FREQ           # uncomment to override folder-name parse

wait
echo "Job finished: $(date)"


#---------------------- For running directly from the command line ------------------------------------------------------
# Load modules first, then run:
#
python compute_modes_wallpressure.py \
    --case_name         "eccStenosis" \
    --mesh_folder       "$SCRATCH/My_Projects/Study2_stenosis/cases/case0_eccStenosis/modelOwais/step1_CFD/data" \
    --input_folder      "$SCRATCH/My_Projects/Study2_stenosis/cases/case0_eccStenosis/modelOwais/step1_CFD/results/rampoffset2mLs/eccStenosis_noisy_ts12000_cy6_saveFreq1" \
    --output_folder     "$SCRATCH/My_Projects/Study2_stenosis/cases/case0_eccStenosis/modelOwais/step2_PostProcess/ModalAnalysis_wallpressure" \
    --slice_xcoord_D    -2 0 3 10 18  \
    --n_wallNodes       16
#    --pipe_axis         0
