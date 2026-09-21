#!/bin/bash
#-----------------------------------------------------------------------------------------------------------------------
# compute_NS_terms_centerline_job.sh
# SLURM wrapper to run compute_NS_terms_centerline.py for the idealized stenosis case.
#
# __author__ = Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__   = 2026-09
#
# PURPOSE:
#   - Extract velocity/pressure at equally-spaced centerline points from CFD results.
#   - Compute the axial NS momentum term budget (temporal, convective, pressure gradient, viscous).
#   - Plot spatial profiles at selected Re values and Re-evolution at selected x/D locations.
#
# EXECUTION:
#   sbatch compute_NS_terms_centerline_job.sh
#
# Copyright (C) 2026 University of Toronto, Biomedical Simulation Lab.
#-----------------------------------------------------------------------------------------------------------------------

#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=64
#SBATCH --time=00:30:00
#SBATCH --job-name NS_centerline
#SBATCH --output=NS_centerline_%j.txt

set -euo pipefail
echo "Job started: $(date)"

# ---------------------------------- Define Paths -----------------------------------------------------------------------
CASE=eccStenosis
BASE_DIR=$SCRATCH/My_Projects/Study2_stenosis/cases/case0_eccStenosis/modelOwais
MESH_FOLDER="$BASE_DIR/step1_CFD/data"                                                       # folder with eccStenosis.xml.gz
INPUT_CFD="$BASE_DIR/step1_CFD/results/rampoffset2mLs/${CASE}_clean_ts12000_cy6_saveFreq1"  # CFD HDF5 results
OUTPUT="$BASE_DIR/step2_PostProcess"

SCRIPT="$SCRATCH/My_Projects/Study2_stenosis/scripts/step2_PostProcess/compute_NS_terms_centerline.py"

# --------------------------------- Load Modules -----------------------------------------------------------------------
module load StdEnv/2023 gcc/12.3 python/3.12.4
source $HOME/virtual_envs/pyvista36/bin/activate
module load vtk/9.3.0

# ------------------------------ Export Directories / env vars ---------------------------------------------------------
mkdir -p "$OUTPUT"
mkdir -p "$SCRATCH/.config/mpl"

export MPLCONFIGDIR=$SCRATCH/.config/mpl
export MPLBACKEND=Agg

# ------------------------------ Run -----------------------------------------------------------------------------------
python "$SCRIPT" \
    --case_name              "$CASE"            \
    --input_folder           "$INPUT_CFD"       \
    --mesh_folder            "$MESH_FOLDER"     \
    --output_folder          "$OUTPUT"          \
    --n_centerline_points    150                \
    --pipe_axis              0                  \
    --centerline_coord1      0.0                \
    --centerline_coord2      0.0                \
    --x_ref_mm               0.0                \
    --pipe_diameter          6.35               \
    --ramp_slope             2.0                \
    --ramp_offset            2.0                \
    --Re_plot_targets        100 300 600 1000   \
    --xD_plot_targets        -2.0 -1.0 0.0 1.0 3.0 \
    --Re_min_plot            50                 \
    --Re_max_plot            1400

# ---------------------- Direct command-line run (comment out sbatch section above) ------------------------------------
# module load StdEnv/2023 gcc/12.3 python/3.12.4
# source $HOME/virtual_envs/pyvista36/bin/activate
# export MPLBACKEND=Agg
#
python compute_NS_terms_centerline.py \
    --case_name              "eccStenosis" \
    --input_folder           "$SCRATCH/My_Projects/Study2_stenosis/cases/case0_eccStenosis/modelOwais/step1_CFD/results/rampoffset2mLs/eccStenosis_clean_ts12000_cy6_saveFreq1" \
    --mesh_folder            "$SCRATCH/My_Projects/Study2_stenosis/cases/case0_eccStenosis/modelOwais/step1_CFD/data" \
    --output_folder          "$SCRATCH/My_Projects/Study2_stenosis/cases/case0_eccStenosis/modelOwais/step2_PostProcess" \
    --n_centerline_points    150  \
    --ramp_slope             2.0  \
    --ramp_offset            2.0  \
    --Re_plot_targets        200 400 600 750 \
    --xD_plot_targets        -1.0 0.0 3.0 10.0

wait
echo "Job finished: $(date)"
