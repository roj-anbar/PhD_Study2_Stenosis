# -----------------------------------------------------------------------------------------------------------------------
# compute_Spectrograms.py 
# To compute the average power spectrogram in dB scale of pressure or velocity from Oasis/BSLSolver CFD outputs.
#
# __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2025-10
#
# PURPOSE:
#   - This script is part of the BSL post-processing pipeline.
#   - Reads CFD HDF5 snapshots for all timesteps, extracts the quantity of interest (wall pressure/velocity) time-series,
#     computes ROI-averaged spectrograms, and saves both .npz data and .png images.
#
# REQUIREMENTS:
#   - h5py, pyvista, vtk, numpy, scipy, matplotlib
#   - On Trillium: virtual environment called "pyvista36"
#
# EXECUTION:
#   - Run using "compute_Spectrograms_job.sh" bash script.
#   - Run directly on a login/debug node as below:
#       > module load StdEnv/2023 gcc/12.3 python/3.12.4
#       > source $HOME/virtual_envs/pyvista36/bin/activate
#       > module load  vtk/9.3.0
#
# EXAMPLE CLI (with required arguments):
#       > python compute_Spectrograms.py \
#           --input_folder       <path_to_CFD_results_folder> \
#           --mesh_folder        <path_to_case_mesh_data>     \
#           --output_folder      <path_to_output_folder>      \
#           --case_name          PTSeg028_base_0p64           \
#           --ROI_center_csv     <path_to_centerline_CSV_file>       \
#           --ROI_radius         4.0                          \
#
#
# INPUTS:
#   - input_folder        Path to results directory with HDF5 snapshots
#   - mesh_folder         Path to the case mesh data files (with Mesh/Wall/{coordinates,topology,pointIds})
#   - case_name           Case name (used only in output filename)
#   - output_folder       Path to output directory to save results (will create subfolders files/, imgs/, ROIs/)
#   --period_seconds      Flow period [s] (if omitted, try to parse from filenames with '_Per<ms>')
#   --timesteps_per_cyc   Timesteps per cycle (if omitted, try to parse from filenames with '_ts<int>')
#   --density             Blood density [kg/m3] (default = 1050 kg/m3)
#   --spec_quantity       Quantity to compute spectrogram from: ['wallpressure', 'velocity', 'qcriterion']
#   --ROI_type            (default = 'cylinder')
#   --ROI_center_coord    X Y Z center of spherical ROI (mesh units).
#   --ROI_center_csv      Path to CSV file containing the coordinates of multiple points for ROI center.
#   --ROI_radius          Sphere radius (mesh units). If 0, treat ROI_center as the **point ID** to sample.
#   --flag_save_ROI       Flag to save the ROI.vtp surface file (If it's included in args then it's True if not it's False)
#   --flag_multi_ROI      Flag to compute spectrogram in a segment based on multiple ROIs (If it's included in args then it's True if not it's False)
#   --window_length       STFT window length (samples, i.e., snapshots)
#   --n_fft               STFT FFT length (bins)
#   --overlap_frac        STFT noverlap = overlap_frac * window_length --> Overlap fraction between consequent windows (0-1)
#   --window              STFT window type
#   --pad_mode            Edge padding ('cycle','constant','odd','even','none')
#   --detrend             STFT detrend ('linear','constant', or False)
#   --cutoff_db           Minimum threshold for calculated power in SPL dB (default: 0) --> anything below that will be set to this value
#   --cutoff_freq         Maximum frequency threshold in Hz for filtering high frequencies (default: 1500 Hz) --> anything above this frequency is cut from the spectrogram
#   --n_process           Number of worker processes (default: #logical CPUs)
#
# OUTPUTS:
#   1) Processed spectrograms saved as .npz (output_folder/files)
#   2) Spectrogram images saved as .png     (output_folder/imgs)
#   3) ROI surface file saved as .vtp       (output_folder/ROIs)
#
# NOTES:
#   - Sampling rate inferred as: fs = timesteps_per_cyc / period_seconds [Hz].
#   - Filename helpers expect snapshots containing '_curcyc_' and optionally '_ts<int>' and '_Per<ms>'.
#   - Pressure in Oasis is p/rho; multiply by density (default 1050 kg/m^3) to get Pa if desired.
#
# Adapted from BSL-tools repository (Dan Macdonald 2022) and make_wall_pressure_specs.py (Anna Haley 2024). 
# Copyright (C) 2025 University of Toronto, Biomedical Simulation Lab.
# -----------------------------------------------------------------------------------------------------------------------

import sys
import gc
import h5py
import warnings
import argparse
from pathlib import Path
import multiprocessing as mp
from multiprocessing import sharedctypes
from collections import defaultdict

import re   # for text manupulation

import vtk
import numpy as np
from scipy.signal import stft, find_peaks
from scipy.ndimage import uniform_filter1d #for cleaning signal
import pyvista as pv
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning) 


# ======================================================================================================
# GENERAL UTILITIES
# ======================================================================================================

# -------------------------------- Shared-memory Utilities ---------------------------------------------

def create_shared_array(size, dtype = np.float64):
    """Create a ctypes-backed shared array filled with zeros."""
    ctype_array = np.ctypeslib.as_ctypes( np.zeros(size, dtype=dtype) )
    return sharedctypes.Array(ctype_array._type_, ctype_array,  lock=False)

def view_shared_array(shared_obj):
    """Get a NumPy view (no copy) onto a shared ctypes array created by create_shared_array."""
    return np.ctypeslib.as_array(shared_obj)


# ---------------------------------------- Helper Utilities -----------------------------------------------------

def extract_timestep_from_h5_filename(h5_file: Path) -> int:
    """Extract integer timestep values of the current file from filename pattern '*_ts=<int>_...'.

    Used as a sort key to order HDF5 snapshots chronologically.
    Example: 'result_curcyc_ts=0042_up.h5' → 42
    """
    match = re.search(r'_ts=(\d+)', h5_file.stem)
    if match is None:
        raise ValueError(f"Filename '{h5_file.name}' does not contain expected '_ts=<int>' pattern.")

    return int(match.group(1))


def extract_sim_params_from_h5_filename(h5_file: Path) -> tuple[float, int]:
    """Parse timesteps-per-cycle from a snapshot filename.

    Expected patterns:
      '_ts<int>'   — timesteps per cycle             (e.g. '_ts500_')

    Returns:
    timesteps_per_cyc : int
    """
    stem = h5_file.stem

    match_ts = re.search(r'_ts(\d+)', stem)
    if match_ts is None:
        raise ValueError(
            f"Filename '{h5_file.name}' has no '_ts<int>' pattern. "
            "Supply --timesteps_per_cyc on the CLI instead."
        )

    timesteps_per_cyc = int(match_ts.group(1))

    return timesteps_per_cyc


# ---------------------------------------- Mesh Utilities -----------------------------------------------------

def load_surface_mesh(mesh_file:Path) -> pv.PolyData:
    """
    Build a Pyvista PolyData surface from the wall mesh stored in a BSLSolver-style HDF5 file.

    Expects HDF5 layout:
      Mesh/Wall/coordinates : (Npoints, 3) float    – XYZ node positions
      Mesh/Wall/topology    : (Ncells, 3 or 4) int  – triangle connectivity
      Mesh/Wall/pointIds    : (Npoints,) int        – global volume-mesh point IDs
    """

    with h5py.File(mesh_file, 'r') as h5:
        coords = np.array(h5['Mesh/Wall/coordinates'])      # coords of wall points (n_points, 3)
        cells  = np.array(h5['Mesh/Wall/topology'])         # connectivity of wall points (n_cells, 3) -> triangles
        point_ids   = np.array(h5['Mesh/Wall/pointIds'])    # mapping to volume point IDs (n_points,)
        
    # Create connectivity array compatible with VTK --> requires a size prefix per cell (here '3' for triangles)
    n_cells        = cells.shape[0]
    node_per_cell  = 3                                                      # the surface cells are triangles with size of 3 (3 nodes per elem)
    cell_size      = np.full((n_cells, 1), node_per_cell, dtype=np.int64)   # array of size (n_cells, 1) filled with 3 
    cells_vtk      = np.hstack([cell_size, cells]).ravel()                  # horizontal stacking of arrays / ravel: flattens the array into a 1d array
        
    # Build surface and attach point ID
    surf = pv.PolyData(coords, cells_vtk)
    surf.point_data['vtkOriginalPtIds'] = point_ids

    return surf


def load_volume_mesh(mesh_file: Path) -> pv.UnstructuredGrid:
    """
    Create a PyVista UnstructuredGrid from the volumetric mesh stored in a BSLSolver-style HDF5.

    Expects datasets:
      Mesh/coordinates : (Npoints, 3) float
      Mesh/topology    : (Ncells,  4) int  - tetrahedra expected
    """
    
    with h5py.File(mesh_file, 'r') as h5:
        coords = np.array(h5['Mesh/coordinates'])  # coords of volumetric points (n_points, 3)
        cells  = np.array(h5['Mesh/topology'])     # connectivity of volumetric points (n_cells, 4) -> tetrahedrons
        
    # Create connectivity array compatible with VTK --> requires a size prefix per cell (here '4' for tetrahedrons)
    # VTK cell array layout = [nverts, v0, v1, v2, v3, nverts, ...]
    n_cells        = cells.shape[0]
    node_per_cell  = 4      # the volumetric cells are tets with size of 4 (4 nodes per elem)
    cell_types     = np.full(n_cells, pv.CellType.TETRA, dtype=np.uint8)
    cell_size      = np.full((n_cells, 1), node_per_cell, dtype = np.int64)  # array of size (n_cells, 1) filled with 4 
    cells_vtk      = np.hstack([cell_size, cells]).ravel() # horizontal stacking of array / ravel: flattens the array into a 1d array


    # Build grid and attach points
    vol_mesh = pv.UnstructuredGrid(cells_vtk, cell_types, coords)

    return vol_mesh, cells

# --------------------------------- Parallel File Reader -----------------------------------------------

def read_wallpressure_from_h5_files(file_ids, wall_pids, h5_files, shared_pressure_ctype, density):
    """
    Reads a *chunk* of time-snapshot HDF5 files, extracts wall pressures, and writes into the shared array.

    Arguments:
      file_ids               : list of snapshot indices assigned to this worker
      wall_pids              : wall point indices used to slice the full pressure field
      h5_files               : list of Path objects to HDF5 snapshots (all timesteps)
      shared_pressure_ctype  : shared ctypes array; viewed as (n_points, n_times)
      density                : blood density [kg/m³] — multiplied because Oasis stores p/rho
    """

    # Create a numpy view of shared (across processes) array of wall-pressure time-series
    shared_pressure = view_shared_array(shared_pressure_ctype)
    
    for t_index in file_ids:
        with h5py.File(h5_files[t_index], 'r') as h5:
            #pressure = np.array(h5['Solution']['p']) * density
            #pressure_wall = pressure[wall_pids].flatten() # shape: (n_points,)
            pressure_wall = np.array(h5['Solution']['p'])[wall_pids].flatten() * density
        shared_pressure[:, t_index] = pressure_wall

def read_wallpressure_from_h5_files_parallel(CFD_h5_files, wall_mesh, n_process, density):
    """
    Read all wall-pressure snapshots in parallel and return a (n_points, n_times) array.

    Spawns n_process workers, each reading a contiguous chunk of HDF5 files into a shared-memory array.
    Workers write directly into shared memory.
    """
    
    n_snapshots = len(CFD_h5_files)                      # total number of saved frames    
    wall_pids = wall_mesh.point_data['vtkOriginalPtIds'] # wall point IDs and sizes
    n_points  = len(wall_mesh.points)


    # Create and allocate shared arrays
    # Array to hold pressures (n_points, n_times) - written by worker processes
    shared_pressure_ctype = create_shared_array([n_points, n_snapshots])

    print(f"\nReading {n_snapshots} CFD results HDF5 files in parallel into 1 array of shape [{n_points}, {n_snapshots}] ... \n")
        
    # Divide all snapshot files into chunks and spread across workers
    time_indices    = list(range(n_snapshots))
    time_chunk_size = max(n_snapshots // n_process, 1)
    time_groups     = [time_indices[i : i + time_chunk_size] for i in range(0, n_snapshots, time_chunk_size)]
        
    processes_list = []
    for idx, group in enumerate(time_groups):
        proc = mp.Process(
            target = read_wallpressure_from_h5_files,
            name = f"Reader{idx}",
            args = (group, wall_pids, CFD_h5_files, shared_pressure_ctype, density))
        processes_list.append(proc)

    # Start all readers
    for proc in processes_list:
        proc.start()

    # Wait for all readers to finish
    for proc in processes_list:
        proc.join()

    wall_pressure = view_shared_array(shared_pressure_ctype) # (n_points, n_times)
    
    #gc.collect() # Free up memory

    return wall_pressure


# ---------------------------------- Fourier Transform Utilities -------------------------------------------
# Used to determine STFT params if not given
def shift_bit_length(x: int) -> int:
    """ Round up to nearest power of 2.
    Notes: See https://stackoverflow.com/questions/14267555/find-the-smallest-power-of-2-greater-than-n-in-python
    """
    return 1<<(x-1).bit_length()

def short_time_fourier(data,
                        sampling_rate: float,
                        window_type:   str,
                        window_length: int,
                        overlap_frac:  float,
                        n_fft:         int,
                        pad_mode:      str,
                        detrend:       str,
                        ):

    """
    Compute the windowed FFT for a timeseries data.
    All scipy.signal STFT parameters are set to defaults if they are None.
    See here for scipy.signal.stft documentation:  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.stft.html 
   
    
    Arguments: 
        data: Timeseries data for the point of interest -> shape (n_points, n_frames)
        sampling_rate: Number of samples per second [Hz]
        window_type : Window type for stft
        window_length (nperseg): Number of time samples in each window 
        overlap_frac: Fraction (0-1) of overlap between segments
        n_fft: Number of FFT bins in each segment (>= window_length) -> if a zero padded FFT is desired, if None, is equal to window_length (nperseg) 
        pad_mode : Optional padding strategy to reduce edge artifacts {'cycle','constant','odd','even',None}
        detrend : {'linear','constant', False}
    
    Returns:
        freqs: Frequency vector in [Hz] -> shape (n_freqs,)
        bins : Time vector in [seconds] -> shape (n_frames,)
        Z    : Complex STFT output      -> shape (n_freqs, n_frames)
    """

    n_frames = data.shape[1]

    # Define defaults
    if window_length is None: window_length = shift_bit_length(int(n_frames / 10))
    if n_fft is None: n_fft = window_length

    if pad_mode == 'cycle':
        pad_size = window_length // 2
        front_pad = data[:,-pad_size:]
        back_pad = data[:,:pad_size]
        data = np.concatenate([front_pad, data, back_pad], axis=1)
        boundary = None 

    elif pad_mode == 'constant':
        pad_size = window_length // 2
        front_pad = np.zeros((data.shape[0], pad_size)) + data[:,0][:,None]
        back_pad = np.zeros((data.shape[0], pad_size)) + data[:,-1][:,None]
        data = np.concatenate([front_pad, data, back_pad], axis=1)
        boundary = None 

    elif pad_mode in ['odd', 'even', 'none', None]:
        boundary = pad_mode
    

    stft_params = {
        'fs' : sampling_rate,
        'window' : window_type,
        'nperseg' : window_length,
        'noverlap' : int(overlap_frac * window_length),   # number of overlapping samples
        'nfft' : n_fft,
        'detrend' : detrend,
        'return_onesided' : True,
        'boundary' : boundary,
        'padded' : True,
        'axis' : -1,
        }


    # All the below S arrays have shape (n_freq, n_frames)
    freqs, bins, Z = stft(x=data, **stft_params) #data[0] will be the first row

    return freqs, bins, Z


# ---------------------------------- ROI Utilities -------------------------------------------
def read_ROI_centerlines_from_csv(csv_path: str, ROI_type: str) -> np.ndarray:
    """
    Read a CSV of ROI centerline points with columns:
    - Points:0, Points:1, Points:2
    - FrenetTangent:0, FrenetTangent:1, FrenetTangent:2
    Uses header names to locate columns.
    Returns
    - coords  : (n_points, 3) array of XYZ coordinates
    - normals : (n_points, 3) array of normals
    """

    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    # Field names from the header (quotes in CSV are handled by genfromtxt)
    #column_names = data.dtype.names

    # Change the header names for your dataset
    x = data['Points0']; y = data['Points1']; z = data['Points2']
    
    # We need the normals just for cylindrical ROI
    if ROI_type == "cylinder":
        normx = data['FrenetTangent0']; normy = data['FrenetTangent1']; normz = data['FrenetTangent2']

    elif ROI_type == "sphere":
        normx, normy, normz = np.zeros_like(x), np.zeros_like(y), np.zeros_like(z) 

    else:
        raise ValueError(f'ROI type {ROI_type} not recognized! Choose from <cylinder> or <sphere>.')

    coords = np.vstack([x,y,z]).T
    normals = np.vstack([normx,normy,normz]).T

    return coords, normals


def read_spec_regions_from_csv(csv_path: str) -> list:
    """
    Read ROI region definitions from a CSV file.
    Required columns : ROI_start_center_id, ROI_end_center_id, ROI_stride, ROI_radius
    Optional columns : region_shortname, ROI_height, flag_multi_ROI, flag_save_ROI
    Any other columns (e.g. region_name) are silently ignored.
    region_shortname is used for constructing the spectrograms labels.
    Returns a list of dicts, one per row, containing only the recognised keys.
    """
    int_keys   = {"ROI_start_center_id", "ROI_end_center_id", "ROI_stride"}
    float_keys = {"ROI_radius", "ROI_height"}
    bool_keys  = {"flag_multi_ROI", "flag_save_ROI"}
    str_keys   = {"region_fullname", "region_shortname"}
    known_keys = int_keys | float_keys | bool_keys | str_keys

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.ndim == 0:
        data = data.reshape(1)   # handle single-row CSV

    spec_regions = []
    for row in data:
        region = {}
        for key in data.dtype.names:
            if key not in known_keys:
                continue
            val = row[key]
            if key in int_keys:
                region[key] = int(val)
            elif key in float_keys:
                region[key] = float(val)
            elif key in bool_keys:
                region[key] = bool(int(val)) if str(val).strip().lstrip('-').isdigit() else str(val).strip().lower() in ("true", "yes")
            elif key in str_keys:
                region[key] = str(val).strip()
        spec_regions.append(region)

    return spec_regions

def read_xrange_regions_from_csv(csv_path: str) -> list:
    """
    Read axial (x-range) region definitions for idealized geometries with a known, straight centerline
    (e.g. inlet: -3D to -1D / stenosis: -1D to 1D / post-stenosis: 1D to 6D / outlet: 6D to end).
    Required columns : x_start_D, x_end_D   (bounds in multiples of pipe diameter D)
    Optional columns : region_shortname, flag_save_ROI
    The strings "start"/"end" in x_start_D/x_end_D mean "extend to the mesh's actual min/max x".
    Returns a list of dicts, one per row.
    """
    float_or_none_keys = {"x_start_D", "x_end_D"}
    bool_keys           = {"flag_save_ROI"}
    str_keys            = {"region_shortname"}
    known_keys          = float_or_none_keys | bool_keys | str_keys

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    if data.ndim == 0:
        data = data.reshape(1)   # handle single-row CSV

    regions = []
    for row in data:
        region = {}
        for key in data.dtype.names:
            if key not in known_keys:
                continue
            val = row[key]
            if key in float_or_none_keys:
                sval = str(val).strip().lower()
                region[key] = None if sval in ("start", "end", "none", "") else float(sval)
            elif key in bool_keys:
                region[key] = bool(int(val)) if str(val).strip().lstrip('-').isdigit() else str(val).strip().lower() in ("true", "yes")
            elif key in str_keys:
                region[key] = str(val).strip()
        regions.append(region)

    return regions


def assemble_quantity_array_for_one_ROI(output_folder_ROIs, surf_mesh, vol_mesh, var_name, var_array, ROI_params, return_indices=False):
    """
    Select mesh points inside a ROI with defined shape (ROI_type) and return the time series of variable of interest for those points.
    Note: Units of the radius should be the same as units of the mesh.
    """

    # Unpack input parameters
    ROI_id            = ROI_params.get("ROI_id")
    ROI_type          = ROI_params.get("ROI_type")
    ROI_center_coord  = ROI_params.get("ROI_center_coord")
    ROI_center_normal = ROI_params.get("ROI_center_normal")
    ROI_radius        = ROI_params.get("ROI_radius")
    ROI_height        = ROI_params.get("ROI_height")
    flag_save_ROI     = ROI_params.get("flag_save_ROI")

    # --- Choose the mesh domain based on the given variable name

    if var_name in {'wallpressure'}:
        mesh = surf_mesh
    elif var_name in {'velocity', 'qcriterion'}:
        mesh = vol_mesh


    # --- Compute coordinate of ROI center
    # Case 1: Obtain spectrograms at a single point
    if ROI_type == 'point':
        ROI_pids = ROI_center_coord

    # Case 2: Obtain spectrograms in a spherical or cylindrical ROI (using pyvista)
    elif ROI_type in ['sphere', 'cylinder']:
        
        # Creates a surface geometry of ROI_type centered at the ROI point
        if ROI_type == 'sphere':
            ROI_geom = pv.Sphere(radius = ROI_radius, center = ROI_center_coord)
        elif ROI_type == 'cylinder':
            # Note: needed to add clean() to the surface to make it compatible with vtk 'select_enclosed_points'
            ROI_geom = pv.Cylinder(center = ROI_center_coord, direction = ROI_center_normal, radius = ROI_radius, height = ROI_height).clean()

        # Selects mesh points inside the ROI geometry surface with a certain tolerance (using pyvista)
        ROI_mesh = mesh.select_enclosed_points(ROI_geom, tolerance=0.01)
        
        # Get indices of the points that falls in the ROI
        points_in_ROI = ROI_mesh.point_data['SelectedPoints'].astype(bool)
        ROI_pids = np.where(points_in_ROI)[0]

        # Save ROI geometry to a .vtp file if requested (for visualization in paraview later)
        if flag_save_ROI:
            ROI_geom.save(f'{output_folder_ROIs}/{ROI_id}_{ROI_type}_c{ROI_center_coord}_r{ROI_radius}.vtp') 

    # For any other types    
    else:
        raise ValueError("--ROI_type is not supported, choose from ['point', 'sphere', 'cylinder'].")


    # --- Sanity check: ensure ROI is not empty ---
    if ROI_pids.size == 0:
        raise ValueError("No mesh points found in ROI. Try increasing --ROI_radius (check mesh units: mm vs m) or choose a different --ROI_center_coord. ")
    else:
        print(f"Found {ROI_pids.size} mesh points in {ROI_id} with center coordinate {ROI_center_coord} ...")

    # Assemble variable array for ROI points
    var_array_ROI = var_array[ROI_pids,:]

    # Return the point ids if requested
    if return_indices:
        return ROI_pids

    return var_array_ROI


        
# ======================================================================================================
# HEMODYNAMICS FUNCTIONS
# ======================================================================================================


# ---------------------------------------- Compute Spectrograms -----------------------------------------------------

def calculate_mean_spectrogram(var_name, var_array, STFT_params):
    """
    Compute an average spectrogram (in dB) based on the given array for the variable of interest with configurable STFT parameters.

    var_array:
    STFT_params: 

    Returns:
    spectrogram_data
    """
    
    # Unpack input parameters
    sampling_rate = STFT_params.get("sampling_rate")
    window_length = STFT_params.get("window_length")
    overlap_frac  = STFT_params.get("overlap_frac")
    n_fft         = STFT_params.get("n_fft")
    pad_mode      = STFT_params.get("pad_mode")
    window_type   = STFT_params.get("window_type")
    detrend       = STFT_params.get("detrend")
    #cutoff_db     = STFT_params.get("cutoff_db")
    #cutoff_freq   = STFT_params.get("cutoff_freq")

    # test: cut signal at Q=8
    #Q_cut       = analysis_params.get("Q_max")  # ml/s 
    signal      = var_array #var_array[:, :int(Q_cut / 2 * sampling_rate)]
    n_points    = signal.shape[0]
    n_snapshots = signal.shape[1] # total number of snapshots

    # Remove slowly-varying baseline (ramp mean) before STFT
    #baseline = uniform_filter1d(signal, size=window_length, axis=1, mode='nearest')
    #signal = signal - baseline

    # If window_length is not defined, divide the signal by 10 by default 
    if window_length is None: window_length = shift_bit_length(int(n_snapshots / 10))


    # Note: All the below S arrays have shape (n_freq, n_frames)

    # Compute FFT for first point. # Pass data as row vectors
    freqs, bins, Z0 = short_time_fourier(signal[0][None,:], sampling_rate, window_type, window_length, overlap_frac, n_fft, pad_mode, detrend)
    power_sum = np.zeros_like(Z0, dtype=np.float64)

    # Case 1: Single point ROI
    if n_points == 1:
        power_avg  = np.abs(Z0)**2

    # Case 2: Multiple points ROI
    else:
        for point in range(n_points):
            # Pass data as row vectors
            _, _, Z_point = short_time_fourier(signal[point][None,:], sampling_rate, window_type, window_length, overlap_frac, n_fft, pad_mode, detrend)
            power_point = np.abs(Z_point)**2
            power_sum += power_point 
            
        power_avg = power_sum / n_points

        
    # Define the reference value for normalizing the power to obtain dB scales
    if var_name == 'wallpressure':
        power_ref = (2e-5)**2 
    else:
        power_ref = np.mean(power_avg) 

    # Convert power to dB scale
    power_avg_db = 10.0 * np.log10(power_avg / power_ref)
    power_avg_db = np.squeeze(power_avg_db)



    if pad_mode in ['cycle', 'even', 'odd']:
        bins = bins - bins[0]


    # Remove last frame to keep edges clean    
    power_avg_db = power_avg_db[:,:-1]
    bins = bins[:-1]

    # Clamp values below a threshold
    #power_avg_db[power_avg_db < cutoff_db] = cutoff_db

    # Set the power for any frequencies above a certain threshold to zero
    #mask = freqs <= cutoff_freq
    #power_avg_db[freqs > cutoff_freq, :] = 0


    # Store all values in spectrogram_data
    spectrogram_data = {
            'power_avg_linear': power_avg,  # (n_freq, n_frames)    — linear scale (|Z|^2), before dB conversion
            'power_avg_dB': power_avg_db,   # (n_freq, n_frames)    — dB scale, clamped/filtered
            'bins': bins,                   # time values (n_frames,1)
            'freqs': freqs,                 # (n_freqs,1)
            'sampling_rate': sampling_rate,
            'n_fft': n_fft,
            'window_length': window_length,
            'overlap_frac': overlap_frac,
        }

    
    return spectrogram_data


# ======================================================================================================
# SPECTROGRAM ANALYSIS: QUANTIFICATION AND CLASSIFICATION FUNCTIONS
# ======================================================================================================

def filter_raw_spectrogram(spectrogram_data, spectral_analysis_params):
    """
    Filter and trim the raw spectrogram data to the analysis window.
    Three operations are applied:
        1. Frequency axis : keep only rows where freqs <= freq_max.
        2. Q axis         : keep only columns where Q_inlet = 2*bins falls in [Q_min, Q_max].
        3. dB floor       : clamp any power values below cutoff_db up to cutoff_db.

    Parameters
    ----------
    spectrogram_data (dict) : Output of calculate_mean_spectrogram.
    spectral_analysis_params (dict) : Must contain 'cutoff_db', 'freq_max', 'Q_min', 'Q_max'.

    Returns
    -------
    spec_filt (dict): same as spectrogram_data but with bins and power arrays restricted to the analysis window.
    """

    # Unpack parameters
    cutoff_db = spectral_analysis_params.get("cutoff_db")
    freq_max  = spectral_analysis_params.get("freq_max")
    Q_min     = spectral_analysis_params.get("Q_min")
    Q_max     = spectral_analysis_params.get("Q_max")

    freqs        = spectrogram_data['freqs']
    bins         = spectrogram_data['bins']
    power_avg_dB = spectrogram_data['power_avg_dB']

    # Build masks
    bins_Q    = 2 * bins                           # Q_inlet = 2*t  (ramp-specific conversion)
    mask_Q    = (bins_Q >= Q_min) & (bins_Q <= Q_max)
    mask_freq = freqs <= freq_max
    
    # Apply both masks simultaneously
    power_filt = power_avg_dB[np.ix_(mask_freq, mask_Q)]

    # Clamp values below the dB floor
    power_filt[power_filt < cutoff_db] = cutoff_db


    # Save the filtered fields to a similar structure as raw spectrogram
    spec_filt = dict(spectrogram_data)   # shallow copy

    spec_filt['freqs']        = freqs[mask_freq]
    spec_filt['bins']         = bins[mask_Q]
    spec_filt['power_avg_dB'] = power_filt
    #spec_filt['power_avg_linear'] = spectrogram_data['power_avg_linear'][:, analysis_mask]

    return spec_filt


def extract_metrics_from_spectrogram_column(freqs, spec_col_dB, f_low, f_mid, f_max):
    """
    Compute simple metrics for one spectrogram column (one time).
    spec_col_dB: 1D array (n_freq,) in dB.
    f_low:       low frequency threshold in Hz (default = 100 Hz).
    f_mid:       mid frequency threshold in Hz (default = 1000 Hz).
    f_max:       max frequency threshold in Hz (default = 5000 Hz).
    
    Returns a dictionary of metrics for each column based on the 3 frequency bands (lowFreq_band: 0-f_low / midFreq_band: f_low-f_mid / highFreq_band: f_mid-f_max)
    - mean_power_lowFreq: Average acoustic power below low frequency f_low.
    - mean_power_midFreq: 
    - mean_power_highFreq: Average acoustic power above mid frequency f_mid and below high frequency f_max.
    - centroid_freq: Spectral centroid defined as the center of mass of the spectral power in Hz.
    """
    
    # Create a mask for each frequency band
    mask_lowFreq  = freqs < f_low
    mask_midFreq  = (freqs >= f_low) & (freqs < f_mid)
    mask_highFreq = (freqs >= f_mid) & (freqs < f_max)
    
    # Filter the spectrogram for each frequency band
    spec_lowFreq  = spec_col_dB[mask_lowFreq]
    spec_midFreq  = spec_col_dB[mask_midFreq]
    spec_highFreq = spec_col_dB[mask_highFreq]

    # Compute the basic spectral metrics:

    # Compute average power for each frequency band
    # Note: it is better to perform averaging in linear space and convert back to dB but this doesn't give good results for my cases
    mean_power_lowFreq  = np.mean(spec_lowFreq)   #10 * np.log10(np.mean(10**(spec_lowFreq/10))) 
    mean_power_midFreq  = np.mean(spec_midFreq)
    mean_power_highFreq = np.mean(spec_highFreq)
    
    """
    # Compute fraction of frequencies with power > 80dB
    #frac_above_80dB  = np.mean(spec_above_f_mid > 80)  

    # Compute spectral flatness (0 = very peaky, 1 = white noise)
    linear_power_highFreq = 10.0**(spec_highFreq/10.0)
    eps = 1e-12
    geometric_mean    = np.exp(np.mean(np.log(linear_power_highFreq + eps)))
    arithmetic_mean   = np.mean(linear_power_highFreq + eps)
    flatness_highFreq = geometric_mean / arithmetic_mean


    # Compute peak (dominant) frequency
    #mask_analysis = (freqs < f_max)  & (spec_col_dB > 50)   # Only use frequencies up to f_high to stay within analysis band
    #if np.any(mask_analysis):
    #    peak_freq = freqs[mask_analysis].max()
    #else:
    #    peak_freq = np.nan
    """

    # Compute spectral centroid (center of mass of spectrum)
    spec_col_linear = 10.0**(spec_col_dB/10.0)    
    centroid_freq = np.sum(freqs * spec_col_linear) / np.sum(spec_col_linear)

    spec_col_metrics = dict(mean_power_lowFreq  = mean_power_lowFreq,
                            mean_power_midFreq  = mean_power_midFreq,
                            mean_power_highFreq = mean_power_highFreq,
                            centroid_freq       = centroid_freq)


    #print(f"above_flow, f_high: {mean_power_above_f_low:.2f}, {mean_power_above_f_mid:.2f}\n")  
    return spec_col_metrics


def classify_spectrogram_phases(spectrogram_data, spectral_analysis_params):
    """
    Map metrics dict -> phase {0,1,2,3}.
    
    f_low:       low frequency threshold in Hz (default = 100 Hz).
    f_mid:       mid frequency threshold in Hz (default = 1000 Hz).
    f_max:       max frequency threshold in Hz (default = 5000 Hz) --> 5000 is the Nyquist limit.

    Intended meaning:
      0: quiet / nothing
      1: weak low-frequency activity (laminar)
      2: mid-frequency activity (harmonics)
      3: strong high-frequency activity (turbulent-like)
    """

    # Unpack parameters
    f_low  = spectral_analysis_params.get("freq_low")
    f_mid  = spectral_analysis_params.get("freq_mid")
    f_max  = spectral_analysis_params.get("freq_max")


    bins    = spectrogram_data['bins']
    freqs   = spectrogram_data['freqs']
    spec_dB = spectrogram_data['power_avg_dB']

    bins_Q = 2*bins            # for ramp Q = 2t
    n_cols = spec_dB.shape[1]  # total number of columns of spectrogram (#times)

    # Initialize arrays
    phases   = np.zeros(n_cols, dtype=int)
    spectral_metrics = defaultdict(list)

    # Loop over each frame (column) and calculate spectral metrics for it
    for col in range(n_cols):
        metrics_column = extract_metrics_from_spectrogram_column(freqs, spec_dB[:, col], f_low, f_mid, f_max)
        
        # Append the metrics for each column to the overall metrics array
        for key, value in metrics_column.items():
            spectral_metrics[key].append(value)

  
    # Convert the metrics to numpy array
    spectral_metrics = {k: np.array(v) for k, v in spectral_metrics.items()}


    #------------ Define each phase --------------------

    Q_phases = np.full(3, np.nan) # Initialize Qphases as NaNs

    # PHASE 1: First rise in midFreq power
    idx_nonzero_midFreq_power = np.where(spectral_metrics['mean_power_midFreq'] > 0.1)[0] # array of indices of positive midFreq powers
    #idx_nonzero_spectral_centroid = np.where(spectral_metrics['centroid_freq'] > 1)[0]  # array of indices of positive spectral centroid

    if len(idx_nonzero_midFreq_power) > 0:
        Q_phases[0] = bins_Q[idx_nonzero_midFreq_power[0]]      # first rise in midFreq power
        #Q_phases[0] = bins_Q[idx_nonzero_spectral_centroid[0]]  # first rise in centroid

    # PHASE 2: First rise in highFreq power
    idx_nonzero_highFreq_power = np.where(spectral_metrics['mean_power_highFreq'] > 0.1)[0] # array of indices of positive highFreq powers

    if len(idx_nonzero_highFreq_power) > 0:
        Q_phases[1] = bins_Q[idx_nonzero_highFreq_power[0]] # first rise in power


    # PHASE 3: Sustained centroid freq above f_low
    # centroid_above_lowFreq = spectral_metrics['centroid_freq'] > f_low
    # idx_centroid_below_lowFreq = np.where(~centroid_above_lowFreq)[0]

    # if centroid_above_lowFreq[-1]:  # centroid stays above f_low until end
    #     start = idx_centroid_below_lowFreq[-1] + 1
    #     idx_phase3 = start - 1
    #     Q_phases[2] = bins_Q[idx_phase3]


    return Q_phases, spectral_metrics


def plot_spectrogram_and_metrics(output_folder_imgs, case_name, spectrogram_data, Q_phases, spectral_metrics, analysis_params, plot_title, flag_plot_phases=True):
    """
    Plot and save spectrograms and spectral metrics as PNG files.
    """

    # Extract relevant data for plotting
    bins  = spectrogram_data['bins']
    freqs = spectrogram_data['freqs']
    spectrogram_signal = spectrogram_data['power_avg_dB']

    # JUST FOR PT_RAMP:
    # Create bins to show Q_inlet (instead of time) --> specify based on the ramp slope
    bins_Q = 2*bins # Q_in = 2*t

    # Setting plot properties
    font_size = 20
    plt.rc('axes',   titlesize=font_size)     # fontsize of the title
    plt.rc('font',   size=font_size)          # controls default text size
    plt.rc('xtick',  labelsize=font_size)    # fontsize of the x tick labels
    plt.rc('ytick',  labelsize=font_size)    # fontsize of the y tick labels
    plt.rc('legend', fontsize=font_size)    # fontsize of the legend
    plt.rc('axes',   labelsize=18)     # fontsize of the x and y labels


    #fig, ax = plt.subplots(1, 3, figsize=(20, 5)) #(8,18) #(20,6)
    fig, ax = plt.subplots(3, 1, figsize=(8, 16), sharex=True) #, gridspec_kw={'hspace': 0.05})

    fig.suptitle(plot_title, fontweight='bold', y=0.99)             # y adds distance to the title's location


    # ------------------------ Subplot 0: Spectrogram ----------------------------
    spectrogram = ax[0].pcolormesh(bins_Q, freqs, spectrogram_signal, shading='gouraud', cmap='inferno')
    # Set the limit for power colormap
    spectrogram.set_clim(analysis_params['SPL_db_min'], analysis_params['SPL_db_max'])


    ax[0].set_ylabel('Frequency (Hz)',   fontweight='bold', fontsize=font_size, labelpad=10)

    # Set different y limits based on the case
    if 'PTSeg043' in case_name:
        ax[0].set_ylim([0, analysis_params['freq_mid']/2])
    else:
        ax[0].set_ylim([0, 3000]) #analysis_params['freq_max']])

    # Adding the colorbar
    #cbar = fig.colorbar(spectrogram, ax=ax[0], orientation='vertical') #pad=0.5
    #cbar.set_label('SPL (dB)', rotation=270, labelpad=15, size=16, fontweight='bold')


    # ------------------------ Subplot 1: Mean power ----------------------------
    ax[1].plot(bins_Q, spectral_metrics['mean_power_lowFreq'],  label='low-freq',  linewidth = 4, color='tab:green') #(160/255,230/255,245/255)) #RGB:'#A6CEE3'
    ax[1].plot(bins_Q, spectral_metrics['mean_power_midFreq'],  label='mid-freq',  linewidth = 4, color='tab:blue') #deepskyblue
    ax[1].plot(bins_Q, spectral_metrics['mean_power_highFreq'], label='high-freq', linewidth = 4, color='tab:red') #'mediumblue'

    ax[1].set_ylim([-1, analysis_params['SPL_db_max']])
    ax[1].set_ylabel('Mean SPL power (dB)', fontweight='bold', labelpad=20, fontsize=font_size)
    #ax[1].legend(loc = 'upper left', fontsize=font_size)

    # ------------------------ Subplot 2: Spectral Centroid ----------------------------
    ax[2].plot(bins_Q, spectral_metrics['centroid_freq'], linewidth = 4, color='black')
    ax[2].set_ylim([-1, 300])
    ax[2].set_ylabel('Spectral Centroid (Hz)', fontweight='bold', fontsize=font_size, labelpad=10)



    #------- Common x-axis settings
    for a in ax:
        a.set_xlim([analysis_params['Q_min'], analysis_params['Q_cut']])
        a.tick_params(direction='in')
        #a.set_xlabel('Flow rate (mL/s)', fontweight='bold', labelpad=10)
    ax[2].set_xlabel('Flow rate (mL/s)', fontweight='bold', fontsize=font_size, labelpad=10)

    #--------- Adding phase lines 
    if flag_plot_phases:
        for (phase, Qphase) in enumerate(Q_phases, start=1):
            if not np.isnan(Qphase):
                print(f'Inlet flowrate of onset Phase {phase} = {Qphase:.2f} mL/s')
                for a in ax:
                    a.axvline(Qphase, color="darkgray", linestyle="dashed", linewidth=3, zorder = 5, alpha=0.8)


    #----- For customizing the colorbar and axis for figures ----
    
    #ax.set_xticks([0, 0.9])
    #ax.set_xticklabels(['0.0', '0.9'])
    #ax.set_yticks([0, 600, 800])
    #ax.set_yticklabels(['0', '600', '800'])

    # Define the ticks you want
    #ticks = [40, 60, 80, 100]
    #cbar.set_ticks(ticks)
    #cbar.set_ticklabels([str(t) for t in ticks])   # optional if you want custom text

    #cbar.ax.xaxis.tick_top()
    #cbar.ax.tick_params(labelsize=46)
    #cbar.ax.xaxis.set_label_position('top')
    #cbar.set_label('Power (dB)', rotation=270, labelpad=15, size=16, fontweight='bold')
    

    plt.tight_layout()
    plt.savefig(Path(output_folder_imgs) / f"{plot_title}.png")#, transparent=True)
    plt.close(fig)



def compute_and_save_spectrogram_for_all_ROIs(
                    case_name: str,
                    output_folder_files: Path,
                    output_folder_imgs: Path,
                    output_folder_ROIs: Path,
                    surf_mesh: pv.PolyData,
                    vol_mesh: pv.PolyData,
                    spec_quantity: str,
                    spec_quantity_array: np.array,
                    period_seconds: float, 
                    timesteps_per_cyc: int,
                    ROI_params: dict,
                    STFT_params: dict,
                    spectral_analysis_params: dict):
    """
    Computes and saves spectrograms for:
      1) Multiple ROIs defined by roi_params["ROI_center_csv"], optionally:
         a) one regional (multi-ROI) spectrogram
         b) one spectrogram per ROI
      2) A single ROI defined by roi_params["ROI_center_coord"], if provided

    """
    
    # Unpack input parameters
    ROI_type             = ROI_params.get("ROI_type")
    ROI_center_coord     = ROI_params.get("ROI_center_coord")
    ROI_center_csv       = ROI_params.get("ROI_center_csv")
    ROI_start_center_id  = ROI_params.get("ROI_start_center_id")
    ROI_end_center_id    = ROI_params.get("ROI_end_center_id")
    ROI_stride           = ROI_params.get("ROI_stride")

    window_length        = STFT_params.get("window_length")
    overlap_frac         = STFT_params.get("overlap_frac")
    n_fft                = STFT_params.get("n_fft")


    # Compute sampling rate and add to STFT_params
    sampling_rate = timesteps_per_cyc/period_seconds # Hz
    STFT_params["sampling_rate"] = sampling_rate 

    print (f"Computing {spec_quantity} spectrograms for {ROI_type} ROIs with STFT parameters: \n"
           f"window_length (samples) = {window_length} \n"
           f"overlap_fraction        = {overlap_frac} \n")


    #----------------- Case 1: CSV mode --------------------
    # Coordinates of multiple points given in a CSV file
    if ROI_center_csv is not None:
        ROI_centers, ROI_normals = read_ROI_centerlines_from_csv(ROI_center_csv, ROI_type)
        
        #print(f"Loaded {ROI_centers.shape[0]} ROI points from {ROI_center_csv}: \n")
        
        # Loop over all center points (or with a stride)

        #-----Case 1A: Sweeping method (flag_multi_ROI = True).
        # Generates a single spectrogram for a segment chosen based on multiple ROIs
        # Here we assemble the pressure data for multiple ROIs first and then extract the average spectrogram
        if ROI_params["flag_multi_ROI"]:

            ROI_point_indices = []

            # Loop over center points in the specified region
            for i in range(ROI_start_center_id, ROI_end_center_id+1, ROI_stride): #to be inclusive of ROI_end
                center = ROI_centers[i]
                normal = ROI_normals[i]
                ROI_id = f"ROI{i:02d}"

                # Make a *copy* of ROI_params so we don't overwrite in-place
                ROI_params_multi = ROI_params.copy()
                ROI_params_multi["ROI_id"] = ROI_id
                ROI_params_multi["ROI_center_coord"]  = center
                ROI_params_multi["ROI_center_normal"] = normal

                # Obtain the point ID of all ROIs combined
                #ROI_point_indices.extend(assemble_wall_pressure_for_one_ROI(output_folder_ROIs, surf_mesh, wall_pressure, ROI_params_multi, return_indices=True))
                ROI_point_indices.extend(assemble_quantity_array_for_one_ROI(output_folder_ROIs, surf_mesh, vol_mesh, spec_quantity, spec_quantity_array, ROI_params_multi, return_indices=True))

            # Keep only the unique indices
            ROI_point_indices = np.unique(ROI_point_indices)
            spec_quantity_array_ROI_multi = spec_quantity_array[ROI_point_indices, :]
            
            # [DEBUG] Save raw node signals for the ROI region
            #raw_signals_npz = Path(output_folder_files) / f"raw_signal_{ROI_id}.npz"
            #np.savez(raw_signals_npz, quantity_array=spec_quantity_array_ROI_multi, point_indices=ROI_point_indices)

            print(f"Found {len(ROI_point_indices)} unique mesh points in total in the specified region. \n")

            # Calculate average spectrogram for all the ROIs combined
            spectrogram_data = calculate_mean_spectrogram(
                                var_name= spec_quantity,
                                var_array = spec_quantity_array_ROI_multi,
                                STFT_params = STFT_params)
            
            # Construct the title: use region_shortname from CSV if available, else fall back to ROI ID range
            region_name = ROI_params.get("region_shortname")
            region_label = region_name if region_name else f'ROI{ROI_start_center_id}to{ROI_end_center_id}'
            spectrogram_title = f'{case_name}_win{window_length}_region{region_label}'

            # Save full spectrogram data
            spec_output_npz = Path(output_folder_files) / f"{spectrogram_title}.npz"
            np.savez(spec_output_npz, spectrogram_data)

            # Cut spectrogram to analysis window
            spectrogram_data_filt = filter_raw_spectrogram(spectrogram_data, spectral_analysis_params)

            # Classify spectrogram phases
            Q_phases, spectral_metrics = classify_spectrogram_phases(spectrogram_data_filt, spectral_analysis_params)

            # Plot spectrograms and metrics
            plot_spectrogram_and_metrics(output_folder_imgs, case_name, spectrogram_data_filt, Q_phases, spectral_metrics, spectral_analysis_params, spectrogram_title)

            """
            # -------- For plotting the wall pressure signal for individual nodes -------------
            # Generate figs for a couple of points
            #print('Generating figures of wall-pressure signals at some nodes ...')
            # Create time array
            # create index array
            #array_timesteps = np.arange(len(spec_quantity_array_ROI_multi[1,:]))
            #array_Qin = 2 * (array_timesteps * period_seconds/timesteps_per_cyc)

            #for id in range(0,1000,100):

            #    fig, ax = plt.subplots(1,1, figsize=(16,8))
            #    ax.plot(array_Qin, spec_quantity_array_ROI_multi[id,:])
            #    ax.set_xlabel('time (s)', fontweight='bold', fontsize=16, labelpad=0)
            #    ax.set_ylabel('wall pressure (Pa)', fontweight='bold', fontsize=16, labelpad=0)
            #    ax.set_xlim([2,10])
            #    plt.tight_layout()
            #    plt.savefig(Path(output_folder_imgs) / f"signal_wallPressure_node{id}.png") 
            """
        
        #-----Case 1B: Generate one spectrogram per ROI
        else:
            print("[DEBUG] SCRIPT GOT HERE!!!!")
            for i in range(ROI_start_center_id, ROI_end_center_id, ROI_stride): #range(0, len(ROI_centers), 1):
                
                center = ROI_centers[i]
                normal = ROI_normals[i]
                ROI_id = f"ROI{i:02d}"

                # Add extra fields to ROI_params
                ROI_params["ROI_id"] = ROI_id
                ROI_params["ROI_center_coord"]  = center
                ROI_params["ROI_center_normal"] = normal

                # Assemble pressure data for each ROI
                spec_quantity_array_ROI = assemble_quantity_array_for_one_ROI(output_folder_ROIs, surf_mesh, vol_mesh, spec_quantity, spec_quantity_array, ROI_params)

                # Construct the title
                spectrogram_title = f'{case_name}_win{window_length}_{ROI_id}' 
                
                # Calculate average spectrogram for each ROI
                spectrogram_data = calculate_mean_spectrogram(
                                    var_name = spec_quantity,
                                    var_array = spec_quantity_array_ROI,
                                    STFT_params = STFT_params)
                
                # Save full spectrogram data
                spec_output_npz = Path(output_folder_files) / f"{spectrogram_title}.npz"
                np.savez(spec_output_npz, spectrogram_data)

                # Cut spectrogram to analysis window
                spectrogram_data_filt = filter_raw_spectrogram(spectrogram_data, spectral_analysis_params)

                # Classify spectrogram phases
                Q_phases, spectral_metrics = classify_spectrogram_phases(spectrogram_data_filt, spectral_analysis_params)

                # Plot spectrograms and metrics
                plot_spectrogram_and_metrics(output_folder_imgs, case_name, spectrogram_data_filt, Q_phases, spectral_metrics, spectral_analysis_params, spectrogram_title)

                """
                # -------- For plotting the wall pressure signal for individual nodes -------------
                # Generate figs for a couple of points
                print('Generating figures of wall-pressure signals at some nodes ...')
                # Create time array
                array_timesteps = np.arange(len(spec_quantity_array_ROI[1,:]))
                array_Qin = 2 * (array_timesteps * period_seconds/timesteps_per_cyc)

                for id in range(0,1000,200):
                    fig, ax = plt.subplots(1,1, figsize=(16,8))
                    ax.plot(array_Qin, spec_quantity_array_ROI[id,:])
                    ax.set_xlabel('Q_inlet (ml/s)', fontweight='bold', fontsize=16, labelpad=0)
                    ax.set_ylabel('wall pressure (Pa)', fontweight='bold', fontsize=16, labelpad=0)
                    ax.set_xlim([2,10])
                    plt.tight_layout()
                    plt.savefig(Path(output_folder_imgs) / f"signal_wallPressure_node{id}.png") 
                """


    #------- Case 2: Coords mode
    # Single ROI center coordinates provided
    elif ROI_center_coord is not None:
        ROI_center_coord = np.array(ROI_center_coord, dtype=float)

        # Add extra fields to ROI_params
        ROI_params["ROI_id"] = "single"
        ROI_params["ROI_center_coord"] = ROI_center_coord

        # Assemble pressure data for each ROI
        spec_quantity_array_ROI = assemble_quantity_array_for_one_ROI(output_folder_ROIs, surf_mesh, vol_mesh, spec_quantity, spec_quantity_array, ROI_params)

        # Construct title
        spectrogram_title = f'{case_name}_specP_win{window_length}' 
        
        spectrogram_data = calculate_mean_spectrogram(
                            var_name = spec_quantity,
                            var_array = spec_quantity_array_ROI,
                            STFT_params = STFT_params)

        # Save full spectrogram data
        spec_output_npz = Path(output_folder_files) / f"{spectrogram_title}.npz"
        np.savez(spec_output_npz, spectrogram_data)
        
        # Cut spectrogram to analysis window
        spectrogram_data_filt = filter_raw_spectrogram(spectrogram_data, spectral_analysis_params)
        
        # Classify spectrogram phases
        Q_phases, spectral_metrics = classify_spectrogram_phases(spectrogram_data_filt, spectral_analysis_params)

        # Plot spectrograms and metrics
        plot_spectrogram_and_metrics(output_folder_imgs, case_name, spectrogram_data_filt, Q_phases, spectral_metrics, spectral_analysis_params, spectrogram_title)
    

    print (f'\nFinished computing and saving spectrograms.')
     

# ======================================================================================================
# MAIN
# ======================================================================================================

# ---------------------------------------- Run the script -----------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_folder",   required=True,       help="Results folder with CFD .h5 files")
    ap.add_argument("--mesh_folder",    required=True,       help="Case mesh folder containing HDF5 mesh file")
    ap.add_argument("--output_folder",  required=True,       help="Output directory for SPI VTP file")
    ap.add_argument("--case_name",      required=True,       help="Case name")
    ap.add_argument("--n_process",      type=int,            help="Number of parallel processes", default=max(1, mp.cpu_count() - 1))

    ap.add_argument("--density",           type=float,  default=1057,   help="Blood density [kg/m3] (default: 1057)")
    ap.add_argument("--period_seconds",    type=float,  default=0.915,  help="Period in seconds")
    ap.add_argument("--timesteps_per_cyc", type=int,                    help="Number of timesteps per cycle")
    ap.add_argument("--spec_quantity",     type=str,    required=True,  choices=["wallpressure","velocity","qcriterion"], help="Quantity of interest used for spectrogram")
    

    # ROI parameters: It allows for either a single center OR a CSV of center
    ROI_group = ap.add_mutually_exclusive_group(required=True)
    ROI_group.add_argument("--ROI_center_coord", nargs=3,  type=float, metavar=("X", "Y", "Z"), help="XYZ coordinates for a single ROI center (mesh units)")
    ROI_group.add_argument("--ROI_center_csv",   type=str, help="CSV file with multiple ROI points; coords columns = Points:0/1/2")
    
    ap.add_argument("--spec_regions_csv",    type=str,   default=None,      help="CSV file defining multiple anatomical regions to process for spectrograms. Required columns: region_name, ROI_start_center_id, ROI_end_center_id, ROI_stride, ROI_radius. Optional columns: ROI_height, flag_multi_ROI, flag_save_ROI.")
    ap.add_argument("--ROI_type",            type=str,   default="cylinder", choices=["point","sphere","cylinder"], help="Type of ROI shape")
    ap.add_argument("--ROI_radius",          type=float, default=None,       help="Radius of ROI in mesh units (mm in most cases). Required unless --spec_regions_csv is used.")
    ap.add_argument("--ROI_height",          type=float, default=2,          help="Height of cylindrical ROI in mesh units (mm in most cases)")
    ap.add_argument("--ROI_start_center_id", type=int,   default=1,          help="ROI center ID of the start of the region of inerest")
    ap.add_argument("--ROI_end_center_id",   type=int,   default=10,         help="ROI center ID of the end of the region of inerest")
    ap.add_argument("--ROI_stride",          type=int,   default=1,          help="Stride between ROIs to sweep the region of inerest")
    ap.add_argument("--flag_save_ROI",       action="store_true",            help="Flag to save ROI.vtp surface file")
    ap.add_argument("--flag_multi_ROI",      action="store_true",            help="Flag to compute spectrogram in a segment based on multiple ROIs")



    # Spectrogram specific parameters (including Short-time Fourier Transform control)
    ap.add_argument("--window_length",    type=int,   default=None,     help="Length of FFT window in samples (number of snapshots for each window)")
    ap.add_argument("--n_fft",            type=int,   default=None,     help="FFT length (bins)")
    ap.add_argument("--overlap_fraction", type=float, default=0.9,      help="Overlap fraction between consequent windows [0,1] (default: 0.9)")
    ap.add_argument("--window_type",      type=str,   default='hann',   choices=["hann","hamming","boxcar","blackman","bartlett"], help="Window type for STFT (default: hann)")
    ap.add_argument("--pad_mode",         type=str,   default='even',   choices=["cycle","constant","odd","even","none"], help="Padding strategy to reduce edge artifacts (default: even)")
    ap.add_argument("--detrend",          type=str,   default='linear', help="Detrend option for STFT: 'linear', 'constant', or False (default: linear)")


    # Spectral analysis and visualization parameters
    ap.add_argument("--cutoff_db",          type=float, default=0.0,      help="Minimum dB floor for visualization")
    ap.add_argument("--freq_low",           type=float, default=100,      help="Upper threshold for low-frequency band in Hz (default: 100 Hz)")
    ap.add_argument("--freq_mid",           type=float, default=1000,     help="Upper threshold for mid-frequency band in Hz (default: 1000 Hz)")
    ap.add_argument("--freq_max",           type=float, default=5000,     help="Maximum frequency to filter spectrogram in Hz (default: 5000 Hz)")
    ap.add_argument("--flowrate_min",       type=float, default=2.0,      help="Lower inlet flowrate limit for analysis window in mL/s (default: 2.0)")
    ap.add_argument("--flowrate_max",       type=float, default=10.0,      help="Upper inlet flowrate limit for analysis window in mL/s (default: 10.0)")
    ap.add_argument("--flowrate_cut",       type=float, default=8.0,      help="Upper inlet flowrate limit for figures in mL/s (default: 8.0)")
    ap.add_argument("--power_SPL_db_min",   type=float, default=20.0,     help="Lower SPL power limit for spectrogram colormap in dB (default: 20)")
    ap.add_argument("--power_SPL_db_max",   type=float, default=120.0,    help="Upper SPL power limit for spectrogram colormap in dB (default: 120)")

    return ap.parse_args()


def main():
    args          = parse_args()
    input_folder  = Path(args.input_folder)
    mesh_folder   = Path(args.mesh_folder)
    output_folder = Path(f'{args.output_folder}/Spectrogram_{args.spec_quantity}')
    
    # Create paths
    if not Path(output_folder).exists():
        Path(output_folder).mkdir(parents=True, exist_ok=True)

    output_folder_files = Path(f"{output_folder}/window{args.window_length}_overlap{args.overlap_fraction}_ROI{args.ROI_type}_multiROI{args.flag_multi_ROI}/files")
    output_folder_imgs  = Path(f"{output_folder}/window{args.window_length}_overlap{args.overlap_fraction}_ROI{args.ROI_type}_multiROI{args.flag_multi_ROI}/imgs")
    output_folder_ROIs  = Path(f"{output_folder}/window{args.window_length}_overlap{args.overlap_fraction}_ROI{args.ROI_type}_multiROI{args.flag_multi_ROI}/ROIs")
    
    output_folder_files.mkdir(parents=True, exist_ok=True)
    output_folder_imgs.mkdir(parents=True, exist_ok=True)
    if args.flag_save_ROI: output_folder_ROIs.mkdir(parents=True, exist_ok=True)

    # Put input arguments into dictionaries
    ROI_params = {
        "ROI_type": args.ROI_type,
        "ROI_center_coord": args.ROI_center_coord,
        "ROI_center_csv": args.ROI_center_csv,
        "ROI_radius": args.ROI_radius,
        "ROI_height": args.ROI_height,
        "ROI_start_center_id": args.ROI_start_center_id,
        "ROI_end_center_id": args.ROI_end_center_id,
        "ROI_stride": args.ROI_stride,
        "flag_save_ROI": args.flag_save_ROI,
        "flag_multi_ROI": args.flag_multi_ROI}

    short_time_fourier_params = {
        "window_length": args.window_length,
        "n_fft": args.n_fft,
        "overlap_frac": args.overlap_fraction,
        "window_type": args.window_type,
        "pad_mode": args.pad_mode,
        "detrend": args.detrend}
    
    spectral_analysis_params = {
        "cutoff_db":  args.cutoff_db,
        "freq_low":   args.freq_low,
        "freq_mid":   args.freq_mid,
        "freq_max":   args.freq_max,
        "Q_min":      args.flowrate_min,
        "Q_max":      args.flowrate_max,
        "Q_cut":      args.flowrate_cut,
        "SPL_db_min": args.power_SPL_db_min,
        "SPL_db_max": args.power_SPL_db_max}

    # Load mesh
    mesh_file = list(Path(mesh_folder).glob('*.h5'))[0]
    surf_mesh    = load_surface_mesh(mesh_file)
    vol_mesh, _  = load_volume_mesh(mesh_file)

    # Printing info to log
    print("=" * 200 + "\n")
    print("compute_Spectrogram.py")
    print(f"\n[info] Mesh file:                         {mesh_file}")
    print(f"[info] Read CFD results from:             {input_folder}")
    print(f"[info] Write spectrograms to:             {output_folder}")

    if args.spec_regions_csv is not None:
        print(f"[info] Read spectrogram regions from:  {args.spec_regions_csv}")

    if args.ROI_center_csv is not None:
        print(f"[info] Read ROI centers from:          {args.ROI_center_csv} \n")
    else:
        print(f"[info] Read ROI center from:           {args.ROI_center_coord} \n")
    
    print("=" * 200 + "\n")

    
    # Reading the input files for quantity used to generate spectrograms
    input_path = Path(input_folder)

    # Sanity check
    if not any(input_path.iterdir()):
        print(f'No files found in {input_folder}!')
        sys.exit()

    if args.spec_quantity in ['wallpressure', 'velocity']:
        # Find & sort CFD results h5 files by timestep 
        CFD_h5_files = sorted(input_path.glob('*_curcyc_*up.h5'), key = extract_timestep_from_h5_filename)

        # Assemble variable array
        if args.spec_quantity == 'wallpressure':
            spec_quantity_array = read_wallpressure_from_h5_files_parallel(CFD_h5_files, surf_mesh, args.n_process, args.density) 
        elif args.spec_quantity == 'velocity':
            raise ValueError(f'Not implemented yet for velocity spectrograms!')


    elif args.spec_quantity == 'qcriterion':
        Q_h5_file = input_path / f"{args.case_name}_Qcriterion.h5"

        print(f"Reading Qcriterion file ...")

        with h5py.File(Q_h5_file, 'r') as h5:
            spec_quantity_array = np.array(h5['Data']['Q']) 
        

            
    # Obtain simulation temporal parameters from filename (if not given as input argument)
    timesteps_per_cyc = args.timesteps_per_cyc
    period_seconds    = args.period_seconds

    if timesteps_per_cyc is None:
        timesteps_per_cyc = extract_sim_params_from_h5_filename(CFD_h5_files[0])
        print(f"Found timesteps_per_cycle = {timesteps_per_cyc} from CFD results HDF5 file names. \n")


    # If --spec_regions_csv is provided, load all regions from the CSV so H5 files are reused for every region without re-reading from disk.
    # Each CSV row overrides the matching ROI_params keys for that region.
    # Without --spec_regions_csv the list holds a single empty dict, so the CLI args are used for single-region usage.
    if args.spec_regions_csv is not None:
        spec_regions = read_spec_regions_from_csv(args.spec_regions_csv)
    else:
        spec_regions = [{}]   # single region, no overrides


    # Run post-processing of assembled CFD results
    print (f"Performing post-processing computation on {args.n_process} cores ... \n" )

    # Computing spectrograms
    for region_idx, region_params in enumerate(spec_regions):
        if len(spec_regions) > 1:
            print(f"\n---------------------- Region {region_idx + 1}/{len(spec_regions)} --------------------------------")
            print(f"{region_params['region_fullname']}: ROI {region_params['ROI_start_center_id']} to {region_params['ROI_end_center_id']} \n")

        # Override the CLI ROI params if present in the spec_regions_csv file
        region_ROI_params = dict(ROI_params)
        region_ROI_params.update(region_params)

        compute_and_save_spectrogram_for_all_ROIs(
                            case_name                = args.case_name,
                            output_folder_files      = output_folder_files,
                            output_folder_imgs       = output_folder_imgs,
                            output_folder_ROIs       = output_folder_ROIs,
                            surf_mesh                = surf_mesh,
                            vol_mesh                 = vol_mesh,
                            spec_quantity            = args.spec_quantity,
                            spec_quantity_array      = spec_quantity_array,
                            period_seconds           = period_seconds,
                            timesteps_per_cyc        = timesteps_per_cyc,
                            ROI_params               = region_ROI_params,
                            STFT_params              = short_time_fourier_params,
                            spectral_analysis_params = spectral_analysis_params)

if __name__ == '__main__':
    main()