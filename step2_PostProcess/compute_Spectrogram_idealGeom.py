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


def extract_sim_params_from_foldername(input_path: Path) -> tuple[int, int | None]:
    """Parse timesteps-per-cycle and save frequency from the results folder path.

    Expected patterns (anywhere in the full path string):
      '_ts<int>'        — timesteps per cycle  (e.g. 'run_ts500_...')
      'saveFreq(<int>)' — save frequency       (e.g. 'run_saveFreq(10)')

    Returns:
      timesteps_per_cyc : int
      save_freq         : int or None (None if pattern absent)
    """
    path_str = str(input_path)

    match_ts = re.search(r'_ts(\d+)', path_str)
    if match_ts is None:
        raise ValueError(
            f"Folder path '{input_path}' has no '_ts<int>' pattern. "
            "Supply --timesteps_per_cyc on the CLI instead."
        )
    timesteps_per_cyc = int(match_ts.group(1))

    match_sf = re.search(r'saveFreq\((\d+)\)', path_str)
    save_freq = int(match_sf.group(1)) if match_sf else None

    return timesteps_per_cyc, save_freq


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


def load_surface_mesh_from_xmlgz(xml_gz_path: str) -> pv.PolyData:
    """
    Load wall surface from a DOLFIN XML.gz mesh file without requiring FEniCS.

    Wall facets are entries in <domains>/<mesh_value_collection dim="2"> with value="0".
    DOLFIN local face rule: face i of tet (v0,v1,v2,v3) uses all vertices except v[i].

    Returns PyVista PolyData matching load_surface_mesh() output:
      - points                      : XYZ of wall nodes
      - point_data['vtkOriginalPtIds'] : global volume-mesh node IDs
    """
    import gzip
    import xml.etree.ElementTree as ET

    FACE_VERTS = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]

    print(f"[mesh] Parsing XML.gz: {xml_gz_path} ...")

    n_verts = 0
    n_tets  = 0
    all_coords: np.ndarray = None
    tet_verts:  np.ndarray = None
    wall_tris        = []
    wall_vertex_set  = set()
    in_domain_mvc    = False

    with gzip.open(xml_gz_path, 'rt') as f:
        for event, elem in ET.iterparse(f, events=('start', 'end')):
            tag = elem.tag

            if event == 'start':
                if tag == 'vertices':
                    n_verts    = int(elem.get('size'))
                    all_coords = np.zeros((n_verts, 3), dtype=np.float64)
                elif tag == 'cells':
                    n_tets    = int(elem.get('size'))
                    tet_verts = np.zeros((n_tets, 4), dtype=np.int64)
                elif tag == 'mesh_value_collection':
                    in_domain_mvc = (elem.get('dim') == '2')

            else:  # event == 'end'
                if tag == 'vertex':
                    idx = int(elem.get('index'))
                    all_coords[idx] = [float(elem.get('x')), float(elem.get('y')), float(elem.get('z'))]
                    elem.clear()
                elif tag == 'tetrahedron':
                    idx = int(elem.get('index'))
                    tet_verts[idx] = [int(elem.get('v0')), int(elem.get('v1')),
                                      int(elem.get('v2')), int(elem.get('v3'))]
                    elem.clear()
                elif tag == 'value' and in_domain_mvc:
                    if int(elem.get('value')) == 0:
                        ci   = int(elem.get('cell_index'))
                        le   = int(elem.get('local_entity'))
                        verts = tet_verts[ci][FACE_VERTS[le]]
                        wall_tris.append(verts.copy())
                        wall_vertex_set.update(verts.tolist())
                    elem.clear()
                elif tag == 'mesh_value_collection':
                    in_domain_mvc = False

    wall_point_ids = np.array(sorted(wall_vertex_set), dtype=np.int64)
    wall_coords    = all_coords[wall_point_ids]

    g2l           = {gid: lid for lid, gid in enumerate(wall_point_ids)}
    wall_topology = np.array([[g2l[v] for v in tri] for tri in wall_tris], dtype=np.int64)

    n_wall_cells = len(wall_topology)
    cells_vtk    = np.hstack([np.full((n_wall_cells, 1), 3, dtype=np.int64), wall_topology]).ravel()

    surf = pv.PolyData(wall_coords, cells_vtk)
    surf.point_data['vtkOriginalPtIds'] = wall_point_ids

    print(f"[mesh] Wall surface: {len(wall_point_ids)} nodes, {n_wall_cells} triangles")
    return surf


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

def read_spec_regions_from_csv_idealGeom(csv_path: str) -> list:
    """
    Read axial (x-range) region definitions for idealized geometries with a known, straight centerline
    (e.g. inlet: -3D to -1D / stenosis: -1D to 1D / post-stenosis: 1D to 6D / outlet: 6D to end).
    Required columns : x_start_D, x_end_D   (bounds in multiples of pipe diameter D)
    Optional columns : region_shortname, flag_save_ROI
    The strings "start"/"end" in x_start_D/x_end_D mean "extend to the mesh's actual min/max x".
    Returns a list of dicts, one per row.
    """
    float_keys = {"x_start_D", "x_end_D"}
    bool_keys  = {"flag_save_ROI"}
    str_keys   = {"region_shortname"}
    known_keys = float_keys | bool_keys | str_keys

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
            if key in float_keys:
                sval = str(val).strip().lower()
                region[key] = None if sval in ("start", "end", "none", "") else float(sval)
            elif key in bool_keys:
                region[key] = bool(int(val)) if str(val).strip().lstrip('-').isdigit() else str(val).strip().lower() in ("true", "yes")
            elif key in str_keys:
                region[key] = str(val).strip()
        regions.append(region)

    return regions



def estimate_pipe_diameter_from_mesh(surf_mesh: pv.PolyData, pipe_axis: int = 0) -> float:
    """
    Estimate the inner pipe diameter from the wall mesh bounding box.

    For a straight cylindrical pipe the two axes perpendicular to pipe_axis
    each span exactly one diameter; this returns their average extent.
    """
    perp_axes = [i for i in range(3) if i != pipe_axis]
    extents   = [surf_mesh.points[:, ax].ptp() for ax in perp_axes]
    diameter  = float(np.mean(extents))
    axis_label = {0: "X", 1: "Y", 2: "Z"}.get(pipe_axis, str(pipe_axis))
    print(f"[mesh] Estimated pipe diameter from mesh bounding box: {diameter:.4f}  (pipe_axis={axis_label})")
    return diameter


def extract_wall_points_perROI_idealGeom(surf_mesh: pv.PolyData, x_lo: float, x_hi: float, pipe_axis: int = 0) -> np.ndarray:
    """
    Return indices of wall surface points whose coordinate along `axis` falls in [x_lo, x_hi].
    axis: 0=X, 1=Y, 2=Z — the axis along which the pipe centerline runs.
    """
    coords = surf_mesh.points[:, pipe_axis]
    mask = (coords >= x_lo) & (coords <= x_hi)
    pids = np.where(mask)[0]
    if pids.size == 0:
        axis_label = {0: "X", 1: "Y", 2: "Z"}.get(pipe_axis, str(pipe_axis))
        raise ValueError(
            f"No wall points found in {axis_label}-range [{x_lo:.4f}, {x_hi:.4f}]. "
            "Check pipe_diameter, pipe_axis, and region bounds."
        )
    return pids



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
    cutoff_db   = spectral_analysis_params.get("cutoff_db")
    freq_max    = spectral_analysis_params.get("freq_max")
    Q_min       = spectral_analysis_params.get("Q_min")
    Q_max       = spectral_analysis_params.get("Q_max")
    ramp_slope  = spectral_analysis_params.get("ramp_slope")
    ramp_offset = spectral_analysis_params.get("ramp_offset")

    freqs        = spectrogram_data['freqs']
    bins         = spectrogram_data['bins']
    power_avg_dB = spectrogram_data['power_avg_dB']

    # Build masks
    bins_Q    = ramp_slope * bins + ramp_offset    # Q_inlet = ramp_slope * t + ramp_offset
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
    f_low       = spectral_analysis_params.get("freq_low")
    f_mid       = spectral_analysis_params.get("freq_mid")
    f_max       = spectral_analysis_params.get("freq_max")
    ramp_slope  = spectral_analysis_params.get("ramp_slope")
    ramp_offset = spectral_analysis_params.get("ramp_offset")

    bins    = spectrogram_data['bins']
    freqs   = spectrogram_data['freqs']
    spec_dB = spectrogram_data['power_avg_dB']

    bins_Q = ramp_slope * bins + ramp_offset
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


def plot_spectrogram_and_metrics(output_folder_imgs, case_name, spectrogram_data, Q_phases, spectral_metrics, analysis_params, plot_title, flag_plot_phases=False):
    """
    Plot and save spectrograms and spectral metrics as PNG files.
    """

    # Extract relevant data for plotting
    bins  = spectrogram_data['bins']
    freqs = spectrogram_data['freqs']
    spectrogram_signal = spectrogram_data['power_avg_dB']

    bins_Q = bins #analysis_params.get("ramp_slope") * bins + analysis_params.get("ramp_offset")

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
    ax[0].set_ylim([0, 2000]) #analysis_params['freq_max']])

    # Adding the colorbar
    cbar = fig.colorbar(spectrogram, ax=ax[0], orientation='vertical') #pad=0.5
    cbar.set_label('SPL (dB)', rotation=270, labelpad=15, size=16, fontweight='bold')


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
    # for a in ax:
    #     a.set_xlim([analysis_params['Q_min'], analysis_params['Q_cut']])
    #     a.tick_params(direction='in')
    #     a.set_xlabel('Flow rate (mL/s)', fontweight='bold', labelpad=10)
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


def compute_and_save_spectrogram_perROI_for_idealGeom(
        case_name: str,
        output_folder_files: Path,
        output_folder_imgs: Path,
        output_folder_ROIs: Path,
        surf_mesh: pv.PolyData,
        wall_pressure: np.ndarray,
        spec_region: dict,
        pipe_diameter: float,
        pipe_axis: int,
        period_seconds: float,
        timesteps_per_cyc: int,
        save_freq: int,
        STFT_params: dict,
        spectral_analysis_params: dict):
    """
    Compute and save the spectrogram for one axial region of an idealized geometry.

    region keys: x_start_D, x_end_D (multiples of D; None → mesh min/max), region_shortname, flag_save_ROI.
    Wall points are selected by coordinate along pipe_axis (0=X, 1=Y, 2=Z).
    """

    STFT_params = dict(STFT_params)  # avoid mutating caller's dict
    STFT_params["sampling_rate"] = timesteps_per_cyc / period_seconds / save_freq
    window_length = STFT_params.get("window_length")

    axis_label  = {0: "X", 1: "Y", 2: "Z"}.get(pipe_axis, str(pipe_axis))
    wall_coords = surf_mesh.points[:, pipe_axis]
    x_mesh_min  = wall_coords.min()
    x_mesh_max  = wall_coords.max()

    x_start_D = spec_region.get("x_start_D")
    x_end_D   = spec_region.get("x_end_D")
    shortname = spec_region.get("region_shortname", "region")
    save_roi  = spec_region.get("flag_save_ROI", False)

    x1_region = x_mesh_min if x_start_D is None else x_start_D * pipe_diameter
    x2_region = x_mesh_max if x_end_D   is None else x_end_D   * pipe_diameter

    print(f"    {axis_label}-range: [{x1_region:.4f}, {x2_region:.4f}]  (x_start={x_start_D}D, x_end={x_end_D}D)")

    pids = extract_wall_points_perROI_idealGeom(surf_mesh, x1_region, x2_region, pipe_axis=pipe_axis)
    print(f"    Found {pids.size} wall points.")


    wall_pressure_region   = wall_pressure[pids, :]
    spectrogram_title = f"{case_name}_win{window_length}_region{shortname}"

    spectrogram_data = calculate_mean_spectrogram(
        var_name    = "wallpressure",
        var_array   = wall_pressure_region,
        STFT_params = STFT_params)

    np.savez(output_folder_files / f"{spectrogram_title}.npz", spectrogram_data)

    spectrogram_data_filt = filter_raw_spectrogram(spectrogram_data, spectral_analysis_params)
    Q_phases, spectral_metrics = classify_spectrogram_phases(spectrogram_data_filt, spectral_analysis_params)
    plot_spectrogram_and_metrics(output_folder_imgs, case_name,
                                spectrogram_data_filt, Q_phases, spectral_metrics,
                                spectral_analysis_params, spectrogram_title)


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
    ap.add_argument("--period_seconds",    type=float,  default=1,  help="Period in seconds")
    ap.add_argument("--timesteps_per_cyc", type=int,   default=None,    help="Number of timesteps per cycle (parsed from filename '_ts<int>' if omitted)")
    ap.add_argument("--save_freq",         type=int,   default=None,    help="Snapshot save frequency: every Nth timestep saved (parsed from filename 'saveFreq(<int>)' if omitted)")
    ap.add_argument("--spec_quantity",     type=str,    required=True,  choices=["wallpressure","velocity","qcriterion"], help="Quantity of interest used for spectrogram")
    ap.add_argument("--spec_regions_csv",  type=str,    default=None,   help="CSV defining anatomical regions. idealized: columns x_start_D, x_end_D, region_shortname, flag_save_ROI. patient_specific: columns ROI_start_center_id, ROI_end_center_id, ROI_stride, ROI_radius.")   

    # Geometry type — controls which region-definition method is used
    ap.add_argument("--geometry_type", type=str, default="patient_specific", choices=["idealized", "patient_specific"],
                    help="'idealized': straight pipe, regions defined by x-range in units of D; "
                         "'patient_specific': real geometry, regions defined by centerline ROI CSV.")

    # Idealized-geometry parameters (used when --geometry_type idealized)
    ap.add_argument("--pipe_diameter",      type=float, default=None, help="Pipe inner diameter [mesh units]. If omitted with --geometry_type idealized, estimated from the mesh bounding box.")
    ap.add_argument("--pipe_axis",          type=int,   default=0,   choices=[0, 1, 2], help="Axis along which the pipe centerline runs: 0=X, 1=Y, 2=Z (default: 0)")

    # Patient-specific ROI parameters: either a single center OR a CSV of centers
    ROI_group = ap.add_mutually_exclusive_group(required=False)
    ROI_group.add_argument("--ROI_center_coord", nargs=3,  type=float, metavar=("X", "Y", "Z"), help="XYZ coordinates for a single ROI center (mesh units)")
    ROI_group.add_argument("--ROI_center_csv",   type=str,   help="CSV file with multiple ROI points; coords columns = Points:0/1/2")
    
    ap.add_argument("--ROI_type",                type=str,   default="cylinder", choices=["point","sphere","cylinder"], help="Type of ROI shape")
    ap.add_argument("--ROI_radius",              type=float, default=None,       help="Radius of ROI in mesh units (mm in most cases). Required unless --spec_regions_csv is used.")
    ap.add_argument("--ROI_height",              type=float, default=2,          help="Height of cylindrical ROI in mesh units (mm in most cases)")
    ap.add_argument("--ROI_start_center_id",     type=int,   default=1,          help="ROI center ID of the start of the region of inerest")
    ap.add_argument("--ROI_end_center_id",       type=int,   default=10,         help="ROI center ID of the end of the region of inerest")
    ap.add_argument("--ROI_stride",              type=int,   default=1,          help="Stride between ROIs to sweep the region of inerest")
    ap.add_argument("--flag_save_ROI",           action="store_true",            help="Flag to save ROI.vtp surface file")
    ap.add_argument("--flag_multi_ROI",          action="store_true",            help="Flag to compute spectrogram in a segment based on multiple ROIs")



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
    ap.add_argument("--ramp_slope",         type=float, default=2.0,      help="Slope of the flow-rate ramp [mL/s]: Q = ramp_slope * t + ramp_offset (default: 2.0)")
    ap.add_argument("--ramp_offset",        type=float, default=2.0,      help="Offset of the flow-rate ramp [mL/s2]: Q = ramp_slope * t + ramp_offset (default: 2.0)")

    return ap.parse_args()


def main():
    args          = parse_args()

    # Validate geometry-type-specific required arguments
    if args.geometry_type == "idealized":
        if args.spec_regions_csv is None:
            raise ValueError("--spec_regions_csv is required when --geometry_type idealized.")
    elif args.geometry_type == "patient_specific":
        if args.ROI_center_coord is None and args.ROI_center_csv is None:
            raise ValueError("--ROI_center_coord or --ROI_center_csv is required when --geometry_type patient_specific.")

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
        "cutoff_db":   args.cutoff_db,
        "freq_low":    args.freq_low,
        "freq_mid":    args.freq_mid,
        "freq_max":    args.freq_max,
        "Q_min":       args.flowrate_min,
        "Q_max":       args.flowrate_max,
        "Q_cut":       args.flowrate_cut,
        "SPL_db_min":  args.power_SPL_db_min,
        "SPL_db_max":  args.power_SPL_db_max,
        "ramp_slope":  args.ramp_slope,
        "ramp_offset": args.ramp_offset}

    # Load mesh
    if args.geometry_type == "idealized":
        h5_files     = list(Path(mesh_folder).glob('*.h5'))
        xml_gz_files = list(Path(mesh_folder).glob('*.xml.gz'))
        if h5_files:
            mesh_file = h5_files[0]
            surf_mesh = load_surface_mesh(mesh_file)
        elif xml_gz_files:
            mesh_file = xml_gz_files[0]
            surf_mesh = load_surface_mesh_from_xmlgz(str(mesh_file))
        else:
            raise FileNotFoundError(f"No .h5 or .xml.gz mesh file found in {mesh_folder}")
        vol_mesh = None
        if args.pipe_diameter is None:
            print('Pipe diameter not defined by user ...')
            args.pipe_diameter = estimate_pipe_diameter_from_mesh(surf_mesh, args.pipe_axis)
    else:
        mesh_file    = list(Path(mesh_folder).glob('*.h5'))[0]
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

    if args.geometry_type == "idealized":
        print(f"[info] spec_regions_csv:               {args.spec_regions_csv}")
        print(f"[info] pipe_diameter:                  {args.pipe_diameter}")
        print(f"[info] pipe_axis:                      {args.pipe_axis} \n")
    else:
        if args.ROI_center_csv is not None:
            print(f"[info] Read ROI centers from:      {args.ROI_center_csv} \n")
        else:
            print(f"[info] Read ROI center from:       {args.ROI_center_coord} \n")
    
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
    save_freq         = args.save_freq
    period_seconds    = args.period_seconds

    if timesteps_per_cyc is None or save_freq is None:
        ts_parsed, sf_parsed = extract_sim_params_from_foldername(input_path)
        if timesteps_per_cyc is None:
            timesteps_per_cyc = ts_parsed
            print(f"[info] timesteps_per_cycle = {timesteps_per_cyc}  (parsed from folder name)")
        if save_freq is None:
            if sf_parsed is None:
                raise ValueError("Could not find 'saveFreq(<int>)' in folder path and --save_freq was not supplied.")
            save_freq = sf_parsed
            print(f"[info] save_freq           = {save_freq}  (parsed from folder name)")


    # Run post-processing of assembled CFD results
    print (f"Performing post-processing computation on {args.n_process} cores ... \n" )

    # ---- Idealized geometry: x-range region slicing ----
    if args.geometry_type == "idealized":
        spec_regions = read_spec_regions_from_csv_idealGeom(args.spec_regions_csv)

        axis_label = {0: "X", 1: "Y", 2: "Z"}.get(args.pipe_axis, str(args.pipe_axis))
        print(f"pipe_diameter D = {args.pipe_diameter}  |  pipe_axis = {axis_label}\n")

        for region_idx, region in enumerate(spec_regions):
            print(f"\n--- Region {region_idx + 1}/{len(spec_regions)}: '{region.get('region_shortname', f'region{region_idx}')}' ---")
            compute_and_save_spectrogram_perROI_for_idealGeom(
                case_name                = args.case_name,
                output_folder_files      = output_folder_files,
                output_folder_imgs       = output_folder_imgs,
                output_folder_ROIs       = output_folder_ROIs,
                surf_mesh                = surf_mesh,
                wall_pressure            = spec_quantity_array,
                spec_region              = region,
                pipe_diameter            = args.pipe_diameter,
                pipe_axis                = args.pipe_axis,
                period_seconds           = period_seconds,
                timesteps_per_cyc        = timesteps_per_cyc,
                save_freq                = save_freq,
                STFT_params              = short_time_fourier_params,
                spectral_analysis_params = spectral_analysis_params)

        print(f"\nFinished computing spectrograms for all idealized regions.")

    # ---- Patient-specific geometry: centerline ROI sweeping ----
    else:
        if args.spec_regions_csv is not None:
            spec_regions = read_spec_regions_from_csv_patientGeom(args.spec_regions_csv)
        else:
            spec_regions = [{}]   # single region; CLI ROI args used directly

        for region_idx, region_params in enumerate(spec_regions):
            if len(spec_regions) > 1:
                print(f"\n---------------------- Region {region_idx + 1}/{len(spec_regions)} --------------------------------")
                print(f"{region_params['region_fullname']}: ROI {region_params['ROI_start_center_id']} to {region_params['ROI_end_center_id']} \n")

            # Override the CLI ROI params if present in the spec_regions_csv file
            region_ROI_params = dict(ROI_params)
            region_ROI_params.update(region_params)

            compute_and_save_spectrogram_perROI_for_patientGeom(
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