# -----------------------------------------------------------------------------------------------------------------------
# compute_NS_terms_centerline.py
# Compute Navier-Stokes equation term budget along the pipe centerline from CFD results.
#
# __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2026-09
#
# PURPOSE:
#   - Extract velocity and pressure at N equally-spaced points on the pipe centerline.
#   - Compute all axial-momentum NS terms via finite differences.
#   - Plot spatial budgets vs x/D (at selected Re) and Re-evolution (at selected x/D).
#   - Save 2-D maps of each term over the (x/D, Re) plane.
#
# NS EQUATION — axial component (pipe-axis direction):
#
#   ρ ∂u/∂t  +  ρ u ∂u/∂x  =  -∂p/∂x  +  μ ∂²u/∂x²
#   [temporal]  [convective]  [pres_grad]  [viscous]
#
#   Residual = temporal + convective + pressure_gradient + viscous  (should be ~0)
#
# NOTES:
#   - Field values are extracted via nearest-mesh-node lookup (brute-force Euclidean search on the volume mesh).
#   - Spatial derivatives: uniform 2nd-order central finite differences along the pipe axis
#     (centerline points from np.linspace → equal spacing dx).
#   - Temporal derivative: 2nd-order central finite difference in time.
#   - The viscous term is a 1-D approximation (axial curvature only; transverse Laplacian
#     contributions from the cross-sectional profile are not captured from centerline data alone).
#   - Oasis stores p/ρ → multiply by density to obtain pressure in Pa.
#   - Velocity units: mm/ms == m/s (same numerical value, no conversion needed).
#
# REQUIREMENTS:
#   - h5py, numpy, matplotlib
#   - On Trillium: virtual environment "pyvista36"
#
# EXECUTION:
#   - Run via compute_NS_terms_centerline_job.sh, or directly on a login/debug node:
#       > module load StdEnv/2023 gcc/12.3 python/3.12.4
#       > source $HOME/virtual_envs/pyvista36/bin/activate
#
# EXAMPLE CLI:
#   python compute_NS_terms_centerline.py \
#       --input_folder  <path_to_CFD_results>          \
#       --mesh_folder   <path_to_data_dir_with_xmlgz>  \
#       --output_folder <path_to_output>               \
#       --case_name     eccStenosis                    \
#       --n_centerline_points 150
#
# INPUTS:
#   --input_folder          Path to CFD results folder with HDF5 snapshots
#   --mesh_folder           Path to folder containing mesh file (.xml.gz preferred, or .h5)
#   --output_folder         Output directory
#   --case_name             Case name (used in output filenames)
#   --n_centerline_points   Number of equally-spaced sample points on centerline (default: 100)
#   --pipe_axis             Axis along which the pipe runs: 0=X, 1=Y, 2=Z (default: 0)
#   --pipe_diameter         Pipe inner diameter [mm] (estimated from mesh if omitted)
#   --centerline_coord1     Coordinate on 1st perpendicular axis [mm] (default: 0.0)
#   --centerline_coord2     Coordinate on 2nd perpendicular axis [mm] (default: 0.0)
#   --x_start_mm            Start of centerline region [mm] (default: mesh x_min)
#   --x_end_mm              End of centerline region [mm]   (default: mesh x_max)
#   --x_ref_mm              Reference x for x/D = 0 (default: 0.0 = mesh origin / stenosis throat)
#   --density               Blood density [kg/m³] (default: 1057)
#   --dynamic_viscosity     Dynamic viscosity [Pa·s] (default: 0.0037)
#   --period_ms             Period [ms] — used to compute dt; inferred from last filename if omitted
#   --timesteps_per_cyc     Timesteps per cycle (parsed from folder name if omitted)
#   --save_freq             Snapshot save frequency (parsed from folder name if omitted)
#   --ramp_slope            Ramp slope for Re axis [mL/s per second] (default: 2.0)
#   --ramp_offset           Ramp offset at t=0 [mL/s] (default: 2.0)
#   --Re_plot_targets       Re values for spatial budget plots (default: 3 auto-selected values)
#   --xD_plot_targets       x/D values for Re-evolution plots (default: -1 0 1 3)
#   --Re_min_plot           Minimum Re for output plots (default: no limit)
#   --Re_max_plot           Maximum Re for output plots (default: no limit)
#   --n_process             Number of parallel reader processes (default: #CPUs - 1)
#
# OUTPUTS:
#   output_folder/NS_terms_centerline/
#     data/  <case_name>_NS_terms_centerline.npz   — raw term arrays + x_D + Re_axis
#     imgs/  <case_name>_NSterms_vs_x.png          — spatial profiles at selected Re values
#            <case_name>_NSterms_vs_Re.png          — Re-evolution at selected x/D locations
#            <case_name>_NSterms_2Dmap.png          — 2-D grid of all 4 terms
#            <case_name>_NSterm_<name>.png          — individual 2-D map per term
#
# Copyright (C) 2026 University of Toronto, Biomedical Simulation Lab.
# -----------------------------------------------------------------------------------------------------------------------

import sys
import re
import gzip
import gc
import warnings
import argparse
import xml.etree.ElementTree as ET
import multiprocessing as mp
from multiprocessing import sharedctypes
from pathlib import Path

import h5py
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=DeprecationWarning)


# ======================================================================================================
# SHARED MEMORY UTILITIES
# ======================================================================================================

def create_shared_array(size, dtype=np.float64):
    """Allocate a ctypes-backed shared memory array of zeros."""
    ctype_array = np.ctypeslib.as_ctypes(np.zeros(size, dtype=dtype))
    return sharedctypes.Array(ctype_array._type_, ctype_array, lock=False)

def view_shared_array(shared_obj):
    """Return a NumPy view (no copy) over a shared ctypes array."""
    return np.ctypeslib.as_array(shared_obj)


# ======================================================================================================
# GENERAL UTILITIES
# ======================================================================================================

def extract_timestep_from_h5_filename(h5_file: Path) -> int:
    """Extract integer timestep from filename pattern '*_ts=<int>_...'."""
    match = re.search(r'_ts=(\d+)', h5_file.stem)
    if match is None:
        raise ValueError(f"'{h5_file.name}' has no '_ts=<int>' pattern.")
    return int(match.group(1))


def extract_sim_time_ms_from_h5_filename(h5_file: Path) -> float:
    """Extract simulation time [ms] from filename pattern '*_t=<float>_...'."""
    match = re.search(r'_t=(\d+\.\d+)', h5_file.stem)
    if match is None:
        return None
    return float(match.group(1))


def extract_sim_params_from_foldername(input_path: Path):
    """Parse timesteps_per_cyc and save_freq from folder path string."""
    path_str = str(input_path)
    m_ts = re.search(r'_ts(\d+)', path_str)
    if m_ts is None:
        raise ValueError(f"No '_ts<int>' pattern in path '{input_path}'.")
    m_sf = re.search(r'_saveFreq(\d+)', path_str)
    return int(m_ts.group(1)), (int(m_sf.group(1)) if m_sf else None)


def flowrate_mLs_to_reynolds(Q_mLs, density, dynamic_viscosity, pipe_diameter_mm):
    """Re = 4ρQ / (πDμ).  Q in mL/s, D in mm, returns dimensionless Re."""
    Q_m3s = np.asarray(Q_mLs, dtype=float) * 1e-6
    return 4.0 * density * Q_m3s / (np.pi * (pipe_diameter_mm * 1e-3) * dynamic_viscosity)


# ======================================================================================================
# MESH LOADING
# ======================================================================================================

def load_surface_mesh(mesh_file: Path) -> pv.PolyData:
    """Load wall surface from a BSLSolver-style HDF5 mesh file."""
    with h5py.File(mesh_file, 'r') as h5:
        coords    = np.array(h5['Mesh/Wall/coordinates'])
        cells     = np.array(h5['Mesh/Wall/topology'])
        point_ids = np.array(h5['Mesh/Wall/pointIds'])

    n_cells   = cells.shape[0]
    cells_vtk = np.hstack([np.full((n_cells, 1), 3, dtype=np.int64), cells]).ravel()
    surf      = pv.PolyData(coords, cells_vtk)
    surf.point_data['vtkOriginalPtIds'] = point_ids
    return surf


def load_surface_mesh_from_xmlgz(xml_gz_path: str) -> pv.PolyData:
    """
    Load all volume nodes from a DOLFIN XML.gz mesh file.
    Returns a pv.PolyData point cloud of every mesh vertex.
    vtkOriginalPtIds is the identity mapping (local index == volume-mesh index).
    Note: unlike the wall-only version in compute_SpatialModes/Spectrograms,
    all nodes are kept here so centerline (r=0) nodes can be found.
    """
    print(f"[mesh] Parsing XML.gz: {xml_gz_path} ...")
    all_coords = None

    with gzip.open(xml_gz_path, 'rt') as f:
        for event, elem in ET.iterparse(f, events=('start', 'end')):
            if event == 'start' and elem.tag == 'vertices':
                n_verts    = int(elem.get('size'))
                all_coords = np.zeros((n_verts, 3), dtype=np.float64)
            if event == 'end' and elem.tag == 'vertex':
                idx = int(elem.get('index'))
                all_coords[idx] = [float(elem.get('x')),
                                   float(elem.get('y')),
                                   float(elem.get('z'))]
                elem.clear()

    surf = pv.PolyData(all_coords)
    surf.point_data['vtkOriginalPtIds'] = np.arange(n_verts, dtype=np.int64)
    print(f"[mesh] Loaded {n_verts} volume nodes from XML.gz.")
    return surf


def estimate_pipe_diameter(surf_mesh: pv.PolyData, pipe_axis: int = 0) -> float:
    """
    Estimate pipe inner diameter from the mesh bounding box.
    Averages the extents of the two axes perpendicular to pipe_axis.
    """
    perp_axes = [i for i in range(3) if i != pipe_axis]
    extents   = [surf_mesh.points[:, ax].ptp() for ax in perp_axes]
    diameter  = float(np.mean(extents))
    print(f"[mesh] Pipe diameter estimated from bounding box: {diameter:.5f}")
    return diameter

# ======================================================================================================
# CENTERLINE DEFINITION
# ======================================================================================================

def define_centerline_points(surf_mesh:   pv.PolyData,
                               n_points:    int,
                               pipe_axis:   int,
                               center1:     float,
                               center2:     float,
                               x_start_mm:  float = None,
                               x_end_mm:    float = None):
    """
    Create N equally-spaced sample points along the pipe axis.

    Arguments:
      surf_mesh   : pv.PolyData mesh (all volume nodes from xmlgz, or wall surface from h5)
      n_points    : number of sample points
      pipe_axis   : axis index along which the pipe runs (0=X, 1=Y, 2=Z)
      center1/2   : perpendicular-axis coordinates of the centerline [mm]
      x_start_mm  : start of sampling region [mm] (default: mesh minimum)
      x_end_mm    : end   of sampling region [mm] (default: mesh maximum)

    Returns:
      cl_points   : (n_points, 3) [mm]
      x_axial_mm  : (n_points,)  [mm]  — axial coordinate of each sample point
    """
    perp = [i for i in range(3) if i != pipe_axis]

    x_all = surf_mesh.points[:, pipe_axis]
    x0 = x_all.min() if x_start_mm is None else x_start_mm
    x1 = x_all.max() if x_end_mm   is None else x_end_mm

    x_vals = np.linspace(x0, x1, n_points)

    cl_points                = np.zeros((n_points, 3))
    cl_points[:, pipe_axis]  = x_vals
    cl_points[:, perp[0]]    = center1
    cl_points[:, perp[1]]    = center2

    ax_label = {0: 'X', 1: 'Y', 2: 'Z'}.get(pipe_axis, str(pipe_axis))
    print(f"[centerline] {n_points} points along {ax_label}: [{x0:.3f}, {x1:.3f}] mm  "
          f"(perp: {perp[0]}={center1:.3f}, {perp[1]}={center2:.3f} mm)")
    return cl_points, x_vals


def find_nearest_mesh_nodes(surf_mesh: pv.PolyData,
                             cl_points: np.ndarray):
    """
    For each centerline sample point, find the nearest mesh node (brute-force
    Euclidean search — same approach as sample_circumferential_nodes in
    compute_SpatialModes_wallpressure.py).

    Returns:
      node_indices : (n_cl,) int   — local PolyData point indices
      vol_node_ids : (n_cl,) int   — global volume-mesh indices (via vtkOriginalPtIds)
      distances    : (n_cl,) float — Euclidean distance to nearest node [mm]
    """
    raw_vol_ids = surf_mesh.point_data['vtkOriginalPtIds']
    n_cl        = len(cl_points)
    node_indices = np.empty(n_cl, dtype=np.int64)
    distances    = np.empty(n_cl, dtype=np.float64)

    for i, pt in enumerate(cl_points):
        d                = np.linalg.norm(surf_mesh.points - pt, axis=1)
        node_indices[i]  = int(np.argmin(d))
        distances[i]     = d[node_indices[i]]

    vol_node_ids = raw_vol_ids[node_indices]

    n_unique = np.unique(vol_node_ids).size
    if n_unique < n_cl:
        warnings.warn(
            f"{n_cl - n_unique} of {n_cl} centerline points mapped to the same mesh node. "
            "Consider reducing --n_centerline_points.",
            UserWarning, stacklevel=2,
        )
    print(f"[centerline] Nearest-node search: {n_cl} points → {n_unique} unique mesh nodes")
    return node_indices, vol_node_ids, distances


# ======================================================================================================
# PARALLEL CFD DATA READER
# ======================================================================================================

def _read_fields_worker(file_ids, sorted_node_ids, unsort_order,
                         h5_files, shared_u_ctype, shared_p_ctype,
                         n_nodes, n_times, density):
    """
    Worker: extract u and p at the requested nodes from a chunk of HDF5 snapshots
    and write into shared memory arrays.

    shared_u is interpreted as (n_nodes, 3, n_times).
    shared_p is interpreted as (n_nodes, n_times).

    sorted_node_ids : node indices sorted in ascending order (required by h5py fancy indexing)
    unsort_order    : argsort(argsort(original_node_ids)) to restore original order
    """
    shared_u = view_shared_array(shared_u_ctype).reshape(n_nodes, 3, n_times)
    shared_p = view_shared_array(shared_p_ctype).reshape(n_nodes, n_times)

    for t_idx in file_ids:
        with h5py.File(h5_files[t_idx], 'r') as h5:
            # Fancy indexing with sorted indices — reads only the requested rows from disk
            u_sorted = np.array(h5['Solution']['u'][sorted_node_ids, :])  # (n_nodes, 3) [m/s]
            p_sorted = np.array(h5['Solution']['p'][sorted_node_ids, 0]) * density  # (n_nodes,) [Pa]

        # Restore order matching the original cl_points ordering
        shared_u[:, :, t_idx] = u_sorted[unsort_order, :]
        shared_p[:, t_idx]    = p_sorted[unsort_order]


def read_centerline_fields_parallel(h5_files, node_ids, n_process, density):
    """
    Read velocity and pressure at centerline nodes from all HDF5 snapshots in parallel.

    Returns:
      u_cl : (n_nodes, 3, n_times)  velocity [m/s]    (mm/ms == m/s)
      p_cl : (n_nodes, n_times)     pressure [Pa]
    """
    n_nodes = len(node_ids)
    n_times = len(h5_files)

    # Pre-sort node_ids — h5py fancy indexing requires non-decreasing order
    sort_order   = np.argsort(node_ids)
    sorted_ids   = node_ids[sort_order]
    unsort_order = np.argsort(sort_order)  # inverse permutation

    shared_u = create_shared_array(n_nodes * 3 * n_times)
    shared_p = create_shared_array(n_nodes * n_times)

    mem_u_MB = n_nodes * 3 * n_times * 8 / 1e6
    mem_p_MB = n_nodes * n_times * 8 / 1e6
    print(f"\n[read] {n_nodes} centerline nodes × {n_times} snapshots on {n_process} workers "
          f"(u: {mem_u_MB:.1f} MB, p: {mem_p_MB:.1f} MB)")

    chunk  = max(n_times // n_process, 1)
    groups = [list(range(n_times))[i:i + chunk] for i in range(0, n_times, chunk)]

    procs = [
        mp.Process(
            target=_read_fields_worker,
            name=f"R{i}",
            args=(g, sorted_ids, unsort_order,
                  h5_files, shared_u, shared_p,
                  n_nodes, n_times, density))
        for i, g in enumerate(groups)
    ]
    for p in procs: p.start()
    for p in procs: p.join()

    u_cl = view_shared_array(shared_u).reshape(n_nodes, 3, n_times).copy()
    p_cl = view_shared_array(shared_p).reshape(n_nodes, n_times).copy()
    return u_cl, p_cl


# ======================================================================================================
# NAVIER-STOKES TERM COMPUTATION
# ======================================================================================================

def compute_ns_terms(u_cl, p_cl, x_axial_mm, pipe_axis, dt_s, density, dynamic_viscosity):
    """
    Compute the axial NS momentum terms at all centerline points and all timesteps.

    Axial NS equation:
        ρ ∂u/∂t  +  ρ u ∂u/∂x  =  −∂p/∂x  +  μ ∂²u/∂x²

    Spatial derivatives: uniform 2nd-order central FD (points from np.linspace → equal dx).
    Temporal derivative: 2nd-order central FD in time (interior timesteps only).
    Boundary points/timesteps remain NaN.

    Arguments:
      u_cl            : (n_pts, 3, n_times)  velocity [m/s]
      p_cl            : (n_pts, n_times)     pressure [Pa]
      x_axial_mm      : (n_pts,)             axial coordinate [mm]
      pipe_axis       : int (0/1/2)
      dt_s            : time between consecutive saved snapshots [s]
      density         : [kg/m³]
      dynamic_viscosity: [Pa·s]

    Returns dict with (n_pts, n_times) arrays, all in [Pa/m]:
        'temporal', 'convective', 'pressure_gradient', 'viscous', 'residual'
    """
    n_pts, _, n_times = u_cl.shape

    x_m  = x_axial_mm * 1e-3           # convert mm → m
    u_ax = u_cl[:, pipe_axis, :]       # axial velocity (n_pts, n_times) [m/s]

    temporal          = np.full((n_pts, n_times), np.nan)
    convective        = np.full((n_pts, n_times), np.nan)
    pressure_gradient = np.full((n_pts, n_times), np.nan)
    viscous           = np.full((n_pts, n_times), np.nan)

    # ---- Spatial derivatives — uniform central FD, loop over interior points ----
    # Centerline points come from np.linspace so spacing is constant.
    # First derivative  : (f[i+1] - f[i-1]) / (2 dx)
    # Second derivative : (f[i+1] - 2 f[i] + f[i-1]) / dx²
    dx = float(np.mean(np.diff(x_m)))   # [m] — constant spacing

    for i in range(1, n_pts - 1):
        du_dx_i   = (u_ax[i + 1, :] - u_ax[i - 1, :]) / (2.0 * dx)                       # [1/s]
        d2u_dx2_i = (u_ax[i + 1, :] - 2.0 * u_ax[i, :] + u_ax[i - 1, :]) / dx ** 2      # [1/(m·s)]
        dp_dx_i   = (p_cl[i + 1, :] - p_cl[i - 1, :]) / (2.0 * dx)                       # [Pa/m]

        convective[i, :]        = density * u_ax[i, :] * du_dx_i       # [Pa/m]
        pressure_gradient[i, :] = -dp_dx_i                              # [Pa/m]
        viscous[i, :]           = dynamic_viscosity * d2u_dx2_i         # [Pa/m]

    # ---- Temporal derivative: central FD in time----
    # ρ ∂u/∂t ≈ ρ (u[t+1] - u[t-1]) / (2 dt)
    for t in range(1, n_times - 1):
        du_dt_t        = (u_ax[:, t + 1] - u_ax[:, t - 1]) / (2.0 * dt_s)  # [m/s²]
        temporal[:, t] = density * du_dt_t                                    # [Pa/m]

    # ---- Residual ----
    residual = temporal + convective + pressure_gradient + viscous

    return dict(
        temporal          = temporal,
        convective        = convective,
        pressure_gradient = pressure_gradient,
        viscous           = viscous,
        residual          = residual,
    )


def build_re_axis(h5_files, period_ms, timesteps_per_cyc,
                   ramp_slope, ramp_offset,
                   density, dynamic_viscosity, pipe_diameter_mm):
    """
    Compute inlet Reynolds number for each saved HDF5 snapshot.

    Uses the simulation time encoded in the filename ('_t=<float>_' in ms) when available;
    falls back to reconstructing time from the '_ts=<int>' value and dt_ms.
    """
    dt_ms = period_ms / timesteps_per_cyc

    t_ms_vals = np.zeros(len(h5_files))
    for i, f in enumerate(h5_files):
        t_parsed = extract_sim_time_ms_from_h5_filename(f)
        if t_parsed is not None:
            t_ms_vals[i] = t_parsed
        else:
            t_ms_vals[i] = extract_timestep_from_h5_filename(f) * dt_ms

    t_s_vals = t_ms_vals * 1e-3                            # [s]
    Q_vals   = ramp_slope * t_s_vals + ramp_offset         # [mL/s]
    Re_vals  = flowrate_mLs_to_reynolds(Q_vals, density, dynamic_viscosity, pipe_diameter_mm)
    return Re_vals


# ======================================================================================================
# PLOTTING
# ======================================================================================================

_TERM_META = {
    'temporal':          dict(label=r'$\rho\,\partial u/\partial t$',      color='tab:blue',   ls='-'),
    'convective':        dict(label=r'$\rho\,u\,\partial u/\partial x$',   color='tab:orange', ls='-'),
    'pressure_gradient': dict(label=r'$-\partial p/\partial x$',            color='tab:green',  ls='-'),
    'viscous':           dict(label=r'$\mu\,\partial^2 u/\partial x^2$',   color='tab:red',    ls='-'),
    'residual':          dict(label=r'Residual',                             color='black',      ls='--'),
}


def _set_plot_style(font_size=14):
    plt.rc('font',   size=font_size)
    plt.rc('axes',   titlesize=font_size)
    plt.rc('axes',   labelsize=font_size)
    plt.rc('xtick',  labelsize=font_size - 2)
    plt.rc('ytick',  labelsize=font_size - 2)
    plt.rc('legend', fontsize=font_size - 4)


def plot_terms_vs_x(ns_terms, x_D, Re_axis, Re_targets,
                     output_path: Path, case_name: str):
    """
    Spatial NS budget plots: all terms vs x/D, one panel per selected Re value.
    """
    _set_plot_style()
    n_Re  = len(Re_targets)
    terms = list(ns_terms.keys())

    fig, axes = plt.subplots(n_Re, 1, figsize=(12, 4 * n_Re), sharex=True)
    if n_Re == 1:
        axes = [axes]

    for ax, Re_t in zip(axes, Re_targets):
        t_idx  = int(np.argmin(np.abs(Re_axis - Re_t)))
        Re_act = Re_axis[t_idx]

        for name in terms:
            col = ns_terms[name][:, t_idx]
            m   = _TERM_META[name]
            ax.plot(x_D, col, label=m['label'], color=m['color'],
                    lw=2, ls=m['ls'])

        ax.axhline(0, color='gray', lw=0.8, ls=':', zorder=0)
        ax.set_ylabel('NS term [Pa/m]', fontweight='bold')
        ax.set_title(f'Re ≈ {Re_act:.0f}', fontweight='bold')
        ax.legend(loc='best', ncol=3)
        ax.tick_params(direction='in')

    axes[-1].set_xlabel('x / D', fontweight='bold')
    fig.suptitle(f'{case_name}: NS budget along centerline', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved: {output_path}")


def plot_terms_vs_Re(ns_terms, x_D, Re_axis, xD_targets,
                      output_path: Path, case_name: str):
    """
    Re-evolution plots: all terms vs Re, one panel per selected x/D location.
    """
    _set_plot_style()
    n_x   = len(xD_targets)
    terms = list(ns_terms.keys())

    fig, axes = plt.subplots(n_x, 1, figsize=(12, 4 * n_x), sharex=True)
    if n_x == 1:
        axes = [axes]

    for ax, xD_t in zip(axes, xD_targets):
        xi     = int(np.argmin(np.abs(x_D - xD_t)))
        xD_act = x_D[xi]

        for name in terms:
            row = ns_terms[name][xi, :]
            m   = _TERM_META[name]
            ax.plot(Re_axis, row, label=m['label'], color=m['color'],
                    lw=2, ls=m['ls'])

        ax.axhline(0, color='gray', lw=0.8, ls=':', zorder=0)
        ax.set_ylabel('NS term [Pa/m]', fontweight='bold')
        ax.set_title(f'x/D ≈ {xD_act:.2f}', fontweight='bold')
        ax.legend(loc='best', ncol=3)
        ax.tick_params(direction='in')

    axes[-1].set_xlabel('Inlet Reynolds number', fontweight='bold')
    fig.suptitle(f'{case_name}: NS budget evolution with Re', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[plot] Saved: {output_path}")


def plot_2d_grid(ns_terms, x_D, Re_axis, output_path: Path, case_name: str):
    """
    2×2 subplot grid: x/D (x-axis) × Re (y-axis), color = NS term magnitude.
    RdBu_r colormap, centred at zero.
    """
    _set_plot_style(font_size=12)
    terms_to_plot = ['temporal', 'convective', 'pressure_gradient', 'viscous']

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
    axes_flat  = axes.ravel()

    for ax, name in zip(axes_flat, terms_to_plot):
        arr  = ns_terms[name]          # (n_pts, n_times)
        m    = _TERM_META[name]

        # pcolormesh expects (n_Re, n_x), so transpose
        pcm = ax.pcolormesh(x_D, Re_axis, arr.T, cmap='RdBu_r', shading='gouraud')
        finite = arr[np.isfinite(arr)]
        vmax   = float(np.percentile(np.abs(finite), 99)) if len(finite) > 0 else 1.0
        pcm.set_clim(-vmax, vmax)

        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label('[Pa/m]', rotation=270, labelpad=14)

        ax.set_title(m['label'], fontweight='bold')
        ax.tick_params(direction='in')

    for ax in axes[1, :]:
        ax.set_xlabel('x / D', fontweight='bold')
    for ax in axes[:, 0]:
        ax.set_ylabel('Inlet Re', fontweight='bold')

    fig.suptitle(f'{case_name}: NS term 2-D maps', fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved: {output_path}")


def plot_2d_individual(ns_terms, x_D, Re_axis, output_folder: Path, case_name: str):
    """
    Individual 2-D map for each NS term. Saves one PNG per term.
    """
    _set_plot_style()
    terms_to_plot = ['temporal', 'convective', 'pressure_gradient', 'viscous']

    for name in terms_to_plot:
        arr = ns_terms[name]
        m   = _TERM_META[name]

        fig, ax = plt.subplots(figsize=(11, 6))
        pcm = ax.pcolormesh(x_D, Re_axis, arr.T, cmap='RdBu_r', shading='gouraud')
        finite = arr[np.isfinite(arr)]
        vmax   = float(np.percentile(np.abs(finite), 99)) if len(finite) > 0 else 1.0
        pcm.set_clim(-vmax, vmax)

        cbar = fig.colorbar(pcm, ax=ax)
        cbar.set_label('NS term [Pa/m]', rotation=270, labelpad=16, fontweight='bold')

        ax.set_xlabel('x / D', fontweight='bold')
        ax.set_ylabel('Inlet Reynolds number', fontweight='bold')
        ax.set_title(f'{case_name}: {m["label"]}', fontweight='bold')
        ax.tick_params(direction='in')

        out = output_folder / f"{case_name}_NSterm_{name}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"[plot] Saved: {out}")


# ======================================================================================================
# CLI
# ======================================================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Compute NS term budget along the pipe centerline from CFD results.")

    # --- I/O ---
    ap.add_argument("--input_folder",  required=True,
                    help="CFD results folder with HDF5 snapshots (*_curcyc_*up.h5)")
    ap.add_argument("--mesh_folder",   required=True,
                    help="Folder containing mesh file (.xml.gz preferred, or results-folder .h5)")
    ap.add_argument("--output_folder", required=True, help="Output root directory")
    ap.add_argument("--case_name",     required=True, help="Case name (used in output filenames)")
    ap.add_argument("--n_process",     type=int,
                    default=max(1, mp.cpu_count() - 1),
                    help="Parallel reader processes (default: #CPUs - 1)")

    # --- Fluid ---
    ap.add_argument("--density",           type=float, default=1057,   help="Blood density [kg/m³]")
    ap.add_argument("--dynamic_viscosity", type=float, default=0.0037, help="Dynamic viscosity [Pa·s]")

    # --- Temporal ---
    ap.add_argument("--period_ms",         type=float, default=None,
                    help="Simulation period [ms] (inferred from last file's ts if omitted)")
    ap.add_argument("--timesteps_per_cyc", type=int,   default=None,
                    help="Timesteps per cycle (parsed from folder name if omitted)")
    ap.add_argument("--save_freq",         type=int,   default=None,
                    help="Snapshot save frequency N (parsed from folder name if omitted)")

    # --- Inflow ramp ---
    ap.add_argument("--ramp_slope",  type=float, default=2.0,
                    help="Ramp slope [mL/s per second] for Re axis computation (default: 2.0)")
    ap.add_argument("--ramp_offset", type=float, default=2.0,
                    help="Inflow rate at t=0 [mL/s] for Re axis computation (default: 2.0)")

    # --- Geometry ---
    ap.add_argument("--pipe_axis",     type=int, default=0, choices=[0, 1, 2],
                    help="Pipe axis: 0=X, 1=Y, 2=Z (default: 0)")
    ap.add_argument("--pipe_diameter", type=float, default=None,
                    help="Pipe inner diameter [mm] (estimated from mesh bounding box if omitted)")
    ap.add_argument("--centerline_coord1", type=float, default=0.0,
                    help="Centerline position on 1st perpendicular axis [mm] (default: 0.0)")
    ap.add_argument("--centerline_coord2", type=float, default=0.0,
                    help="Centerline position on 2nd perpendicular axis [mm] (default: 0.0)")
    ap.add_argument("--x_start_mm", type=float, default=None,
                    help="Centerline sampling start [mm] (default: mesh x_min)")
    ap.add_argument("--x_end_mm",   type=float, default=None,
                    help="Centerline sampling end [mm] (default: mesh x_max)")
    ap.add_argument("--x_ref_mm",   type=float, default=0.0,
                    help="x coordinate where x/D = 0 [mm] (default: 0.0 = stenosis throat at mesh origin)")

    # --- Centerline sampling ---
    ap.add_argument("--n_centerline_points", type=int, default=100,
                    help="Number of equally-spaced centerline sample points (default: 100)")

    # --- Plot controls ---
    ap.add_argument("--Re_plot_targets", type=float, nargs='+', default=None,
                    help="Re values at which to draw spatial budget profiles "
                         "(default: 3 auto-selected values spanning the simulation)")
    ap.add_argument("--xD_plot_targets", type=float, nargs='+', default=None,
                    help="x/D values for Re-evolution plots (default: -1 0 1 3)")
    ap.add_argument("--Re_min_plot", type=float, default=None,
                    help="Minimum Re shown in all plots (default: no lower limit)")
    ap.add_argument("--Re_max_plot", type=float, default=None,
                    help="Maximum Re shown in all plots (default: no upper limit)")

    return ap.parse_args()


# ======================================================================================================
# MAIN
# ======================================================================================================

def main():
    args = parse_args()

    input_folder  = Path(args.input_folder)
    output_root   = Path(args.output_folder) / "NS_terms_centerline"
    imgs_dir      = output_root / "imgs"
    data_dir      = output_root / "data"
    for d in (output_root, imgs_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 120)
    print("compute_NS_terms_centerline.py")
    print(f"  case:          {args.case_name}")
    print(f"  input folder:  {input_folder}")
    print(f"  mesh folder:   {args.mesh_folder}")
    print(f"  output folder: {output_root}")
    print("=" * 120 + "\n")

    # ------------------------------------------------------------------
    # 1. Load mesh
    # ------------------------------------------------------------------
    mesh_folder   = Path(args.mesh_folder)
    xml_gz_files  = sorted(mesh_folder.glob('*.xml.gz'))
    h5_mesh_files = sorted(mesh_folder.glob('*.h5'))

    if xml_gz_files:
        surf_mesh = load_surface_mesh_from_xmlgz(str(xml_gz_files[0]))
    elif h5_mesh_files:
        print("[warn] No .xml.gz found; using .h5 wall mesh — centerline nodes may be inaccurate.")
        surf_mesh = load_surface_mesh(h5_mesh_files[0])
    else:
        sys.exit(f"[error] No .xml.gz or .h5 mesh file found in {mesh_folder}")

    # ------------------------------------------------------------------
    # 2. Pipe diameter
    # ------------------------------------------------------------------
    pipe_diameter = (args.pipe_diameter
                     if args.pipe_diameter is not None
                     else estimate_pipe_diameter(surf_mesh, args.pipe_axis))
    print(f"[info] pipe_diameter = {pipe_diameter:.4f} mm\n")

    # ------------------------------------------------------------------
    # 3. Define centerline sample points + find nearest mesh nodes
    # ------------------------------------------------------------------
    cl_points, x_axial_mm = define_centerline_points(
        surf_mesh, args.n_centerline_points, args.pipe_axis,
        args.centerline_coord1, args.centerline_coord2,
        args.x_start_mm, args.x_end_mm)

    x_D = (x_axial_mm - args.x_ref_mm) / pipe_diameter  # normalised axial coordinate

    node_indices, node_ids, node_distances = find_nearest_mesh_nodes(surf_mesh, cl_points)

    # Free the mesh — no longer needed
    del surf_mesh
    gc.collect()

    # ------------------------------------------------------------------
    # 4. Discover and sort HDF5 snapshots
    # ------------------------------------------------------------------
    CFD_h5_files = sorted(
        input_folder.glob('*_curcyc_*up.h5'),
        key=extract_timestep_from_h5_filename)

    if not CFD_h5_files:
        sys.exit(f"[error] No '*_curcyc_*up.h5' files found in {input_folder}")
    print(f"[info] Found {len(CFD_h5_files)} HDF5 snapshots.")

    # ------------------------------------------------------------------
    # 5. Resolve temporal parameters
    # ------------------------------------------------------------------
    timesteps_per_cyc = args.timesteps_per_cyc
    save_freq         = args.save_freq
    period_ms         = args.period_ms

    if timesteps_per_cyc is None or save_freq is None:
        ts_parsed, sf_parsed = extract_sim_params_from_foldername(input_folder)
        if timesteps_per_cyc is None:
            timesteps_per_cyc = ts_parsed
            print(f"[info] timesteps_per_cyc = {timesteps_per_cyc}  (from folder name)")
        if save_freq is None:
            if sf_parsed is None:
                sys.exit("[error] Cannot parse save_freq from folder name; use --save_freq.")
            save_freq = sf_parsed
            print(f"[info] save_freq         = {save_freq}  (from folder name)")

    if period_ms is None:
        # Reconstruct from last file: ts_last × dt_ms == total simulation time per cycle
        # For a ramp simulation the "period" is the total time of one run cycle.
        last_ts   = extract_timestep_from_h5_filename(CFD_h5_files[-1])
        last_t_ms = extract_sim_time_ms_from_h5_filename(CFD_h5_files[-1])
        if last_t_ms is not None:
            period_ms = last_t_ms / (last_ts / timesteps_per_cyc)
        else:
            period_ms = 1000.0   # default: 1 s per cycle
        print(f"[info] period_ms         = {period_ms:.2f} ms  "
              f"({'inferred from last filename' if last_t_ms is not None else 'default — override with --period_ms'})")

    dt_saved_s = (period_ms * 1e-3 / timesteps_per_cyc) * save_freq

    print(f"[info] dt between saved snapshots = {dt_saved_s*1000:.4f} ms  "
          f"(period={period_ms} ms, ts={timesteps_per_cyc}, saveFreq={save_freq})\n")

    # ------------------------------------------------------------------
    # 6. Read u and p at centerline nodes (parallel)
    # ------------------------------------------------------------------
    u_cl, p_cl = read_centerline_fields_parallel(CFD_h5_files, node_ids, args.n_process, args.density)

    # ------------------------------------------------------------------
    # 7. Build Re axis
    # ------------------------------------------------------------------
    Re_axis = build_re_axis(
        CFD_h5_files, period_ms, timesteps_per_cyc,
        args.ramp_slope, args.ramp_offset,
        args.density, args.dynamic_viscosity, pipe_diameter)

    print(f"\n[info] Re range: {Re_axis.min():.1f} – {Re_axis.max():.1f}\n")

    # ------------------------------------------------------------------
    # 8. Compute NS terms
    # ------------------------------------------------------------------
    print("[compute] Computing NS terms along centerline …")
    ns_terms = compute_ns_terms(
        u_cl, p_cl, x_axial_mm,
        args.pipe_axis, dt_saved_s,
        args.density, args.dynamic_viscosity)
    print("[compute] Done.")

    # ------------------------------------------------------------------
    # 9. Save raw data
    # ------------------------------------------------------------------
    npz_path = data_dir / f"{args.case_name}_NS_terms_centerline.npz"
    np.savez(npz_path,
             x_D=x_D,
             Re_axis=Re_axis,
             node_distances=node_distances,
             **ns_terms)
    print(f"\n[data] Saved: {npz_path}")

    # ------------------------------------------------------------------
    # 10. Restrict to plotting Re range
    # ------------------------------------------------------------------
    re_mask = np.ones(len(Re_axis), dtype=bool)
    if args.Re_min_plot is not None: re_mask &= (Re_axis >= args.Re_min_plot)
    if args.Re_max_plot is not None: re_mask &= (Re_axis <= args.Re_max_plot)

    Re_plot = Re_axis[re_mask]
    ns_plot = {k: v[:, re_mask] for k, v in ns_terms.items()}

    # ------------------------------------------------------------------
    # 11. Determine plot targets
    # ------------------------------------------------------------------
    Re_targets = args.Re_plot_targets
    if Re_targets is None:
        idx_s  = np.linspace(0, len(Re_plot) - 1, 3, dtype=int)
        Re_targets = Re_plot[idx_s].tolist()

    xD_targets = args.xD_plot_targets if args.xD_plot_targets is not None else [-1.0, 0.0, 1.0, 3.0]

    # ------------------------------------------------------------------
    # 12. Generate plots
    # ------------------------------------------------------------------
    plot_terms_vs_x(
        ns_plot, x_D, Re_plot, Re_targets,
        imgs_dir / f"{args.case_name}_NSterms_vs_x.png",
        args.case_name)

    plot_terms_vs_Re(
        ns_plot, x_D, Re_plot, xD_targets,
        imgs_dir / f"{args.case_name}_NSterms_vs_Re.png",
        args.case_name)

    plot_2d_grid(
        ns_plot, x_D, Re_plot,
        imgs_dir / f"{args.case_name}_NSterms_2Dmap.png",
        args.case_name)

    #plot_2d_individual(ns_plot, x_D, Re_plot, imgs_dir, args.case_name)

    print(f"\n[done] All outputs written to: {output_root}")


if __name__ == '__main__':
    main()
