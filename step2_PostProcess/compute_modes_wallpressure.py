# -----------------------------------------------------------------------------------------------------------------------
# extract_circumferential_pressure.py
#
# Extract wall-pressure time-series at N evenly-spaced circumferential nodes
# located at a user-defined axial cross-section of an idealized stenosis geometry.
#
# STEPS:
#   Step 1: Load mesh, sample N circumferential wall nodes at a given axial slice,
#                       save selected nodes as .vtp for inspection in ParaView.
#   Step 2: Read pressure H5 snapshots for those nodes.
#   Step 3: Save outputs and full CLI.
#
# __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2026-09
# -----------------------------------------------------------------------------------------------------------------------

import re
import h5py
import warnings
import argparse
from pathlib import Path

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


# ======================================================================================================
# MESH LOADING  (reused from compute_Spectrogram_idealGeom.py)
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
    """Load wall surface from a DOLFIN XML.gz mesh file (no FEniCS required)."""
    import gzip
    import xml.etree.ElementTree as ET

    FACE_VERTS = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
    print(f"[mesh] Parsing XML.gz: {xml_gz_path} ...")

    all_coords: np.ndarray = None
    tet_verts:  np.ndarray = None
    wall_tris       = []
    wall_vertex_set = set()
    in_domain_mvc   = False

    with gzip.open(xml_gz_path, 'rt') as f:
        for event, elem in ET.iterparse(f, events=('start', 'end')):
            tag = elem.tag
            if event == 'start':
                if tag == 'vertices':
                    all_coords = np.zeros((int(elem.get('size')), 3), dtype=np.float64)
                elif tag == 'cells':
                    tet_verts  = np.zeros((int(elem.get('size')), 4), dtype=np.int64)
                elif tag == 'mesh_value_collection':
                    in_domain_mvc = (elem.get('dim') == '2')
            else:
                if tag == 'vertex':
                    idx = int(elem.get('index'))
                    all_coords[idx] = [float(elem.get('x')), float(elem.get('y')), float(elem.get('z'))]
                    elem.clear()
                elif tag == 'tetrahedron':
                    idx = int(elem.get('index'))
                    tet_verts[idx] = [int(elem.get(f'v{k}')) for k in range(4)]
                    elem.clear()
                elif tag == 'value' and in_domain_mvc:
                    if int(elem.get('value')) == 0:
                        ci    = int(elem.get('cell_index'))
                        le    = int(elem.get('local_entity'))
                        verts = tet_verts[ci][FACE_VERTS[le]]
                        wall_tris.append(verts.copy())
                        wall_vertex_set.update(verts.tolist())
                    elem.clear()
                elif tag == 'mesh_value_collection':
                    in_domain_mvc = False

    wall_point_ids = np.array(sorted(wall_vertex_set), dtype=np.int64)
    wall_coords    = all_coords[wall_point_ids]
    g2l            = {gid: lid for lid, gid in enumerate(wall_point_ids)}
    wall_topology  = np.array([[g2l[v] for v in tri] for tri in wall_tris], dtype=np.int64)
    n_wall_cells   = len(wall_topology)
    cells_vtk      = np.hstack([np.full((n_wall_cells, 1), 3, dtype=np.int64), wall_topology]).ravel()
    surf           = pv.PolyData(wall_coords, cells_vtk)
    surf.point_data['vtkOriginalPtIds'] = wall_point_ids

    #print(f"[mesh] Wall surface: {len(wall_point_ids)} nodes, {n_wall_cells} triangles")
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
# FILENAME / SIMULATION-PARAMETER UTILITIES  (reused from compute_Spectrogram_idealGeom.py)
# ======================================================================================================

def extract_timestep_from_h5_filename(h5_file: Path) -> int:
    """Sort key: extract integer timestep from '*_ts=<int>_...' filename pattern."""
    match = re.search(r'_ts=(\d+)', h5_file.stem)
    if match is None:
        raise ValueError(f"Filename '{h5_file.name}' does not contain expected '_ts=<int>' pattern.")
    return int(match.group(1))


def extract_sim_params_from_foldername(input_path: Path) -> tuple[int, int | None]:
    """Parse timesteps-per-cycle and save frequency from the results folder path.

    Expected patterns anywhere in the full path string:
      '_ts<int>'       — timesteps per cycle   (e.g. 'run_ts500_...')
      '_saveFreq<int>' — save frequency        (e.g. 'run_saveFreq10')

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

    match_sf = re.search(r'_saveFreq(\d+)', path_str)
    return int(match_ts.group(1)), (int(match_sf.group(1)) if match_sf else None)


# ======================================================================================================
# STEP 1 — CIRCUMFERENTIAL NODE SAMPLING
# ======================================================================================================

##FIX: CHANGE TO ACCOUNT FOR CENTERLINE NOT ON AXIS EXACTLY (INPUT CX,CY)
def sample_circumferential_nodes(surf_mesh:     pv.PolyData,
                                  slice_xcoord:   float,
                                  n_points:      int,
                                  pipe_diameter: float,
                                  pipe_axis:     int = 0,
                                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find n_points evenly-spaced circumferential wall nodes at a given axial cross-section.

    Assumes the pipe centreline passes through (slice_xcoord, 0, 0) — i.e. the two axes
    perpendicular to pipe_axis are centred at 0.

    Approach
    --------
    1. For each of n_points evenly-spaced angles θ_i = i·2π/n, compute the ideal target
       point on a circle of radius pipe_diameter/2 centred at (slice_xcoord, 0, 0).
    2. Find the closest mesh node (whole mesh, Euclidean 3-D) to each target.

    Parameters
    ----------
    surf_mesh     : pv.PolyData  Wall surface mesh.
    slice_xcoord   : float        Axial coordinate of the slice (mesh units).
    n_points      : int          Number of evenly-spaced circumferential sample points.
    pipe_diameter : float        Pipe inner diameter (mesh units).
    pipe_axis     : int          Axis along the pipe: 0=X, 1=Y, 2=Z.

    Returns
    -------
    node_indices      : np.ndarray (n_points,)   — surface-mesh point indices of chosen nodes.
    target_angles_deg : np.ndarray (n_points,)   — target angles [°], 0–360.
    target_coords     : np.ndarray (n_points, 3) — XYZ of the ideal target points.
    node_coords       : np.ndarray (n_points, 3) — XYZ of the chosen mesh nodes.
    """

    perpendicular_axes = [i for i in range(3) if i != pipe_axis]
    ax0, ax1  = perpendicular_axes
    radius    = pipe_diameter / 2.0

    # ---- Ideal target points (centre assumed at perpendicular coords = 0) ----
    target_angles = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)

    target_coords               = np.zeros((n_points, 3))
    target_coords[:, pipe_axis] = slice_xcoord
    target_coords[:, ax0]       = radius * np.sin(target_angles)
    target_coords[:, ax1]       = radius * np.cos(target_angles)

    # ---- Closest mesh node to each target (full mesh, Euclidean 3-D) ----
    node_indices = np.empty(n_points, dtype=int)
    for i, tgt in enumerate(target_coords):
        dists           = np.linalg.norm(surf_mesh.points - tgt, axis=1)
        node_indices[i] = int(np.argmin(dists))

    # ---- Duplicate warning ----
    n_unique = np.unique(node_indices).size
    if n_unique < n_points:
        warnings.warn(
            f"{n_points - n_unique} of {n_points} angles mapped to the same mesh node. \n Consider reducing --n_wallNodes.",
            UserWarning, stacklevel=2,
        )

    node_coords       = surf_mesh.points[node_indices]
    target_angles_deg = np.degrees(target_angles)

    print(f"[slice] slice_xcoord={slice_xcoord}  |  radius={radius:.5f}  |  unique nodes={n_unique}/{n_points}")

    return node_indices, target_angles_deg, target_coords, node_coords


def save_selected_nodes_vtp(output_path: Path,
                             node_indices:      np.ndarray,
                             target_angles_deg: np.ndarray,
                             target_coords:     np.ndarray,
                             node_coords:       np.ndarray,
                             surf_mesh:         pv.PolyData) -> None:
    """
    Save the selected circumferential nodes as a VTP PolyData file for ParaView inspection.

    Point data arrays written:
      - angle_deg       : target angle for each node [°]
      - dist_to_target  : Euclidean distance from the chosen node to its ideal target [mesh units]
      - node_index      : index into the wall surface mesh
      - vol_point_id    : global volume-mesh point ID (if available in surf_mesh)
    """
    dists = np.linalg.norm(node_coords - target_coords, axis=1)

    cloud = pv.PolyData(node_coords)
    cloud.point_data['angle_deg']      = target_angles_deg.astype(np.float32)
    cloud.point_data['dist_to_target'] = dists.astype(np.float32)
    cloud.point_data['node_index']     = node_indices.astype(np.int32)

    vol_ids = surf_mesh.point_data.get('vtkOriginalPtIds', None)
    if vol_ids is not None:
        cloud.point_data['vol_point_id'] = vol_ids[node_indices].astype(np.int32)

    cloud.save(str(output_path))
    print(f"[out]  Saved {len(node_indices)} selected nodes → {output_path}")


# ======================================================================================================
# STEP 2 — PRESSURE EXTRACTION AT SAMPLE NODES
# ======================================================================================================

import multiprocessing as mp
from multiprocessing import sharedctypes


def _create_shared_array(shape, dtype=np.float64):
    ctype_array = np.ctypeslib.as_ctypes(np.zeros(shape, dtype=dtype).ravel())
    return sharedctypes.Array(ctype_array._type_, ctype_array, lock=False), shape


def _view_shared_array(shared_obj, shape):
    return np.ctypeslib.as_array(shared_obj).reshape(shape)


def _read_pressure_worker(file_ids, vol_point_ids, h5_files, shared_ctype, shape, density):
    """Worker: read a chunk of snapshot files and write into the shared pressure array."""
    pressure = _view_shared_array(shared_ctype, shape)
    for t in file_ids:
        with h5py.File(h5_files[t], 'r') as h5:
            pressure[:, t] = np.array(h5['Solution']['p'])[vol_point_ids].ravel() * density


def read_pressure_at_sample_nodes(input_folder:  Path,
                                   vol_point_ids: np.ndarray,
                                   density:       float,
                                   n_process:     int,
                                   ) -> tuple[np.ndarray, list[Path]]:
    """
    Read pressure time-series for the N selected circumferential nodes from CFD HDF5 snapshots
    in parallel, distributing snapshots across n_process workers.

    Parameters
    ----------
    input_folder  : Path    Folder containing '*_curcyc_*up.h5' snapshot files.
    vol_point_ids : (N,) int  Global volume-mesh point IDs for the N sample nodes.
    density       : float   Blood density [kg/m³] — multiplied because Oasis stores p/ρ.
    n_process     : int     Number of parallel worker processes.

    Returns
    -------
    pressure : np.ndarray  shape (N, n_snapshots) in Pa.
    h5_files : list[Path]  Sorted list of snapshot files.
    """
    h5_files = sorted(input_folder.glob('*_curcyc_*up.h5'),
                      key=extract_timestep_from_h5_filename)
    if not h5_files:
        raise FileNotFoundError(f"No '*_curcyc_*up.h5' files found in {input_folder}")

    n_snapshots = len(h5_files)
    n_nodes     = len(vol_point_ids)
    shape       = (n_nodes, n_snapshots)

    shared_ctype, shape = _create_shared_array(shape)

    print(f"[step2] Reading {n_snapshots} snapshots for {n_nodes} nodes across {n_process} workers ...")

    chunk_size = max(n_snapshots // n_process, 1)
    chunks     = [list(range(n_snapshots))[i : i + chunk_size]
                  for i in range(0, n_snapshots, chunk_size)]

    procs = [mp.Process(target=_read_pressure_worker,
                        args=(chunk, vol_point_ids, h5_files, shared_ctype, shape, density))
             for chunk in chunks]
    for p in procs: p.start()
    for p in procs: p.join()

    pressure = _view_shared_array(shared_ctype, shape).copy()
    #print(f"[step2] Pressure array shape: {pressure.shape}  (nodes × snapshots)")
    return pressure, h5_files


def save_pressure_npz(output_path:      Path,
                      pressure:         np.ndarray,
                      target_angles_deg: np.ndarray,
                      node_indices:     np.ndarray,
                      vol_point_ids:    np.ndarray,
                      node_coords:      np.ndarray,
                      slice_xcoord:     float,
                      sampling_rate:    float,
                      ) -> None:
    """
    Save circumferential pressure time-series and metadata to a compressed .npz file.

    Saved arrays
    ------------
    pressure        : (n_nodes, n_snapshots) float64  [Pa]
    angles_deg      : (n_nodes,)  float64             [°]
    node_indices    : (n_nodes,)  int32               surface-mesh indices
    vol_point_ids   : (n_nodes,)  int64               global volume-mesh point IDs
    node_coords     : (n_nodes, 3) float64            XYZ [mesh units]
    slice_xcoord    : scalar float                    axial coordinate of the slice
    sampling_rate   : scalar float                    [Hz]
    """
    np.savez_compressed(
        output_path,
        pressure       = pressure.astype(np.float64),
        angles_deg     = target_angles_deg.astype(np.float64),
        node_indices   = node_indices.astype(np.int32),
        vol_point_ids  = vol_point_ids.astype(np.int64),
        node_coords    = node_coords.astype(np.float64),
        slice_xcoord   = np.float64(slice_xcoord),
        sampling_rate  = np.float64(sampling_rate),
    )
    print(f"[out]  Saved pressure time-series → {output_path}.npz")


# ======================================================================================================
# STEP 3 — SPATIAL FOURIER TRANSFORM (CIRCUMFERENTIAL MODES)
# ======================================================================================================

def compute_spatial_fourier_coefficients(pressure: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the spatial DFT over N evenly-spaced circumferential nodes at every timestep.

    The raw rfft output is divided by N so that |coeffs[m, t]| gives the true
    sinusoidal amplitude of mode m in the same units as the input (Pa).

    Parameters
    ----------
    pressure : (N, n_snapshots)  Wall pressure [Pa] at the N sample nodes over time.

    Returns
    -------
    coeffs     : (n_modes, n_snapshots) complex128
                 Normalised one-sided DFT coefficients; n_modes = N // 2 + 1.
                 Row m holds the complex amplitude of circumferential mode m at each timestep.
                   m = 0  →  mean (axisymmetric)
                   m = 1  →  first circumferential harmonic (single-lobe asymmetry)
                   m = 2  →  second harmonic, etc.
    mode_numbers : (n_modes,) int  Wavenumber indices [0, 1, ..., N//2].
    """
    n_nodes      = pressure.shape[0]
    coeffs       = np.fft.rfft(pressure, axis=0) / n_nodes   # (n_modes, n_snapshots), complex, [Pa]
    mode_numbers = np.arange(coeffs.shape[0], dtype=int)
    print(f"[step3] Spatial FFT: {n_nodes} nodes → {coeffs.shape[0]} modes  |  shape {coeffs.shape}")
    return coeffs, mode_numbers


def plot_mode_amplitudes(output_path:   Path,
                         coeffs:        np.ndarray,
                         mode_numbers:  np.ndarray,
                         sampling_rate: float,
                         case_name:     str,
                         slice_xcoord:  float,
                         pipe_diameter: float,
                         ) -> None:
    """
    Plot the time evolution of all circumferential mode amplitudes on one figure.

    amplitude[m, t] = |coeffs[m, t]|  [Pa]  (coeffs already normalised by N)
    """
    amplitude   = np.abs(coeffs)                           # (n_modes, n_snapshots) [Pa]
    n_snapshots = coeffs.shape[1]
    time        = np.arange(n_snapshots) / sampling_rate   # [s]
    inlet_flowrate = time*2 + 2
    cmap   = plt.get_cmap('Set2')

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"{case_name}  |  slice x={slice_xcoord/pipe_diameter}D  |  Wall-pressure mode amplitudes", fontsize=13, fontweight='bold')

    for m in mode_numbers[1:3]:
        ax.plot(inlet_flowrate, amplitude[m, :], color=cmap((m - 1) % 10), linewidth=1, label=f'm = {m}')

    ax.set_ylim([0,40])
    ax.set_xlabel('Inlet Flowrate [mL/s]', fontweight='bold')
    ax.set_ylabel('Amplitude [Pa]', fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)
    ax.tick_params(direction='in')

    plt.tight_layout()
    save_path = Path(str(output_path) + '.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[out]  Saved mode amplitude plot → {save_path}")


# ======================================================================================================
# MAIN  (Step 1: mesh loading + circumferential node sampling + VTP output)
# ======================================================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Extract wall-pressure time-series at N evenly-spaced circumferential nodes.")
    # Step 1 — mesh / slice
    ap.add_argument("--mesh_folder",       required=True,  help="Folder with mesh .h5 or .xml.gz file")
    ap.add_argument("--output_folder",     required=True,  help="Output folder for .vtp and .npz files")
    ap.add_argument("--case_name",         required=True,  help="Case name prefix for output files")
    ap.add_argument("--slice_xcoord_D",    required=True,  type=float, help="Axial position of the slice in pipe-diameter units (e.g. 10 → x = 10 × D)")
    ap.add_argument("--n_wallNodes",       required=True,  type=int,   help="Number of evenly-spaced sample points on the wall")
    ap.add_argument("--pipe_axis",         type=int,       default=0,  choices=[0, 1, 2], help="Axis along which the pipe runs: 0=X, 1=Y, 2=Z (default: 0)")
    ap.add_argument("--pipe_diameter",     type=float,     default=None, help="Pipe inner diameter [mesh units]. Estimated from bounding box if omitted.")
    # Step 2 — CFD results / pressure extraction
    ap.add_argument("--input_folder",      required=True,  help="Folder containing CFD '*_curcyc_*up.h5' snapshot files")
    ap.add_argument("--density",           type=float,     default=1057, help="Blood density [kg/m³] — multiplied because Oasis stores p/ρ (default: 1057)")
    ap.add_argument("--period_seconds",    type=float,     default=1.0, help="Flow period [s] (default: 1.0)")
    ap.add_argument("--timesteps_per_cyc", type=int,       default=None, help="Timesteps per cycle (parsed from folder name '_ts<int>' if omitted)")
    ap.add_argument("--save_freq",         type=int,       default=None, help="Save frequency: every Nth timestep saved (parsed from folder name '_saveFreq<int>' if omitted)")
    ap.add_argument("--n_process",         type=int,       default=max(1, mp.cpu_count() - 1), help="Number of parallel worker processes (default: n_CPUs - 1)")
    return ap.parse_args()


def main():
    args = parse_args()


    # ------------------------------ Load mesh ----------------------------------------
    mesh_folder  = Path(args.mesh_folder)
    h5_files     = list(mesh_folder.glob('*.h5'))
    xml_gz_files = list(mesh_folder.glob('*.xml.gz'))

    if h5_files:
        print(f"[mesh] Loading {h5_files[0].name} ...")
        surf_mesh = load_surface_mesh(h5_files[0])
    elif xml_gz_files:
        print(f"[mesh] Loading {xml_gz_files[0].name} ...")
        surf_mesh = load_surface_mesh_from_xmlgz(str(xml_gz_files[0]))
    else:
        raise FileNotFoundError(f"No .h5 or .xml.gz mesh file found in {mesh_folder}")

    print(f"[mesh] {surf_mesh.n_points} wall nodes  |  pipe axis = {'XYZ'[args.pipe_axis]}")

    # ---- Pipe diameter ----
    pipe_diameter = args.pipe_diameter
    if pipe_diameter is None:
        pipe_diameter = estimate_pipe_diameter(surf_mesh, args.pipe_axis)

    slice_xcoord = args.slice_xcoord_D * pipe_diameter
    print(f"[step1] slice_xcoord = {args.slice_xcoord_D} D = {slice_xcoord:.5f} (mesh units)")

    # ------------------------------ Find parameters ----------------------------------------       
    # Resolve temporal parameters
    timesteps_per_cyc = args.timesteps_per_cyc
    save_freq         = args.save_freq
    if timesteps_per_cyc is None or save_freq is None:
        ts_parsed, sf_parsed = extract_sim_params_from_foldername(Path(args.input_folder))
        if timesteps_per_cyc is None:
            timesteps_per_cyc = ts_parsed
            print(f"[step2] timesteps_per_cyc = {timesteps_per_cyc}  (parsed from folder name)")
        if save_freq is None:
            save_freq = sf_parsed
            print(f"[step2] save_freq         = {save_freq}  (parsed from folder name)")

    sampling_rate = timesteps_per_cyc / args.period_seconds / save_freq
    print(f"[step2] sampling_rate = {sampling_rate:.2f} Hz")


    # ------------------------ Step 1: sample circumferential nodes -----------------------------
    node_indices, target_angles_deg, target_coords, node_coords = sample_circumferential_nodes(
        surf_mesh     = surf_mesh,
        slice_xcoord  = slice_xcoord,
        n_points      = args.n_wallNodes,
        pipe_diameter = pipe_diameter,
        pipe_axis     = args.pipe_axis,
    )

    # ---- Save selected nodes as VTP ----
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    vtp_path = output_folder / f"{args.case_name}_slice{args.slice_xcoord_D}D_n{args.n_wallNodes}_nodes.vtp"
    #save_selected_nodes_vtp(vtp_path, node_indices, target_angles_deg, target_coords, node_coords, surf_mesh)

    # ------------------------ Step 2: Extract pressure at sampled nodes -----------------------------

    # Resolve vol_point_ids for the selected surface-mesh nodes
    raw_vol_ids = surf_mesh.point_data.get('vtkOriginalPtIds', None)
    vol_point_ids = raw_vol_ids[node_indices]

    pressure, _ = read_pressure_at_sample_nodes(
        input_folder  = Path(args.input_folder),
        vol_point_ids = vol_point_ids,
        density       = args.density,
        n_process     = args.n_process,
    )

    # npz_stem = output_folder / f"{args.case_name}_slice{args.slice_xcoord}_n{args.n_wallNodes}_pressure"
    # save_pressure_npz(
    #     output_path       = npz_stem,
    #     pressure          = pressure,
    #     target_angles_deg = target_angles_deg,
    #     node_indices      = node_indices,
    #     vol_point_ids     = vol_point_ids,
    #     node_coords       = node_coords,
    #     slice_xcoord      = args.slice_xcoord,
    #     sampling_rate     = sampling_rate,
    # )

    # ------------------------ Step 3: Spatial Fourier transform (circumferential modes) ----------
    coeffs, mode_numbers = compute_spatial_fourier_coefficients(pressure)

    plot_save_path = output_folder / f"{args.case_name}_slice{args.slice_xcoord_D}D_n{args.n_wallNodes}_modes"
    plot_mode_amplitudes(
        output_path   = plot_save_path,
        coeffs        = coeffs,
        mode_numbers  = mode_numbers,
        sampling_rate = sampling_rate,
        case_name     = args.case_name,
        slice_xcoord  = slice_xcoord,
        pipe_diameter = pipe_diameter,
    )


if __name__ == '__main__':
    main()
