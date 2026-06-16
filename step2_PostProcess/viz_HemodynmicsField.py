# -----------------------------------------------------------------------------------------------------------------------
# viz_HemodynamicsField.py
# Visualizes velocity streamlines, velocity magnitude isosurface, and Qcriterion
# isosurface from a single CFD snapshot resolved by target simulation time.
#
# __author__: Rojin Anbarafshan <rojin.anbar@gmail.com>
# __date__:   2026-05
#
# PURPOSE:
#   - Locates the HDF5 snapshot closest to a user-supplied target time [s] given
#     save_freq and timesteps-per-cycle (parsed from filename if not provided).
#   - Reads only that one snapshot (velocity field u); no other files are loaded.
#   - Renders velocity streamlines seeded from a sphere source (center + radius).
#   - Renders a 3D velocity magnitude isosurface at a user-specified isovalue (single color).
#   - Renders Qcriterion isosurface (via PyVista's built-in compute_derivative filter)
#     in a single color.
#   - Overlays geometry silhouette outline on every figure for spatial reference.
#   - Camera parameters (position, focal point, view-up, parallel scale) apply to all figures.
#
# REQUIREMENTS:
#   - h5py, numpy, pyvista, pyyaml
#   - On Trillium: virtual environment called "pyvista36"
#
# EXECUTION:
#   - Run directly on a login/debug node:
#       > module load StdEnv/2023 gcc/12.3 python/3.12.4
#       > source $HOME/virtual_envs/pyvista36/bin/activate
#
# EXAMPLE CLI:
#       > python viz_HemodynmicsField.py                     \
#           --config_file        <path_to_viz_config.yaml>        \
#           --mesh_folder   <path_to_mesh_folder>            \
#           --input_folder  <path_to_CFD_results_folder>     \
#           --output_folder <path_to_output_folder>          \
#           --case_name     PTSeg028_base_0p64               \
#           --target_time   2.53                             \
#           --save_freq     5                                \
#           --velocity_isovalue 0.5                          \
#           --qcri_isovalue     8000
#
# INPUTS (CLI — change per run):
#   --config_file             Path to YAML config file with case-specific defaults  [optional]
#   --mesh_folder        Folder containing BSLSolver HDF5 mesh file            [REQUIRED]
#   --input_folder       Folder with CFD HDF5 snapshots (*_curcyc_*up.h5)     [REQUIRED]
#   --output_folder      Output directory for PNG files                        [REQUIRED]
#   --case_name          Case identifier used in filenames and figure titles   [REQUIRED]
#   --target_time        Desired simulation time [s] to visualize              [REQUIRED]
#   --save_freq          Snapshot save frequency: every Nth timestep           (default: 5)
#   --velocity_isovalue  Velocity magnitude isosurface value [m/s]            (default: 0.5)
#   --qcri_isovalue      Qcriterion isosurface value(s) [1/s2]               (default: 1000)
#
# INPUTS (config file — case-specific, rarely change):
#   period_seconds       Flow period [s]
#   timesteps_per_cyc    Timesteps per cycle (parsed from filename if omitted)
#   stream_seed_coord    Seed point [x y z] for streamline sphere source
#   cam_position         [x y z]
#   cam_focal_point      [x y z]
#   cam_view_up          [x y z]
#   cam_parallel_scale   Enables parallel (orthographic) projection at this scale
#   window_size          Render window [W H]
#
# OUTPUTS:
#   - <case_name>_<figure_label>_streamlines.png
#   - <case_name>_<figure_label>_velocityIso<value>.png
#   - <case_name>_<figure_label>_QcriIso<value>.png
#
# NOTES:
#   - Velocity in HDF5 is (N_nodes, 3) [m/s].  Mesh coordinates are in [mm].
#   - Coordinates are converted mm -> m before Qcriterion computation so that
#     Q has physical units of [1/s2].
#
# Copyright (C) 2026 University of Toronto, Biomedical Simulation Lab.
# -----------------------------------------------------------------------------------------------------------------------

import re
import sys
import argparse
import numpy as np
import h5py
import pyvista as pv
pv.start_xvfb()        # start virtual X11 framebuffer for headless rendering on compute nodes
from pathlib import Path


# ======================================================================================================
# SNAPSHOT LOOKUP
# ======================================================================================================

def _extract_timestep(h5_file: Path) -> int:
    """Extract integer timestep from filename pattern '*_ts=<int>_*'."""
    match = re.search(r'_ts=(\d+)', h5_file.stem)
    if match is None:
        raise ValueError(f"'{h5_file.name}' has no '_ts=<int>' pattern.")
    return int(match.group(1))


def _extract_timesteps_per_cyc(h5_file: Path) -> int:
    """Parse timesteps-per-cycle from filename pattern '_ts<int>'."""
    match = re.search(r'_ts(\d+)', h5_file.stem)
    if match is None:
        raise ValueError(
            f"'{h5_file.name}' has no '_ts<int>' pattern. "
            "Supply --timesteps_per_cyc on the CLI instead."
        )
    return int(match.group(1))


def find_snapshot_at_time(input_folder: Path,
                          target_time: float,
                          save_freq: int,
                          period_seconds: float,
                          timesteps_per_cyc):
    """
    Locate the CFD snapshot closest to target_time [s].

    Snapshots are sorted by the '_ts=<int>' counter in their filenames.
    dt per saved snapshot = save_freq * period_seconds / timesteps_per_cyc.

    Returns (snapshot_file, frame_index, actual_time, timesteps_per_cyc, h5_files).
    """
    h5_files = sorted(input_folder.glob('*_curcyc_*up.h5'), key=_extract_timestep)
    if not h5_files:
        raise FileNotFoundError(f"No CFD snapshots (*_curcyc_*up.h5) found in {input_folder}")

    if timesteps_per_cyc is None:
        timesteps_per_cyc = _extract_timesteps_per_cyc(h5_files[0])

    dt         = save_freq * period_seconds / timesteps_per_cyc
    time_array = np.arange(len(h5_files)) * dt

    frame_idx   = int(np.argmin(np.abs(time_array - target_time)))
    actual_time = float(time_array[frame_idx])

    return h5_files[frame_idx], frame_idx, actual_time, timesteps_per_cyc, h5_files


# ======================================================================================================
# MESH & DATA I/O
# ======================================================================================================

# VTK cell type IDs used in the <DataArray Name="types"> section of .vtu files;
# tells VTK how to interpret each cell's connectivity list.
_VTK_TETRA = 10  # tetrahedron  (4-node 3D element)
_VTK_HEXA  = 12  # hexahedron   (8-node brick/cube element)


def load_mesh_from_h5(mesh_file: Path) -> pv.UnstructuredGrid:
    """Load volume mesh from BSLSolver/FEniCS HDF5 mesh file.

    Expects:
      Mesh/coordinates : (N, 3) float -- node coordinates [mm]
      Mesh/topology    : (M, k) int   -- cell connectivity (0-indexed)
    """
    with h5py.File(mesh_file, 'r') as h5:
        coords = np.array(h5['Mesh/coordinates'], dtype=np.float64)
        topo   = np.array(h5['Mesh/topology'],   dtype=np.int64)

    n_cells, nodes_per_cell = topo.shape
    if nodes_per_cell == 4:
        vtk_type = _VTK_TETRA
    elif nodes_per_cell == 8:
        vtk_type = _VTK_HEXA
    else:
        raise ValueError(f"Unsupported element: {nodes_per_cell} nodes/cell. "
                         "Expected 4 (tet) or 8 (hex).")

    cells      = np.column_stack([np.full(n_cells, nodes_per_cell, dtype=np.int64), topo]).ravel()
    cell_types = np.full(n_cells, vtk_type, dtype=np.uint8)

    return pv.UnstructuredGrid(cells, cell_types, coords)


def load_velocity_snapshot(snapshot_h5: Path) -> np.ndarray:
    """Read velocity field from one HDF5 snapshot.  Returns (N, 3) float [m/s]."""
    with h5py.File(snapshot_h5, 'r') as h5:
        u = np.array(h5['Solution']['u'], dtype=np.float64)
    if u.ndim == 1:
        u = u.reshape(-1, 3)
    return u


def attach_fields(grid: pv.UnstructuredGrid, u: np.ndarray) -> pv.UnstructuredGrid:
    """Attach velocity vector and magnitude as point arrays to the mesh."""
    grid['velocity']           = u
    grid['velocity_magnitude'] = np.linalg.norm(u, axis=1)
    return grid


# ======================================================================================================
# QCRITERION
# ======================================================================================================

def compute_qcriterion(grid: pv.UnstructuredGrid) -> np.ndarray:
    """
    Compute Qcriterion [1/s2] using PyVista's built-in compute_derivative filter.

    Converts coordinates mm -> m before differentiation so that the velocity
    gradient du_i/dx_j has units [1/s] and Q is in [1/s2].
    """
    grid_m = grid.copy()
    grid_m.points = grid.points * 1e-3    # mm -> m

    result = grid_m.compute_derivative(scalars='velocity', qcriterion=True)
    return result['qcriterion']


# ======================================================================================================
# CAMERA & RENDERING HELPERS
# ======================================================================================================

def make_plotter(window_size: list) -> pv.Plotter:
    """Create an off-screen PyVista plotter with a white background."""
    pl = pv.Plotter(off_screen=True, window_size=window_size)
    pl.set_background('white')
    pl.enable_anti_aliasing('ssaa')
    pl.enable_depth_peeling()   #improves rendering of translucent geometry

    return pl


def apply_camera(plotter: pv.Plotter, cam_params: dict):
    """Set camera position, focal point, view-up, and optional parallel scale.

    If none of the params are provided, falls back to PyVista's reset_camera()
    which auto-fits the view to the full mesh bounds.
    """
    any_set = False
    plotter.camera.parallel_projection = True
    if cam_params.get('position') is not None:
        plotter.camera.position = tuple(cam_params['position'])
        any_set = True
    if cam_params.get('focal_point') is not None:
        plotter.camera.focal_point = tuple(cam_params['focal_point'])
        any_set = True
    if cam_params.get('view_up') is not None:
        plotter.camera.up = tuple(cam_params['view_up'])
        any_set = True
    if cam_params.get('parallel_scale') is not None:
        plotter.camera.parallel_scale = cam_params['parallel_scale']
        any_set = True

    if any_set:
        plotter.camera_set = True   # prevent auto reset_camera() from overriding user params
    else:
        plotter.reset_camera()      # auto-fit to mesh bounds when no camera params given


def add_geometry_outline(plotter: pv.Plotter, grid: pv.UnstructuredGrid,
                         color: str = 'black', line_width: float = 1.5, opacity: float = 0.9):
    """Overlay the silhouette outline of the geometry (surface is invisible; only the outline is drawn)."""
    surface = grid.extract_surface()
    plotter.add_mesh(surface, color='white', opacity=0.1, show_scalar_bar=False,
                     silhouette=dict(color='silver', line_width=1, opacity=0.5))


def _scalar_bar_args(title: str) -> dict:
    return {'title': title, 'n_labels': 5, 'color': 'black', 'bold': True}


def save_screenshot(plotter: pv.Plotter, output_path: Path, scale: int = 2):
    plotter.screenshot(str(output_path), scale=scale)   # renders at scale× window_size
    plotter.close()


# ======================================================================================================
# FIGURE 1: VELOCITY STREAMLINES
# ======================================================================================================

def render_streamlines(grid: pv.UnstructuredGrid,
                       seed_coord: list,
                       seed_radius: float,
                       cam_params: dict,
                       output_path: Path,
                       title: str,
                       window_size: list,
                       plot_seed_cloud: bool = True):
    """Render velocity streamlines from a sphere seed source as lines."""

    streamlines = grid.streamlines(
        vectors='velocity',
        source_center=seed_coord,
        source_radius=seed_radius,
        n_points=120,
        initial_step_length=0.05,
        min_step_length=0.01,
        max_step_length=0.2,
        max_steps=50000,
        terminal_speed=0.05,            # stop tracing when nearly stagnant 
        interpolator_type='cell',       # more robust than 'point' for complex unstructured meshes
        integrator_type=45,             # Runge-Kutta 4/5
        integration_direction="forward",
    )
    

    pl = make_plotter(window_size)
    apply_camera(pl, cam_params)
    add_geometry_outline(pl, grid)

    if streamlines.n_points > 0:
        # Render streamlines as tubes
        tubes = streamlines.tube(radius=0.1)   # radius in mesh units (mm)

        # Clamp colorscale to jet velocity range — don't let slow recirculation dominate
        vel_max = 2             # from your max (m/s) — or compute: streamlines['velocity_magnitude'].max()

        pl.add_mesh(tubes,
                    scalars='velocity_magnitude',
                    cmap='turbo',
                    clim=[0, vel_max],
                    specular=0.8,
                    specular_power=30,
                    ambient=0.15,
                    smooth_shading=True,
                    show_scalar_bar=True,
                    scalar_bar_args=dict(
                        title='Velocity (m/s)',
                        n_labels=5,                 # number of colorbar ticks
                        fmt='%.2f',                 # format colorbar ticks
                        height=0.15,
                        width=0.02,
                        vertical=True,
                        position_x=0.95,
                        position_y=0.8,
                        )
        )
    else:
        print("  Warning: no streamlines generated. Check stream_seed_coord.")

    if plot_seed_cloud:
        seed_sphere = pv.Sphere(center=seed_coord, radius=5)
        pl.add_mesh(seed_sphere, color='yellow', opacity=0.3, show_scalar_bar=False)

    #pl.add_title(title, color='black', font_size=16)
    save_screenshot(pl, output_path)


# ======================================================================================================
# FIGURE 2: VELOCITY MAGNITUDE ISOSURFACE
# ======================================================================================================

def render_velocity_isosurface(grid: pv.UnstructuredGrid,
                               isovalue: float,
                               cam_params: dict,
                               output_path: Path,
                               title: str,
                               window_size: list):
    """Render the |u| = isovalue isosurface in a single color."""
    iso = grid.contour(isosurfaces=[isovalue], scalars='velocity_magnitude')

    pl = make_plotter(window_size)
    apply_camera(pl, cam_params)
    add_geometry_outline(pl, grid)

    if iso.n_points > 0:
        pl.add_mesh(iso, color='red', specular=1.0, specular_power=50, ambient=0.1, smooth_shading=True, show_scalar_bar=False)
    else:
        print(f"  Warning: no isosurface at |u| = {isovalue} m/s. "
              "Adjust --velocity_isovalue (velocity range printed above).")

    #pl.add_title(title, color='black', font_size=20)
    save_screenshot(pl, output_path)


# ======================================================================================================
# FIGURE 3: QCRITERION ISOSURFACE
# ======================================================================================================

# Colors for past / center / future frames in the Qcriterion overlay

_QCRI_FRAME_COLORS = {-1: '#440154', 0: '#E31A8C', 1: "#FFCC00"}  #FF8C00
_QCRI_FRAME_OPACITY = {-1: 0.95, 0: 0.6, 1: 0.4} #-1: 0.5, 0: 0.4, 1: 0.3   -1: 0.95, 0: 0.6, 1: 0.4

def render_qcriterion(frames: list,
                      qcri_isovalue,
                      cam_params: dict,
                      output_path: Path,
                      title: str,
                      window_size: list):
    """Overlay Qcriterion isosurfaces from multiple timesteps.

    frames : list of (grid, time_float, rel_pos) where rel_pos is -1/0/+1
             (past / center / future) and each grid has 'velocity' point data.
    """

    pl = make_plotter(window_size)
    apply_camera(pl, cam_params)
    add_geometry_outline(pl, frames[0][0])

    for grid, t_val, rel_pos in frames:
        print(f"  Computing Qcriterion for t = {t_val:.4f} s ...")
        Q = compute_qcriterion(grid)
        g = grid.copy()
        g['Qcriterion'] = Q
        print(f"    Q range: {Q.min():.3g}  to  {Q.max():.3g}  [1/s2]")
        isosurface = g.contour(isosurfaces=qcri_isovalue, scalars='Qcriterion')
        isosurface = isosurface.smooth(n_iter=50, relaxation_factor=0.5) # to smooth the isosurfaces
        if isosurface.n_points > 0:
            pl.add_mesh(isosurface, color=_QCRI_FRAME_COLORS[rel_pos], smooth_shading=True, specular=0.5, specular_power=20, ambient=0.2, opacity = _QCRI_FRAME_OPACITY[rel_pos], show_scalar_bar=False)
        else:
            print(f"    Warning: no isosurface at Q = {qcri_isovalue} for t = {t_val:.4f} s.")

    #pl.add_title(title, color='black', font_size=16)
    save_screenshot(pl, output_path)


# ======================================================================================================
# ARGUMENT PARSING
# ======================================================================================================

def parse_args():
    import yaml

    # Pre-parse to get --config before building the full parser
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config_file', default=None)
    pre_args, _ = pre.parse_known_args()

    ap = argparse.ArgumentParser(
        description="Visualize streamlines, velocity contour, and Qcriterion from one CFD snapshot.")

    # ---- Config file ----
    ap.add_argument('--config_file', default=None,
                    help="Path to YAML config file with case-specific defaults (CLI args override)")

    # ---- Mesh & data (required) ----
    ap.add_argument('--mesh_folder',    required=True, help="Folder with BSLSolver HDF5 mesh (Mesh/coordinates + Mesh/topology)")
    ap.add_argument('--input_folder',   required=True, help="Folder with CFD HDF5 snapshots (*_curcyc_*up.h5)")
    ap.add_argument('--output_folder',  required=True, help="Output directory for PNG files")
    ap.add_argument('--case_name',      required=True, help="Case identifier used in filenames and figure titles")


    # ---- CFD cycle parameters (config) ----
    ap.add_argument('--period_seconds',    type=float, default=None, help="Flow period [s]")
    ap.add_argument('--timesteps_per_cyc', type=int,   default=None, help="Timesteps per cycle (parsed from filename if omitted)")
    ap.add_argument('--save_freq',         type=int,   default=5, help="Snapshot save frequency: every Nth timestep (default: 5)")

    # ---- Isosurface values (CLI) ----
    ap.add_argument('--velocity_isovalue', type=float, default=0.5,    help="Velocity magnitude isosurface value [m/s] (default: 0.5)")
    ap.add_argument('--qcri_isovalue',     type=float, default=1000.0, help="Qcriterion isosurface value(s) [1/s2] (default: 1000)")
    ap.add_argument('--frame_spacing',     type=int,   default=10,     help="Frame offset for Qcriterion overlay: plots target±frame_spacing (default: 10); use 0 for single timestep")
    ap.add_argument('--target_flowrate',   type=float, nargs='+', default=None, help="Target flowrate(s) [mL/s]. Accepts one or more values.")

    # ---- Streamline seed (config) ----
    ap.add_argument('--stream_seed_coord',  type=float, nargs=3, default=None, help="Seed point [x y z] for streamline sphere source")
    ap.add_argument('--stream_seed_radius', type=float,          default=None, help="Radius of seed point cloud (in mesh units)")

    # ---- Camera (config) ----
    ap.add_argument('--cam_position',       type=float, nargs=3, default=None, help="Camera position [x y z]")
    ap.add_argument('--cam_focal_point',    type=float, nargs=3, default=None, help="Camera focal point [x y z]")
    ap.add_argument('--cam_view_up',        type=float, nargs=3, default=None, help="Camera view-up vector [x y z]")
    ap.add_argument('--cam_parallel_scale', type=float,          default=None, help="Enable parallel projection at this scale")

    # ---- Output (config) ----
    ap.add_argument('--window_size', type=int, nargs=2, default=None, help="Render window size [W H]")

    # Load config as defaults so CLI args always win
    if pre_args.config_file:
        with open(pre_args.config_file) as f:
            cfg = yaml.safe_load(f)
        ap.set_defaults(**cfg)

    return ap.parse_args()


# ======================================================================================================
# MAIN
# ======================================================================================================

def main():
    args = parse_args()

    input_folder  = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    window_size = args.window_size if args.window_size is not None else [1920, 1080]
    period      = args.period_seconds if args.period_seconds is not None else 0.915

    # ---- Determine which target times to render ----
    if not args.target_flowrate:
        print("ERROR: provide --target_flowrate <Q1> [Q2 ...] (in mL/s).")
        sys.exit(1)
    run_configs = [(q / 2, q) for q in args.target_flowrate]

    print("=" * 100)
    print("viz_HemodynamicsField.py")
    print(f"  Config           : {args.config_file}")
    print(f"  Input folder     : {input_folder}")
    print(f"  Output folder    : {output_folder}")
    print(f"  Case name        : {args.case_name}")
    print(f"  Target flowrates : {args.target_flowrate} mL/s")
    print(f"  Target times     : {[q/2 for q in args.target_flowrate]} s")
    print("=" * 100 + "\n")

    # ---- Load mesh once (shared across all target times) ----
    mesh_file = next(Path(args.mesh_folder).glob('*.h5'), None)
    if mesh_file is None:
        print(f"ERROR: no .h5 mesh file found in {args.mesh_folder}")
        sys.exit(1)
    print(f"Loading mesh ...")
    grid = load_mesh_from_h5(mesh_file)

    # ---- Shared camera dict ----
    cam_params = {
        'position':       args.cam_position,
        'focal_point':    args.cam_focal_point,
        'view_up':        args.cam_view_up,
        'parallel_scale': args.cam_parallel_scale,
    }

    # ---- Loop over target times ----
    for target_time, flowrate in run_configs:
        print(f"\n{'─' * 80}")
        print(f"Processing flowrate = {flowrate:.4f} mL/s  ...")

        snapshot_file, frame_idx, actual_time, timesteps_per_cyc, h5_files = find_snapshot_at_time(
            input_folder, target_time,
            save_freq         = args.save_freq,
            period_seconds    = period,
            timesteps_per_cyc = args.timesteps_per_cyc,
        )
        dt = args.save_freq * period / timesteps_per_cyc
        print(f"  Target time    : {target_time:.4f} s")
        print(f"  Resolved frame : {frame_idx}  (0-based index in sorted snapshot list)")
        print(f"  Actual time    : {actual_time:.4f} s  (dt = {dt:.4f} s/frame)")
        print(f"  Time error     : {abs(actual_time - target_time) * 1e3:.3f} ms")
        print(f"  Snapshot file  : {snapshot_file.name}\n")

        # --------------------- Load velocity for this snapshot -----------------
        print("Loading velocity ...")
        u = load_velocity_snapshot(snapshot_file)
        vel_mag = np.linalg.norm(u, axis=1)
        print(f"  Velocity magnitude:  max={vel_mag.max():.2f} [m/s]\n")
        attach_fields(grid, u)

        # ----------------------------- Figure 1: Streamlines --------------------------
        print("Rendering streamlines ...")
        out_stream = output_folder / f"{args.case_name}_Qin{flowrate:.3f}mLs_streamlines.png"
        render_streamlines(
            grid,
            seed_coord  = args.stream_seed_coord,
            seed_radius = args.stream_seed_radius,
            cam_params  = cam_params,
            output_path = out_stream,
            title       = f"{args.case_name}  --  Velocity Streamlines  |  Flowrate = {flowrate:.3f} mL/s  |  t = {actual_time:.4f} s",
            window_size = window_size,
        )

        # --------------------------- Figure 2: Velocity isosurface ---------------------
        print("\nRendering velocity isosurface ...")
        out_vel = output_folder / f"{args.case_name}_Qin{flowrate:.3f}mLs_velocityIso{args.velocity_isovalue}.png"
        render_velocity_isosurface(
            grid,
            isovalue    = args.velocity_isovalue,
            cam_params  = cam_params,
            output_path = out_vel,
            title       = f"{args.case_name}  --  |u| = {args.velocity_isovalue} m/s  |  Flowrate = {flowrate:.3f} mL/s |  t = {actual_time:.4f} s  ",
            window_size = window_size,
        )

        # ---------------------------- Figure 3: Qcriterion ----------------------------
        print("\nRendering Qcriterion isosurface ...")
        spacing  = args.frame_spacing
        n_frames = len(h5_files)
        neighbor_indices = sorted({frame_idx - spacing, frame_idx, frame_idx + spacing})

        if neighbor_indices[0] < 0 or neighbor_indices[-1] >= n_frames:
            print(f"  Note: frame_spacing={spacing} hits boundary; overlaying {len(neighbor_indices)} frame(s).")

        qcri_frames = []
        for fidx in neighbor_indices:
            t_f = fidx * dt
            rel = 0 if fidx == frame_idx else (-1 if fidx < frame_idx else 1)
            if fidx == frame_idx:
                g = grid
            else:
                g = grid.copy()
                attach_fields(g, load_velocity_snapshot(h5_files[fidx]))
            qcri_frames.append((g, t_f, rel))

        out_qcrit = output_folder / f"{args.case_name}_Qin{flowrate:.3f}mLs_Qcri{args.qcri_isovalue:.0f}_frSpace{spacing}.png"
        render_qcriterion(
            qcri_frames,
            qcri_isovalue = args.qcri_isovalue,
            cam_params    = cam_params,
            output_path   = out_qcrit,
            title         = f"{args.case_name}  --  Qcriterion  |  Flowrate = {flowrate:.3f} mL/s  |  t = {actual_time:.4f} s  ",
            window_size   = window_size,
        )

    print("\nDone.\n")


if __name__ == '__main__':
    main()
