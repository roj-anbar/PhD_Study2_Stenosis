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

import h5py
import warnings
import argparse
from pathlib import Path

import numpy as np
import pyvista as pv


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

    print(f"[mesh] Wall surface: {len(wall_point_ids)} nodes, {n_wall_cells} triangles")
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
            f"{n_points - n_unique} of {n_points} angles mapped to the same mesh node. "
            "Consider reducing --n_circumferential.",
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
# MAIN  (Step 1: mesh loading + circumferential node sampling + VTP output)
# ======================================================================================================

def parse_args():
    ap = argparse.ArgumentParser(
        description="Step 1: Sample N circumferential wall nodes at an axial slice.")
    ap.add_argument("--mesh_folder",       required=True,  help="Folder with mesh .h5 or .xml.gz file")
    ap.add_argument("--output_folder",     required=True,  help="Output folder for .vtp file")
    ap.add_argument("--case_name",         required=True,  help="Case name prefix for output files")
    ap.add_argument("--slice_xcoord",      required=True,  type=float, help="Axial coordinate of the slice (mesh units)")
    ap.add_argument("--n_circumferential", required=True,  type=int, help="Number of evenly-spaced circumferential sample points")
    ap.add_argument("--pipe_axis",         type=int, default=0, choices=[0, 1, 2], help="Axis along which the pipe runs: 0=X, 1=Y, 2=Z (default: 0 = X)")
    ap.add_argument("--pipe_diameter",     type=float, default=None, help="Pipe inner diameter [mesh units]. Estimated from bounding box if omitted.")
    return ap.parse_args()


def main():
    args = parse_args()

    # ---- Load mesh ----
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

    # ---- Step 1: sample circumferential nodes ----
    node_indices, target_angles_deg, target_coords, node_coords = sample_circumferential_nodes(
        surf_mesh     = surf_mesh,
        slice_xcoord   = args.slice_xcoord,
        n_points      = args.n_circumferential,
        pipe_diameter = pipe_diameter,
        pipe_axis     = args.pipe_axis,
    )

    # ---- Save selected nodes as VTP ----
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    vtp_path = output_folder / f"{args.case_name}_slice{args.slice_xcoord}_n{args.n_circumferential}_nodes.vtp"
    save_selected_nodes_vtp(vtp_path, node_indices, target_angles_deg,
                            target_coords, node_coords, surf_mesh)

    print("\nStep 1 complete. Inspect the .vtp in ParaView, then proceed to Step 2.")


if __name__ == '__main__':
    main()
