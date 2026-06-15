"""
STL to Volumetric Mesh Converter
=================================
Reads a closed vessel surface mesh from STL,
generates a 3-D tetrahedral volumetric mesh using gmsh,
identifies boundary patches (inlet / outlet / wall),
and saves outputs in:

  *.h5      – HDF5 with Mesh/coordinates, Mesh/topology,
               Mesh/Wall, Mesh/ID_1, Mesh/ID_2 patches
  *.xml.gz  – FEniCS/DOLFIN XML (tetrahedral mesh + boundary markers)
  *.info    – boundary meta-data (center, normal, radius, area, FR/AR)

Usage
-----
Set STL_PATH, OUTPUT_DIR, MODEL_NAME at the bottom of this file
and run:
    python mesh_eccentricStenosicPipe.py

Requirements
------------
    gmsh, h5py, numpy, pyvista
"""

import os
import gzip
import numpy as np
import h5py
import gmsh
import pyvista as pv
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Paths & Directories
# ─────────────────────────────────────────────────────────────
BASE_DIR   = '/Users/rojin/Library/CloudStorage/OneDrive-UniversityofToronto/Education/PhD/My_Projects/Study2_stenosis/models/eccentricStenosis'
STL_PATH   = f'{BASE_DIR}/D=10/eccStenosis.stl'
OUTPUT_DIR = f'{BASE_DIR}/D=10/mesh'
MODEL_NAME = 'eccStenosis'


# ─────────────────────────────────────────────────────────────
# Tunable parameters
# ─────────────────────────────────────────────────────────────
MESH_SIZE   = 0.05   # target edge length (same units as STL)
ANGLE_DEG   = 40.0   # surface-classification dihedral-angle threshold
# For vessels whose axis is roughly aligned with X, flat caps at the
# ends have |normal_x| close to 1.  Increase AXIS_THRESH if the
# classifier mistakenly labels wall patches as caps.
AXIS_THRESH = 0.85

# Boundary-layer parameters
N_BOUNDARY_LAYERS = 0      # number of prismatic layers near the wall
BL_SIZE_FACTOR    = 0.05   # first-layer thickness = MESH_SIZE * BL_SIZE_FACTOR
BL_RATIO          = 1.2    # growth ratio between successive layers


# ─────────────────────────────────────────────────────────────
# 1.  GMSH MESHING
# ─────────────────────────────────────────────────────────────
def mesh_stl(stl_path: str,
             mesh_size: float = MESH_SIZE,
             angle_deg: float = ANGLE_DEG,
             n_bl: int = N_BOUNDARY_LAYERS,
             bl_size_factor: float = BL_SIZE_FACTOR,
             bl_ratio: float = BL_RATIO) -> tuple:
    """
    Load a closed STL surface with gmsh, classify surfaces, create a
    volume and generate a tetrahedral mesh.

    Returns
    -------
    coords      : (N, 3) float64  – node coordinates
    tets        : (M, 4) int64    – tet connectivity (0-based)
    tag_to_tris : dict            – {surface_tag: (T, 3) int64 triangles}
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    gmsh.model.add("vessel")

    gmsh.merge(stl_path)

    angle_rad = angle_deg * np.pi / 180.0
    gmsh.model.mesh.classifySurfaces(
        angle_rad,   # dihedral-angle threshold
        True,        # include boundary
        True,        # for reparametrisation
        2 * np.pi    # curve-angle threshold (don't split curves)
    )
    gmsh.model.mesh.createGeometry()

    surfaces = gmsh.model.getEntities(2)
    surf_tags = [s[1] for s in surfaces]
    print(f"  classifySurfaces found {len(surf_tags)} surface(s): {surf_tags}")

    # Build closed volume
    sl  = gmsh.model.geo.addSurfaceLoop(surf_tags)
    gmsh.model.geo.addVolume([sl])
    gmsh.model.geo.synchronize()

    # Mesh controls
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size * 0.1)
    gmsh.option.setNumber("Mesh.Algorithm3D",             1)   # 3D Delaunay
    gmsh.option.setNumber("Mesh.Optimize",                1)

    # ── Boundary layers on wall surfaces ──────────────────────
    if n_bl > 0:
        # Classify surfaces geometrically via bounding boxes (no mesh needed)
        xmin_g, _, _, xmax_g, _, _ = gmsh.model.getBoundingBox(-1, -1)
        x_range  = xmax_g - xmin_g
        flat_tol = x_range * 0.02   # caps have negligible x-extent
        end_tol  = x_range * 0.05   # caps sit within 5 % of the ends

        wall_surf_tags = []
        for stag in surf_tags:
            bb = gmsh.model.getBoundingBox(2, stag)
            x_extent = bb[3] - bb[0]
            x_center = (bb[0] + bb[3]) / 2.0
            is_flat        = x_extent < flat_tol
            is_near_inlet  = abs(x_center - xmin_g) < end_tol
            is_near_outlet = abs(x_center - xmax_g) < end_tol
            if not (is_flat and (is_near_inlet or is_near_outlet)):
                wall_surf_tags.append(stag)

        # gmsh's BoundaryLayer field is 2D-only; for 3D tet meshes use a
        # Distance + Threshold size field to create graded refinement near the
        # wall — equivalent to prismatic BL resolution with tets.
        first = mesh_size * bl_size_factor
        # Total BL thickness = geometric series: first * (r^n - 1) / (r - 1)
        bl_thickness = first * (bl_ratio ** n_bl - 1.0) / (bl_ratio - 1.0)

        dist_f = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(dist_f, "SurfacesList", wall_surf_tags)
        gmsh.model.mesh.field.setNumber(dist_f, "Sampling", 100)

        thr_f = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(thr_f, "InField",  dist_f)
        gmsh.model.mesh.field.setNumber(thr_f, "SizeMin",  first)
        gmsh.model.mesh.field.setNumber(thr_f, "SizeMax",  mesh_size)
        gmsh.model.mesh.field.setNumber(thr_f, "DistMin",  0.0)
        gmsh.model.mesh.field.setNumber(thr_f, "DistMax",  bl_thickness)

        gmsh.model.mesh.field.setAsBackgroundMesh(thr_f)
        # Allow elements as small as the first BL layer
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", first)

        print(f"  Wall surfaces for BL: {wall_surf_tags}")
        print(f"  {n_bl} boundary layers: first = {first:.4f}, "
              f"total thickness = {bl_thickness:.4f}, ratio = {bl_ratio}")

    print("  Running 3-D mesh generation ...")
    gmsh.model.mesh.generate(3)

    # ── Extract nodes ──────────────────────────────────────────
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    coords = node_coords.reshape(-1, 3)
    tag2idx = {int(t): i for i, t in enumerate(node_tags)}

    # ── Extract tetrahedra ─────────────────────────────────────
    _, _, tet_conn = gmsh.model.mesh.getElements(3)
    tets = np.array([tag2idx[int(t)] for t in tet_conn[0]],
                    dtype=np.int64).reshape(-1, 4)

    # ── Extract triangles per surface ─────────────────────────
    tag_to_tris = {}
    for stag in surf_tags:
        _, _, tri_conn = gmsh.model.mesh.getElements(2, stag)
        if not tri_conn or len(tri_conn[0]) == 0:
            continue
        tris = np.array([tag2idx[int(t)] for t in tri_conn[0]],
                        dtype=np.int64).reshape(-1, 3)
        tag_to_tris[stag] = tris

    gmsh.finalize()
    print(f"  Mesh: {len(coords):,} nodes, {len(tets):,} tets")
    return coords, tets, tag_to_tris


# ─────────────────────────────────────────────────────────────
# 2.  SURFACE CLASSIFICATION
# ─────────────────────────────────────────────────────────────
def classify_surfaces(coords: np.ndarray,
                      tag_to_tris: dict,
                      axis_thresh: float = AXIS_THRESH) -> dict:
    """
    Label each gmsh surface as 'inlet', 'outlet', or 'wall'.

    The eccentric-stenosis model has its vessel axis along X.
    A surface is a cap (inlet or outlet) only if:
      1. |avg_normal_x| > axis_thresh  (nearly perpendicular to X)
      2. centroid_x is within 2 % of the total X range from the
         extreme end  (caps sit at the very ends of the domain)
      3. the X spread of its vertices is < 1 % of the X range
         (caps are planar, wall patches near the throat are not)

    Returns
    -------
    dict: {surface_tag -> {'label': str, 'centroid': (3,), 'normal': (3,),
                            'area': float}}
    """
    x_min = float(coords[:, 0].min())
    x_max = float(coords[:, 0].max())
    x_range = x_max - x_min
    end_tol  = x_range * 0.02   # 2 % of total length
    flat_tol = x_range * 0.01   # 1 % – caps are truly flat

    info = {}
    for stag, tris in tag_to_tris.items():
        v0, v1, v2 = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)                # (T, 3) area-weighted
        face_areas   = np.linalg.norm(cross, axis=1) / 2.0
        face_normals = cross / (2 * face_areas[:, None] + 1e-15)

        total_area = float(face_areas.sum())
        avg_normal = (face_normals * face_areas[:, None]).sum(0) / total_area
        avg_normal /= np.linalg.norm(avg_normal) + 1e-15

        patch_pts = coords[np.unique(tris.ravel())]
        centroid  = patch_pts.mean(0)
        x_spread  = float(patch_pts[:, 0].ptp())   # peak-to-peak in X

        nx = avg_normal[0]
        is_cap = (abs(nx) > axis_thresh) and (x_spread < flat_tol)
        if is_cap and abs(centroid[0] - x_min) < end_tol and nx < 0:
            label = 'inlet'
        elif is_cap and abs(centroid[0] - x_max) < end_tol and nx > 0:
            label = 'outlet'
        else:
            label = 'wall'

        info[stag] = {
            'label':    label,
            'centroid': centroid,
            'normal':   avg_normal,
            'area':     total_area,
        }
    return info


def merge_patch_triangles(coords: np.ndarray,
                          tag_to_tris: dict,
                          surface_info: dict,
                          label: str) -> np.ndarray:
    """Concatenate all triangles whose surface has the given label."""
    parts = [tris for stag, tris in tag_to_tris.items()
             if surface_info[stag]['label'] == label]
    if not parts:
        raise RuntimeError(
            f"No surfaces labelled '{label}' found. "
            "Check AXIS_THRESH or the STL vessel-axis orientation."
        )
    return np.vstack(parts)


# ─────────────────────────────────────────────────────────────
# 3.  BOUNDARY STATISTICS
# ─────────────────────────────────────────────────────────────
def cap_statistics(coords: np.ndarray, tris: np.ndarray,
                   domain_center_x: float) -> dict:
    """
    Compute center, outward normal, equivalent radius and area of a cap.
    The outward normal points *away* from the domain centre.
    """
    pts    = coords[np.unique(tris.ravel())]
    center = pts.mean(axis=0)

    v0, v1, v2 = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
    cross  = np.cross(v1 - v0, v2 - v0)
    areas  = np.linalg.norm(cross, axis=1) / 2.0
    total_area = float(areas.sum())
    n_unit = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-15)
    avg_n  = (n_unit * areas[:, None]).sum(0) / total_area
    avg_n /= np.linalg.norm(avg_n) + 1e-15

    # Ensure the normal points away from the interior
    to_center = np.array([domain_center_x, 0.0, 0.0]) - center
    if np.dot(avg_n, to_center) > 0:
        avg_n = -avg_n

    radius = float(np.sqrt(total_area / np.pi))
    return {'center': center, 'normal': avg_n,
            'radius': radius, 'area': total_area}


# ─────────────────────────────────────────────────────────────
# 4.  HDF5 OUTPUT
# ─────────────────────────────────────────────────────────────
def _patch_datasets(coords: np.ndarray, tris: np.ndarray):
    """
    Build the four datasets stored per boundary patch.

    topology    – triangles with local (patch-level) vertex indices
    cellIds     – triangles with global vertex indices
    pointIds    – mapping  local_idx -> global_idx
    coordinates – 3-D positions of patch vertices (local order)
    """
    global_ids = np.unique(tris.ravel()).astype(np.int64)   # sorted
    g2l        = {g: l for l, g in enumerate(global_ids)}
    local_topo = np.array([[g2l[v] for v in tri] for tri in tris],
                           dtype=np.int64)
    cell_ids   = tris.astype(np.int64)
    bnd_coords = coords[global_ids]
    return local_topo, cell_ids, global_ids, bnd_coords


def _vertex_normals(coords: np.ndarray, tris: np.ndarray,
                    global_ids: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals for a surface patch."""
    v0, v1, v2 = coords[tris[:, 0]], coords[tris[:, 1]], coords[tris[:, 2]]
    face_n = np.cross(v1 - v0, v2 - v0)   # area-weighted face normals

    g2l = {g: l for l, g in enumerate(global_ids)}
    vn  = np.zeros((len(global_ids), 3), dtype=np.float64)
    for i, tri in enumerate(tris):
        for v in tri:
            vn[g2l[v]] += face_n[i]
    norms = np.linalg.norm(vn, axis=1, keepdims=True)
    vn   /= norms + 1e-15
    return vn.astype(np.float32)


def save_h5(coords: np.ndarray, tets: np.ndarray,
            wall_tris: np.ndarray,
            inlet_tris: np.ndarray,
            outlet_tris: np.ndarray,
            out_path: str) -> None:
    """Save volumetric mesh to HDF5 matching the reference layout."""
    with h5py.File(out_path, 'w') as f:
        grp = f.create_group('Mesh')
        grp.create_dataset('coordinates', data=coords.astype(np.float64))
        grp.create_dataset('topology',    data=tets.astype(np.int64))

        patches = {
            'Wall': (wall_tris,   True),
            'ID_1': (inlet_tris,  False),
            'ID_2': (outlet_tris, False),
        }
        for name, (tris, with_normals) in patches.items():
            local_topo, cell_ids, point_ids, bnd_coords = \
                _patch_datasets(coords, tris)
            pg = grp.create_group(name)
            pg.create_dataset('coordinates', data=bnd_coords.astype(np.float64))
            pg.create_dataset('topology',    data=local_topo.astype(np.int64))
            pg.create_dataset('pointIds',    data=point_ids.astype(np.int64))
            pg.create_dataset('cellIds',     data=cell_ids.astype(np.int64))
            if with_normals:
                pg.create_dataset('normal',
                                  data=_vertex_normals(coords, tris, point_ids))
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 5.  DOLFIN XML.GZ OUTPUT
# ─────────────────────────────────────────────────────────────
def _tet_face_lookup(tets: np.ndarray) -> dict:
    """
    Build: sorted-3-tuple -> (tet_index, local_face_index)

    DOLFIN local-facet convention for tet (v0,v1,v2,v3):
      facet 0: opposite v0 -> vertices (v1,v2,v3)
      facet 1: opposite v1 -> vertices (v0,v2,v3)
      facet 2: opposite v2 -> vertices (v0,v1,v3)
      facet 3: opposite v3 -> vertices (v0,v1,v2)
    """
    n = len(tets)
    local_verts = [(1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2)]

    # Build all faces as (n*4, 3), then sort each row for canonical form
    all_faces = np.empty((n * 4, 3), dtype=np.int64)
    for lf, (a, b, c) in enumerate(local_verts):
        all_faces[lf::4, 0] = tets[:, a]
        all_faces[lf::4, 1] = tets[:, b]
        all_faces[lf::4, 2] = tets[:, c]
    all_faces_sorted = np.sort(all_faces, axis=1)

    tet_idx   = np.repeat(np.arange(n, dtype=np.int64), 4)
    local_idx = np.tile(np.arange(4,  dtype=np.int64), n)

    lookup = {}
    for i in range(len(all_faces_sorted)):
        key = (int(all_faces_sorted[i, 0]),
               int(all_faces_sorted[i, 1]),
               int(all_faces_sorted[i, 2]))
        lookup[key] = (int(tet_idx[i]), int(local_idx[i]))
    return lookup


def save_xml_gz(coords: np.ndarray, tets: np.ndarray,
                wall_tris: np.ndarray,
                inlet_tris: np.ndarray,
                outlet_tris: np.ndarray,
                out_path: str) -> None:
    """
    Save FEniCS/DOLFIN XML mesh (gzip-compressed).
    Boundary markers: wall=0, inlet(ID_1)=1, outlet(ID_2)=2
    """
    print("  Building tet-face lookup ... (may take a moment)")
    lookup = _tet_face_lookup(tets)

    # Collect (cell_index, local_entity, marker) for every boundary face
    face_markers = []
    for tris, marker in [(wall_tris, 0), (inlet_tris, 1), (outlet_tris, 2)]:
        sorted_tris = np.sort(tris, axis=1)
        for row in sorted_tris:
            key = (int(row[0]), int(row[1]), int(row[2]))
            hit = lookup.get(key)
            if hit is not None:
                face_markers.append((*hit, marker))
            else:
                print(f"    Warning: boundary tri {key} not found in tet faces")

    n_bnd = len(face_markers)
    print(f"  Boundary faces tagged: {n_bnd:,}")

    with gzip.open(out_path, 'wt', compresslevel=6) as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<dolfin xmlns:dolfin="http://www.fenicsproject.org">\n')
        f.write('  <mesh celltype="tetrahedron" dim="3">\n')

        f.write(f'    <vertices size="{len(coords)}">\n')
        for i, (x, y, z) in enumerate(coords):
            f.write(f'      <vertex index="{i}" x="{x:.7g}" '
                    f'y="{y:.7g}" z="{z:.7g}" />\n')
        f.write('    </vertices>\n')

        f.write(f'    <cells size="{len(tets)}">\n')
        for i, (v0, v1, v2, v3) in enumerate(tets):
            f.write(f'      <tetrahedron index="{i}" '
                    f'v0="{v0}" v1="{v1}" v2="{v2}" v3="{v3}" />\n')
        f.write('    </cells>\n')

        f.write('    <domains>\n')
        f.write(f'      <mesh_value_collection type="uint" dim="2" '
                f'size="{n_bnd}">\n')
        for ci, lf, val in face_markers:
            f.write(f'        <value cell_index="{ci}" '
                    f'local_entity="{lf}" value="{val}" />\n')
        f.write('      </mesh_value_collection>\n')
        f.write('    </domains>\n')
        f.write('  </mesh>\n')
        f.write('</dolfin>\n')

    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 6.  VTU OUTPUT
# ─────────────────────────────────────────────────────────────
def save_vtu(coords: np.ndarray, tets: np.ndarray,
             wall_tris: np.ndarray,
             inlet_tris: np.ndarray,
             outlet_tris: np.ndarray,
             out_path: str) -> None:
    """
    Save the volumetric mesh as an unstructured VTU file for ParaView.

    A cell-data array 'BoundaryMarker' is attached to every cell:
      interior tetrahedra  = -1
      wall triangles       =  0
      inlet triangles      =  1
      outlet triangles     =  2
    This lets you colour/filter patches directly in ParaView.
    """
    import pyvista as pv

    # ── Volumetric tets ──────────────────────────────────────
    # pyvista UnstructuredGrid cell array: [n_pts, v0, v1, v2, v3, ...]
    n_tets = len(tets)
    tet_cells  = np.hstack([np.full((n_tets, 1), 4, dtype=np.int64), tets])
    tet_types  = np.full(n_tets, 10, dtype=np.uint8)   # VTK_TETRA = 10
    tet_marker = np.full(n_tets, -1, dtype=np.int32)   # interior

    # ── Boundary triangles ───────────────────────────────────
    patches = [(wall_tris, 0), (inlet_tris, 1), (outlet_tris, 2)]
    tri_cells_list, tri_types_list, tri_markers_list = [], [], []
    for tris, marker in patches:
        n = len(tris)
        tri_cells_list.append(
            np.hstack([np.full((n, 1), 3, dtype=np.int64), tris]))
        tri_types_list.append(np.full(n, 5, dtype=np.uint8))  # VTK_TRIANGLE = 5
        tri_markers_list.append(np.full(n, marker, dtype=np.int32))

    all_cells_flat = np.concatenate(
        [tet_cells.ravel()] + [c.ravel() for c in tri_cells_list])
    all_markers = np.concatenate([tet_marker] + tri_markers_list)

    grid = pv.UnstructuredGrid(all_cells_flat, all_types, coords)
    grid.cell_data['BoundaryMarker'] = all_markers

    grid.save(out_path)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 7.  INFO FILE OUTPUT
# ─────────────────────────────────────────────────────────────
def save_info(model_name: str,
              inlets: list,
              outlets: list,
              out_path: str) -> None:
    """
    Save .info file matching the reference format.

    Each inlet/outlet dict must contain:
        id, name, center (3,), normal (3,), radius, area
    Optional keys: fr (inlet flow-rate ratio), ar (outlet area ratio).
    """
    def fmt_vec(v):
        return f"({v[0]:.12f},{v[1]:.12f},{v[2]:.12f})"

    lines = [
        "# id, wave, center, normal, radius, area, FR(inlet)/AR(outlet)",
        "",
        model_name,
        "",
        "<INLETS>",
    ]
    for s in inlets:
        lines.append(
            f"{s['id']}   {s['name']}   "
            f"{fmt_vec(s['center'])}   {fmt_vec(s['normal'])}   "
            f"{s['radius']:.12f}   {s['area']:.12f}   "
            f"{s.get('fr', 1.0):.12f}"
        )
    lines += ["", "<OUTLETS>"]
    for s in outlets:
        lines.append(
            f"{s['id']}   {s['name']}   "
            f"{fmt_vec(s['center'])}   {fmt_vec(s['normal'])}   "
            f"{s['radius']:.12f}   {s['area']:.12f}   "
            f"{s.get('ar', 1.0):.12f}"
        )
    lines.append("")

    with open(out_path, 'w') as fh:
        fh.write('\n'.join(lines))
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────────────────────


if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prefix = os.path.join(OUTPUT_DIR, MODEL_NAME)

    # ── 1. Volumetric meshing ─────────────────────────────────
    print("[1/5] Generating tetrahedral mesh with gmsh ...")
    coords, tets, tag_to_tris = mesh_stl(STL_PATH, MESH_SIZE, ANGLE_DEG,
                                          N_BOUNDARY_LAYERS, BL_SIZE_FACTOR, BL_RATIO)

    # ── 2. Classify boundary surfaces ────────────────────────
    print("[2/5] Classifying boundary surfaces ...")
    surface_info = classify_surfaces(coords, tag_to_tris, AXIS_THRESH)
    for stag, si in surface_info.items():
        print(f"  tag={stag:3d}  {si['label']:8s}  "
              f"centroid_x={si['centroid'][0]:+.3f}  "
              f"avg_nx={si['normal'][0]:+.3f}  "
              f"area={si['area']:.4f}  "
              f"tris={len(tag_to_tris[stag]):,}")

    wall_tris   = merge_patch_triangles(coords, tag_to_tris, surface_info, 'wall')
    inlet_tris  = merge_patch_triangles(coords, tag_to_tris, surface_info, 'inlet')
    outlet_tris = merge_patch_triangles(coords, tag_to_tris, surface_info, 'outlet')
    print(f"  Wall: {len(wall_tris):,} tris  |  "
          f"Inlet: {len(inlet_tris):,} tris  |  "
          f"Outlet: {len(outlet_tris):,} tris")

    # Estimate domain centre for outward-normal convention
    dom_cx = float((coords[:, 0].min() + coords[:, 0].max()) / 2.0)
    inlet_stats  = cap_statistics(coords, inlet_tris,  dom_cx)
    outlet_stats = cap_statistics(coords, outlet_tris, dom_cx)

    # ── 3. Save HDF5 ─────────────────────────────────────────
    print("[3/6] Writing HDF5 ...")
    save_h5(coords, tets, wall_tris, inlet_tris, outlet_tris,
            f"{prefix}.h5")

    # ── 4. Save DOLFIN XML.gz ────────────────────────────────
    print("[4/6] Writing XML.gz ...")
    save_xml_gz(coords, tets, wall_tris, inlet_tris, outlet_tris,
                f"{prefix}.xml.gz")

    # ── 5. Save VTU ──────────────────────────────────────────
    print("[5/6] Writing VTU ...")
    save_vtu(coords, tets, wall_tris, inlet_tris, outlet_tris,
             f"{prefix}.vtu")

    # ── 6. Save .info ────────────────────────────────────────
    print("[6/6] Writing .info ...")
    inlets = [{
        'id':     1,
        'name':   'INLET',
        'center': inlet_stats['center'],
        'normal': inlet_stats['normal'],
        'radius': inlet_stats['radius'],
        'area':   inlet_stats['area'],
        'fr':     1.0,
    }]
    outlets = [{
        'id':     2,
        'name':   'None',
        'center': outlet_stats['center'],
        'normal': outlet_stats['normal'],
        'radius': outlet_stats['radius'],
        'area':   outlet_stats['area'],
        'ar':     1.0,
    }]
    save_info(MODEL_NAME, inlets, outlets, f"{prefix}.info")

    print(f"\n  Done.  Output written to {OUTPUT_DIR}/")
    print(f"    {MODEL_NAME}.h5")
    print(f"    {MODEL_NAME}.xml.gz")
    print(f"    {MODEL_NAME}.vtu")
    print(f"    {MODEL_NAME}.info")
