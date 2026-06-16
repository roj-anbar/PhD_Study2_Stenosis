"""
Eccentric Stenosis Model Generator
===================================
Based on: Varghese, Frankel & Fischer (2005)
"Direct numerical simulation of stenotic flows, Part 1: Steady flow"

Geometry equations from Section 2.1 of the paper:
- Axisymmetric stenosis shape:
    S(x) = (D/2) * [1 - s0*(1 + cos(2*pi*(x - x0)/L))]
    y = S(x)*cos(theta), z = S(x)*sin(theta)

- Eccentric offset (added in the x-z plane, y=0):
    E(x) = (s0/10) * (1 + cos(2*pi*(x - x0)/L))
    y = S(x)*cos(theta), z = E(x) + S(x)*sin(theta)

Parameters (matching the paper):
- D = 1.0 (vessel diameter, normalizing length scale)
- s0 = 0.25 (gives 75% area reduction at throat)
- L = 2D (stenosis length)
- x0 = 0 (stenosis center)
- Upstream length: 3D from throat
- Downstream length: 16D from throat
- Eccentricity offset: 0.05D at throat
"""

import numpy as np
from stl import mesh as stl_mesh
import pyvista as pv
import os


OUTPUT_DIR = '/Users/rojin/Library/CloudStorage/OneDrive-UniversityofToronto/Education/PhD/My_Projects/Study2_stenosis/models/eccentricStenosis'

# ─────────────────────────────────────────────
# 1.  PARAMETERS  (all lengths normalised by D)
# ─────────────────────────────────────────────
D   = 10.0          # vessel diameter
R   = D / 2.0      # vessel radius
s0  = 0.25         # stenosis severity  → 75 % area reduction
L   = 2.0 * D      # stenosis axial length
x0  = 0.0          # stenosis centre

x_upstream   = -3.0 * D  # upstream inlet  (3D from throat)
x_downstream = 16.0 * D  # downstream exit (16D from throat)

# Discretisation
N_theta = 128        # circumferential points
N_x     = 400       # axial points
N_r     = 20        # radial layers for volumetric mesh


# ─────────────────────────────────────────────
# 2.  STENOSIS PROFILE FUNCTIONS
# ─────────────────────────────────────────────
def in_stenosis(x):
    """Returns True if x is within the stenosis region."""
    return (x >= x0 - L/2) & (x <= x0 + L/2)

def S(x):
    """
    Wall radius at axial position x.
    Equation (2.1): S(x) = (D/2)*[1 - s0*(1 + cos(2π(x-x0)/L))]
    Outside the stenosis region S(x) = R (full vessel radius).
    """
    r = np.full_like(np.asarray(x, dtype=float), R) # initiate the radius array as equal to R
    mask_stenosis = in_stenosis(np.asarray(x, dtype=float))
    r[mask_stenosis] = (D/2.0) * (1.0 - s0 * (1.0 + np.cos(2*np.pi*(x[mask_stenosis] - x0)/L)))
    return r

def E(x):
    """
    Eccentricity offset in the z-direction.
    Equation (2.2): E(x) = (s0/10)*(1 + cos(2π(x-x0)/L))
    Applied only inside the stenosis region.
    """
    e = np.zeros_like(np.asarray(x, dtype=float))
    mask_stenosis = in_stenosis(np.asarray(x, dtype=float))
    e[mask_stenosis] = (s0 / 10.0) * (1.0 + np.cos(2*np.pi*(x[mask_stenosis] - x0)/L))
    return e


# ─────────────────────────────────────────────
# 3.  GENERATE SURFACE MESH (wall only)
# ─────────────────────────────────────────────
def build_wall_surface(N_x=N_x, N_theta=N_theta, eccentric=True):
    """
    Build the cylindrical wall surface of the stenosed tube.
    Returns arrays X, Y, Z of shape (N_x, N_theta).
    """
    x_arr     = np.linspace(x_upstream, x_downstream, N_x)
    theta_arr = np.linspace(0, 2*np.pi, N_theta, endpoint=False)

    X = np.zeros((N_x, N_theta))
    Y = np.zeros((N_x, N_theta))
    Z = np.zeros((N_x, N_theta))

    s_vals = S(x_arr)
    e_vals = E(x_arr) if eccentric else np.zeros_like(x_arr)

    for i, xi in enumerate(x_arr):
        for j, th in enumerate(theta_arr):
            X[i, j] = xi
            Y[i, j] = s_vals[i] * np.cos(th)          # eq 2.2: y = S(x)*cos(θ)
            Z[i, j] = e_vals[i] + s_vals[i] * np.sin(th)  # eq 2.2: z = E(x) + S(x)*sin(θ)

    return X, Y, Z, x_arr, theta_arr


# ─────────────────────────────────────────────
# 4.  EXPORT SURFACE AS STL
# ─────────────────────────────────────────────
def surface_to_stl(X, Y, Z, filename):
    """
    Convert structured grid (N_x × N_theta) wall + two end-caps to STL.
    Each quad cell → 2 triangles.
    """
    Nx, Nt = X.shape
    triangles = []

    # ── lateral (cylindrical) wall ──
    for i in range(Nx - 1):
        for j in range(Nt):
            jn = (j + 1) % Nt  # wrap around circumference
            p00 = np.array([X[i,   j ], Y[i,   j ], Z[i,   j ]])
            p10 = np.array([X[i+1, j ], Y[i+1, j ], Z[i+1, j ]])
            p01 = np.array([X[i,   jn], Y[i,   jn], Z[i,   jn]])
            p11 = np.array([X[i+1, jn], Y[i+1, jn], Z[i+1, jn]])
            triangles.append([p00, p10, p01])
            triangles.append([p10, p11, p01])

    # ── inlet cap (x = x_upstream) ──
    cx_in = np.mean(X[0,  :])
    cy_in = np.mean(Y[0,  :])
    cz_in = np.mean(Z[0,  :])
    centroid_in = np.array([cx_in, cy_in, cz_in])
    for j in range(Nt):
        jn = (j + 1) % Nt
        triangles.append([centroid_in,
                          np.array([X[0, jn], Y[0, jn], Z[0, jn]]),
                          np.array([X[0, j ], Y[0, j ], Z[0, j ]])])

    # ── outlet cap (x = x_downstream) ──
    cx_out = np.mean(X[-1, :])
    cy_out = np.mean(Y[-1, :])
    cz_out = np.mean(Z[-1, :])
    centroid_out = np.array([cx_out, cy_out, cz_out])
    for j in range(Nt):
        jn = (j + 1) % Nt
        triangles.append([centroid_out,
                          np.array([X[-1, j ], Y[-1, j ], Z[-1, j ]]),
                          np.array([X[-1, jn], Y[-1, jn], Z[-1, jn]])])

    n_tri = len(triangles)
    solid  = stl_mesh.Mesh(np.zeros(n_tri, dtype=stl_mesh.Mesh.dtype))
    for k, tri in enumerate(triangles):
        for v in range(3):
            solid.vectors[k][v] = tri[v]

    solid.save(filename)
    print(f"  STL saved → {filename}  ({n_tri} triangles)")
    return solid


# ─────────────────────────────────────────────
# 5.  VOLUMETRIC MESH (structured hex-like)
# ─────────────────────────────────────────────
def build_volumetric_mesh(N_x=N_x, N_theta=N_theta, N_r=N_r,
                          eccentric=True, filename="eccentric_stenosis_volume.vtk"):
    """
    Build a structured hexahedral volumetric mesh by sweeping radially
    from the vessel centreline out to the wall.

    Grid point (i, j, k):
      i → axial   (0 … N_x-1)
      j → radial  (0 … N_r-1),  0 = centreline, N_r-1 = wall
      k → azimuthal (0 … N_theta-1)
    """
    print("  Building volumetric mesh …")

    x_arr     = np.linspace(x_upstream, x_downstream, N_x)
    theta_arr = np.linspace(0, 2*np.pi, N_theta, endpoint=False)
    r_frac    = np.linspace(0.0, 1.0, N_r)   # 0 = axis, 1 = wall

    # Allocate coordinate arrays: shape (N_x, N_r, N_theta)
    Xv = np.zeros((N_x, N_r, N_theta))
    Yv = np.zeros((N_x, N_r, N_theta))
    Zv = np.zeros((N_x, N_r, N_theta))

    s_vals = S(x_arr)
    e_vals = E(x_arr) if eccentric else np.zeros_like(x_arr)

    for i, xi in enumerate(x_arr):
        for k, th in enumerate(theta_arr):
            for j, rf in enumerate(r_frac):
                # Wall position for this (x, theta)
                y_wall = s_vals[i] * np.cos(th)
                z_wall = e_vals[i] + s_vals[i] * np.sin(th)

                # Centreline shifts with eccentricity offset
                # (centre of the tube cross-section at each x)
                y_ctr = 0.0
                z_ctr = e_vals[i] / 2.0  # linear interp between axis & wall centre

                # Interpolate radially from centreline to wall
                Xv[i, j, k] = xi
                Yv[i, j, k] = y_ctr + rf * (y_wall - y_ctr)
                Zv[i, j, k] = z_ctr + rf * (z_wall - z_ctr)

    # ── PyVista StructuredGrid ──
    # PyVista expects shape (N_x * N_r * N_theta,) flat arrays
    # with grid dimensions (N_x, N_r, N_theta)
    grid = pv.StructuredGrid(Xv, Yv, Zv)
    grid.save(filename)
    print(f"  Volumetric mesh saved → {filename}")
    print(f"    Dimensions : {N_x} × {N_r} × {N_theta}  "
          f"({N_x * N_r * N_theta:,} points, "
          f"{(N_x-1)*(N_r-1)*(N_theta-1):,} hex cells)")
    return grid


# ─────────────────────────────────────────────
# 6.  DIAGNOSTIC PLOTS  (saved as PNG)
# ─────────────────────────────────────────────
def plot_geometry(x_arr, eccentric=True):
    """Save cross-section profile and throat cross-section as PNG."""
    import matplotlib.pyplot as plt

    s_vals = S(x_arr)
    e_vals = E(x_arr) if eccentric else np.zeros_like(x_arr)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Eccentric Stenosis Geometry  —  Varghese et al. (2005)", fontsize=13)

    # ── Side view (x-z plane, y = 0, theta = pi/2 & 3pi/2) ──
    ax = axes[0]
    ax.set_title("Side profile  (x–z plane, y = 0)")
    z_top    =  e_vals + s_vals          # θ = π/2
    z_bottom =  e_vals - s_vals          # θ = 3π/2
    ax.fill_between(x_arr, z_bottom, z_top, alpha=0.25, color="steelblue", label="Fluid domain")
    ax.plot(x_arr,  z_top,    "b-",  lw=1.5, label="Wall (eccentric)")
    ax.plot(x_arr,  z_bottom, "b-",  lw=1.5)
    # Also show axisymmetric for comparison
    ax.plot(x_arr,  s_vals,   "r--", lw=1.0, label="Wall (axisymmetric)")
    ax.plot(x_arr, -s_vals,   "r--", lw=1.0)
    ax.axvline(-L/2, color="gray", ls=":", lw=0.8)
    ax.axvline( L/2, color="gray", ls=":", lw=0.8, label="Stenosis extent")
    ax.axvline(0,    color="k",    ls="--", lw=0.8, label="Throat (x=0)")
    ax.set_xlabel("x / D")
    ax.set_ylabel("z / D")
    ax.set_xlim(x_upstream, x_downstream)
    ax.set_ylim(-0.65, 0.65)
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, ls=":", alpha=0.4)

    # ── Cross-sections at throat ──
    ax2 = axes[1]
    ax2.set_title("Cross-section at throat  (x = 0)")
    theta_fine = np.linspace(0, 2*np.pi, 360)
    x_th = np.zeros_like(theta_fine)     # x = 0
    s_th = float(S(np.array([0.0]))[0])
    e_th = float(E(np.array([0.0]))[0]) if eccentric else 0.0
    y_axi =  s_th * np.cos(theta_fine)
    z_axi =  s_th * np.sin(theta_fine)
    y_ecc =  s_th * np.cos(theta_fine)
    z_ecc =  e_th + s_th * np.sin(theta_fine)
    # Upstream full vessel
    y_full = R * np.cos(theta_fine)
    z_full = R * np.sin(theta_fine)

    ax2.plot(y_full, z_full, "k-",  lw=1.5, label=f"Upstream vessel (R = {R:.2f}D)")
    ax2.plot(y_axi,  z_axi,  "r--", lw=1.5, label=f"Axisymmetric throat (R_t = {s_th:.3f}D)")
    ax2.plot(y_ecc,  z_ecc,  "b-",  lw=1.5, label=f"Eccentric throat (offset = {e_th:.3f}D)")
    ax2.axhline(0, color="gray", ls=":", lw=0.6)
    ax2.axvline(0, color="gray", ls=":", lw=0.6)
    ax2.set_xlabel("y / D")
    ax2.set_ylabel("z / D")
    ax2.set_aspect("equal")
    ax2.legend(fontsize=8)
    ax2.grid(True, ls=":", alpha=0.4)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/stenosis_geometry.png", dpi=150)
    plt.close()
    print("  Profile plot saved → stenosis_geometry.png")


def plot_stenosis_profile(x_arr):
    """Plot S(x), E(x), area reduction along the axis."""
    import matplotlib.pyplot as plt

    s_vals = S(x_arr)
    e_vals = E(x_arr)
    area_reduction = 1.0 - (s_vals / R)**2   # fractional area reduction

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Stenosis Profile Functions  —  Varghese et al. (2005)", fontsize=13)

    axes[0].plot(x_arr, s_vals, "b-", lw=1.5)
    axes[0].axhline(R,    color="k", ls="--", lw=0.8, label="Full radius R")
    axes[0].axhline(R/2,  color="r", ls="--", lw=0.8, label="Throat radius (75% area)")
    axes[0].set_ylabel("S(x) / D")
    axes[0].set_title("Wall radius profile S(x)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, ls=":", alpha=0.4)

    axes[1].plot(x_arr, e_vals, "g-", lw=1.5)
    axes[1].set_ylabel("E(x) / D")
    axes[1].set_title("Eccentricity offset E(x)  (z-direction)")
    axes[1].grid(True, ls=":", alpha=0.4)

    axes[2].plot(x_arr, area_reduction * 100, "r-", lw=1.5)
    axes[2].axhline(75, color="k", ls="--", lw=0.8, label="75% area reduction")
    axes[2].set_ylabel("Area reduction (%)")
    axes[2].set_xlabel("x / D")
    axes[2].set_title("Local cross-sectional area reduction")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, ls=":", alpha=0.4)

    # Mark stenosis extent
    for ax in axes:
        ax.axvspan(-L/2, L/2, alpha=0.08, color="orange", label="Stenosis region")
        ax.axvline(0, color="gray", ls=":", lw=0.8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/stenosis_profiles.png", dpi=150)
    plt.close()
    print("  Profile functions plot saved → stenosis_profiles.png")


# ─────────────────────────────────────────────
# 7.  VERIFY GEOMETRY PARAMETERS
# ─────────────────────────────────────────────
def print_geometry_summary():
    """Print key geometric quantities to verify against the paper."""
    x_throat = np.array([0.0])
    s_throat = float(S(x_throat)[0])
    e_throat = float(E(x_throat)[0])
    area_full   = np.pi * R**2
    area_throat = np.pi * s_throat**2
    area_red    = (1 - area_throat / area_full) * 100

    print("\n" + "="*55)
    print("  GEOMETRY SUMMARY  (lengths in units of D)")
    print("="*55)
    print(f"  Vessel diameter D          : {D:.3f}")
    print(f"  Vessel radius R            : {R:.3f}")
    print(f"  Stenosis parameter s0      : {s0:.3f}")
    print(f"  Stenosis length L          : {L:.3f} D")
    print(f"  Upstream section           : {abs(x_upstream):.1f} D")
    print(f"  Downstream section         : {x_downstream:.1f} D")
    print(f"  Total tube length          : {x_downstream - x_upstream:.1f} D")
    print(f"  --- At throat (x = 0) ---")
    print(f"  Stenosis radius S(0)       : {s_throat:.4f} D")
    print(f"  Eccentricity offset E(0)   : {e_throat:.4f} D  "
          f"(paper: {s0/10 * 2:.4f} D)")
    print(f"  Area reduction             : {area_red:.1f}%  (paper: 75%)")
    print("="*55 + "\n")


# ─────────────────────────────────────────────
# 8.  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print_geometry_summary()

    x_arr = np.linspace(x_upstream, x_downstream, N_x)

    # ── Diagnostic plots ──
    print("[1/4] Generating diagnostic plots …")
    plot_geometry(x_arr, eccentric=True)
    plot_stenosis_profile(x_arr)

    # ── Surface mesh → STL ──
    print("[2/4] Building wall surface …")
    X, Y, Z, x_arr, theta_arr = build_wall_surface(N_x=N_x, N_theta=N_theta, eccentric=True)

    print("[3/4] Exporting STL …")
    surface_to_stl(X, Y, Z, f"{OUTPUT_DIR}/eccentric_stenosis.stl")

    # ── Volumetric mesh → VTK ──
    print("[4/4] Building volumetric mesh …")
    grid = build_volumetric_mesh(
        N_x=N_x, N_theta=N_theta, N_r=N_r,
        eccentric=True,
        filename=f"{OUTPUT_DIR}/eccentric_stenosis_volume.vtk"
    )

    print("\n✓  All done.  Output files:")
    for file in ["stenosis_geometry.png",
              "stenosis_profiles.png",
              "eccentric_stenosis.stl",
              "eccentric_stenosis_volume.vtk"]:
        path = f"{OUTPUT_DIR}/{file}"
        size = os.path.getsize(path) / 1024

