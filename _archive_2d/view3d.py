"""มุมมอง 3 มิติของลำน้ำ: ภูมิประเทศ + ผิวน้ำจากแบบจำลอง 2D และส่งออกโมเดล 3D (.obj)

วาดพื้นดินและผิวน้ำเป็นพื้นผิวเดียวกัน (matplotlib เรียงลำดับการวาดได้ถูกต้อง)
และวาด 'ผนังตัด' รอบขอบพื้นที่ เพื่อให้เห็นความลึกน้ำแบบหน้าตัดในภาพ 3 มิติ
"""
from __future__ import annotations

import os

import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap, Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .geometry import River, Section
from .terrain import Terrain

LAND_CMAP = LinearSegmentedColormap.from_list("land3d", ["#b89c66", "#a3ad63", "#7c9650", "#8f7752", "#d8cfc0"])
SOIL = (0.50, 0.38, 0.24, 1.0)
WATER_SIDE = (0.16, 0.42, 0.80, 0.88)


def extract_region(ter: Terrain, river: River, sec: Section, h: np.ndarray, field: np.ndarray,
                   station: int, span_B: float, full: bool, max_n: int = 170) -> dict:
    """ตัดพื้นที่รอบจุดที่เลือก (หรือทั้งลำน้ำ) และลดความละเอียดให้วาด 3D ได้เร็ว"""
    ny, nx = ter.shape
    if full:
        i0, i1, j0, j1 = 0, ny, 0, nx
    else:
        half = max(span_B * sec.B / 2, sec.Tb)
        ix, iy = ter.cell_of(river.x[station], river.y[station])
        r = int(np.ceil(half / ter.dx))
        i0, i1 = max(int(iy) - r, 0), min(int(iy) + r + 1, ny)
        j0, j1 = max(int(ix) - r, 0), min(int(ix) + r + 1, nx)
    st = max(1, int(np.ceil(max(i1 - i0, j1 - j0) / max_n)))
    sl = (slice(i0, i1, st), slice(j0, j1, st))
    wall = ter.wall[sl]
    z = np.where(wall, np.nan, ter.z[sl].astype(float))
    hh = np.where(wall, 0.0, h[sl].astype(float))
    wet = (hh > 0.02) & ~wall
    eta = z + hh
    # ตัดผนังหุบเขาที่สูงเกินไปออก เพื่อให้เห็นลำน้ำชัด
    bank_top = np.nanmax(np.where(ter.channel[sl], z, np.nan)) if ter.channel[sl].any() else np.nanmax(z)
    ref = max(np.nanmax(eta[wet]) if wet.any() else bank_top, bank_top)
    zcap = ref + max(0.4 * sec.Hb, 0.5)
    z = np.where(np.isnan(z), zcap, np.minimum(z, zcap))
    X = ter.x0 + (np.arange(j0, j1, st) + 0.5) * ter.dx
    Y = ter.y0 + (np.arange(i0, i1, st) + 0.5) * ter.dx
    XX, YY = np.meshgrid(X, Y)
    return dict(X=XX, Y=YY, z=z, eta=np.where(wet, eta, z), h=hh, wet=wet, field=field[sl].astype(float),
                step=st * ter.dx, full=full)


def auto_ve(reg: dict) -> float:
    X, Y = reg["X"], reg["Y"]
    span = max(X.max() - X.min(), Y.max() - Y.min())
    relief = max(np.nanmax(reg["eta"]) - np.nanmin(reg["z"]), 1e-3)
    return float(np.clip(0.12 * span / relief, 1.0, 200.0))


def _edge_panels(xs, ys, zg, top, wet, base):
    """ผนังตัดตามขอบ: ดิน (base→พื้น) และน้ำ (พื้น→ผิวน้ำ)"""
    soil, water = [], []
    for k in range(len(xs) - 1):
        a, b = (xs[k], ys[k]), (xs[k + 1], ys[k + 1])
        soil.append([(a[0], a[1], base), (b[0], b[1], base), (b[0], b[1], zg[k + 1]), (a[0], a[1], zg[k])])
        if wet[k] and wet[k + 1]:
            water.append([(a[0], a[1], zg[k]), (b[0], b[1], zg[k + 1]), (b[0], b[1], top[k + 1]), (a[0], a[1], top[k])])
    return soil, water


def draw_3d(ax, reg: dict, ve: float, cmap, norm, title: str) -> float:
    """วาดภาพ 3 มิติ คืนค่าอัตราขยายแนวตั้งที่ใช้จริง (ve<=0 = อัตโนมัติ)"""
    ax.clear()
    X, Y, z, eta, wet = reg["X"], reg["Y"], reg["z"], reg["eta"], reg["wet"]
    if ve <= 0:
        ve = auto_ve(reg)
    zmin = float(np.nanmin(z))
    base = -0.15 * max(np.nanmax(eta) - zmin, 0.5) * ve
    Zg = (z - zmin) * ve
    Zt = (np.where(wet, np.maximum(eta, z), z) - zmin) * ve

    ls = LightSource(azdeg=315, altdeg=45)
    shade = ls.hillshade(Zt, dx=reg["step"], dy=reg["step"])
    cols = LAND_CMAP(Normalize(np.nanmin(z), np.nanmax(z))(z))
    wcol = cmap(norm(np.nan_to_num(reg["field"])))
    cols = np.where(wet[..., None], wcol, cols)
    cols[..., :3] *= (0.6 + 0.4 * shade[..., None])
    ax.plot_surface(X, Y, Zt, facecolors=cols, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)

    soil, water = [], []
    for xs, ys, zg, zt, w in ((X[0], Y[0], Zg[0], Zt[0], wet[0]), (X[-1], Y[-1], Zg[-1], Zt[-1], wet[-1]),
                              (X[:, 0], Y[:, 0], Zg[:, 0], Zt[:, 0], wet[:, 0]),
                              (X[:, -1], Y[:, -1], Zg[:, -1], Zt[:, -1], wet[:, -1])):
        s, wp = _edge_panels(xs, ys, zg, zt, w, base)
        soil += s
        water += wp
    ax.add_collection3d(Poly3DCollection(soil, facecolors=SOIL, edgecolors="none"))
    if water:
        ax.add_collection3d(Poly3DCollection(water, facecolors=WATER_SIDE, edgecolors="none"))

    dxr, dyr = X.max() - X.min(), Y.max() - Y.min()
    dzr = max(np.nanmax(Zt) - base, 1e-3)
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(base, base + dzr)
    if reg["full"]:  # ลำน้ำยาวมาก: บีบสัดส่วนแนวยาวเพื่อให้มองเห็นได้
        bx, by = max(dxr, 0.3 * dyr), max(dyr, 0.3 * dxr)
    else:
        bx, by = dxr, dyr
    ax.set_box_aspect((bx, by, max(dzr, 0.06 * max(bx, by))))
    ax.set_xlabel("x (ม.)")
    ax.set_ylabel("y (ม.)")
    ax.set_zlabel(f"ระดับ (ขยาย ×{ve:.0f})")
    ax.set_zticks([])
    ax.set_title(title + ("  [แกนแนวนอนไม่ได้สัดส่วนจริง]" if reg["full"] else ""), fontsize=9)
    return ve


def export_obj(path: str, reg: dict, ve: float = 1.0):
    """บันทึกภูมิประเทศและผิวน้ำเป็นไฟล์ Wavefront OBJ (+ .mtl) เปิดใน Blender / 3D Viewer ได้"""
    X, Y, z, eta, wet = reg["X"], reg["Y"], reg["z"], reg["eta"], reg["wet"]
    ny, nx = X.shape
    x0, y0, z0 = X.min(), Y.min(), float(np.nanmin(z))
    mtl = os.path.splitext(path)[0] + ".mtl"
    verts: list[str] = []

    def grid(Z, name, mask=None):
        base = len(verts)
        for i in range(ny):
            for j in range(nx):
                verts.append(f"v {X[i, j]-x0:.2f} {(Z[i, j]-z0)*ve:.3f} {-(Y[i, j]-y0):.2f}")
        faces = [f"o {name}", f"usemtl {name}"]
        for i in range(ny - 1):
            for j in range(nx - 1):
                if mask is not None and not (mask[i, j] and mask[i + 1, j] and mask[i, j + 1] and mask[i + 1, j + 1]):
                    continue
                a = base + i * nx + j + 1
                faces.append(f"f {a} {a + nx} {a + nx + 1} {a + 1}")
        return faces

    f1 = grid(z, "terrain")
    f2 = grid(eta, "water", wet)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join([f"# Flood Sim 3D export (vertical exaggeration x{ve:g})",
                           f"mtllib {os.path.basename(mtl)}"] + verts + f1 + f2) + "\n")
    with open(mtl, "w", encoding="utf-8") as f:
        f.write("newmtl terrain\nKd 0.62 0.55 0.38\n\nnewmtl water\nKd 0.18 0.45 0.85\nd 0.8\n")
