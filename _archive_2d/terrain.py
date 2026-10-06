"""สร้างแบบจำลองภูมิประเทศ (DEM แบบกริด) จากเส้นแม่น้ำ + หน้าตัด สำหรับแบบจำลอง 2 มิติ"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

from .geometry import River, Section


@dataclass
class Terrain:
    z: np.ndarray          # ระดับพื้น (ny, nx)
    n: np.ndarray          # สัมประสิทธิ์แมนนิ่งรายเซลล์
    wall: np.ndarray       # เซลล์นอกโดเมน (ปิด)
    inflow: np.ndarray     # เซลล์ที่ป้อนน้ำเข้า
    outflow: np.ndarray    # เซลล์ทางออก (ไหลออกแบบความลึกปกติ)
    idx: np.ndarray        # ดัชนีจุดบนแนวร่องน้ำที่ใกล้ที่สุด
    dist: np.ndarray       # ระยะห่างจากแนวร่องน้ำ (ม.)
    lat: np.ndarray        # ระยะตามแนวขวางแบบมีเครื่องหมาย (+ = ฝั่งซ้ายเมื่อมองตามน้ำ)
    zc: np.ndarray         # ระดับท้องน้ำของจุดแนวร่องน้ำที่ใกล้ที่สุด
    channel: np.ndarray    # เซลล์ร่องน้ำ (ภายในตลิ่ง)
    dx: float
    x0: float
    y0: float
    S_out: float
    river_ix: np.ndarray   # ตำแหน่งเซลล์ของแต่ละจุดบนแนวร่องน้ำ
    river_iy: np.ndarray
    wall_slope: float

    @property
    def shape(self):
        return self.z.shape

    @property
    def extent(self):
        ny, nx = self.z.shape
        return (self.x0, self.x0 + nx * self.dx, self.y0, self.y0 + ny * self.dx)

    def cell_of(self, x, y):
        ny, nx = self.z.shape
        ix = np.clip(((np.asarray(x) - self.x0) / self.dx).astype(int), 0, nx - 1)
        iy = np.clip(((np.asarray(y) - self.y0) / self.dx).astype(int), 0, ny - 1)
        return ix, iy


def plan_grid(river: River, sec: Section, cells_per_B: float, max_cells: int, hmax: float,
              wall_slope: float = 0.3):
    """คืนค่า (dx, margin, nx, ny) — ขยาย dx อัตโนมัติถ้าเซลล์เกินขีดจำกัด"""
    dx = sec.B / max(cells_per_B, 1.0)
    for _ in range(30):
        margin = max(6 * dx, 1.5 * max(hmax - sec.Hb, 0.5) / wall_slope + 3 * dx)
        half = sec.Tb / 2 + sec.Bf + margin
        W = river.x.max() - river.x.min() + 2 * half
        H = river.y.max() - river.y.min() + 2 * half
        nx, ny = int(np.ceil(W / dx)), int(np.ceil(H / dx))
        if nx * ny <= max_cells:
            break
        dx *= 1.05 * np.sqrt(nx * ny / max_cells)
    return dx, margin, nx, ny


def build_terrain(river: River, sec: Section, cells_per_B=6.0, max_cells=2_000_000,
                  hmax=5.0, wall_slope=0.3) -> Terrain:
    dx, margin, nx, ny = plan_grid(river, sec, cells_per_B, max_cells, hmax, wall_slope)
    half = sec.Tb / 2 + sec.Bf + margin
    x0 = river.x.min() - half
    y0 = river.y.min() - half

    # แนวร่องน้ำแบบละเอียด (ระยะห่าง ≤ dx/2) เพื่อหาจุดใกล้ที่สุดได้แม่น
    if river.ds > dx / 2:
        m = int(np.ceil(river.L / (dx / 2))) + 1
        sd = np.linspace(0, river.L, m)
    else:
        sd = river.s
    xd = np.interp(sd, river.s, river.x)
    yd = np.interp(sd, river.s, river.y)
    th = np.interp(sd, river.s, river.theta)
    node = np.clip(np.round(sd / river.ds).astype(int), 0, river.N - 1)

    X = x0 + (np.arange(nx) + 0.5) * dx
    Y = y0 + (np.arange(ny) + 0.5) * dx
    XX, YY = np.meshgrid(X, Y)
    tree = cKDTree(np.column_stack([xd, yd]))
    dist, j = tree.query(np.column_stack([XX.ravel(), YY.ravel()]), workers=-1)
    dist = dist.reshape(ny, nx)
    j = j.reshape(ny, nx)
    along = (XX - xd[j]) * np.cos(th[j]) + (YY - yd[j]) * np.sin(th[j])
    lat = -(XX - xd[j]) * np.sin(th[j]) + (YY - yd[j]) * np.cos(th[j])
    beyond = ((j == 0) & (along < -0.5 * dx)) | ((j == len(sd) - 1) & (along > 0.5 * dx))

    idx = node[j]
    zc = river.zb[idx]
    B2, Tb2 = sec.B / 2, sec.Tb / 2
    fp_edge = Tb2 + sec.Bf
    bank_top = gaussian_filter(zc + sec.Hb, sigma=max(1.0, sec.Tb / dx))
    if sec.z > 0:
        side = zc + np.minimum((dist - B2) / sec.z, sec.Hb)
    else:
        side = zc + sec.Hb
    z = np.where(dist <= B2, zc,
                 np.where(dist <= Tb2, side,
                          np.where(dist <= fp_edge, bank_top,
                                   bank_top + (dist - fp_edge) * wall_slope)))
    zmax = float(z[~beyond].max()) if (~beyond).any() else float(z.max())
    z = np.where(beyond, zmax + 50.0, z)

    channel = (dist <= Tb2) & ~beyond
    n = np.where(channel, sec.n, sec.nf)
    s_cell = river.s[idx]
    inflow = (s_cell <= max(2 * dx, river.ds)) & (dist <= B2 + 0.5 * dx) & ~beyond
    if not inflow.any():
        inflow = (j == 0) & (dist <= dist[j == 0].min() + dx)
    outflow = (s_cell >= river.L - 2 * dx) & ~beyond & (dist <= fp_edge + margin)
    rix = np.clip(((river.x - x0) / dx).astype(int), 0, nx - 1)
    riy = np.clip(((river.y - y0) / dx).astype(int), 0, ny - 1)
    return Terrain(z=z.astype(np.float32), n=n.astype(np.float32), wall=beyond, inflow=inflow,
                   outflow=outflow, idx=idx.astype(np.int32), dist=dist.astype(np.float32), lat=lat.astype(np.float32),
                   zc=zc.astype(np.float32), channel=channel, dx=float(dx), x0=float(x0), y0=float(y0),
                   S_out=float(max(river.slope[-1], 1e-5)), river_ix=rix, river_iy=riy,
                   wall_slope=wall_slope)
