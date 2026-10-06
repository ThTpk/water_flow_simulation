"""แบบจำลองน้ำท่วม 2 มิติ (local inertial / LISFLOOD-FP, Bates et al. 2010; de Almeida et al. 2012)

ตัวแปรบนกริด:  h (ความลึก) ที่กลางเซลล์,  qx / qy (อัตราการไหลต่อหน่วยกว้าง ม²/วิ) ที่หน้าเซลล์
โมเมนตัม:      q⁺ = [θq + (1-θ)/2(q₋+q₊) − g·h_f·Δt·∂η/∂x] / [1 + g·Δt·n²·|q| / h_f^(7/3)]
มวล:           h⁺ = h + Δt/Δx·(q_in − q_out) + แหล่งน้ำเข้า − ทางออก
ทุกการคำนวณทำผ่าน backend (CuPy/PyTorch บน CUDA หรือ NumPy บน CPU)
"""
from __future__ import annotations

import numpy as np

from .backend import Backend
from .terrain import Terrain

G = 9.81


class Solver2D:
    def __init__(self, ter: Terrain, be: Backend, theta=1.0, fr_max=1.2, hmin=1e-3, cfl=0.6,
                 outlet_rating=None, outlet_bed=None):
        self.ter, self.be = ter, be
        self.theta, self.fr_max, self.hmin, self.cfl = theta, fr_max, hmin, cfl
        self.dx = ter.dx
        z = ter.z.astype(np.float64)
        n2 = ter.n.astype(np.float64) ** 2
        A = be.asarray
        self.z = A(z)
        self.zmx = A(np.maximum(z[:, :-1], z[:, 1:]))
        self.zmy = A(np.maximum(z[:-1, :], z[1:, :]))
        self.n2x = A(0.5 * (n2[:, :-1] + n2[:, 1:]))
        self.n2y = A(0.5 * (n2[:-1, :] + n2[1:, :]))
        self.active = A((~ter.wall).astype(np.float32))
        nin = max(int(ter.inflow.sum()), 1)
        self.w_in = A(ter.inflow.astype(np.float64) / (nin * ter.dx * ter.dx))
        self.out_coef = A(ter.outflow * np.sqrt(ter.S_out) / ter.n / ter.dx)
        self.inflow_ids = np.flatnonzero(ter.inflow.ravel())
        # ทางออกปลายน้ำ: ปรับอัตราการไหลออกให้ตรงกับ rating curve 1 มิติของหน้าตัดเดียวกัน
        # (สูตรแมนนิ่งรายเซลล์อย่างเดียวจะระบายน้ำเร็วเกินและทำให้ผิวน้ำปลายน้ำลดต่ำผิดจริง)
        self.outlet_rating = outlet_rating
        self.outlet_bed = outlet_bed
        oid = np.flatnonzero((ter.outflow & ter.channel).ravel())
        self.out_ids_np = oid if len(oid) else np.flatnonzero(ter.outflow.ravel())
        self.out_all_np = np.flatnonzero(ter.outflow.ravel())
        self.out_ids = be.int_array(self.out_ids_np)
        self.out_all = be.int_array(self.out_all_np)
        self.z_out = ter.z.ravel()[self.out_all_np].astype(np.float64)
        self.coef_out = (ter.outflow * np.sqrt(ter.S_out) / ter.n / ter.dx).ravel()[self.out_all_np]
        self.out_scale = 1.0
        ny, nx = ter.z.shape
        self.ny, self.nx = ny, nx
        self.h = be.zeros((ny, nx))
        self.qx = be.zeros((ny, nx + 1))
        self.qy = be.zeros((ny + 1, nx))
        self.hmax_dev = None
        self.engine = "generic"
        self._setup_fast(n2)
        self.reset_time()
        self.particles = None

    def _setup_fast(self, n2):
        """ใช้ CUDA kernel (CuPy) หรือ Numba (CPU) ถ้ามี ไม่เช่นนั้นใช้การคำนวณอาเรย์ทั่วไป"""
        be = self.be
        try:
            if be.name == "cupy":
                from .kernels import CudaKernels
                self._k = CudaKernels(be.cp)
                self.engine = "cuda-kernel"
            elif be.name == "numpy":
                from .kernels import make_numba_step
                self._nb = make_numba_step()
                self.engine = "numba-cpu"
        except Exception as e:  # noqa: BLE001
            self.engine = "generic"
            self.engine_note = f"{type(e).__name__}: {e}"
            return
        if self.engine != "generic":
            ny, nx = self.ny, self.nx
            self.n2 = be.asarray(n2)
            self.qxn = be.zeros((ny, nx + 1))
            self.qyn = be.zeros((ny + 1, nx))
            self.fac = be.zeros((ny, nx))
            self.w_in = be.asarray(be.to_np(self.w_in))

    # ------------------------------------------------------------ state
    def reset_time(self):
        self.t = 0.0
        self.nstep = 0
        self.vol_out = 0.0
        self.vol_in = 0.0
        self._out_acc = self.be.zeros((1,))
        self.update_dt()

    def set_depth(self, h_np: np.ndarray, qcx=None, qcy=None):
        """ตั้งความลึก (และอัตราการไหลต่อหน่วยกว้างที่กลางเซลล์ ถ้ามี) แล้วรีเซ็ตเวลา"""
        act = ~self.ter.wall
        h_np = np.where(act, h_np, 0).astype(np.float32)
        qx = np.zeros((self.ny, self.nx + 1), np.float32)
        qy = np.zeros((self.ny + 1, self.nx), np.float32)
        if qcx is not None:
            wet = (h_np > self.hmin) & act
            bx = wet[:, :-1] & wet[:, 1:]
            by = wet[:-1, :] & wet[1:, :]
            qx[:, 1:-1] = np.where(bx, 0.5 * (qcx[:, :-1] + qcx[:, 1:]), 0)
            qy[1:-1, :] = np.where(by, 0.5 * (qcy[:-1, :] + qcy[1:, :]), 0)
        self.h = self.be.asarray(np.ascontiguousarray(h_np))
        self.qx = self.be.asarray(qx)
        self.qy = self.be.asarray(qy)
        self.reset_time()

    def update_outlet(self):
        if self.outlet_rating is None:
            return
        be = self.be
        hf = self.h.reshape(-1)
        hc = be.to_np(hf[self.out_ids]).astype(np.float64)
        ha = be.to_np(hf[self.out_all]).astype(np.float64)
        zc = self.ter.z.ravel()[self.out_ids_np]
        wet = hc > self.hmin
        if not wet.any():
            self.out_scale = 1.0
            return
        stage = float(np.mean((zc + hc)[wet]))
        depth = max(stage - self.outlet_bed, 0.0)
        q_target = float(self.outlet_rating(depth))
        raw = float(np.sum(self.coef_out * ha ** (5 / 3))) * self.dx * self.dx
        self.out_scale = float(np.clip(q_target / raw, 0.02, 50.0)) if raw > 1e-9 else 1.0

    def update_dt(self):
        self.update_outlet()
        hmax = max(self.be.fmax(self.h), 0.05)
        self.hmax = hmax
        self.dt = self.cfl * self.dx / np.sqrt(G * hmax * 1.3)

    # ------------------------------------------------------------ numerics
    def _momentum(self, q, qpad_lo, qpad_hi, eta_lo, eta_hi, zm, n2):
        be, dt, dx = self.be, self.dt, self.dx
        hf = be.clip(be.maximum(eta_lo, eta_hi) - zm, 0.0, None)
        th = self.theta
        qs = th * q + (1 - th) * 0.5 * (qpad_lo + qpad_hi) if th < 1 else q
        hs = be.maximum(hf, self.hmin)
        qn = (qs - G * hf * dt * (eta_hi - eta_lo) / dx) / (1 + G * dt * n2 * abs(q) / hs ** (7.0 / 3.0))
        lim = self.fr_max * hf * be.sqrt(G * hf)
        qn = be.minimum(be.maximum(qn, -lim), lim)
        return be.where(hf > self.hmin, qn, 0.0)

    def step(self, Qin: float):
        if self.engine == "cuda-kernel":
            self._k.step(self, Qin)
            self._post_step(Qin)
            return
        if self.engine == "numba-cpu":
            acc = self._nb(self.h, self.z, self.n2, self.qx, self.qy, self.qxn, self.qyn, self.fac,
                           self.w_in, self.out_coef, self.active, self.dt, self.dx, self.theta,
                           self.fr_max, self.hmin, Qin, self.out_scale)
            self._out_acc = self._out_acc + acc
            self._post_step(Qin)
            return
        self._step_generic(Qin)

    def _post_step(self, Qin):
        self.vol_in += Qin * self.dt
        self.t += self.dt
        self.nstep += 1
        if self.nstep % 10 == 0:
            self.update_dt()

    def _step_generic(self, Qin: float):
        be, dt, dx = self.be, self.dt, self.dx
        h = self.h
        eta = self.z + h
        qx, qy = self.qx, self.qy
        qxn = self._momentum(qx[:, 1:-1], qx[:, :-2], qx[:, 2:], eta[:, :-1], eta[:, 1:], self.zmx, self.n2x)
        qyn = self._momentum(qy[1:-1, :], qy[:-2, :], qy[2:, :], eta[:-1, :], eta[1:, :], self.zmy, self.n2y)
        qx[:, 1:-1] = qxn
        qy[1:-1, :] = qyn

        # จำกัดไม่ให้ปริมาณน้ำที่ไหลออกจากเซลล์เกินปริมาณที่มีอยู่ (กันความลึกติดลบ)
        outv = (be.clip(qx[:, 1:], 0.0, None) + be.clip(-qx[:, :-1], 0.0, None)
                + be.clip(qy[1:, :], 0.0, None) + be.clip(-qy[:-1, :], 0.0, None)) * (dt / dx)
        fac = be.where(outv > h, h / be.maximum(outv, 1e-12), 1.0)
        qx[:, 1:-1] = be.where(qxn > 0, qxn * fac[:, :-1], qxn * fac[:, 1:])
        qy[1:-1, :] = be.where(qyn > 0, qyn * fac[:-1, :], qyn * fac[1:, :])

        h = h + (dt / dx) * (qx[:, :-1] - qx[:, 1:] + qy[:-1, :] - qy[1:, :])
        h = be.clip(h, 0.0, None)
        if Qin > 0:
            h = h + (Qin * dt) * self.w_in
        out = be.minimum(h, (dt * self.out_scale) * self.out_coef * h ** (5.0 / 3.0))
        self._out_acc = self._out_acc + out.sum()
        self.h = (h - out) * self.active
        self._post_step(Qin)

    def advance(self, sim_seconds: float, Qin_func, max_steps=100000) -> int:
        """เดินเวลาไปข้างหน้าเท่ากับ sim_seconds (หรือจนครบ max_steps)"""
        t_end = self.t + sim_seconds
        k = 0
        while self.t < t_end and k < max_steps:
            self.step(float(Qin_func(self.t)))
            k += 1
        return k

    def flush_outflow(self) -> float:
        """อ่านปริมาตรน้ำที่ไหลออกสะสม (ทำให้ GPU sync จึงเรียกเป็นครั้งคราว)"""
        v = float(self.be.to_np(self._out_acc)[0]) * self.dx * self.dx
        self._out_acc = self.be.zeros((1,))
        self.vol_out += v
        return v

    # ------------------------------------------------------------ fields
    def velocity(self):
        be = self.be
        hs = be.maximum(self.h, 0.05)
        wet = self.h > 0.05
        u = be.where(wet, 0.5 * (self.qx[:, :-1] + self.qx[:, 1:]) / hs, 0.0)
        v = be.where(wet, 0.5 * (self.qy[:-1, :] + self.qy[1:, :]) / hs, 0.0)
        return u, v

    def fields_np(self):
        """คืนค่า (h, speed, froude) เป็น NumPy"""
        be = self.be
        u, v = self.velocity()
        sp = be.sqrt(u * u + v * v)
        fr = sp / be.sqrt(G * be.maximum(self.h, 0.01))
        return be.to_np(self.h), be.to_np(sp), be.to_np(fr)

    def volume(self) -> float:
        return self.be.fsum(self.h) * self.dx * self.dx

    # ------------------------------------------------------------ tracers
    def init_particles(self, n: int):
        be = self.be
        self.np_ = n
        self.px = be.zeros((n,))
        self.py = be.zeros((n,))
        self.page = be.zeros((n,))
        self._respawn(be.asarray(np.ones(n, np.float32)) > 0, frac_inflow=0.0)
        self.particles = True

    def _respawn(self, dead, frac_inflow=0.8):
        """วางอนุภาคใหม่: ส่วนใหญ่ที่ต้นน้ำ ที่เหลือสุ่มบนพื้นที่เปียก"""
        be, ter = self.be, self.ter
        n = self.np_
        wet = be.to_np(self.h > 0.05).ravel()
        wet_ids = np.flatnonzero(wet)
        if len(wet_ids) == 0:
            wet_ids = self.inflow_ids
        rng = np.random.default_rng()
        pick_in = rng.random(n) < frac_inflow
        ids = np.where(pick_in, rng.choice(self.inflow_ids, n), rng.choice(wet_ids, n))
        iy, ix = np.divmod(ids, self.nx)
        nx_ = ter.x0 + (ix + rng.random(n)) * ter.dx
        ny_ = ter.y0 + (iy + rng.random(n)) * ter.dx
        dead_f = be.where(dead, 1.0, 0.0)
        self.px = self.px * (1 - dead_f) + be.asarray(nx_) * dead_f
        self.py = self.py * (1 - dead_f) + be.asarray(ny_) * dead_f
        self.page = self.page * (1 - dead_f)

    def move_particles(self, dt_sim: float):
        if not self.particles:
            return
        be, ter = self.be, self.ter
        u, v = self.velocity()
        umax = 3.0
        nsub = int(np.clip(np.ceil(dt_sim * umax / ter.dx), 1, 12))
        ddt = dt_sim / nsub
        for _ in range(nsub):
            ix = be.asint(be.clip(be.floor((self.px - ter.x0) / ter.dx), 0, self.nx - 1))
            iy = be.asint(be.clip(be.floor((self.py - ter.y0) / ter.dx), 0, self.ny - 1))
            self.px = self.px + u[iy, ix] * ddt
            self.py = self.py + v[iy, ix] * ddt
        ix = be.asint(be.clip(be.floor((self.px - ter.x0) / ter.dx), 0, self.nx - 1))
        iy = be.asint(be.clip(be.floor((self.py - ter.y0) / ter.dx), 0, self.ny - 1))
        self.page = self.page + dt_sim
        hloc = self.h[iy, ix]
        sp = be.sqrt(u[iy, ix] ** 2 + v[iy, ix] ** 2)
        dead = (hloc < 0.03) | (self.out_coef[iy, ix] > 0) | ((sp < 0.005) & (self.page > 60))
        if bool(be.to_np(dead.any())):
            self._respawn(dead)

    def particles_np(self):
        be, ter = self.be, self.ter
        u, v = self.velocity()
        ix = be.asint(be.clip(be.floor((self.px - ter.x0) / ter.dx), 0, self.nx - 1))
        iy = be.asint(be.clip(be.floor((self.py - ter.y0) / ter.dx), 0, self.ny - 1))
        sp = be.sqrt(u[iy, ix] ** 2 + v[iy, ix] ** 2)
        return be.to_np(self.px), be.to_np(self.py), be.to_np(sp)
