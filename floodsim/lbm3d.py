"""แบบจำลองการไหล 3 มิติ: Lattice Boltzmann D3Q19 + LES (Smagorinsky) บน GPU

- การชนแบบ regularized BGK (เสถียรที่ Reynolds สูง) + แบบจำลองความปั่นป่วน Smagorinsky
- แรงภายนอก (ใบพัดเรือ) แบบ Guo forcing
- ผนัง/ท้องน้ำ/ตัวเรือ: bounce-back + wall function (แรงเฉือนตามกฎลอการิทึมจากความขรุขระ),
  ผิวน้ำ: rigid lid แบบลื่น (specular reflection)
- ทางเข้า: กำหนดความเร็ว, ทางออก: กำหนดความดัน (non-equilibrium extrapolation)

ลำดับเก็บข้อมูล: f[i*N + cell], cell = x + nx*(y + ny*z)  (x ต่อเนื่องในหน่วยความจำ)
หน่วยแลตทิซ: dx = 1, dt = 1, ρ ≈ 1
"""
from __future__ import annotations

import numpy as np

CX = np.array([0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0])
CY = np.array([0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1])
CZ = np.array([0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1])
W = np.array([1 / 3] + [1 / 18] * 6 + [1 / 36] * 12)
OPP = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])
MIRZ = np.array([0, 1, 2, 3, 4, 6, 5, 7, 8, 9, 10, 13, 14, 11, 12, 17, 18, 15, 16])


def _carr(name, a, typ):
    return f"__constant__ {typ} {name}[19] = {{{', '.join(str(float(v)) + 'f' if typ == 'float' else str(int(v)) for v in a)}}};"


CUDA_SRC = "\n".join([_carr("CX", CX, "int"), _carr("CY", CY, "int"), _carr("CZ", CZ, "int"),
                      _carr("W", W, "float"), _carr("OPP", OPP, "int"), _carr("MIRZ", MIRZ, "int")]) + r"""

extern "C" __global__ void lbm_step(const float* __restrict__ fs, float* __restrict__ fd,
        const unsigned char* __restrict__ solid,
        const float* __restrict__ Fx, const float* __restrict__ Fy, const float* __restrict__ Fz,
        float* __restrict__ rho_o, float* __restrict__ ux_o, float* __restrict__ uy_o, float* __restrict__ uz_o,
        const float* __restrict__ cfw, const float* __restrict__ uref,
        const int nx, const int ny, const int nz, const float tau0, const float csmag2)
{
    const long N = (long)nx * ny * nz;
    const long c = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= N) return;
    const int x = (int)(c % nx);
    const int y = (int)((c / nx) % ny);
    const int z = (int)(c / ((long)nx * ny));
    if (solid[c]) { rho_o[c] = 1.f; ux_o[c] = 0.f; uy_o[c] = 0.f; uz_o[c] = 0.f; return; }

    float f[19];
    unsigned int wmask = 0u;
    float nwx = 0.f, nwy = 0.f, nwz = 0.f;          // direction towards the wall
    #pragma unroll
    for (int i = 0; i < 19; ++i) {
        int xs = x - CX[i], ys = y - CY[i], zs = z - CZ[i];
        int ii = i;
        if (xs < 0 || xs >= nx) { f[i] = fs[(long)i * N + c]; continue; }   // inlet/outlet: fixed later by lbm_bc
        if (zs >= nz) { zs = z; ii = MIRZ[i]; }                              // free-slip surface (rigid lid)
        bool wall = (ys < 0 || ys >= ny || zs < 0);
        long s = 0;
        if (!wall) { s = xs + (long)nx * (ys + (long)ny * zs); wall = solid[s]; }
        if (wall) { wmask |= (1u << i); nwx -= CX[i]; nwy -= CY[i]; nwz -= CZ[i]; }
        f[i] = wall ? fs[(long)OPP[i] * N + c] : fs[(long)ii * N + s];
    }
    // wall function (momentum exchange): moving-wall bounce-back with a tangential wall speed u_s chosen
    // so that the momentum the cell gives to the wall per step equals the log-law shear on its wall faces:
    //   lost_t = -sum_wall (c_i.t)(2 f*_opp + 6 w_i rho (c_i.t) u_s) = rho*cf*U^2 * A_faces
    // U = uref[c] (column-mean speed, Manning-type cf) when > 0, otherwise the local tangential speed (log law)
    // Wall area of a stair-step cell = |(n_x-faces, n_y-faces, n_z-faces)| (not the face count).
    // (A residual body force for the part the clamped slip cannot deliver was tried and is unstable.)
    const float cf = cfw[c];
    if (wmask && cf > 0.f) {
        const float nn = sqrtf(nwx * nwx + nwy * nwy + nwz * nwz);
        const float afx = (float)(((wmask >> 1) & 1u) + ((wmask >> 2) & 1u));
        const float afy = (float)(((wmask >> 3) & 1u) + ((wmask >> 4) & 1u));
        const float afz = (float)(((wmask >> 5) & 1u) + ((wmask >> 6) & 1u));
        const float area = sqrtf(afx * afx + afy * afy + afz * afz);
        if (nn > 0.f && area > 0.f) {
            const float ex = nwx / nn, ey = nwy / nn, ez = nwz / nn;
            const float px = ux_o[c], py = uy_o[c], pz = uz_o[c];          // previous step (own cell)
            const float un = px * ex + py * ey + pz * ez;
            float tx = px - un * ex, ty = py - un * ey, tz = pz - un * ez;
            const float ut = sqrtf(tx * tx + ty * ty + tz * tz);
            if (ut > 1e-7f) {
                tx /= ut; ty /= ut; tz /= ut;
                const float rw = rho_o[c];
                float D = 0.f, K = 0.f;
                #pragma unroll
                for (int i = 0; i < 19; ++i)
                    if (wmask & (1u << i)) {
                        const float ct = CX[i] * tx + CY[i] * ty + CZ[i] * tz;
                        D += ct * f[i];                     // f[i] = f*_opp (bounced back)
                        K += W[i] * ct * ct;
                    }
                if (K > 0.f) {
                    const float U = uref[c] > 0.f ? uref[c] : ut;
                    const float us0 = (-2.f * D - rw * cf * U * U * area) / (6.f * rw * K);
                    const float us = fminf(fmaxf(us0, -ut), ut);
                    #pragma unroll
                    for (int i = 0; i < 19; ++i)
                        if (wmask & (1u << i)) f[i] += 6.f * W[i] * rw * (CX[i] * tx + CY[i] * ty + CZ[i] * tz) * us;
                }
            }
        }
    }

    float r = 0.f, mx = 0.f, my = 0.f, mz = 0.f;
    #pragma unroll
    for (int i = 0; i < 19; ++i) { r += f[i]; mx += f[i] * CX[i]; my += f[i] * CY[i]; mz += f[i] * CZ[i]; }
    const float fx = Fx[c], fy = Fy[c], fz = Fz[c];
    const float ux = (mx + 0.5f * fx) / r, uy = (my + 0.5f * fy) / r, uz = (mz + 0.5f * fz) / r;
    const float usq = ux * ux + uy * uy + uz * uz;

    float pxx = 0.f, pyy = 0.f, pzz = 0.f, pxy = 0.f, pxz = 0.f, pyz = 0.f;
    float feq[19];
    #pragma unroll
    for (int i = 0; i < 19; ++i) {
        const float cu = 3.f * (CX[i] * ux + CY[i] * uy + CZ[i] * uz);
        feq[i] = W[i] * r * (1.f + cu + 0.5f * cu * cu - 1.5f * usq);
        const float fn = f[i] - feq[i];
        pxx += CX[i] * CX[i] * fn; pyy += CY[i] * CY[i] * fn; pzz += CZ[i] * CZ[i] * fn;
        pxy += CX[i] * CY[i] * fn; pxz += CX[i] * CZ[i] * fn; pyz += CY[i] * CZ[i] * fn;
    }
    const float qn = sqrtf(pxx * pxx + pyy * pyy + pzz * pzz + 2.f * (pxy * pxy + pxz * pxz + pyz * pyz));
    const float tau = 0.5f * (tau0 + sqrtf(tau0 * tau0 + 18.f * 1.41421356f * csmag2 * qn / r));
    const float om = 1.f / tau;
    const float sf = 1.f - 0.5f * om;
    #pragma unroll
    for (int i = 0; i < 19; ++i) {
        const float cxx = CX[i] * CX[i] - 1.f / 3.f, cyy = CY[i] * CY[i] - 1.f / 3.f, czz = CZ[i] * CZ[i] - 1.f / 3.f;
        const float fneq = 4.5f * W[i] * (cxx * pxx + cyy * pyy + czz * pzz
                          + 2.f * (CX[i] * CY[i] * pxy + CX[i] * CZ[i] * pxz + CY[i] * CZ[i] * pyz));
        const float cu = CX[i] * ux + CY[i] * uy + CZ[i] * uz;
        const float cF = CX[i] * fx + CY[i] * fy + CZ[i] * fz;
        const float src = sf * W[i] * (3.f * ((CX[i] - ux) * fx + (CY[i] - uy) * fy + (CZ[i] - uz) * fz)
                          + 9.f * cu * cF);
        // first moment of f_neq = -F/2 (Guo); regularization drops it -> only (3-omega)/2 of the force would act
        const float fneq1 = -1.5f * W[i] * cF;
        fd[(long)i * N + c] = feq[i] + (1.f - om) * (fneq + fneq1) + src;
    }
    rho_o[c] = r; ux_o[c] = ux; uy_o[c] = uy; uz_o[c] = uz;
}

__device__ inline float feq_i(int i, float r, float ux, float uy, float uz) {
    const float cu = 3.f * (CX[i] * ux + CY[i] * uy + CZ[i] * uz);
    return W[i] * r * (1.f + cu + 0.5f * cu * cu - 1.5f * (ux * ux + uy * uy + uz * uz));
}

extern "C" __global__ void lbm_bc(float* __restrict__ fd, const unsigned char* __restrict__ solid,
        const float* __restrict__ uin, const int nx, const int ny, const int nz, const float rho_out)
{
    const long N = (long)nx * ny * nz;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny * nz) return;
    const int y = j % ny, z = j / ny;
    const long row = (long)nx * (y + (long)ny * z);
    // inlet x=0: prescribed velocity profile, density from the next cell
    {
        const long c0 = row, c1 = row + 1;
        if (!solid[c0] && !solid[c1]) {
            float r = 0.f, mx = 0.f, my = 0.f, mz = 0.f;
            for (int i = 0; i < 19; ++i) { const float v = fd[(long)i * N + c1]; r += v; mx += v * CX[i]; my += v * CY[i]; mz += v * CZ[i]; }
            const float ux = mx / r, uy = my / r, uz = mz / r;
            for (int i = 0; i < 19; ++i)
                fd[(long)i * N + c0] = feq_i(i, r, uin[j], 0.f, 0.f) + (fd[(long)i * N + c1] - feq_i(i, r, ux, uy, uz));
        }
    }
    // outlet x=nx-1: fixed pressure, velocity from the previous cell
    {
        const long c0 = row + nx - 1, c1 = row + nx - 2;
        if (!solid[c0] && !solid[c1]) {
            float r = 0.f, mx = 0.f, my = 0.f, mz = 0.f;
            for (int i = 0; i < 19; ++i) { const float v = fd[(long)i * N + c1]; r += v; mx += v * CX[i]; my += v * CY[i]; mz += v * CZ[i]; }
            const float ux = mx / r, uy = my / r, uz = mz / r;
            for (int i = 0; i < 19; ++i)
                fd[(long)i * N + c0] = feq_i(i, rho_out, ux, uy, uz) + (fd[(long)i * N + c1] - feq_i(i, r, ux, uy, uz));
        }
    }
}
"""


def make_numba_kernels():
    """เวอร์ชัน CPU (Numba) ของเคอร์เนลเดียวกัน — ช้ากว่ามาก ใช้เมื่อไม่มี GPU"""
    from numba import njit, prange

    cx, cy, cz, w, opp, mirz = CX.astype(np.int64), CY.astype(np.int64), CZ.astype(np.int64), \
        W.astype(np.float32), OPP.astype(np.int64), MIRZ.astype(np.int64)

    @njit(parallel=True, fastmath=True, cache=True)
    def step(fs, fd, solid, Fx, Fy, Fz, rho_o, ux_o, uy_o, uz_o, cfw, uref, nx, ny, nz, tau0, csmag2):
        N = nx * ny * nz
        for c in prange(N):
            x = c % nx
            y = (c // nx) % ny
            z = c // (nx * ny)
            if solid[c]:
                rho_o[c] = 1.0
                ux_o[c] = 0.0
                uy_o[c] = 0.0
                uz_o[c] = 0.0
                continue
            f = np.empty(19, np.float32)
            wl = np.zeros(19, np.bool_)
            nwx = 0.0
            nwy = 0.0
            nwz = 0.0
            for i in range(19):
                xs, ys, zs = x - cx[i], y - cy[i], z - cz[i]
                ii = i
                if xs < 0 or xs >= nx:
                    f[i] = fs[i * N + c]
                    continue
                if zs >= nz:
                    zs = z
                    ii = mirz[i]
                wall = ys < 0 or ys >= ny or zs < 0
                s = 0
                if not wall:
                    s = xs + nx * (ys + ny * zs)
                    wall = solid[s] != 0
                if wall:
                    wl[i] = True
                    nwx -= cx[i]
                    nwy -= cy[i]
                    nwz -= cz[i]
                f[i] = fs[opp[i] * N + c] if wall else fs[ii * N + s]
            cf = cfw[c]
            nn = np.sqrt(nwx * nwx + nwy * nwy + nwz * nwz)
            afx = float(wl[1]) + float(wl[2])
            afy = float(wl[3]) + float(wl[4])
            afz = float(wl[5]) + float(wl[6])
            area = np.sqrt(afx * afx + afy * afy + afz * afz)
            if cf > 0.0 and nn > 0.0 and area > 0.0:      # wall function (ดูคำอธิบายในเวอร์ชัน CUDA)
                ex, ey, ez = nwx / nn, nwy / nn, nwz / nn
                px, py, pz = ux_o[c], uy_o[c], uz_o[c]
                un = px * ex + py * ey + pz * ez
                tx, ty, tz = px - un * ex, py - un * ey, pz - un * ez
                ut = np.sqrt(tx * tx + ty * ty + tz * tz)
                if ut > 1e-7:
                    tx, ty, tz = tx / ut, ty / ut, tz / ut
                    rw = rho_o[c]
                    D = 0.0
                    K = 0.0
                    for i in range(19):
                        if wl[i]:
                            ct = cx[i] * tx + cy[i] * ty + cz[i] * tz
                            D += ct * f[i]
                            K += w[i] * ct * ct
                    if K > 0.0:
                        U = uref[c] if uref[c] > 0.0 else ut
                        us0 = (-2.0 * D - rw * cf * U * U * area) / (6.0 * rw * K)
                        us = min(max(us0, -ut), ut)
                        for i in range(19):
                            if wl[i]:
                                f[i] += 6.0 * w[i] * rw * (cx[i] * tx + cy[i] * ty + cz[i] * tz) * us
            r = 0.0
            mx = 0.0
            my = 0.0
            mz = 0.0
            for i in range(19):
                r += f[i]
                mx += f[i] * cx[i]
                my += f[i] * cy[i]
                mz += f[i] * cz[i]
            fx, fy, fz = Fx[c], Fy[c], Fz[c]
            ux, uy, uz = (mx + 0.5 * fx) / r, (my + 0.5 * fy) / r, (mz + 0.5 * fz) / r
            usq = ux * ux + uy * uy + uz * uz
            feq = np.empty(19, np.float32)
            pxx = pyy = pzz = pxy = pxz = pyz = 0.0
            for i in range(19):
                cu = 3.0 * (cx[i] * ux + cy[i] * uy + cz[i] * uz)
                feq[i] = w[i] * r * (1.0 + cu + 0.5 * cu * cu - 1.5 * usq)
                fn = f[i] - feq[i]
                pxx += cx[i] * cx[i] * fn
                pyy += cy[i] * cy[i] * fn
                pzz += cz[i] * cz[i] * fn
                pxy += cx[i] * cy[i] * fn
                pxz += cx[i] * cz[i] * fn
                pyz += cy[i] * cz[i] * fn
            qn = np.sqrt(pxx * pxx + pyy * pyy + pzz * pzz + 2.0 * (pxy * pxy + pxz * pxz + pyz * pyz))
            tau = 0.5 * (tau0 + np.sqrt(tau0 * tau0 + 18.0 * 1.41421356 * csmag2 * qn / r))
            om = 1.0 / tau
            sf = 1.0 - 0.5 * om
            for i in range(19):
                fneq = 4.5 * w[i] * ((cx[i] * cx[i] - 1 / 3) * pxx + (cy[i] * cy[i] - 1 / 3) * pyy
                                     + (cz[i] * cz[i] - 1 / 3) * pzz + 2.0 * (cx[i] * cy[i] * pxy
                                     + cx[i] * cz[i] * pxz + cy[i] * cz[i] * pyz))
                cu = cx[i] * ux + cy[i] * uy + cz[i] * uz
                cF = cx[i] * fx + cy[i] * fy + cz[i] * fz
                src = sf * w[i] * (3.0 * ((cx[i] - ux) * fx + (cy[i] - uy) * fy + (cz[i] - uz) * fz)
                                   + 9.0 * cu * cF)
                fneq1 = -1.5 * w[i] * cF          # โมเมนต์ที่ 1 ของ f_neq = −F/2 (ดูเวอร์ชัน CUDA)
                fd[i * N + c] = feq[i] + (1.0 - om) * (fneq + fneq1) + src
            rho_o[c] = r
            ux_o[c] = ux
            uy_o[c] = uy
            uz_o[c] = uz

    @njit(cache=True)
    def feq1(i, r, ux, uy, uz):
        cu = 3.0 * (cx[i] * ux + cy[i] * uy + cz[i] * uz)
        return w[i] * r * (1.0 + cu + 0.5 * cu * cu - 1.5 * (ux * ux + uy * uy + uz * uz))

    @njit(parallel=True, cache=True)
    def bc(fd, solid, uin, nx, ny, nz, rho_out):
        N = nx * ny * nz
        for j in prange(ny * nz):
            y = j % ny
            z = j // ny
            row = nx * (y + ny * z)
            for side in range(2):
                c0 = row if side == 0 else row + nx - 1
                c1 = row + 1 if side == 0 else row + nx - 2
                if solid[c0] or solid[c1]:
                    continue
                r = 0.0
                mx = 0.0
                my = 0.0
                mz = 0.0
                for i in range(19):
                    v = fd[i * N + c1]
                    r += v
                    mx += v * cx[i]
                    my += v * cy[i]
                    mz += v * cz[i]
                ux, uy, uz = mx / r, my / r, mz / r
                for i in range(19):
                    if side == 0:
                        fd[i * N + c0] = feq1(i, r, uin[j], 0.0, 0.0) + (fd[i * N + c1] - feq1(i, r, ux, uy, uz))
                    else:
                        fd[i * N + c0] = feq1(i, rho_out, ux, uy, uz) + (fd[i * N + c1] - feq1(i, r, ux, uy, uz))

    return step, bc


class LBM3D:
    """ตัวแก้สมการ LBM บน GPU (CuPy) หรือ CPU (Numba)

    solid: bool (nz, ny, nx), F: แรงต่อหน่วยปริมาตร (หน่วยแลตทิซ) 3 อาเรย์ (nz, ny, nx)
    uin: ความเร็วขาเข้า (หน่วยแลตทิซ) (nz, ny)
    cf_wall: สัมประสิทธิ์แรงเสียดทานผนัง τ = ρ·cf·U² ของเซลล์ติดผนัง (nz, ny, nx); None/0 = bounce-back ไม่ลื่นธรรมดา
             U = self.uref (ความเร็วอ้างอิง เช่น ความเร็วเฉลี่ยตามความลึก) ถ้า > 0 ไม่เช่นนั้นใช้ความเร็วขนานผนังของเซลล์
    """

    def __init__(self, be, solid, Fx, Fy, Fz, uin, tau0, cs=0.17, cf_wall=None):
        self.be = be
        self.nz, self.ny, self.nx = solid.shape
        self.N = solid.size
        self.tau0 = float(tau0)
        self.csmag2 = float(cs * cs)
        xp = be.xp if be.name in ("cupy", "numpy") else None
        if be.name == "torch":
            raise RuntimeError("แบบจำลอง 3D รองรับ CuPy (GPU) หรือ NumPy/Numba (CPU)")
        self.xp = xp
        A = lambda a, dt=np.float32: xp.asarray(np.ascontiguousarray(a).ravel(), dtype=dt)  # noqa: E731
        self.solid = A(solid.astype(np.uint8), np.uint8)
        self.Fx, self.Fy, self.Fz = A(Fx), A(Fy), A(Fz)
        self.uin = A(uin)
        self.rho = xp.ones(self.N, np.float32)
        self.ux = xp.zeros(self.N, np.float32)
        self.uy = xp.zeros(self.N, np.float32)
        self.uz = xp.zeros(self.N, np.float32)
        self.cfw = A(cf_wall if cf_wall is not None else np.zeros(solid.shape, np.float32))
        self.uref = xp.zeros(self.N, np.float32)
        u0 = np.broadcast_to(uin[:, :, None], solid.shape).copy()
        u0[solid] = 0
        self.f = self._equilibrium(np.ones(solid.shape, np.float32), u0)
        self.f2 = xp.empty_like(self.f)
        self.nstep = 0
        if be.name == "cupy":
            cp = be.cp
            mod = cp.RawModule(code=CUDA_SRC, options=("--use_fast_math",))
            self.k_step = mod.get_function("lbm_step")
            self.k_bc = mod.get_function("lbm_bc")
            self.engine = "cuda-lbm"
        else:
            self.n_step, self.n_bc = make_numba_kernels()
            self.engine = "numba-lbm"

    def _equilibrium(self, r, ux):
        ux = ux.ravel().astype(np.float32)
        r = r.ravel()
        f = np.empty(19 * self.N, np.float32)
        for i in range(19):
            cu = 3 * CX[i] * ux
            f[i * self.N:(i + 1) * self.N] = W[i] * r * (1 + cu + 0.5 * cu * cu - 1.5 * ux * ux)
        return self.xp.asarray(f)

    def set_force(self, Fx, Fy, Fz):
        A = lambda a: self.xp.asarray(np.ascontiguousarray(a).ravel(), dtype=np.float32)  # noqa: E731
        self.Fx, self.Fy, self.Fz = A(Fx), A(Fy), A(Fz)

    def step(self, n=1):
        nx, ny, nz, N = self.nx, self.ny, self.nz, self.N
        for a in (self.Fx, self.Fy, self.Fz, self.uin, self.uref):
            if a.dtype != np.float32 or a.size != (N if a is not self.uin else ny * nz):
                raise TypeError(f"LBM3D: force/inlet arrays must be float32 of the correct size (got {a.dtype}, {a.size})")
        if self.engine == "cuda-lbm":
            i32, f32 = np.int32, np.float32
            blk = 256
            grid = ((N + blk - 1) // blk,)
            gbc = ((ny * nz + blk - 1) // blk,)
            for _ in range(n):
                self.k_step(grid, (blk,), (self.f, self.f2, self.solid, self.Fx, self.Fy, self.Fz,
                                           self.rho, self.ux, self.uy, self.uz, self.cfw, self.uref,
                                           i32(nx), i32(ny), i32(nz), f32(self.tau0), f32(self.csmag2)))
                self.k_bc(gbc, (blk,), (self.f2, self.solid, self.uin, i32(nx), i32(ny), i32(nz), f32(1.0)))
                self.f, self.f2 = self.f2, self.f
        else:
            for _ in range(n):
                self.n_step(self.f, self.f2, self.solid, self.Fx, self.Fy, self.Fz, self.rho, self.ux, self.uy,
                            self.uz, self.cfw, self.uref, nx, ny, nz, np.float32(self.tau0), np.float32(self.csmag2))
                self.n_bc(self.f2, self.solid, self.uin, nx, ny, nz, np.float32(1.0))
                self.f, self.f2 = self.f2, self.f
        self.nstep += n

    def shape3(self, a):
        return a.reshape(self.nz, self.ny, self.nx)

    def memory_mb(self):
        return self.N * (19 * 2 * 4 + 4 * 10 + 1) / 1e6
