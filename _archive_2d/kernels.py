"""เคอร์เนลเร่งความเร็วของ local inertial solver

- CUDA (CuPy RawKernel): 4 เคอร์เนลต่อหนึ่งก้าวเวลา
- CPU (Numba parallel): ตรรกะเดียวกัน ใช้เมื่อไม่มี GPU
ตัวแปรเรียงแบบ row-major: เซลล์ (i,j) -> i*nx+j, qx (ny, nx+1), qy (ny+1, nx)
"""
from __future__ import annotations

import numpy as np

CUDA_SRC = r"""
extern "C" {
__global__ void momentum(const float* h, const float* z, const float* n2,
                         const float* qx, const float* qy, float* qxn, float* qyn,
                         const int nx, const int ny, const float dt, const float dx,
                         const float theta, const float frmax, const float hmin)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    const int i = blockDim.y * blockIdx.y + threadIdx.y;
    const float g = 9.81f;
    if (i < ny && j <= nx) {
        const int f = i * (nx + 1) + j;
        float qnew = 0.f;
        if (j >= 1 && j <= nx - 1) {
            const int cl = i * nx + j - 1, cr = cl + 1;
            const float el = z[cl] + h[cl], er = z[cr] + h[cr];
            const float hf = fmaxf(el, er) - fmaxf(z[cl], z[cr]);
            if (hf > hmin) {
                const float q = qx[f];
                const float qs = theta * q + 0.5f * (1.f - theta) * (qx[f - 1] + qx[f + 1]);
                const float nn = 0.5f * (n2[cl] + n2[cr]);
                qnew = (qs - g * hf * dt * (er - el) / dx) / (1.f + g * dt * nn * fabsf(q) / powf(hf, 7.f / 3.f));
                const float lim = frmax * hf * sqrtf(g * hf);
                qnew = fminf(fmaxf(qnew, -lim), lim);
            }
        }
        qxn[f] = qnew;
    }
    if (i <= ny && j < nx) {
        const int f = i * nx + j;
        float qnew = 0.f;
        if (i >= 1 && i <= ny - 1) {
            const int cb = (i - 1) * nx + j, ct = cb + nx;
            const float eb = z[cb] + h[cb], et = z[ct] + h[ct];
            const float hf = fmaxf(eb, et) - fmaxf(z[cb], z[ct]);
            if (hf > hmin) {
                const float q = qy[f];
                const float qs = theta * q + 0.5f * (1.f - theta) * (qy[f - nx] + qy[f + nx]);
                const float nn = 0.5f * (n2[cb] + n2[ct]);
                qnew = (qs - g * hf * dt * (et - eb) / dx) / (1.f + g * dt * nn * fabsf(q) / powf(hf, 7.f / 3.f));
                const float lim = frmax * hf * sqrtf(g * hf);
                qnew = fminf(fmaxf(qnew, -lim), lim);
            }
        }
        qyn[f] = qnew;
    }
}

__global__ void outfac(const float* h, const float* qx, const float* qy, float* fac,
                       const int nx, const int ny, const float dt, const float dx)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    const int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= ny || j >= nx) return;
    const int c = i * nx + j;
    const float ql = qx[i * (nx + 1) + j], qr = qx[i * (nx + 1) + j + 1];
    const float qb = qy[i * nx + j], qt = qy[(i + 1) * nx + j];
    const float out = (fmaxf(qr, 0.f) + fmaxf(-ql, 0.f) + fmaxf(qt, 0.f) + fmaxf(-qb, 0.f)) * dt / dx;
    fac[c] = out > h[c] ? h[c] / fmaxf(out, 1e-12f) : 1.f;
}

__global__ void limitq(const float* fac, const float* qxn, const float* qyn, float* qx, float* qy,
                       const int nx, const int ny)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    const int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < ny && j <= nx) {
        const int f = i * (nx + 1) + j;
        const float q = qxn[f];
        float r = 0.f;
        if (j >= 1 && j <= nx - 1) r = q > 0.f ? q * fac[i * nx + j - 1] : q * fac[i * nx + j];
        qx[f] = r;
    }
    if (i <= ny && j < nx) {
        const int f = i * nx + j;
        const float q = qyn[f];
        float r = 0.f;
        if (i >= 1 && i <= ny - 1) r = q > 0.f ? q * fac[(i - 1) * nx + j] : q * fac[i * nx + j];
        qy[f] = r;
    }
}

__global__ void continuity(float* h, const float* qx, const float* qy, const float* w_in,
                           const float* coef, const float* active, float* acc,
                           const int nx, const int ny, const float dt, const float dx, const float Qin,
                           const float oscale)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    const int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= ny || j >= nx) return;
    const int c = i * nx + j;
    float hn = h[c] + dt / dx * (qx[i * (nx + 1) + j] - qx[i * (nx + 1) + j + 1]
                                 + qy[i * nx + j] - qy[(i + 1) * nx + j]);
    hn = fmaxf(hn, 0.f) + Qin * dt * w_in[c];
    if (coef[c] > 0.f && hn > 0.f) {
        const float o = fminf(hn, dt * oscale * coef[c] * powf(hn, 5.f / 3.f));
        atomicAdd(acc, o);
        hn -= o;
    }
    h[c] = hn * active[c];
}
}
"""


class CudaKernels:
    def __init__(self, cp):
        self.cp = cp
        mod = cp.RawModule(code=CUDA_SRC, options=("--use_fast_math",))
        self.k_mom = mod.get_function("momentum")
        self.k_fac = mod.get_function("outfac")
        self.k_lim = mod.get_function("limitq")
        self.k_con = mod.get_function("continuity")

    def step(self, s, Qin):
        cp = self.cp
        nx, ny = s.nx, s.ny
        blk = (32, 8, 1)
        gf = ((nx + 1 + 31) // 32, (ny + 1 + 7) // 8, 1)
        gc = ((nx + 31) // 32, (ny + 7) // 8, 1)
        f32, i32 = np.float32, np.int32
        self.k_mom(gf, blk, (s.h, s.z, s.n2, s.qx, s.qy, s.qxn, s.qyn, i32(nx), i32(ny), f32(s.dt),
                             f32(s.dx), f32(s.theta), f32(s.fr_max), f32(s.hmin)))
        self.k_fac(gc, blk, (s.h, s.qxn, s.qyn, s.fac, i32(nx), i32(ny), f32(s.dt), f32(s.dx)))
        self.k_lim(gf, blk, (s.fac, s.qxn, s.qyn, s.qx, s.qy, i32(nx), i32(ny)))
        self.k_con(gc, blk, (s.h, s.qx, s.qy, s.w_in, s.out_coef, s.active, s._out_acc,
                             i32(nx), i32(ny), f32(s.dt), f32(s.dx), f32(Qin), f32(s.out_scale)))


# ---------------------------------------------------------------- Numba (CPU)
def make_numba_step():
    from numba import njit, prange

    @njit(parallel=True, fastmath=True, cache=True)
    def step(h, z, n2, qx, qy, qxn, qyn, fac, w_in, coef, active, dt, dx, theta, frmax, hmin, Qin, oscale):
        ny, nx = h.shape
        g = 9.81
        for i in prange(ny):
            for j in range(1, nx):
                el = z[i, j - 1] + h[i, j - 1]
                er = z[i, j] + h[i, j]
                hf = max(el, er) - max(z[i, j - 1], z[i, j])
                qn = 0.0
                if hf > hmin:
                    q = qx[i, j]
                    qs = theta * q + 0.5 * (1 - theta) * (qx[i, j - 1] + qx[i, j + 1])
                    nn = 0.5 * (n2[i, j - 1] + n2[i, j])
                    qn = (qs - g * hf * dt * (er - el) / dx) / (1 + g * dt * nn * abs(q) / hf ** (7.0 / 3.0))
                    lim = frmax * hf * np.sqrt(g * hf)
                    qn = min(max(qn, -lim), lim)
                qxn[i, j] = qn
        for i in prange(1, ny):
            for j in range(nx):
                eb = z[i - 1, j] + h[i - 1, j]
                et = z[i, j] + h[i, j]
                hf = max(eb, et) - max(z[i - 1, j], z[i, j])
                qn = 0.0
                if hf > hmin:
                    q = qy[i, j]
                    qs = theta * q + 0.5 * (1 - theta) * (qy[i - 1, j] + qy[i + 1, j])
                    nn = 0.5 * (n2[i - 1, j] + n2[i, j])
                    qn = (qs - g * hf * dt * (et - eb) / dx) / (1 + g * dt * nn * abs(q) / hf ** (7.0 / 3.0))
                    lim = frmax * hf * np.sqrt(g * hf)
                    qn = min(max(qn, -lim), lim)
                qyn[i, j] = qn
        for i in prange(ny):
            for j in range(nx):
                out = (max(qxn[i, j + 1], 0.0) + max(-qxn[i, j], 0.0) + max(qyn[i + 1, j], 0.0)
                       + max(-qyn[i, j], 0.0)) * dt / dx
                fac[i, j] = h[i, j] / max(out, 1e-12) if out > h[i, j] else 1.0
        for i in prange(ny):
            for j in range(1, nx):
                q = qxn[i, j]
                qx[i, j] = q * fac[i, j - 1] if q > 0 else q * fac[i, j]
        for i in prange(1, ny):
            for j in range(nx):
                q = qyn[i, j]
                qy[i, j] = q * fac[i - 1, j] if q > 0 else q * fac[i, j]
        acc = 0.0
        for i in prange(ny):
            for j in range(nx):
                hn = h[i, j] + dt / dx * (qx[i, j] - qx[i, j + 1] + qy[i, j] - qy[i + 1, j])
                hn = max(hn, 0.0) + Qin * dt * w_in[i, j]
                if coef[i, j] > 0 and hn > 0:
                    o = min(hn, dt * oscale * coef[i, j] * hn ** (5.0 / 3.0))
                    acc += o
                    hn -= o
                h[i, j] = hn * active[i, j]
        return acc

    return step
