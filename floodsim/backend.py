"""เลือก backend สำหรับคำนวณอาเรย์: CuPy (CUDA) -> PyTorch (CUDA) -> NumPy (CPU)

ตัวแก้สมการ 2 มิติเรียกใช้ฟังก์ชันผ่านอ็อบเจกต์ Backend ตัวเดียว
จึงรันได้ทั้งบน GPU และ CPU โดยไม่ต้องแก้โค้ดส่วนอื่น
"""
from __future__ import annotations

import warnings

import numpy as np

warnings.filterwarnings("ignore", message="CUDA path could not be detected")


class Backend:
    name = "numpy"
    device_name = "CPU"
    is_gpu = False

    def __init__(self):
        self.xp = np
        self.dtype = np.float32

    # --- การสร้าง/แปลงอาเรย์
    def asarray(self, a, dtype=None):
        return self.xp.asarray(a, dtype=dtype or self.dtype)

    def zeros(self, shape):
        return self.xp.zeros(shape, dtype=self.dtype)

    def to_np(self, a):
        return np.asarray(a)

    def asint(self, a):
        return a.astype(np.int64)

    def int_array(self, a):
        return self.xp.asarray(a, dtype=np.int64)

    # --- คณิตศาสตร์
    def maximum(self, a, b):
        return self.xp.maximum(a, b)

    def minimum(self, a, b):
        return self.xp.minimum(a, b)

    def clip(self, a, lo=None, hi=None):
        return self.xp.clip(a, lo, hi)

    def where(self, c, a, b):
        return self.xp.where(c, a, b)

    def sqrt(self, a):
        return self.xp.sqrt(a)

    def floor(self, a):
        return self.xp.floor(a)

    def fsum(self, a) -> float:
        return float(a.sum())

    def fmax(self, a) -> float:
        return float(a.max())

    def rand(self, n):
        return self.xp.random.random(n).astype(self.dtype)

    def randint(self, hi, n):
        return self.xp.random.randint(0, hi, n)

    def sync(self):
        pass


class CupyBackend(Backend):
    name = "cupy"
    is_gpu = True

    def __init__(self):
        import cupy as cp

        cp.zeros(4).sum()  # ทดสอบว่าเรียก kernel ได้จริง (เช่น สถาปัตยกรรม GPU รองรับ)
        self.xp = cp
        self.cp = cp
        self.dtype = np.float32
        props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        name = props["name"]
        self.device_name = name.decode() if isinstance(name, bytes) else str(name)

    def to_np(self, a):
        return self.cp.asnumpy(a)

    def rand(self, n):
        return self.cp.random.random(n, dtype=self.cp.float32)

    def sync(self):
        self.cp.cuda.Stream.null.synchronize()


class TorchBackend(Backend):
    name = "torch"
    is_gpu = True

    def __init__(self):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch ไม่พบ CUDA")
        self.torch = torch
        self.dev = torch.device("cuda")
        torch.zeros(4, device=self.dev).sum().item()
        self.dtype = torch.float32
        self.device_name = torch.cuda.get_device_name(0)

    def asarray(self, a, dtype=None):
        if isinstance(a, self.torch.Tensor):
            return a.to(self.dev, dtype or self.dtype)
        return self.torch.as_tensor(np.asarray(a), dtype=dtype or self.dtype, device=self.dev)

    def zeros(self, shape):
        return self.torch.zeros(shape, dtype=self.dtype, device=self.dev)

    def to_np(self, a):
        return a.detach().cpu().numpy()

    def asint(self, a):
        return a.long()

    def int_array(self, a):
        return self.torch.as_tensor(np.asarray(a), dtype=self.torch.int64, device=self.dev)

    def maximum(self, a, b):
        return self.torch.maximum(a, b)

    def minimum(self, a, b):
        return self.torch.minimum(a, b)

    def clip(self, a, lo=None, hi=None):
        return self.torch.clamp(a, min=lo, max=hi)

    def where(self, c, a, b):
        return self.torch.where(c, a, b)

    def sqrt(self, a):
        return self.torch.sqrt(a)

    def floor(self, a):
        return self.torch.floor(a)

    def rand(self, n):
        return self.torch.rand(n, device=self.dev, dtype=self.dtype)

    def randint(self, hi, n):
        return self.torch.randint(0, hi, (n,), device=self.dev)

    def sync(self):
        self.torch.cuda.synchronize()


_ORDER = {"auto": ["cupy", "torch", "numpy"], "cupy": ["cupy"], "torch": ["torch"], "numpy": ["numpy"]}
_CLASSES = {"cupy": CupyBackend, "torch": TorchBackend, "numpy": Backend}


def get_backend(pref: str = "auto") -> tuple[Backend, list[str]]:
    """คืนค่า (backend, บันทึกเหตุผลที่ข้ามตัวเลือกอื่น)"""
    notes = []
    for key in _ORDER.get(pref, _ORDER["auto"]):
        try:
            return _CLASSES[key](), notes
        except Exception as e:  # noqa: BLE001 - ต้องการ fallback ทุกกรณี
            notes.append(f"{key}: {type(e).__name__}: {str(e).splitlines()[0][:120]}")
    b = Backend()
    return b, notes
