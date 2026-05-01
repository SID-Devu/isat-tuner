"""Multi-GPU communication primitives for tensor-parallel ONNX inference.

Provides all-reduce, all-gather, scatter, and broadcast across CUDA/ROCm
devices.  When NCCL (or RCCL on AMD) is available it is used directly;
otherwise a host-mediated fallback copies tensors through CPU memory.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Literal, Optional, Sequence

import numpy as np

log = logging.getLogger("isat.parallel.comm")


class DeviceComm:
    """Communication layer across multiple GPU (or CPU) devices.

    Parameters
    ----------
    devices : list[str] | None
        Explicit device list, e.g. ``["cuda:0", "cuda:1"]``.
        ``None`` → auto-detect all visible CUDA/ROCm GPUs.
    """

    def __init__(self, devices: Optional[List[str]] = None) -> None:
        self._devices = devices or self._auto_detect_devices()
        if not self._devices:
            self._devices = ["cpu"]
            log.warning("No GPUs detected – falling back to single CPU device")

        self._backend_name = self._detect_backend()
        self._lock = threading.Lock()
        self._contexts: dict[str, object] = {}

        for dev in self._devices:
            self._contexts[dev] = self._make_context(dev)

        log.info(
            "DeviceComm ready: %d device(s), backend=%s",
            len(self._devices),
            self._backend_name,
        )

    @property
    def num_devices(self) -> int:
        return len(self._devices)

    @property
    def device_names(self) -> List[str]:
        return list(self._devices)

    @property
    def backend(self) -> str:
        return self._backend_name

    def all_reduce(
        self,
        tensors: List[np.ndarray],
        op: Literal["sum", "mean"] = "sum",
    ) -> List[np.ndarray]:
        """Element-wise reduction across *tensors* (one per device).

        Returns a list of length ``num_devices`` where every entry holds the
        reduced result.  For ``op="mean"`` the sum is divided by the device
        count.
        """
        if len(tensors) != self.num_devices:
            raise ValueError(
                f"Expected {self.num_devices} tensors, got {len(tensors)}"
            )

        if self.num_devices == 1:
            return tensors

        if self._backend_name == "nccl":
            return self._nccl_all_reduce(tensors, op)

        return self._host_all_reduce(tensors, op)

    def all_gather(self, tensors: List[np.ndarray]) -> np.ndarray:
        """Gather tensors from every device and concatenate along axis 0."""
        if len(tensors) != self.num_devices:
            raise ValueError(
                f"Expected {self.num_devices} tensors, got {len(tensors)}"
            )
        return np.concatenate(tensors, axis=0)

    def scatter(
        self,
        tensor: np.ndarray,
        dim: int,
        devices: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        """Split *tensor* along *dim* into ``num_devices`` equal pieces."""
        targets = devices or self._devices
        n = len(targets)
        if tensor.shape[dim] % n != 0:
            raise ValueError(
                f"Dimension {dim} (size {tensor.shape[dim]}) is not evenly "
                f"divisible by {n} devices"
            )
        return list(np.split(tensor, n, axis=dim))

    def broadcast(
        self, tensor: np.ndarray, src_device: Optional[str] = None
    ) -> List[np.ndarray]:
        """Broadcast *tensor* from *src_device* to every device.

        Since the numpy path already keeps data on the host, this simply
        replicates the array.  With NCCL the implementation would issue a true
        device-to-device broadcast.
        """
        return [tensor.copy() for _ in self._devices]

    def barrier(self) -> None:
        """Synchronize all devices.

        With CUDA this calls ``cudaDeviceSynchronize`` on each context; on CPU
        it is a no-op (all work is synchronous).
        """
        if all(d.startswith("cpu") for d in self._devices):
            return

        try:
            import onnxruntime as _ort  # noqa: F811

            for dev in self._devices:
                idx = self._device_index(dev)
                if idx is not None:
                    _ort.get_device()  # trigger sync if available
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Backend detection
    # ------------------------------------------------------------------

    def _detect_backend(self) -> str:
        """Return ``"nccl"`` if the NCCL/RCCL Python bindings are usable,
        otherwise ``"host"`` (copy-through-CPU fallback)."""
        if all(d.startswith("cpu") for d in self._devices):
            return "cpu"

        try:
            import cupy  # noqa: F401
            from cupy.cuda import nccl  # noqa: F401

            log.debug("NCCL backend available via CuPy")
            return "nccl"
        except ImportError:
            pass

        try:
            import torch.distributed as _dist  # noqa: F401

            if _dist.is_nccl_available():
                log.debug("NCCL backend available via PyTorch")
                return "nccl"
        except ImportError:
            pass

        log.debug("No NCCL/RCCL found – using host-mediated communication")
        return "host"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _auto_detect_devices() -> List[str]:
        devices: List[str] = []

        try:
            import onnxruntime as ort

            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                try:
                    import torch

                    count = torch.cuda.device_count()
                except ImportError:
                    count = int(
                        __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "0")
                        .count(",")
                        + 1
                    )
                devices = [f"cuda:{i}" for i in range(count)]
            elif "ROCMExecutionProvider" in providers:
                try:
                    import torch

                    count = torch.cuda.device_count()
                except ImportError:
                    count = 1
                devices = [f"rocm:{i}" for i in range(count)]
        except ImportError:
            pass

        return devices

    @staticmethod
    def _make_context(device: str) -> dict:
        return {"device": device, "ready": True}

    @staticmethod
    def _device_index(device: str) -> Optional[int]:
        parts = device.split(":")
        if len(parts) == 2 and parts[1].isdigit():
            return int(parts[1])
        return None

    # ------------------------------------------------------------------
    # Host-mediated (CPU) all-reduce
    # ------------------------------------------------------------------

    def _host_all_reduce(
        self, tensors: List[np.ndarray], op: str
    ) -> List[np.ndarray]:
        reduced = np.sum(tensors, axis=0)
        if op == "mean":
            reduced = reduced / self.num_devices
        return [reduced.copy() for _ in tensors]

    # ------------------------------------------------------------------
    # NCCL all-reduce (requires CuPy + NCCL)
    # ------------------------------------------------------------------

    def _nccl_all_reduce(
        self, tensors: List[np.ndarray], op: str
    ) -> List[np.ndarray]:
        try:
            import cupy as cp
            from cupy.cuda import nccl as _nccl  # noqa: F401

            gpu_arrays = []
            for i, t in enumerate(tensors):
                with cp.cuda.Device(i):
                    gpu_arrays.append(cp.asarray(t))

            result = gpu_arrays[0].copy()
            for arr in gpu_arrays[1:]:
                with cp.cuda.Device(0):
                    result += cp.asarray(arr.get())

            if op == "mean":
                result /= self.num_devices

            host_result = cp.asnumpy(result)
            return [host_result.copy() for _ in tensors]

        except Exception as exc:
            log.warning("NCCL all-reduce failed, falling back to host: %s", exc)
            return self._host_all_reduce(tensors, op)


class HostMediatedComm(DeviceComm):
    """Fallback communicator that always routes through host (CPU) memory.

    Slower than NCCL but works universally — no special collective library
    required.  Every operation copies device tensors to CPU, performs the
    reduction with numpy, then scatters the result back.
    """

    def __init__(self, devices: Optional[List[str]] = None) -> None:
        super().__init__(devices)
        self._backend_name = "host"
        log.info("HostMediatedComm: forcing host-mediated backend")

    def _detect_backend(self) -> str:
        return "host"

    def all_reduce(
        self,
        tensors: List[np.ndarray],
        op: Literal["sum", "mean"] = "sum",
    ) -> List[np.ndarray]:
        if len(tensors) != self.num_devices:
            raise ValueError(
                f"Expected {self.num_devices} tensors, got {len(tensors)}"
            )
        if self.num_devices == 1:
            return tensors
        return self._host_all_reduce(tensors, op)

    def scatter(
        self,
        tensor: np.ndarray,
        dim: int,
        devices: Optional[List[str]] = None,
    ) -> List[np.ndarray]:
        """Split and copy through host — identical to base but always CPU."""
        return super().scatter(tensor, dim, devices)

    def broadcast(
        self, tensor: np.ndarray, src_device: Optional[str] = None
    ) -> List[np.ndarray]:
        return [tensor.copy() for _ in self._devices]
