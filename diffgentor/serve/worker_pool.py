# Copyright 2026 Ruihang Li.
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for details.

"""Multi-GPU worker pool for diffgentor serve mode.

Provides two pool implementations:

- ``InProcessPool``: Loads N backend replicas in the main process, each pinned to
  a different ``cuda:i`` device.  Suited for models that fit on a single GPU.

- ``SubprocessPool``: Spawns N child processes, each with an isolated
  ``CUDA_VISIBLE_DEVICES`` slice.  Required for models that use
  ``device_map="auto"`` to shard across multiple GPUs (e.g. emu35,
  hunyuan_image_3).
"""

from __future__ import annotations

import asyncio
import io
import logging
import multiprocessing as mp
import os
import signal
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image

from diffgentor.config import BackendConfig, OptimizationConfig

logger = logging.getLogger("diffgentor.serve.pool")


# ---------------------------------------------------------------------------
# Request / Response containers (picklable for subprocess communication)
# ---------------------------------------------------------------------------


@dataclass
class _GenerateRequest:
    prompt: str
    kwargs: Dict[str, Any]


@dataclass
class _EditRequest:
    images_bytes: List[bytes]
    instruction: str
    kwargs: Dict[str, Any]


@dataclass
class _Response:
    images_bytes: Optional[List[bytes]] = None
    error: Optional[str] = None


def _pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class WorkerPool(ABC):
    """Abstract base for multi-GPU worker pools."""

    @abstractmethod
    async def submit_generate(self, prompt: str, **kwargs: Any) -> List[Image.Image]:
        """Submit a T2I generation request."""

    @abstractmethod
    async def submit_edit(
        self,
        images: Union[Image.Image, List[Image.Image]],
        instruction: str,
        **kwargs: Any,
    ) -> List[Image.Image]:
        """Submit an image editing request."""

    @abstractmethod
    async def shutdown(self) -> None:
        """Gracefully shut down all workers."""


# ---------------------------------------------------------------------------
# InProcessPool — one replica per GPU, all in the main process
# ---------------------------------------------------------------------------


class InProcessPool(WorkerPool):
    """Load N backend replicas (one per GPU) inside the current process.

    Each replica is pinned to ``cuda:i``.  An ``asyncio.Queue`` dispatches
    incoming requests to N concurrent worker coroutines that call the backend
    via ``asyncio.to_thread``.
    """

    def __init__(
        self,
        mode: str,
        backend_config: BackendConfig,
        opt_config: OptimizationConfig,
        num_workers: int,
        gpu_ids: List[int],
    ):
        self._mode = mode
        self._backend_config = backend_config
        self._opt_config = opt_config
        self._num_workers = num_workers
        self._gpu_ids = gpu_ids

        self._backends: list = []
        self._queue: asyncio.Queue[Tuple[Any, asyncio.Future]] = asyncio.Queue()
        self._worker_tasks: List[asyncio.Task] = []

    # -- lifecycle -----------------------------------------------------------

    def load(self) -> None:
        """Load backend replicas on each GPU (called from the main thread)."""
        for i, gpu_id in enumerate(self._gpu_ids[: self._num_workers]):
            cfg = BackendConfig(
                backend=self._backend_config.backend,
                model_name=self._backend_config.model_name,
                model_type=self._backend_config.model_type,
                device=f"cuda:{gpu_id}",
                seed=self._backend_config.seed,
            )
            if self._mode == "t2i":
                from diffgentor.backends.registry import get_backend
                backend = get_backend(cfg, self._opt_config)
            else:
                from diffgentor.backends.editing.registry import get_editing_backend
                backend = get_editing_backend(cfg, self._opt_config)
            logger.info("Loading replica %d on cuda:%d ...", i, gpu_id)
            backend.load_model()
            self._backends.append(backend)
            logger.info("Replica %d ready", i)

    def start_workers(self) -> None:
        """Spawn asyncio worker tasks (must be called inside a running loop)."""
        for i, backend in enumerate(self._backends):
            task = asyncio.create_task(self._worker_loop(i, backend))
            self._worker_tasks.append(task)

    async def _worker_loop(self, worker_id: int, backend: Any) -> None:
        """Pull requests from the queue and execute inference."""
        while True:
            request, future = await self._queue.get()
            try:
                if isinstance(request, _GenerateRequest):
                    result = await asyncio.to_thread(
                        backend.generate, request.prompt, **request.kwargs
                    )
                elif isinstance(request, _EditRequest):
                    imgs = [_bytes_to_pil(b) for b in request.images_bytes]
                    input_imgs: Any = imgs if len(imgs) > 1 else imgs[0]
                    result = await asyncio.to_thread(
                        backend.edit, input_imgs, request.instruction, **request.kwargs
                    )
                else:
                    raise TypeError(f"Unknown request type: {type(request)}")
                future.set_result(result)
            except Exception as exc:
                future.set_exception(exc)
            finally:
                self._queue.task_done()

    # -- public API ----------------------------------------------------------

    async def submit_generate(self, prompt: str, **kwargs: Any) -> List[Image.Image]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[List[Image.Image]] = loop.create_future()
        await self._queue.put((_GenerateRequest(prompt=prompt, kwargs=kwargs), future))
        return await future

    async def submit_edit(
        self,
        images: Union[Image.Image, List[Image.Image]],
        instruction: str,
        **kwargs: Any,
    ) -> List[Image.Image]:
        if isinstance(images, Image.Image):
            images = [images]
        images_bytes = [_pil_to_bytes(img) for img in images]
        loop = asyncio.get_running_loop()
        future: asyncio.Future[List[Image.Image]] = loop.create_future()
        await self._queue.put(
            (_EditRequest(images_bytes=images_bytes, instruction=instruction, kwargs=kwargs), future)
        )
        return await future

    async def shutdown(self) -> None:
        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        for backend in self._backends:
            backend.cleanup()
        self._backends.clear()


# ---------------------------------------------------------------------------
# SubprocessPool — one child process per GPU group
# ---------------------------------------------------------------------------

def _subprocess_worker(
    worker_id: int,
    mode: str,
    backend_config_dict: Dict[str, Any],
    opt_config_dict: Dict[str, Any],
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    cuda_visible_devices: str,
) -> None:
    """Entry point for a worker subprocess.

    Runs in a child process with ``CUDA_VISIBLE_DEVICES`` set to a subset
    of GPUs so that ``device_map="auto"`` distributes across exactly
    the intended devices.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    backend_config = BackendConfig(**backend_config_dict)
    opt_config = OptimizationConfig(**opt_config_dict)

    # Force device to cuda (the worker only sees its assigned GPUs)
    backend_config_local = BackendConfig(
        backend=backend_config.backend,
        model_name=backend_config.model_name,
        model_type=backend_config.model_type,
        device="cuda",
        seed=backend_config.seed,
    )

    if mode == "t2i":
        from diffgentor.backends.registry import get_backend
        backend = get_backend(backend_config_local, opt_config)
    else:
        from diffgentor.backends.editing.registry import get_editing_backend
        backend = get_editing_backend(backend_config_local, opt_config)

    backend.load_model()

    while True:
        try:
            item = request_queue.get()
        except (EOFError, OSError):
            break
        if item is None:
            break

        req_id, request = item
        try:
            if isinstance(request, _GenerateRequest):
                result_imgs = backend.generate(request.prompt, **request.kwargs)
            elif isinstance(request, _EditRequest):
                imgs = [_bytes_to_pil(b) for b in request.images_bytes]
                input_imgs: Any = imgs if len(imgs) > 1 else imgs[0]
                result_imgs = backend.edit(input_imgs, request.instruction, **request.kwargs)
            else:
                raise TypeError(f"Unknown request type: {type(request)}")

            resp = _Response(images_bytes=[_pil_to_bytes(img) for img in result_imgs])
        except Exception as exc:
            resp = _Response(error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")

        response_queue.put((req_id, resp))

    backend.cleanup()


class SubprocessPool(WorkerPool):
    """Spawn N child processes, each with its own ``CUDA_VISIBLE_DEVICES``.

    Required for backends that use ``device_map="auto"`` to shard a single
    model across multiple GPUs (emu35, hunyuan_image_3, etc.).
    """

    def __init__(
        self,
        mode: str,
        backend_config: BackendConfig,
        opt_config: OptimizationConfig,
        num_workers: int,
        gpus_per_model: int,
        gpu_ids: List[int],
    ):
        self._mode = mode
        self._backend_config = backend_config
        self._opt_config = opt_config
        self._num_workers = num_workers
        self._gpus_per_model = gpus_per_model
        self._gpu_ids = gpu_ids

        self._processes: List[mp.Process] = []
        self._request_queues: List[mp.Queue] = []
        self._response_queue: mp.Queue = mp.Queue()
        self._pending: Dict[int, asyncio.Future] = {}
        self._next_id = 0
        self._round_robin = 0
        self._reader_task: Optional[asyncio.Task] = None

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Spawn worker subprocesses."""
        ctx = mp.get_context("spawn")
        self._response_queue = ctx.Queue()

        bc_dict = {
            "backend": self._backend_config.backend,
            "model_name": self._backend_config.model_name,
            "model_type": self._backend_config.model_type,
            "device": self._backend_config.device,
            "seed": self._backend_config.seed,
        }
        oc_dict = self._opt_config.to_dict()

        for i in range(self._num_workers):
            start = i * self._gpus_per_model
            worker_gpus = ",".join(
                str(self._gpu_ids[start + j]) for j in range(self._gpus_per_model)
            )
            req_q: mp.Queue = ctx.Queue()
            self._request_queues.append(req_q)

            p = ctx.Process(
                target=_subprocess_worker,
                args=(i, self._mode, bc_dict, oc_dict, req_q, self._response_queue, worker_gpus),
                daemon=True,
            )
            logger.info("Starting subprocess worker %d on GPUs %s", i, worker_gpus)
            p.start()
            self._processes.append(p)

    def start_reader(self) -> None:
        """Start the asyncio task that reads results from child processes."""
        self._reader_task = asyncio.create_task(self._read_responses())

    async def _read_responses(self) -> None:
        """Poll the response queue and resolve pending futures."""
        loop = asyncio.get_running_loop()
        while True:
            try:
                item = await loop.run_in_executor(None, self._response_queue.get)
            except (EOFError, OSError):
                break
            if item is None:
                break
            req_id, response = item
            future = self._pending.pop(req_id, None)
            if future is None:
                continue
            if response.error:
                future.set_exception(RuntimeError(response.error))
            else:
                future.set_result([_bytes_to_pil(b) for b in response.images_bytes])

    # -- public API ----------------------------------------------------------

    async def submit_generate(self, prompt: str, **kwargs: Any) -> List[Image.Image]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[List[Image.Image]] = loop.create_future()
        req_id = self._next_id
        self._next_id += 1
        self._pending[req_id] = future

        worker_idx = self._round_robin % self._num_workers
        self._round_robin += 1
        self._request_queues[worker_idx].put(
            (req_id, _GenerateRequest(prompt=prompt, kwargs=kwargs))
        )
        return await future

    async def submit_edit(
        self,
        images: Union[Image.Image, List[Image.Image]],
        instruction: str,
        **kwargs: Any,
    ) -> List[Image.Image]:
        if isinstance(images, Image.Image):
            images = [images]
        images_bytes = [_pil_to_bytes(img) for img in images]
        loop = asyncio.get_running_loop()
        future: asyncio.Future[List[Image.Image]] = loop.create_future()
        req_id = self._next_id
        self._next_id += 1
        self._pending[req_id] = future

        worker_idx = self._round_robin % self._num_workers
        self._round_robin += 1
        self._request_queues[worker_idx].put(
            (req_id, _EditRequest(images_bytes=images_bytes, instruction=instruction, kwargs=kwargs))
        )
        return await future

    async def shutdown(self) -> None:
        for q in self._request_queues:
            q.put(None)
        for p in self._processes:
            p.join(timeout=30)
            if p.is_alive():
                p.terminate()
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _get_visible_gpu_ids() -> List[int]:
    """Return the list of GPU ids visible to this process."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        return [int(g.strip()) for g in cuda_visible.split(",") if g.strip()]
    try:
        import torch
        return list(range(torch.cuda.device_count()))
    except (ImportError, Exception):
        return [0]


def detect_gpus_per_model(backend: str, model_type: Optional[str] = None) -> int:
    """Detect how many GPUs a single model replica requires.

    Returns the value from the backend-specific ``DG_*_GPUS_PER_MODEL``
    environment variable.  If the backend has no such setting, returns 1.

    A return value of 0 means "use all visible GPUs for one model" (no data
    parallelism).
    """
    effective = model_type or backend
    if effective == "emu35":
        from diffgentor.utils.env import Emu35Env
        return Emu35Env.gpus_per_model()
    elif effective == "hunyuan_image_3":
        from diffgentor.utils.env import HunyuanImage3Env
        return HunyuanImage3Env.gpus_per_model()
    elif effective == "deepgen":
        from diffgentor.utils.env import DeepGenEnv
        return DeepGenEnv.gpus_per_model()
    return 1


def create_worker_pool(
    mode: str,
    backend_config: BackendConfig,
    opt_config: OptimizationConfig,
    num_gpus: Optional[int] = None,
) -> Optional[WorkerPool]:
    """Create a worker pool if multiple replicas can be launched.

    Returns ``None`` when the configuration results in a single worker
    (i.e. the caller should fall back to loading one backend directly).
    """
    gpu_ids = _get_visible_gpu_ids()
    total_gpus = num_gpus if num_gpus is not None else len(gpu_ids)
    if total_gpus <= 0:
        total_gpus = 1

    gpus_per_model = detect_gpus_per_model(backend_config.backend, backend_config.model_type)

    # gpus_per_model == 0 means the model wants ALL visible GPUs for one instance
    if gpus_per_model == 0:
        return None

    num_workers = total_gpus // gpus_per_model
    if num_workers <= 1:
        return None

    gpu_ids = gpu_ids[:total_gpus]

    if gpus_per_model == 1:
        pool = InProcessPool(mode, backend_config, opt_config, num_workers, gpu_ids)
        pool.load()
        return pool
    else:
        pool = SubprocessPool(mode, backend_config, opt_config, num_workers, gpus_per_model, gpu_ids)
        pool.start()
        return pool
