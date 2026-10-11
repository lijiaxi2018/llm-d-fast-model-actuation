#!/usr/bin/env python3
# Copyright 2025 The llm-d Authors.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# 	http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
vLLM Launcher
"""

import asyncio
import ctypes
import logging
import json
import multiprocessing
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from http import HTTPStatus  # HTTP Status Codes
from typing import Dict, List, Literal, Optional

import psutil
import uvloop
from fastapi import FastAPI, Header, HTTPException, Path, Query
from fastapi.responses import (
    JSONResponse,
    PlainTextResponse,
    Response,
    StreamingResponse,
)
from gputranslator import GpuTranslator
from pydantic import BaseModel
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.serve.utils.api_utils import cli_env_setup
from vllm.utils.argparse_utils import FlexibleArgumentParser

MAX_LOG_RESPONSE_BYTES = 1 * 1024 * 1024  # 1 MB default for API response
_MAX_BROADCASTER_EVENTS = 1000
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, force=True)


class LogRangeNotAvailable(Exception):
    """Raised when the requested start_byte is beyond available log content"""

    def __init__(self, start_byte, available_bytes):
        self.start_byte = start_byte
        self.available_bytes = available_bytes
        super().__init__(
            f"start_byte {start_byte} is beyond available content "
            f"({available_bytes} bytes available)"
        )


# Define a the expected JSON structure in dataclass
class VllmConfig(BaseModel):
    options: str
    gpu_uuids: Optional[List[str]] = None
    env_vars: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None


class WatchEvent(BaseModel):
    """Represents an instance lifecycle event for the watch stream."""

    type: str  # "CREATED", "MODIFIED", "STOPPED", "DELETED"
    object: dict


class RevisionTooOld(Exception):
    """Raised when a requested watch revision has fallen out of the buffer."""

    def __init__(self, requested: int, oldest: int):
        super().__init__()
        self.requested = requested
        self.oldest = oldest


class EventBroadcaster:
    """Fans out WatchEvents to all connected watchers using asyncio.Condition.

    Each watcher maintains its own cursor (revision) so late joiners can
    catch up with events still in the buffer.
    """

    def __init__(self):
        self._condition = asyncio.Condition()
        self._events: List[WatchEvent] = []
        # Tracks the revision of the most recently published event
        # (0 = no events published yet).  Revision numbers are assigned
        # by the VllmMultiProcessManager; the broadcaster only records them.
        self._revision: int = 0

    @property
    def revision(self) -> int:
        return self._revision

    @property
    def oldest_revision(self) -> int:
        """Exclusive lower bound: events in the buffer have revision
        strictly greater than this value."""
        return self._revision - len(self._events)

    def _append(self, event: WatchEvent):
        """Synchronously buffer event and advance the revision cursor."""
        self._revision = event.object["revision"]
        self._events.append(event)
        if len(self._events) > _MAX_BROADCASTER_EVENTS:
            self._events = self._events[-_MAX_BROADCASTER_EVENTS:]

    async def _notify(self):
        """Acquire condition lock and wake all watchers."""
        async with self._condition:
            self._condition.notify_all()

    async def watch(self, since_revision: int = 0):
        """Async generator that yields WatchEvents as they arrive.

        :param since_revision: Resume from this revision.  Events with
            revision > since_revision are yielded.
        :raises RevisionTooOld: If/When since_revision is older than the
            oldest event still in the buffer.
        """
        oldest = self.oldest_revision
        if since_revision < oldest:
            raise RevisionTooOld(since_revision, oldest)
        pos = since_revision
        while True:
            async with self._condition:
                while pos >= self._revision:
                    await self._condition.wait()
                offset = pos - self.oldest_revision
                if offset < 0:
                    raise RevisionTooOld(pos, self.oldest_revision)
                new_events = self._events[offset:]
                pos = self._revision
            for event in new_events:
                yield event


class HalfMade(Exception):
    """Raised when something other than start is the first op on a VllmInstance"""

    def __init__(self, instance_id):
        super().__init__()
        self.instance_id = instance_id


def dump_process_group(subject: str, pgpid: int, when: str) -> set[psutil.Process]:
    members = []
    ans: set[psutil.Process] = set()
    for proc in psutil.process_iter(attrs=["pid", "ppid", "name", "cmdline"]):
        pid = proc.info["pid"]
        ppid = proc.info["ppid"]
        pname = proc.info["name"]
        cmdline = proc.info["cmdline"]
        if isinstance(cmdline, list) and len(cmdline) > 4:
            # 4 is enough to resolve the mysteries that we have had
            cmdline = cmdline[:4] + ["..."]
        try:
            proc_pg = os.getpgid(pid)
            if proc_pg == pgpid:
                members.append([pid, ppid, pname, cmdline])
                ans.add(proc)
        except OSError as exn:
            logger.debug(
                f"Failed to os.getpgid({pid} {ppid} {pname}), errno={exn.errno}"
            )
        except Exception as exn:
            logger.debug(f"Failed to os.getpgid({pid} {ppid} {pname}), exn={exn}")
    logger.debug(
        "Members of process group for "
        + subject
        + f" ({pgpid}) "
        + when
        + f" are: {members}"
    )
    return ans

class ManagedProcess:
    """A Linux process, including a root restored as CRIU's sibling."""

    def __init__(self, pid, popen=None):
        self.pid = pid
        self._proc = psutil.Process(pid)
        self._popen = popen
        self.sentinel = os.pidfd_open(pid)
        self._exitcode = None

    def is_alive(self):
        if self._popen is not None and self._popen.poll() is not None:
            self._exitcode = self._popen.returncode
            return False
        try:
            return (
                self._proc.is_running()
                and self._proc.status() != psutil.STATUS_ZOMBIE
            )
        except psutil.NoSuchProcess:
            return False

    @property
    def exitcode(self):
        if not self.is_alive():
            self.join(0)
        return self._exitcode

    def terminate(self):
        self._proc.terminate()

    def join(self, timeout=None):
        try:
            if self._popen is not None:
                self._exitcode = self._popen.wait(timeout=timeout)
            else:
                self._exitcode = self._proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, psutil.TimeoutExpired):
            pass

    def close(self):
        if self.sentinel is not None:
            os.close(self.sentinel)
            self.sentinel = None

class VllmInstance:
    """Represents a single vLLM instance"""

    def __init__(
        self,
        instance_id: str,
        config: VllmConfig,
        gpu_translator: GpuTranslator,
        log_dir: str = "",
        debug_gpu_memory: bool = False,
    ):
        """
        Initialize VllmInstance object
        :param instance_id: Instance id (autogenerated or custom)
        :param config: VllmConfig object
        :param gpu_translator: GpuTranslator object
        :param log_dir: Directory for log files (defaults to /tmp)
        """

        # Check for CUDA device UUIDs and set CUDA_VISIBLE_DEVICES accordingly
        if config.gpu_uuids:
            cuda_indices = []
            for uuid_str in config.gpu_uuids:
                index = gpu_translator.uuid_to_index(uuid_str)
                cuda_indices.append(str(index))
            logger.info(
                f"Translated GPU UUIDs {config.gpu_uuids} to indices {cuda_indices}."
            )

            if config.env_vars is None:
                config.env_vars = {}
            config.env_vars["CUDA_VISIBLE_DEVICES"] = ",".join(cuda_indices)
            logger.info(
                "Set CUDA_VISIBLE_DEVICES to '%s' based on UUIDs.",
                config.env_vars["CUDA_VISIBLE_DEVICES"],
            )

        # Initialize instance variables
        self.instance_id = instance_id
        self.config = config
        self.process: Optional[ManagedProcess] = None
        self.last_revision: Optional[int] = None
        self._sentinel_active = False
        self._log_file_path = os.path.join(
            log_dir or "/tmp",
            f"launcher-{os.getpid()}-vllm-{instance_id}.log",
        )
        self.debug_gpu_memory = debug_gpu_memory
        self.cuda_lock = threading.Lock()
        self._tier = "running"
        self.operation = None

        self._disk_valid = False
        self._disk_dir = None
        self._dump_pids = []

        root = os.path.abspath(
            os.environ.get("FMA_CRIU_ROOT", "/tmp/fma-criu")
        )
        os.makedirs(root, mode=0o700, exist_ok=True)

        # Generated names avoid using the user-supplied instance ID as a path.
        self._work_dir = tempfile.mkdtemp(prefix="instance-", dir=root)

    def _nvidia_fd_report(self):
        if self.process is None or not self.process.is_alive():
            raise HTTPException(409, "The vLLM process tree is not alive")

        report = []

        try:
            root = psutil.Process(self.process.pid)
            processes = [root, *root.children(recursive=True)]

            for proc in processes:
                if (
                    not proc.is_running()
                    or proc.status() == psutil.STATUS_ZOMBIE
                ):
                    raise HTTPException(
                        409,
                        f"Process {proc.pid} exited during inspection",
                    )

                fds = []
                directory = f"/proc/{proc.pid}/fd"

                for name in os.listdir(directory):
                    try:
                        target = os.readlink(f"{directory}/{name}")
                    except FileNotFoundError:
                        # The FD closed between listdir() and readlink().
                        continue

                    if target.startswith("/dev/nvidia"):
                        fds.append({
                            "fd": int(name),
                            "target": target,
                        })

                with open(
                    f"/proc/{proc.pid}/maps",
                    encoding="utf-8",
                ) as stream:
                    mappings = [
                        line.strip()
                        for line in stream
                        if "/dev/nvidia" in line
                    ]

                report.append({
                    "pid": proc.pid,
                    "name": proc.name(),
                    "nvidia_fds": fds,
                    "nvidia_mappings": mappings,
                })

        except (OSError, psutil.Error) as exc:
            raise HTTPException(
                409,
                f"Cannot inspect the complete process tree: {exc}",
            ) from exc

        return report

    def _prepare_tree_for_criu(self):
        report = self._nvidia_fd_report()

        logger.info(
            "NVIDIA resources before CRIU: %s",
            report,
        )

        if any(
            item["nvidia_fds"] or item["nvidia_mappings"]
            for item in report
        ):
            raise HTTPException(
                409,
                {
                    "message": (
                        "NVIDIA resources remain in the process tree; "
                        "CRIU dump was not started"
                    ),
                    "processes": report,
                },
            )

        return report

    def cuda_control(self, action: str = "status") -> dict:
        if action not in {"status", "checkpoint", "restore"}:
            raise HTTPException(400, "Invalid CUDA action")

        if self.process is None or not self.process.is_alive():
            raise HTTPException(409, "The vLLM process tree is not alive")

        started = time.monotonic()
        binary = os.environ.get("CUDA_CHECKPOINT_BIN", "cuda-checkpoint")

        try:
            root = psutil.Process(self.process.pid)
            engines = []

            for proc in root.children(recursive=True):
                titles = [proc.name(), *proc.cmdline()[:1]]
                if any(t.startswith("VLLM::EngineCor") for t in titles):
                    if os.getpgid(proc.pid) == root.pid:
                        engines.append(proc)

            if len(engines) != 1:
                raise HTTPException(
                    409, f"Expected one EngineCore; found {len(engines)}"
                )

            engine = engines[0]
            processes = [root, engine]

            def run_cuda(proc, *args):
                if (
                    not proc.is_running()
                    or proc.status() == psutil.STATUS_ZOMBIE
                ):
                    raise HTTPException(409, f"Process {proc.pid} exited")

                try:
                    cp = subprocess.run(
                        [binary, *args, "--pid", str(proc.pid)],
                        capture_output=True,
                        text=True,
                        timeout=120,
                    )
                except subprocess.TimeoutExpired as exc:
                    raise HTTPException(
                        504, f"CUDA command timed out for PID {proc.pid}"
                    ) from exc
                except OSError as exc:
                    raise HTTPException(503, str(exc)) from exc

                if cp.returncode:
                    raise HTTPException(
                        502,
                        {
                            "pid": proc.pid,
                            "command": list(args),
                            "returncode": cp.returncode,
                            "stdout": cp.stdout[-2000:],
                            "stderr": cp.stderr[-2000:],
                        },
                    )

                return cp.stdout.strip().lower()

            before = {
                proc.pid: run_cuda(proc, "--get-state")
                for proc in processes
            }
            after = dict(before)
            changed = False

            if action != "status":
                if any(
                    state not in {"running", "checkpointed"}
                    for state in before.values()
                ):
                    raise HTTPException(
                        409,
                        {
                            "message": "CUDA state is not stable",
                            "states": before,
                        },
                    )

                target = (
                    "checkpointed" if action == "checkpoint" else "running"
                )
                order = (
                    [engine, root]
                    if action == "checkpoint"
                    else [root, engine]
                )

                # A partial transition must not be reported as successful.
                self._tier = "error"

                for proc in order:
                    if before[proc.pid] != target:
                        run_cuda(proc, "--toggle")
                        changed = True

                after = {
                    proc.pid: run_cuda(proc, "--get-state")
                    for proc in processes
                }

                if any(state != target for state in after.values()):
                    raise HTTPException(
                        502, {"expected": target, "states": after}
                    )

            states = set(after.values())
            cuda_state = (
                next(iter(states)) if len(states) == 1 else "mixed"
            )

            self._tier = {
                "running": "running",
                "checkpointed": "checkpointed_cpu",
            }.get(cuda_state, "error")

            return {
                "instance_id": self.instance_id,
                "engine_pid": engine.pid,
                "api_server_pid": root.pid,
                "cuda_state": cuda_state,
                "changed": changed,
                "processes": [
                    {
                        "pid": proc.pid,
                        "role": (
                            "api_server" if proc.pid == root.pid else "engine"
                        ),
                        "previous_cuda_state": before[proc.pid],
                        "cuda_state": after[proc.pid],
                    }
                    for proc in processes
                ],
                "elapsed_seconds": time.monotonic() - started,
            }

        except (psutil.Error, OSError) as exc:
            self._tier = "error"
            raise HTTPException(
                409, f"Cannot inspect CUDA processes: {exc}"
            ) from exc
        except Exception:
            self._tier = "error"
            raise
    
    def _run_criu(self, command, *args):
        # CUDA is managed explicitly by cuda_control().
        # An empty plugin directory prevents automatic CUDA restore.
        plugins = os.path.join(self._work_dir, "empty-plugins")
        os.makedirs(plugins, exist_ok=True)

        log = os.path.join(self._disk_dir, command + ".log")
        env = os.environ.copy()
        env.pop("CRIU_CONFIG_FILE", None)

        try:
            result = subprocess.run(
                [
                    os.environ.get("CRIU_BIN", "criu"),
                    command,
                    "--no-default-config",
                    "--libdir", plugins,
                    "--images-dir", self._disk_dir,
                    "--log-file", log,
                    "-v4",
                    "--file-locks",
                    "--link-remap",
                    "--tcp-close",
                    *args,
                ],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                text=True,
                env=env,
            )
        except OSError as exc:
            raise HTTPException(503, str(exc)) from exc

        if result.returncode:
            raise HTTPException(
                502,
                {
                    "command": command,
                    "returncode": result.returncode,
                    "stderr": result.stderr[-2000:],
                    "log": log,
                },
            )

    def disk_control(self, action, resume: bool = False):
        # Called only through the serialized manager operation.
        if action == "checkpoint":
            if self._tier == "checkpointed_disk" and self._disk_valid:
                return

            if self._tier not in {"running", "checkpointed_cpu"}:
                raise HTTPException(
                    409, f"Cannot checkpoint to disk from {self._tier}"
                )

            if self.process is None or not self.process.is_alive():
                raise HTTPException(409, "The process tree is not alive")

            if self.cuda_control("checkpoint")["cuda_state"] != "checkpointed":
                raise HTTPException(
                    409, "CUDA must be checkpointed before CRIU"
                )

            self._prepare_tree_for_criu()

            root = psutil.Process(self.process.pid)
            tree = [root, *root.children(recursive=True)]

            if os.getsid(root.pid) != root.pid:
                raise HTTPException(
                    409, "Recreate instance with the new start()"
                )

            # # No active inference connections in this first prototype.
            # for proc in tree:
            #     for conn in proc.net_connections(kind="tcp"):
            #         if conn.status != psutil.CONN_LISTEN:
            #             raise HTTPException(
            #                 409, "Close inference TCP connections first"
            #             )

            self._disk_dir = tempfile.mkdtemp(
                prefix="dump-", dir=self._work_dir
            )
            self._dump_pids = [proc.pid for proc in tree]

            self._tier = "error"

            # Default dump kills the saved tree after a successful checkpoint.
            # Do not use --leave-running or --leave-stopped.
            self._run_criu("dump", "--tree", str(root.pid))

            self.process.join(10)

            for proc in tree[1:]:
                try:
                    proc.wait(timeout=10)
                except psutil.NoSuchProcess:
                    pass

            if self.process.is_alive():
                raise HTTPException(
                    502, "CRIU returned but root is still alive"
                )

            self.process.close()

            # Flush image contents before reporting a completed disk checkpoint.
            for directory, _, names in os.walk(self._disk_dir):
                for name in names:
                    with open(os.path.join(directory, name), "rb") as stream:
                        os.fsync(stream.fileno())

            self._disk_valid = True
            self._tier = "checkpointed_disk"
            return

        if not self._disk_valid or self._tier != "checkpointed_disk":
            raise HTTPException(409, "No unused disk checkpoint")

        occupied = [
            pid for pid in self._dump_pids
            if psutil.pid_exists(pid)
        ]
        if occupied:
            raise HTTPException(
                409, {"restore_pids_occupied": occupied}
            )

        pidfile = os.path.join(self._disk_dir, "restore.pid")

        # Restore can consume remap files even if it fails.
        # This prototype does not blindly retry the same image.
        self._disk_valid = False
        self._tier = "error"

        self._run_criu(
            "restore",
            "--restore-detached",
            "--restore-sibling",
            "--pidfile", pidfile,
        )

        with open(pidfile) as stream:
            pid = int(stream.read().strip())

        self.process = ManagedProcess(pid)

        if not self.process.is_alive():
            raise HTTPException(502, "Restored root exited")

        self._tier = "checkpointed_cpu"

        result = self.cuda_control("status")
        if result["cuda_state"] != "checkpointed":
            raise HTTPException(
                502, "Restored CUDA state was not checkpointed"
            )

        if resume:
            self.cuda_control("restore")

    def _make_state(self, status: str) -> dict:
        return {
            "status": status,
            "instance_id": self.instance_id,
            "revision": self.last_revision,
            "operation": self.operation,
            "checkpoint_dir": self._disk_dir,
            **self.config.model_dump(exclude_none=True),
        }

    def start(self) -> dict:
        if self.process and self.process.is_alive():
            return self.get_status()

        config_path = os.path.join(self._work_dir, "worker.json")
        with open(config_path, "w") as stream:
            json.dump(
                {
                    "config": self.config.model_dump(exclude_none=True),
                    "log": self._log_file_path,
                    "debug": self.debug_gpu_memory,
                },
                stream,
            )
        os.chmod(config_path, 0o600)

        with open(self._log_file_path, "wb") as log:
            child = subprocess.Popen(
                [
                    sys.executable,
                    os.path.abspath(__file__),
                    "--worker-config",
                    config_path,
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                close_fds=True,
                start_new_session=True,
            )

        self.process = ManagedProcess(child.pid, child)
        self._tier = "running"
        return self.get_status()

    def stop(self, timeout: int = 10) -> dict:
        """
        Stop existing vLLM instance
        :param timeout: waits for the process to stop, defaults to 10
        :return: a dictionary with the status "terminated"
        """
        if self.process is None:
            raise HalfMade(self.instance_id)
        if not self.process.is_alive():
            self._cleanup_log_file()
            return self._make_state("not_running")

        vllm_pid = self.process.pid
        assert vllm_pid is not None

        dump_process_group(self.instance_id, vllm_pid, "before process.terminate")
        # psutil.Process hashes according to (PID, create time)
        termd: set[psutil.Process] = {psutil.Process(vllm_pid)}
        killd: set[psutil.Process] = set()

        # Graceful termination — send SIGTERM to the vLLM process,
        # which will propagate shutdown to the EngineCore via vLLM's
        # own cleanup logic.
        try:
            self.process.terminate()
        except psutil.Error as exn:
            logger.error(f"For instance {self.instance_id}, terminate threw {exn}")
        self.process.join(timeout=timeout)
        rounds = 1
        while rounds < 3:
            remains = dump_process_group(
                self.instance_id, vllm_pid, f"after {rounds} rounds of stopping"
            )
            for proc in remains:
                try:
                    if proc.pid == 1:
                        logger.error(
                            f"Somehow PID 1 got into the process group ({vllm_pid}) of"
                            f" instance {self.instance_id}, excepting it from stopping"
                        )
                    elif proc in killd:
                        logger.info(
                            f"Stopping pid {proc.pid} with SIGKILL AGAIN because"
                            " once was not enough"
                        )
                        proc.kill()
                    elif proc in termd:
                        logger.info(
                            f"Stopping pid {proc.pid} with SIGKILL because"
                            " SIGTERM was not enough"
                        )
                        killd.add(proc)
                        proc.kill()
                    else:
                        logger.info(
                            f"Stopping pid {proc.pid} with SIGTERM because it remains"
                        )
                        termd.add(proc)
                        proc.terminate()
                except psutil.Error as exn:
                    logger.error(
                        f"For instance {self.instance_id},"
                        f" terminate/kill({proc.pid} threw {exn})"
                    )
            psutil.wait_procs(remains, timeout=timeout)
            rounds = rounds + 1
        finals = dump_process_group(self.instance_id, vllm_pid, "at end of stopping")
        if len(finals) > 0:
            logger.error(
                f"Giving up on stopping processes for instance {self.instance_id},"
                f" {finals} remain"
            )
        self._cleanup_log_file()
        return self._make_state("terminated")

    def _on_sentinel_exit(self):
        """Handle process exit detected by the sentinel fd.

        Removes the reader, collects the exit code, and invokes the
        registered *_on_exit_callback(instance_id, exitcode)*.
        """
        if not self._sentinel_active:
            return

        loop = asyncio.get_running_loop()
        loop.remove_reader(self.process.sentinel)
        self._sentinel_active = False
        self._on_exit_callback(self.instance_id, self.process.exitcode)

        self.process.close()

    def start_sentinel_watcher(self, on_exit_callback):
        """Register a sentinel fd on the event loop to detect process exit.

        When the child process terminates, the kernel makes the sentinel fd
        readable.  The handler invokes *on_exit_callback(instance_id, exitcode)*.
        """
        if self.process is None:
            raise HalfMade(self.instance_id)

        self._on_exit_callback = on_exit_callback
        loop = asyncio.get_running_loop()
        loop.add_reader(self.process.sentinel, self._on_sentinel_exit)
        self._sentinel_active = True

    def cancel_sentinel_watcher(self):
        """Remove the sentinel reader if it is still registered."""
        if self._sentinel_active and self.process is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.remove_reader(self.process.sentinel)
            except RuntimeError:
                pass
            self._sentinel_active = False

    def _cleanup_log_file(self):
        """Remove the log file if it exists."""
        self._tier = "stopped"
        self._disk_valid = False

        if self.process is not None:
            self.process.close()

        try:
            os.unlink(self._log_file_path)
        except FileNotFoundError:
            pass

    def get_status(self) -> dict:
        """
        Returns the status of the process
        :return: Status of the running process.
        """
        if self.operation:
            return self._make_state(self._tier)

        if self._tier == "checkpointed_disk" and self._disk_valid:
            return self._make_state("checkpointed_disk")

        if self._tier == "error":
            return self._make_state("error")

        if self.process is None:
            raise HalfMade(self.instance_id)

        return self._make_state(
            self._tier if self.process.is_alive() else "stopped"
        )

    def get_log_bytes(
        self, start: int | None = 0, end: int | None = None
    ) -> tuple[int, bytes, int]:
        """
        Retrieve log bytes from the child process.
        :param start: First byte to read (inclusive, 0-based) or None for suffix.
        :param end: if start is None then suffix length otherwise
                    Last byte to read (inclusive, must be >= start).
                    None means up to start + MAX_LOG_RESPONSE_BYTES - 1
                    or EOF, whichever comes first; may not be None if start is None.
        :return: (read_start, content_bytes, current_total_log_length)
        :raises LogRangeNotAvailable: If start is beyond available content
        """
        try:
            total = os.path.getsize(self._log_file_path)
        except FileNotFoundError:
            raise LogRangeNotAvailable(start, 0)

        if start is not None and start >= total:
            raise LogRangeNotAvailable(start, total)

        if start is None:
            if end is None:
                raise LogRangeNotAvailable(None, total)
            read_start = max(total - end, 0)
            read_end = total - 1
        elif end is None:
            read_start = start
            read_end = min(start + MAX_LOG_RESPONSE_BYTES - 1, total - 1)
        else:
            read_start = start
            read_end = min(end, total - 1)

        nbytes = read_end - read_start + 1
        with open(self._log_file_path, "rb") as f:
            f.seek(read_start)
            data = f.read(nbytes)
        return (read_start, data, total)


# Multi-instance vLLM process manager
class VllmMultiProcessManager:
    def __init__(
        self,
        mock_gpus: bool = False,
        mock_gpu_count: int = 8,
        node_name: Optional[str] = None,
        namespace: Optional[str] = None,
        log_dir: str = "",
        debug_gpu_memory: bool = False,
    ):
        self.instances: Dict[str, VllmInstance] = {}
        self.broadcaster = EventBroadcaster()
        # Monotonically increasing counter owned by the manager.
        # The manager stamps every event before handing it to the
        # broadcaster, so revision assignment is synchronous and
        # race-free with respect to the returned API response.
        self._revision: int = 0
        self.gpu_translator = GpuTranslator(
            mock_gpus=mock_gpus,
            node_name=node_name,
            namespace=namespace,
            mock_gpu_count=mock_gpu_count,
        )
        self.log_dir = log_dir
        self.debug_gpu_memory = debug_gpu_memory

    @property
    def revision(self) -> int:
        return self._revision

    def _next_revision(self) -> int:
        self._revision += 1
        return self._revision

    def _on_instance_stopped(self, instance_id: str, exitcode):
        """Sentinel callback: assign revision and publish a STOPPED event."""
        revision = self._next_revision()
        instance = self.instances[instance_id]
        instance.last_revision = revision
        obj = instance.get_status()
        obj["exit_code"] = exitcode
        event = WatchEvent(type="STOPPED", object=obj)
        self.broadcaster._append(event)
        loop = asyncio.get_running_loop()
        loop.create_task(self.broadcaster._notify())

    def create_instance(
        self, vllm_config: VllmConfig, instance_id: Optional[str] = None
    ) -> dict:
        """Create and start a new vLLM instance"""
        if any(
            i._disk_valid or (i.operation or "").startswith("disk_")
            for i in self.instances.values()
        ):
            raise ValueError(
                "Restore the disk-checkpointed instance before creating another"
            )

        if instance_id is None:
            instance_id = str(uuid.uuid4())

        if instance_id in self.instances:
            logger.warning(
                "Rejecting request to create vLLM instance: id=%s already exists",
                instance_id,
            )
            raise ValueError(f"Instance with ID {instance_id} already exists")

        logger.info("Accepted request to create vLLM instance with id=%s", instance_id)

        instance = VllmInstance(
            instance_id,
            vllm_config,
            self.gpu_translator,
            self.log_dir,
            self.debug_gpu_memory,
        )
        self.instances[instance_id] = instance

        try:
            instance.start()
        except Exception:
            self.instances.pop(instance_id, None)
            raise

        revision = self._next_revision()
        instance.last_revision = revision
        result = instance.get_status()
        event = WatchEvent(type="CREATED", object=result)
        self.broadcaster._append(event)
        try:
            loop = asyncio.get_running_loop()
            instance.start_sentinel_watcher(self._on_instance_stopped)
            loop.create_task(self.broadcaster._notify())
        except RuntimeError:
            pass  # No running event loop (e.g. in sync tests)
        return result

    def stop_instance(self, instance_id: str, timeout: int = 10) -> dict:
        instance = self.instances.get(instance_id)
        if instance is None:
            raise KeyError(f"Instance {instance_id} not found")

        if not instance.cuda_lock.acquire(blocking=False):
            raise HTTPException(
                409, "A CUDA operation is in progress; retry deletion later"
            )

        try:
            return self._stop_instance_unlocked(instance_id, timeout)
        finally:
            instance.cuda_lock.release()

    def _stop_instance_unlocked(self, instance_id: str, timeout: int = 10) -> dict:
        """Stop a specific vLLM instance"""
        if instance_id not in self.instances:
            raise KeyError(f"Instance {instance_id} not found")

        instance = self.instances[instance_id]
        instance.cancel_sentinel_watcher()
        instance.stop(timeout)

        revision = self._next_revision()
        instance.last_revision = revision
        result = instance.get_status()

        del self.instances[instance_id]

        event = WatchEvent(type="DELETED", object=result)
        self.broadcaster._append(event)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.broadcaster._notify())
        except RuntimeError:
            pass  # No running event loop (e.g. in sync tests)
        return result

    def stop_all_instances(self, timeout: int = 10) -> dict:
        """Stop all running vLLM instances"""
        results = []
        instance_ids = list(self.instances.keys())

        for instance_id in instance_ids:
            try:
                result = self.stop_instance(instance_id, timeout)
                results.append(result)
            except KeyError:
                continue  # Instance was already removed

        return {
            "status": "all_stopped",
            "stopped_instances": results,
            "total_stopped": len(results),
        }

    def get_instance_status(self, instance_id: str) -> dict:
        """Get status of a specific instance"""
        if instance_id not in self.instances:
            raise KeyError(f"Instance {instance_id} not found")

        return self.instances[instance_id].get_status()

    def get_all_instances_status(self) -> dict:
        """Get status of all instances"""
        instances_status = []
        running_count = 0

        for instance in self.instances.values():
            status = instance.get_status()
            instances_status.append(status)
            if status["status"] == "running":
                running_count += 1

        return {
            "revision": self._revision,
            "total_instances": len(self.instances),
            "running_instances": running_count,
            "instances": instances_status,
        }

    def list_instances(self) -> List[str]:
        """List all instance IDs"""
        return list(self.instances.keys())

    def get_instance_log_bytes(
        self,
        instance_id: str,
        start: int | None = 0,
        end: int | None = None,
    ) -> tuple[int, bytes, int]:
        """
        Get log bytes from a specific instance.
        :param instance_id: ID of the instance
        :param start: First byte to read (inclusive, 0-based), or None for suffix
        :param end: if start is None then suffix length otherwise
                    Last byte to read (inclusive) or None for default limit
        :return: (read_start, content_bytes, total_file_size)
        :raises LogRangeNotAvailable: If start is beyond available content
        """
        if instance_id not in self.instances:
            raise KeyError(f"Instance {instance_id} not found")
        return self.instances[instance_id].get_log_bytes(start, end)

    async def control_instance(self, instance_id, kind, action, resume: bool = False):
        instance = self.instances.get(instance_id)
        if instance is None:
            raise HTTPException(404, "Instance not found")

        if kind == "disk" and len(self.instances) != 1:
            raise HTTPException(
                409, "Disk prototype supports one instance per launcher"
            )

        if not instance.cuda_lock.acquire(blocking=False):
            raise HTTPException(409, "Another operation is in progress")

        # A CRIU dump intentionally exits the tree; suppress the ordinary
        # STOPPED callback until the checkpoint operation has finished.
        instance.cancel_sentinel_watcher()
        instance.operation = f"{kind}_{action}"
        started = time.monotonic()

        try:
            if kind == "cuda":
                work = asyncio.to_thread(instance.cuda_control, action)
            else:
                work = asyncio.to_thread(
                    instance.disk_control, action, resume=resume
                )

            task = asyncio.create_task(work)

            try:
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                # Client disconnection must not release the lock while CRIU
                # is still changing the process tree.
                await task
                raise
        finally:
            instance.operation = None
            try:
                if instance.process and instance.process.is_alive():
                    instance.start_sentinel_watcher(
                        self._on_instance_stopped
                    )

                instance.last_revision = self._next_revision()
                self.broadcaster._append(
                    WatchEvent(
                        type="MODIFIED",
                        object=instance.get_status(),
                    )
                )
                asyncio.create_task(self.broadcaster._notify())
            finally:
                instance.cuda_lock.release()

        return {
            **(result or {}),
            **instance.get_status(),
            "elapsed_seconds": time.monotonic() - started,
        }


# Create global manager instance
vllm_manager = VllmMultiProcessManager()

# Setup logging
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Manage application lifecycle: clean up all vLLM instances on shutdown."""
    yield
    logger.info("Launcher shutting down, stopping all vLLM instances...")
    vllm_manager.stop_all_instances()


# Create FastAPI application
app = FastAPI(
    title="Multi-Instance vLLM Management API",
    version="2.0",
    description="REST API for managing multiple vLLM instances",
    lifespan=lifespan,
)


_RANGE_RE = re.compile(r"^bytes=(\d+)?-(\d+)?$")


def parse_range_header(range_header: str) -> tuple[int | None, int | None]:
    """Parse an HTTP Range header value.

    Accepts ``bytes=START-END`` (both inclusive) or ``bytes=START-`` (open-ended)
    or ``bytes=-SUFFIX``.
    Returns ``(start, end_or_suffix)`` where either may be ``None``.

    Raises :class:`ValueError` for unsupported or malformed values
    (e.g. suffix ranges like ``bytes=-0``).
    """
    m = _RANGE_RE.match(range_header)
    if m is None:
        raise ValueError(f"Unsupported or malformed Range header: {range_header}")
    start = int(m.group(1)) if m.group(1) else None
    # group(2) is the end value; absent in open-ended ranges like "bytes=100-"
    end = int(m.group(2)) if m.group(2) else None
    if start is None:
        if end is None:
            raise ValueError("Range must specify start or end_or_suffix")
        if end <= 0:
            raise ValueError(f"Range suffix must be positive, not {end}")
    else:
        if end is not None and end < start:
            raise ValueError(f"Range end ({end}) must be >= start ({start})")
    return (start, end)


############################################################
# Health Endpoint
############################################################
@app.get("/health")
async def health():
    """Health Status"""
    return JSONResponse(content={"status": "OK"}, status_code=HTTPStatus.OK)


######################################################################
# GET INDEX
######################################################################
@app.get("/")
async def index():
    """Root URL response"""
    return JSONResponse(
        content={
            "name": "Multi-Instance vLLM Management API",
            "version": "2.0",
            "endpoints": {
                "index": "GET /",
                "health": "GET /health",
                "create_instance": "POST /v2/vllm/instances",
                "create_named_instance": "PUT /v2/vllm/instances/{instance_id}",
                "delete_instance": "DELETE /v2/vllm/instances/{instance_id}",
                "delete_all_instances": "DELETE /v2/vllm/instances",
                "get_instance_status": "GET /v2/vllm/instances/{instance_id}",
                "get_all_instances": "GET /v2/vllm/instances",
                "get_instance_logs": "GET /v2/vllm/instances/{instance_id}/log",
                "watch_instances": "GET /v2/vllm/instances/watch",
                "get_gpu_debug": "GET /v2/gpu-debug",
            },
        },
        status_code=HTTPStatus.OK,
    )


######################################################################
# vLLM MANAGEMENT ENDPOINTS
######################################################################


@app.get("/v2/vllm/instances/watch")
async def watch_instances(
    since: Optional[int] = Query(
        None,
        description="Resume watching from this revision. "
        "If omitted, the stream begins with CREATED events for every "
        "instance that currently exists, followed by live events.",
    ),
):
    """Stream instance lifecycle events as NDJSON (Kubernetes watch-style)."""

    if since is not None:
        oldest = vllm_manager.broadcaster.oldest_revision
        if since < oldest:
            raise HTTPException(
                status_code=HTTPStatus.GONE,
                detail=f"Requested revision {since} is no longer available. "
                f"Oldest available: {oldest}.",
            )

    async def event_stream():
        if since is None:
            # Initial state: emit a CREATED event for every existing instance
            # and start streaming from the current revision.
            start_revision = vllm_manager.revision
            for instance in vllm_manager.instances.values():
                state = instance.get_status()
                initial = WatchEvent(
                    type="CREATED",
                    object=state,
                )
                yield initial.model_dump_json() + "\n"
        else:
            start_revision = since

        try:
            async for event in vllm_manager.broadcaster.watch(start_revision):
                yield event.model_dump_json() + "\n"
        except RevisionTooOld:
            return

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        headers={"X-Content-Type-Options": "nosniff"},
    )


@app.post("/v2/vllm/instances")
async def create_vllm_instance(vllm_config: VllmConfig):
    """Create a new vLLM instance with random instance ID"""

    try:
        result = vllm_manager.create_instance(vllm_config)
        return JSONResponse(content=result, status_code=HTTPStatus.CREATED)
    except Exception as e:
        logger.error(f"Failed to create vLLM instance: {e}")
        return PlainTextResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR, content=str(e)
        )


@app.put("/v2/vllm/instances/{instance_id}")
async def create_id_vllm_instance(
    vllm_config: VllmConfig,
    instance_id: str = Path(..., description="Custom instance ID"),
):
    """Create a new vLLM instance with instance ID"""
    try:
        result = vllm_manager.create_instance(vllm_config, instance_id)
        return JSONResponse(content=result, status_code=HTTPStatus.CREATED)
    except ValueError as e:
        raise HTTPException(status_code=HTTPStatus.CONFLICT, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create vLLM instance {instance_id}: {e}")
        return PlainTextResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR, content=str(e)
        )


@app.delete("/v2/vllm/instances/{instance_id}")
async def delete_vllm_instance(
    instance_id: str = Path(..., description="Instance ID to delete")
):
    """Delete a specific vLLM instance"""
    try:
        result = vllm_manager.stop_instance(instance_id)
        return JSONResponse(content=result, status_code=HTTPStatus.OK)
    except KeyError:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail=f"Instance {instance_id} not found"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete vLLM instance {instance_id}: {e}")
        return PlainTextResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR, content=str(e)
        )


@app.delete("/v2/vllm/instances")
async def delete_all_vllm_instances():
    """Delete all vLLM instances"""
    try:
        result = vllm_manager.stop_all_instances()
        return JSONResponse(content=result, status_code=HTTPStatus.OK)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete all vLLM instances: {e}")
        return PlainTextResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR, content=str(e)
        )

@app.get("/v2/vllm/instances/{instance_id}/cuda")
async def get_cuda_state(instance_id: str):
    return await vllm_manager.control_instance(
        instance_id, "cuda", "status"
    )


@app.post("/v2/vllm/instances/{instance_id}/cuda/{action}")
async def change_cuda_state(
    instance_id: str,
    action: Literal["checkpoint", "restore"],
):
    return await vllm_manager.control_instance(
        instance_id, "cuda", action
    )


@app.post("/v2/vllm/instances/{instance_id}/disk/{action}")
async def change_disk_state(
    instance_id: str,
    action: Literal["checkpoint", "restore"],
    resume: bool = False,
):
    return await vllm_manager.control_instance(
        instance_id, "disk", action, resume=resume
    )

@app.get("/v2/gpu-debug")
async def get_gpu_debug():
    try:
        cp = subprocess.run(
            "nvidia-smi --query-gpu=index,uuid,memory.used --format=csv; nvidia-smi",
            shell=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except subprocess.TimeoutExpired:
        logger.error("Timeout trying to use nvidia-smi")
        return PlainTextResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content="debugging commands timed out",
        )
    except Exception as exn:
        logger.error(f"Failed to use nvidia-smi, exception {exn}")
        return PlainTextResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content=f"Running debug commands threw exception {exn}",
        )
    else:
        return JSONResponse(
            status_code=HTTPStatus.OK,
            content=dict(
                stdout=cp.stdout,
                stderr=cp.stderr,
                returncode=cp.returncode,
            ),
        )


@app.get("/v2/vllm/instances")
async def get_all_vllm_instances(detail: bool = True):
    """
    Get information about all vLLM instances

    Query Parameters:
    - detail: If True (default), returns full status of all instances.
              If False, returns only instance IDs.
    """
    if detail:
        result = vllm_manager.get_all_instances_status()
    else:
        instances = vllm_manager.list_instances()
        result = {
            "revision": vllm_manager.revision,
            "instance_ids": instances,
            "count": len(instances),
        }

    return JSONResponse(content=result, status_code=HTTPStatus.OK)


@app.get("/v2/vllm/instances/{instance_id}")
async def get_vllm_instance_status(
    instance_id: str = Path(..., description="Instance ID")
):
    """Get status of a specific vLLM instance"""
    try:
        result = vllm_manager.get_instance_status(instance_id)
        return JSONResponse(content=result, status_code=HTTPStatus.OK)
    except KeyError:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail=f"Instance {instance_id} not found"
        )


@app.get("/v2/vllm/instances/{instance_id}/log")
async def get_vllm_instance_log(
    instance_id: str = Path(..., description="Instance ID"),
    range: str | None = Header(None, alias="Range"),
):
    """
    Get logs from a specific vLLM instance.

    Supports range requests per RFC 9110 §14 (Range Requests).

    Without a Range header the full log (up to 1 MB) is returned with
    200 OK.  With ``Range: bytes=START-END`` or ``Range: bytes=START-``
    or ``Range: bytes=-SUFFIX``
    the requested slice is returned with 206 Partial Content.  In both
    cases the response includes a ``Content-Range`` header indicating the byte range
    and current total log length
    --- except in the case of empty log, when no ``Content-Range`` header is returned.
    """
    try:
        if range is None:
            start, end = 0, None
            partial = False
        else:
            try:
                start, end = parse_range_header(range)
            except ValueError as exc:
                raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail=str(exc))
            partial = True

        read_start, data, total = vllm_manager.get_instance_log_bytes(
            instance_id, start, end
        )

        actual_end = read_start + len(data) - 1
        headers = {
            "Accept-Ranges": "bytes",
        }
        if actual_end >= 0:
            headers["Content-Range"] = f"bytes {read_start}-{actual_end}/{total}"
        # There is no valid Content-Range when the file's length is zero
        if partial:
            status_code = HTTPStatus.PARTIAL_CONTENT
        else:
            status_code = HTTPStatus.OK

        return Response(
            content=data,
            status_code=status_code,
            media_type="application/octet-stream",
            headers=headers,
        )
    except KeyError:
        raise HTTPException(
            status_code=HTTPStatus.NOT_FOUND, detail=f"Instance {instance_id} not found"
        )
    except LogRangeNotAvailable as e:
        return Response(
            content=b"",
            status_code=HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE,
            media_type="application/octet-stream",
            headers={"Content-Range": f"bytes */{e.available_bytes}"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get logs for instance {instance_id}: {e}")
        return PlainTextResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR, content=str(e)
        )


######################################################################
# HELPER FUNCTIONS
######################################################################


def _close_inherited_sockets():
    """Close every socket fd inherited from the parent across fork().

    Iterates /proc/self/fd and uses fstat to identify sockets, leaving
    pipes, regular files, character devices, and anything else
    untouched. fd 0/1/2 are skipped on principle even if they happened
    to be sockets.
    """
    try:
        fd_names = os.listdir("/proc/self/fd")
    except OSError:
        return
    for name in fd_names:
        try:
            fd = int(name)
        except ValueError:
            continue
        if fd <= 2:
            continue
        try:
            if stat.S_ISSOCK(os.fstat(fd).st_mode):
                os.close(fd)
        except OSError:
            # fd was already closed or unstatable; ignore.
            pass

# Function to be executed by the child process
def vllm_kickoff(vllm_config: VllmConfig, log_file_path: str, debug_gpu_memory: bool):
    """
    Child function to kickoff vllm instance
    :param vllm_config: vLLM configuration parameters and env variables
    :param log_file_path: Path to the log file for capturing stdout/stderr
    :param debug_gpu_memory: bool indicating whether to first give evidence of
                             GPU memory usage
    """
    # Isolate this process tree into its own process group so that
    # signals (SIGINT/SIGTERM) sent to the launcher's group do not
    # propagate to the vLLM server or its EngineCore child process.
    # os.setpgrp()

    # Close socket fds inherited from the launcher across fork(). The
    # child must not hold duplicate references to uvicorn's listening
    # socket on :8001 nor to any in-flight TCP connections — the kernel
    # keeps a socket open as long as any process has an fd to it, so a
    # leaked client fd in this child wedges the connection long after
    # uvicorn has closed its own end (see issue #550). Pipes (including
    # multiprocessing's parent-child sentinel pipe used by
    # start_sentinel_watcher) and regular files are deliberately left
    # alone.
    _close_inherited_sockets()

    # Redirect stdout and stderr at the OS level using dup2 so that
    # all output (including from vLLM/uvicorn internal logging and any
    # C extensions writing to fd 1/2) is captured in the log file.
    log_fd = os.open(log_file_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(log_fd, sys.stdout.fileno())
    os.dup2(log_fd, sys.stderr.fileno())
    os.close(log_fd)
    sys.stdout = os.fdopen(1, "w")
    sys.stderr = os.fdopen(2, "w")

    # Set env vars in the current process
    if vllm_config.env_vars:
        set_env_vars(vllm_config.env_vars)

    if debug_gpu_memory:
        try:
            cp = subprocess.run(
                "nvidia-smi --query-gpu=index,uuid,memory.used --format=csv"
                "; nvidia-smi",
                shell=True,
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            logger.warning("nvidia-smi timed out")
        except Exception as exn:
            logger.warning(f"Running nvidia-smi threw exception {exn}")
        else:
            logger.info(
                f"Ran nvidia-smi; got returncode={cp.returncode}"
                f", stdout={cp.stdout}, stderr={cp.stderr}"
            )

    cvd = os.getenv("CUDA_VISIBLE_DEVICES", "<unset>")
    logger.info(
        f"VLLM process (PID: {os.getpid()}, CUDA_VISIBLE_DEVICES: {cvd}) starting."
    )

    # prepare args
    receive_args = vllm_config.options.split()

    cli_env_setup()
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    args = parser.parse_args(receive_args)
    validate_parsed_serve_args(args)

    uvloop.run(run_server(args))


# Function to set env variables
def set_env_vars(env_vars: Dict[str, str]):
    """
    Set environment variables from a dictionary
    :param env_vars: Dict with environment var name as keys and string values
    """

    # Set environment variables from a dictionary
    for key, value in env_vars.items():
        os.environ[key] = value


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="vLLM Launcher Service")
    parser.add_argument(
        "--mock-gpus",
        action="store_true",
        help="Enable mock GPU mode for CPU-only testing environments",
    )
    parser.add_argument(
        "--mock-gpu-count",
        type=int,
        default=8,
        help="Number of mock GPUs to create in mock mode (default: 8)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8001,
        help="Port to bind the server to (default: 8001)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["critical", "error", "warning", "info", "debug"],
        help="Logging level (default: info)",
    )
    parser.add_argument(
        "--debug-gpu-memory",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Log info on accelerator memory usage",
    )

    parser.add_argument("--worker-config", help=argparse.SUPPRESS)

    args = parser.parse_args()

    if args.worker_config:
        with open(args.worker_config) as stream:
            worker = json.load(stream)

        vllm_kickoff(
            VllmConfig(**worker["config"]),
            worker["log"],
            worker["debug"],
        )
        sys.exit(0)

    # Adopt orphaned descendants when CRIU kills the saved tree.
    # Reaping them frees their PIDs for restoration.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(
            ctypes.get_errno(),
            "Cannot become a child subreaper",
        )

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Get node name from environment variable
    node_name = os.getenv("NODE_NAME")
    namespace = os.getenv("NAMESPACE")

    logger.info(
        "Launcher starting with args: mock_gpus=%s, mock_gpu_count=%d"
        ", host=%s, port=%d, log_level=%s, node_name=%s, namespace=%s"
        ", debug_gpu_memory=%s"
        "",
        args.mock_gpus,
        args.mock_gpu_count,
        args.host,
        args.port,
        args.log_level,
        node_name,
        namespace,
        args.debug_gpu_memory,
    )

    # Reinitialize the global manager with mock mode settings
    vllm_manager = VllmMultiProcessManager(
        mock_gpus=args.mock_gpus,
        mock_gpu_count=args.mock_gpu_count,
        node_name=node_name,
        namespace=namespace,
        debug_gpu_memory=args.debug_gpu_memory,
    )

    uvicorn.run(
        app, host=args.host, port=args.port, log_level=args.log_level, log_config=None
    )
