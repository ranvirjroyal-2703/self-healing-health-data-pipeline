"""Isolated sandbox execution with hard resource limits."""

import os
import subprocess
import tempfile
import time
from dataclasses import dataclass

SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "data-pipeline-sandbox")


@dataclass
class SandboxResult:
    success: bool
    stdout: str
    stderr: str
    returncode: int
    duration_sec: float
    timed_out: bool = False
    killed_by_limit: bool = False


class SandboxRunner:
    """Run untrusted Python inside a Docker container. Never inside the app process."""

    def __init__(
        self,
        memory_limit: str = "256m",
        cpu_limit: str = "0.5",
        timeout_sec: int = 15,
        network_disabled: bool = True,
    ):
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.timeout_sec = timeout_sec
        self.network_disabled = network_disabled

    def run(self, code: str) -> SandboxResult:
        with tempfile.TemporaryDirectory() as tmpdir:
            script_path = os.path.join(tmpdir, "task.py")
            with open(script_path, "w", encoding="utf-8") as handle:
                handle.write(code)
            os.chmod(script_path, 0o644)

            cmd = [
                "docker", "run", "--rm",
                "--memory", self.memory_limit,
                "--cpus", self.cpu_limit,
                "--user", "1000:1000",
                "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges",
                "-v", f"{tmpdir}:/workspace:ro",
                "-w", "/workspace",
            ]
            if self.network_disabled:
                cmd.append("--network=none")
            cmd.extend([SANDBOX_IMAGE, "python", "task.py"])

            start = time.time()
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )
                duration = round(time.time() - start, 2)
                stderr = result.stderr or ""
                if "Unable to find image" in stderr or "No such image" in stderr:
                    stderr += (
                        "\nBuild the sandbox image first: "
                        "docker build -t data-pipeline-sandbox -f sandbox/Dockerfile sandbox"
                    )
                return SandboxResult(
                    success=result.returncode == 0,
                    stdout=result.stdout or "",
                    stderr=stderr,
                    returncode=result.returncode,
                    duration_sec=duration,
                    killed_by_limit="Killed" in stderr or result.returncode == 137,
                )
            except subprocess.TimeoutExpired as exc:
                duration = round(time.time() - start, 2)
                stdout = exc.stdout or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                stderr = exc.stderr or ""
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                return SandboxResult(
                    success=False,
                    stdout=stdout,
                    stderr=stderr + "\n[TIMEOUT] Process killed by sandbox. Host was not touched.",
                    returncode=-1,
                    duration_sec=duration,
                    timed_out=True,
                    killed_by_limit=True,
                )
            except FileNotFoundError:
                return SandboxResult(
                    success=False,
                    stdout="",
                    stderr="Docker is not installed or not on PATH.",
                    returncode=-1,
                    duration_sec=0.0,
                )
