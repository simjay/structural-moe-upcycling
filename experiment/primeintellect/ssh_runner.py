"""SSH/SCP wrapper using subprocess (primary) or paramiko (fallback)."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SSHRunner:
    """Execute commands and transfer files over SSH to a remote host.

    Uses the system ``ssh`` / ``scp`` binaries by default so that the user's
    SSH agent and config are respected. Falls back to paramiko when system
    SSH is not available.
    """

    def __init__(
        self,
        host: str,
        port: int = 22,
        user: str = "root",
        key_path: str | None = None,
        connect_timeout: int = 30,
    ) -> None:
        self.host = host
        self.port = port
        self.user = user
        self.key_path = key_path
        self.connect_timeout = connect_timeout
        self._use_paramiko = shutil.which("ssh") is None
        self._paramiko_client = None

        if self._use_paramiko:
            self._init_paramiko()

    # ------------------------------------------------------------------
    # Paramiko fallback
    # ------------------------------------------------------------------

    def _init_paramiko(self) -> None:
        try:
            import paramiko  # noqa: F811
        except ImportError as exc:
            raise RuntimeError(
                "System ssh not found and paramiko is not installed. "
                "Install paramiko: pip install paramiko"
            ) from exc
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kwargs: dict = {
            "hostname": self.host,
            "port": self.port,
            "username": self.user,
            "timeout": self.connect_timeout,
        }
        if self.key_path:
            connect_kwargs["key_filename"] = self.key_path
        client.connect(**connect_kwargs)
        self._paramiko_client = client
        logger.info("Connected via paramiko to %s:%d", self.host, self.port)

    # ------------------------------------------------------------------
    # Common SSH options for subprocess calls
    # ------------------------------------------------------------------

    def _ssh_opts(self) -> list[str]:
        opts = [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={self.connect_timeout}",
            "-p", str(self.port),
        ]
        if self.key_path:
            opts.extend(["-i", self.key_path])
        return opts

    def _target(self) -> str:
        return f"{self.user}@{self.host}"

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def run_command(
        self,
        cmd: str,
        stream_output: bool = True,
        check: bool = True,
    ) -> subprocess.CompletedProcess | tuple[int, str, str]:
        """Execute a command on the remote host.

        When *stream_output* is True, stdout/stderr are forwarded to the
        local terminal in real time (subprocess mode only).

        Returns a CompletedProcess (subprocess) or (exit_code, stdout, stderr)
        tuple (paramiko).
        """
        logger.info("Remote exec: %s", cmd)

        if self._use_paramiko:
            return self._run_paramiko(cmd, stream_output)

        full_cmd = ["ssh", *self._ssh_opts(), self._target(), cmd]
        if stream_output:
            result = subprocess.run(full_cmd, check=False)
        else:
            result = subprocess.run(
                full_cmd, capture_output=True, text=True, check=False
            )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Remote command failed (exit {result.returncode}): {cmd}"
            )
        return result

    def _run_paramiko(
        self, cmd: str, stream_output: bool
    ) -> tuple[int, str, str]:
        assert self._paramiko_client is not None
        _, stdout_ch, stderr_ch = self._paramiko_client.exec_command(cmd)
        if stream_output:
            for line in stdout_ch:
                sys.stdout.write(line)
            for line in stderr_ch:
                sys.stderr.write(line)
        stdout = stdout_ch.read().decode()
        stderr = stderr_ch.read().decode()
        exit_code = stdout_ch.channel.recv_exit_status()
        if exit_code != 0:
            raise RuntimeError(
                f"Remote command failed (exit {exit_code}): {cmd}\n{stderr}"
            )
        return exit_code, stdout, stderr

    # ------------------------------------------------------------------
    # File transfer
    # ------------------------------------------------------------------

    def upload(self, local_path: str | Path, remote_path: str) -> None:
        """SCP a local file or directory to the remote host."""
        local_path = Path(local_path)
        logger.info("Upload %s -> %s:%s", local_path, self.host, remote_path)

        if self._use_paramiko:
            self._upload_paramiko(local_path, remote_path)
            return

        scp_opts = [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(self.port),
        ]
        if self.key_path:
            scp_opts.extend(["-i", self.key_path])
        if local_path.is_dir():
            scp_opts.append("-r")

        subprocess.run(
            ["scp", *scp_opts, str(local_path), f"{self._target()}:{remote_path}"],
            check=True,
        )

    def download(self, remote_path: str, local_path: str | Path) -> None:
        """SCP a remote file or directory to a local path."""
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Download %s:%s -> %s", self.host, remote_path, local_path)

        if self._use_paramiko:
            self._download_paramiko(remote_path, local_path)
            return

        scp_opts = [
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(self.port),
        ]
        if self.key_path:
            scp_opts.extend(["-i", self.key_path])

        subprocess.run(
            ["scp", *scp_opts, "-r", f"{self._target()}:{remote_path}", str(local_path)],
            check=True,
        )

    def _upload_paramiko(self, local_path: Path, remote_path: str) -> None:
        assert self._paramiko_client is not None
        sftp = self._paramiko_client.open_sftp()
        try:
            if local_path.is_dir():
                for child in local_path.rglob("*"):
                    if child.is_file():
                        rel = child.relative_to(local_path)
                        dest = f"{remote_path}/{rel}"
                        self._sftp_makedirs(sftp, str(Path(dest).parent))
                        sftp.put(str(child), dest)
            else:
                sftp.put(str(local_path), remote_path)
        finally:
            sftp.close()

    def _download_paramiko(self, remote_path: str, local_path: Path) -> None:
        assert self._paramiko_client is not None
        sftp = self._paramiko_client.open_sftp()
        try:
            sftp.get(remote_path, str(local_path))
        finally:
            sftp.close()

    @staticmethod
    def _sftp_makedirs(sftp: Any, remote_dir: str) -> None:
        """Recursively create remote directories via SFTP."""
        from stat import S_ISDIR

        dirs_to_create: list[str] = []
        current = remote_dir
        while current and current != "/":
            try:
                if S_ISDIR(sftp.stat(current).st_mode):
                    break
            except FileNotFoundError:
                dirs_to_create.append(current)
                current = str(Path(current).parent)
                continue
            break
        for d in reversed(dirs_to_create):
            sftp.mkdir(d)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._paramiko_client is not None:
            self._paramiko_client.close()
            self._paramiko_client = None
            logger.info("Paramiko connection closed.")

    def __enter__(self) -> SSHRunner:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
