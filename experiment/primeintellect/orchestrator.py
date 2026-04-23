"""Orchestrate full experiment lifecycle on Prime Intellect GPUs."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from experiment.primeintellect.client import PrimeIntellectClient, PrimeIntellectConfig
from experiment.primeintellect.ssh_runner import SSHRunner

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Describes a single experiment to run on a remote GPU."""

    name: str
    train_command: str
    upload_paths: list[str] = field(default_factory=list)
    download_paths: list[str] = field(default_factory=list)
    setup_commands: list[str] = field(default_factory=list)
    remote_workdir: str = "/root/workspace"
    ssh_key_path: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExperimentConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class JobOrchestrator:
    """Provision, run, collect, and teardown — full experiment lifecycle.

    Usage::

        config = PrimeIntellectConfig.from_env()
        orch = JobOrchestrator(config)
        orch.run_experiment(exp)

    Or as a context manager for safe teardown::

        with JobOrchestrator(config) as orch:
            orch.run_experiment(exp)
    """

    def __init__(self, config: PrimeIntellectConfig) -> None:
        self.config = config
        self.client = PrimeIntellectClient(config)
        self._active_pods: list[str] = []

    # ------------------------------------------------------------------
    # Context manager — guarantees teardown
    # ------------------------------------------------------------------

    def __enter__(self) -> JobOrchestrator:
        return self

    def __exit__(self, *exc: object) -> None:
        self._teardown_all()

    def _teardown_all(self) -> None:
        for pod_id in list(self._active_pods):
            try:
                self.client.terminate(pod_id)
            except Exception:
                logger.exception("Failed to terminate pod %s", pod_id[:12])
            self._active_pods.remove(pod_id)

    # ------------------------------------------------------------------
    # Single experiment
    # ------------------------------------------------------------------

    def run_experiment(self, exp: ExperimentConfig) -> Path:
        """Run one experiment end-to-end. Returns local results directory."""
        pod_id: str | None = None
        try:
            # 1. Find cheapest GPU
            offers = self.client.check_availability()
            if not offers:
                raise RuntimeError(
                    f"No GPUs available matching {self.config.gpu_type} "
                    f"in {self.config.regions}"
                )
            offer = offers[0]
            logger.info(
                "Selected offer: %s on %s @ $%.2f/hr",
                offer["cloudId"],
                offer["provider"],
                offer["prices"]["onDemand"],
            )

            # 2. Provision
            pod = self.client.provision(offer, name=exp.name)
            pod_id = pod["id"]
            self._active_pods.append(pod_id)

            # 3. Wait for ready
            pod = self.client.wait_until_ready(pod_id)
            host, port = self.client.get_ssh_info(pod_id)
            logger.info("Pod ready at %s:%d", host, port)

            # 4. SSH setup + training
            results_dir = self._run_on_pod(exp, host, port)

            return results_dir

        finally:
            # 7. Teardown
            if pod_id:
                try:
                    self.client.terminate(pod_id)
                except Exception:
                    logger.exception("Teardown failed for pod %s", pod_id[:12])
                if pod_id in self._active_pods:
                    self._active_pods.remove(pod_id)

    def _run_on_pod(
        self, exp: ExperimentConfig, host: str, port: int
    ) -> Path:
        """SSH into the pod, upload code, run training, download results."""
        results_dir = Path("outputs") / exp.name / time.strftime("%Y%m%d-%H%M%S")
        results_dir.mkdir(parents=True, exist_ok=True)

        with SSHRunner(host, port, key_path=exp.ssh_key_path) as ssh:
            # Create remote workspace
            ssh.run_command(f"mkdir -p {exp.remote_workdir}")

            # Setup commands (install deps, etc.)
            for cmd in exp.setup_commands:
                ssh.run_command(cmd)

            # Upload local paths
            for local_path in exp.upload_paths:
                remote_dest = f"{exp.remote_workdir}/{Path(local_path).name}"
                ssh.upload(local_path, remote_dest)

            # Run training
            logger.info("Starting training: %s", exp.train_command)
            train_cmd = f"cd {exp.remote_workdir} && {exp.train_command}"
            ssh.run_command(train_cmd)
            logger.info("Training complete.")

            # Download results
            for remote_path in exp.download_paths:
                local_dest = results_dir / Path(remote_path).name
                try:
                    ssh.download(remote_path, local_dest)
                except Exception:
                    logger.warning("Could not download %s", remote_path)

        logger.info("Results saved to %s", results_dir)
        return results_dir

    # ------------------------------------------------------------------
    # Batch execution
    # ------------------------------------------------------------------

    def run_batch(
        self,
        experiments: list[ExperimentConfig],
        parallel: bool = False,
        max_workers: int = 4,
    ) -> list[Path]:
        """Run multiple experiments sequentially or in parallel."""
        if not parallel:
            return [self.run_experiment(exp) for exp in experiments]

        results: list[Path] = [Path()] * len(experiments)
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(self.run_experiment, exp): i
                for i, exp in enumerate(experiments)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    logger.exception(
                        "Experiment %s failed", experiments[idx].name
                    )
        return results
