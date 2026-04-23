"""CLI entry point for launching experiments on Prime Intellect.

Usage::

    python -m experiment.primeintellect.launch --config experiment.yaml
    python -m experiment.primeintellect.launch --config experiment.yaml --parallel
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

from experiment.primeintellect.client import PrimeIntellectConfig
from experiment.primeintellect.orchestrator import ExperimentConfig, JobOrchestrator

logger = logging.getLogger(__name__)


def load_experiments(config_path: str) -> list[ExperimentConfig]:
    """Load experiment definitions from a YAML file.

    The YAML should contain either a single experiment dict or a list of
    experiment dicts under an ``experiments`` key.

    Example YAML::

        experiments:
          - name: setting-a-direct-copy
            train_command: "python train.py --init direct"
            upload_paths: ["src/", "configs/"]
            download_paths: ["/root/workspace/outputs/"]
            setup_commands:
              - "pip install -e ."

          - name: setting-b-gaussian
            train_command: "python train.py --init gaussian"
            upload_paths: ["src/", "configs/"]
            download_paths: ["/root/workspace/outputs/"]
            setup_commands:
              - "pip install -e ."
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if isinstance(raw, dict) and "experiments" in raw:
        return [ExperimentConfig.from_dict(e) for e in raw["experiments"]]
    if isinstance(raw, dict):
        return [ExperimentConfig.from_dict(raw)]
    if isinstance(raw, list):
        return [ExperimentConfig.from_dict(e) for e in raw]

    raise ValueError(f"Unexpected YAML structure in {config_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Launch training experiments on Prime Intellect GPUs",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config file",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Run experiments in parallel (one pod each)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Max parallel pods when --parallel is set (default: 4)",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable debug logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    pi_config = PrimeIntellectConfig.from_env(dotenv_path=args.env_file)
    experiments = load_experiments(args.config)
    logger.info("Loaded %d experiment(s) from %s", len(experiments), args.config)

    with JobOrchestrator(pi_config) as orch:
        results = orch.run_batch(
            experiments,
            parallel=args.parallel,
            max_workers=args.max_workers,
        )

    for exp, result_dir in zip(experiments, results):
        logger.info("  %s -> %s", exp.name, result_dir)

    logger.info("All experiments complete.")


if __name__ == "__main__":
    main()
