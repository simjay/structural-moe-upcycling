"""Prime Intellect API client for GPU provisioning and management."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

BASE_URL = "https://api.primeintellect.ai/api/v1"

DEFAULTS = {
    "gpu_type": "A100_80GB",
    "gpu_count": 1,
    "regions": ["united_states", "canada"],
    "disk_size": 300,
    "image": "cuda_12_1_pytorch_2_2",
}


@dataclass
class PrimeIntellectConfig:
    api_key: str
    gpu_type: str = DEFAULTS["gpu_type"]
    gpu_count: int = DEFAULTS["gpu_count"]
    regions: list[str] = field(default_factory=lambda: list(DEFAULTS["regions"]))
    disk_size: int = DEFAULTS["disk_size"]
    image: str = DEFAULTS["image"]

    @classmethod
    def from_env(cls, dotenv_path: str | None = None) -> PrimeIntellectConfig:
        """Load configuration from environment / .env file."""
        load_dotenv(dotenv_path)
        api_key = os.environ.get("PRIME_INTELLECT_API_KEY", "")
        if not api_key:
            raise ValueError(
                "PRIME_INTELLECT_API_KEY is required. "
                "Set it in .env or as an environment variable."
            )
        regions_raw = os.environ.get("PI_REGIONS", "")
        regions = (
            [r.strip() for r in regions_raw.split(",") if r.strip()]
            if regions_raw
            else list(DEFAULTS["regions"])
        )
        return cls(
            api_key=api_key,
            gpu_type=os.environ.get("PI_GPU_TYPE", DEFAULTS["gpu_type"]),
            gpu_count=int(os.environ.get("PI_GPU_COUNT", DEFAULTS["gpu_count"])),
            regions=regions,
            disk_size=int(os.environ.get("PI_DISK_SIZE", DEFAULTS["disk_size"])),
            image=os.environ.get("PI_IMAGE", DEFAULTS["image"]),
        )


class PrimeIntellectClient:
    """Thin wrapper around the Prime Intellect REST API."""

    def __init__(self, config: PrimeIntellectConfig) -> None:
        self.config = config
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def check_availability(
        self,
        gpu_type: str | None = None,
        gpu_count: int | None = None,
        regions: list[str] | None = None,
        **extra_filters: Any,
    ) -> list[dict]:
        """Query GPU availability, sorted cheapest-first.

        Any filter accepted by the API can be passed as a keyword argument.
        """
        params: dict[str, Any] = {
            "gpu_type": gpu_type or self.config.gpu_type,
            "gpu_count": gpu_count or self.config.gpu_count,
        }
        for region in regions or self.config.regions:
            params.setdefault("regions", [])
            params["regions"].append(region)
        params.update(extra_filters)

        resp = self._session.get(f"{BASE_URL}/availability/gpus", params=params)
        resp.raise_for_status()
        items = resp.json().get("items", [])
        items.sort(key=lambda o: o.get("prices", {}).get("onDemand", float("inf")))
        logger.info("Found %d available offers", len(items))
        return items

    # ------------------------------------------------------------------
    # Pod lifecycle
    # ------------------------------------------------------------------

    def provision(
        self,
        offer: dict,
        name: str,
        disk_size: int | None = None,
        image: str | None = None,
    ) -> dict:
        """Provision a pod from an availability offer.

        Required API fields (cloudId, socket, dataCenterId, country, security,
        provider type) are extracted directly from the offer.
        """
        requested_image = image or self.config.image
        available_images = offer.get("images", [])
        if available_images and requested_image not in available_images:
            resolved_image = available_images[0]
            logger.warning(
                "Image %r not available for this offer; using %r instead",
                requested_image,
                resolved_image,
            )
        else:
            resolved_image = requested_image

        pod_def: dict[str, Any] = {
            "name": name,
            "cloudId": offer["cloudId"],
            "gpuType": offer["gpuType"],
            "socket": offer["socket"],
            "gpuCount": offer.get("gpuCount", self.config.gpu_count),
            "image": resolved_image,
            "security": offer.get("security", "secure_cloud"),
        }
        if offer.get("dataCenter"):
            pod_def["dataCenterId"] = offer["dataCenter"]
        if offer.get("country"):
            pod_def["country"] = offer["country"]

        disk_spec = offer.get("disk", {})
        disk_is_customizable = disk_spec.get("maxCount") is not None
        effective_disk = disk_size or self.config.disk_size
        if disk_is_customizable and effective_disk:
            pod_def["diskSize"] = effective_disk

        body: dict[str, Any] = {
            "pod": pod_def,
            "provider": {"type": offer["provider"]},
        }

        logger.info(
            "Provisioning pod %r on %s (%s) image=%s...",
            name,
            offer["provider"],
            offer.get("dataCenter", "?"),
            resolved_image,
        )
        resp = self._session.post(f"{BASE_URL}/pods/", json=body)
        if not resp.ok:
            logger.error("Provision failed: %s %s", resp.status_code, resp.text)
        resp.raise_for_status()
        pod = resp.json()
        logger.info("Pod created: id=%s status=%s", pod["id"], pod["status"])
        return pod

    def get_pod(self, pod_id: str) -> dict:
        """Fetch current pod state."""
        resp = self._session.get(f"{BASE_URL}/pods/{pod_id}")
        resp.raise_for_status()
        return resp.json()

    def list_pods(self) -> list[dict]:
        """List all pods owned by the authenticated user."""
        resp = self._session.get(f"{BASE_URL}/pods/")
        resp.raise_for_status()
        return resp.json()

    def wait_until_ready(
        self,
        pod_id: str,
        timeout: float = 600,
        poll_interval: float = 15,
    ) -> dict:
        """Block until the pod is RUNNING with SSH available."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pod = self.get_pod(pod_id)
            status = pod.get("status", "")
            ssh = pod.get("sshConnection") or pod.get("ip")
            logger.info("Pod %s: status=%s ssh=%s", pod_id[:12], status, bool(ssh))
            if status == "RUNNING" and ssh:
                return pod
            if status in ("FAILED", "TERMINATED"):
                raise RuntimeError(
                    f"Pod {pod_id} entered terminal state: {status}"
                )
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Pod {pod_id} did not become ready within {timeout}s"
        )

    def get_ssh_info(self, pod_id: str) -> tuple[str, int]:
        """Return (host, port) for SSH access to a running pod."""
        pod = self.get_pod(pod_id)
        host = pod.get("ip")
        if not host:
            raise RuntimeError(f"Pod {pod_id} has no IP address yet")
        port = 22
        for mapping in pod.get("primePortMapping", []):
            if mapping.get("usedBy") == "SSH":
                port = int(mapping.get("external", 22))
                break
        return host, port

    def terminate(self, pod_id: str) -> None:
        """Terminate a pod to stop billing."""
        logger.info("Terminating pod %s...", pod_id[:12])
        resp = self._session.delete(f"{BASE_URL}/pods/{pod_id}")
        resp.raise_for_status()
        logger.info("Pod %s terminated.", pod_id[:12])
