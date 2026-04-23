"""Quick CLI helpers for inspecting Prime Intellect availability."""

from __future__ import annotations

import argparse
import json
import sys

from experiment.primeintellect.client import PrimeIntellectClient, PrimeIntellectConfig


def list_offers(client: PrimeIntellectClient) -> None:
    offers = client.check_availability()
    print(f"Found {len(offers)} offers:\n")
    for o in offers:
        images = ", ".join(o.get("images", []))
        price = o["prices"]["onDemand"]
        print(
            f"  {o['cloudId']:30s}  {o['provider']:20s}  "
            f"{o.get('dataCenter', '?'):15s}  ${price:.2f}/hr  "
            f"images: [{images}]"
        )


def list_images(client: PrimeIntellectClient) -> None:
    offers = client.check_availability()
    images = sorted({img for o in offers for img in o.get("images", [])})
    print("Available images across all offers:\n")
    for img in images:
        print(f"  {img}")


def list_pods(client: PrimeIntellectClient) -> None:
    pods = client.list_pods()
    print(json.dumps(pods, indent=2))


def kill_pod(client: PrimeIntellectClient, pod_id: str) -> None:
    client.terminate(pod_id)
    print(f"Pod {pod_id} terminated.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prime Intellect helpers")
    parser.add_argument(
        "--env-file", default=".env", help="Path to .env file"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("offers", help="List available GPU offers")
    sub.add_parser("images", help="List available images")
    sub.add_parser("pods", help="List active pods")
    kill = sub.add_parser("kill", help="Terminate a pod")
    kill.add_argument("pod_id", help="Pod ID to terminate")

    args = parser.parse_args(argv)
    config = PrimeIntellectConfig.from_env(dotenv_path=args.env_file)
    client = PrimeIntellectClient(config)

    if args.command == "offers":
        list_offers(client)
    elif args.command == "images":
        list_images(client)
    elif args.command == "pods":
        list_pods(client)
    elif args.command == "kill":
        kill_pod(client, args.pod_id)


if __name__ == "__main__":
    main()
