"""Print unverified entries from the quickstart claim registry."""

from pathlib import Path

import yaml

REGISTRY = Path(__file__).with_name("claim_registry.yaml")


def main() -> int:
    data = yaml.safe_load(REGISTRY.read_text())
    unverified = [claim for claim in data.get("claims", []) if not claim.get("verified")]
    if not unverified:
        print("All registered claims are verified.")
        return 0
    print("Unverified claims:")
    for claim in unverified:
        print(f"- {claim['id']}: {claim['value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
