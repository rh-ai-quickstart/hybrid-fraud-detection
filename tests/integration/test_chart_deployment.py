"""Render the deployable UI/API graph and model/image configuration boundaries."""
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
HELM = shutil.which("helm")
pytestmark = pytest.mark.skipif(HELM is None, reason="Helm rendering runs in the Helm CI job")


def render(*settings):
    command = [HELM, "template", "example", str(ROOT / "chart")]
    for setting in settings:
        command += ["--set-string", setting]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def documents(*settings):
    result = render(*settings)
    assert result.returncode == 0, result.stderr
    return [row for row in yaml.safe_load_all(result.stdout) if row]


def deployment(rows):
    return next(row for row in rows if row["kind"] == "Deployment")


def test_default_chart_exposes_working_ui_and_api_graph():
    rows = documents()
    spec = deployment(rows)["spec"]["template"]["spec"]
    assert spec["automountServiceAccountToken"] is False
    containers = {row["name"]: row for row in spec["containers"]}
    assert set(containers) == {"scorer", "ui"}
    ui_env = {row["name"]: row for row in containers["ui"]["env"]}
    assert ui_env["SCORER_URL"]["value"] == "http://127.0.0.1:8000"
    for container in containers.values():
        assert container["securityContext"]["runAsNonRoot"] is True
        assert container["securityContext"]["allowPrivilegeEscalation"] is False
    service = next(row for row in rows if row["kind"] == "Service")
    assert {p["name"] for p in service["spec"]["ports"]} == {"http", "ui"}
    routes = {row["metadata"]["name"]: row for row in rows if row["kind"] == "Route"}
    assert set(routes) == {"example-scorer", "example-ui"}
    assert routes["example-ui"]["spec"]["port"]["targetPort"] == "ui"
    assert routes["example-ui"]["spec"]["tls"]["insecureEdgeTerminationPolicy"] == "Redirect"


def test_api_only_mode_omits_ui_container_service_port_and_route():
    # Booleans are typed values, so use --set rather than --set-string.
    result = subprocess.run([HELM, "template", "example", str(ROOT / "chart"), "--set", "ui.enabled=false"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    rows = list(yaml.safe_load_all(result.stdout))
    assert len(deployment(rows)["spec"]["template"]["spec"]["containers"]) == 1
    assert not any(row["kind"] == "Route" and row["metadata"]["name"] == "example-ui" for row in rows)
    service = next(row for row in rows if row["kind"] == "Service")
    assert [port["name"] for port in service["spec"]["ports"]] == ["http"]


def test_both_containers_use_the_same_exact_digest():
    image = "registry.example.test/team/hybrid-fraud@sha256:" + "a" * 64
    rows = documents("app.image=" + image)
    assert {row["image"] for row in deployment(rows)["spec"]["template"]["spec"]["containers"]} == {image}


def test_malformed_digest_is_rejected_before_installation():
    result = render("app.image=registry.example.test/team/hybrid-fraud@sha256:wrong")
    assert result.returncode != 0
    assert "64 lowercase hex characters" in result.stderr


def test_secret_backed_model_configuration_does_not_silently_select_demo_mode():
    result = subprocess.run([HELM, "template", "example", str(ROOT / "chart"), "--set",
                             "model.endpointFromSecret=true", "--set-string", "model.existingSecret=model-access"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    rows = list(yaml.safe_load_all(result.stdout))
    scorer = deployment(rows)["spec"]["template"]["spec"]["containers"][0]
    env = {row["name"]: row for row in scorer["env"]}
    assert env["DEMO_MODE"]["value"] == "false"
    for variable, key in (("MODEL_ENDPOINT", "endpoint"), ("MODEL_NAME", "name"), ("MODEL_API_KEY", "api-key")):
        assert env[variable]["valueFrom"]["secretKeyRef"] == {"name": "model-access", "key": key}
    assert not any(row["kind"] == "Secret" for row in rows)


def test_secret_configuration_requires_an_existing_secret_name():
    result = subprocess.run([HELM, "template", "example", str(ROOT / "chart"), "--set",
                             "model.endpointFromSecret=true"], capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "model.existingSecret is required" in result.stderr
