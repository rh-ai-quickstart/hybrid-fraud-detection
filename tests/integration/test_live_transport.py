"""Exercise configured live routing against a loopback protocol stub, not a real LLM."""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2] / "src"


def test_live_conditional_flow_preserves_skips_and_uses_model_transport():
    calls = []

    class ModelStub(BaseHTTPRequestHandler):
        def do_GET(self):
            assert self.path == "/v1/models"
            self.respond({"data": [{"id": "loopback-stub"}]})

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            calls.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.respond({"choices": [{"message": {"content": '{"score": 72}'}}]})

        def respond(self, payload):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    model = ThreadingHTTPServer(("127.0.0.1", 0), ModelStub)
    worker = threading.Thread(target=model.serve_forever, daemon=True)
    worker.start()
    env = {k: v for k, v in os.environ.items() if k not in {"DEMO_MODE", "MODEL_ENDPOINT", "MODEL_NAME", "MODEL_API_KEY"}}
    env.update(PYTHONPATH=str(SRC_DIR), DEMO_MODE="false", MODEL_NAME="loopback-stub",
               MODEL_ENDPOINT=f"http://127.0.0.1:{model.server_port}/v1")
    code = '''
from fastapi.testclient import TestClient
from scorer import app
with TestClient(app) as client:
    assert client.get('/ready').json()['mode'] == 'live'
    low = client.post('/api/v1/score', json={'amount': 50}).json()
    mid = client.post('/api/v1/score', json={'amount': 5000}).json()
    high = client.post('/api/v1/score', json={'amount': 15000, 'country': 'NG', 'category': 'crypto'}).json()
    assert low['llm_skipped'] and high['llm_skipped']
    assert not mid['llm_skipped'] and mid['llm_score'] == 72
    assert mid['risk_score'] == 37.8
    assert mid['model'] == 'loopback-stub'
    stats = client.get('/api/v1/stats').json()
    assert stats['llm_calls'] == 1 and stats['llm_skips'] == 2 and stats['llm_failures'] == 0
'''
    try:
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=15, check=False)
        assert result.returncode == 0, result.stderr
        assert len(calls) == 1
        assert calls[0]["model"] == "loopback-stub"
    finally:
        model.shutdown()
        model.server_close()
        worker.join(timeout=2)
