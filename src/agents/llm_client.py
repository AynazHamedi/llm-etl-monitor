
from __future__ import annotations

import hashlib
import json
import re
from typing import Optional

import requests


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral:latest",
                 temperature: float = 0.0, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": self.temperature},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["response"]

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except requests.RequestException:
            return False


class MockLLMClient:

    def __init__(self, seed_note: str = "mock"):
        self.seed_note = seed_note

    def generate(self, prompt: str) -> str:
        m = re.search(r"Allowed values:\s*\[([^\]]*)\]", prompt)
        digest = int(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), 16)
        if m:
            options = [o.strip().strip("'\"") for o in m.group(1).split(",") if o.strip()]
            if options:
                value = options[digest % len(options)]
                confidence = 60 + (digest % 35)  # 60-94, varies per prompt
                return json.dumps({"value": value, "confidence": confidence})
        # Free-text fallback.
        return json.dumps({"value": "UNKNOWN", "confidence": 40})

    def is_available(self) -> bool:
        return True


def get_llm_client(cfg: dict, prefer_mock: bool = False):
    llm_cfg = cfg.get("llm", {})
    agents_cfg = cfg.get("agents", {})
    client = OllamaClient(
        base_url=llm_cfg.get("base_url", "http://localhost:11434"),
        model=llm_cfg.get("model_name", "mistral:latest"),
        temperature=llm_cfg.get("temperature", 0.0),
        timeout=agents_cfg.get("llm_timeout_seconds", 30),
    )
    if prefer_mock or not client.is_available():
        return MockLLMClient(), False
    return client, True
