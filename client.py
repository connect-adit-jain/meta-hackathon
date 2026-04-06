"""
Blood Bank Environment – HTTP Client.

Provides a synchronous and asynchronous client for interacting with a
running BloodBank environment server (local Docker or remote HF Space).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import httpx

from models import BloodBankAction, BloodBankObservation, BloodBankState


class BloodBankClient:
    """
    Synchronous HTTP client for the BloodBank environment.

    Usage:
        client = BloodBankClient(base_url="http://localhost:8000")
        obs = client.reset(task_name="basic_compatibility")
        obs, reward, done, info = client.step(BloodBankAction(request_id=0, donor_blood_type="O+"))
        score = client.grade()
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def health(self) -> Dict[str, Any]:
        resp = self._client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    def reset(
        self,
        seed: Optional[int] = None,
        task_name: str = "basic_compatibility",
        max_steps: int = 30,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"task_name": task_name, "max_steps": max_steps}
        if seed is not None:
            payload["seed"] = seed
        resp = self._client.post(f"{self.base_url}/reset", json=payload)
        resp.raise_for_status()
        return resp.json()

    def step(self, action: BloodBankAction) -> Tuple[Dict, float, bool, Dict]:
        payload = asdict(action)
        resp = self._client.post(f"{self.base_url}/step", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["observation"], data["reward"], data["done"], data.get("info", {})

    def get_state(self) -> Dict[str, Any]:
        resp = self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    def grade(self, task_name: Optional[str] = None) -> float:
        payload: Dict[str, Any] = {}
        if task_name:
            payload["task_name"] = task_name
        resp = self._client.post(f"{self.base_url}/grade", json=payload)
        resp.raise_for_status()
        return resp.json()["score"]

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AsyncBloodBankClient:
    """
    Asynchronous HTTP client for the BloodBank environment.

    Usage:
        async with AsyncBloodBankClient("http://localhost:8000") as client:
            obs = await client.reset()
            obs, reward, done, info = await client.step(action)
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def health(self) -> Dict[str, Any]:
        resp = await self._client.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    async def reset(
        self,
        seed: Optional[int] = None,
        task_name: str = "basic_compatibility",
        max_steps: int = 30,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"task_name": task_name, "max_steps": max_steps}
        if seed is not None:
            payload["seed"] = seed
        resp = await self._client.post(f"{self.base_url}/reset", json=payload)
        resp.raise_for_status()
        return resp.json()

    async def step(self, action: BloodBankAction) -> Tuple[Dict, float, bool, Dict]:
        payload = asdict(action)
        resp = await self._client.post(f"{self.base_url}/step", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return data["observation"], data["reward"], data["done"], data.get("info", {})

    async def get_state(self) -> Dict[str, Any]:
        resp = await self._client.get(f"{self.base_url}/state")
        resp.raise_for_status()
        return resp.json()

    async def grade(self, task_name: Optional[str] = None) -> float:
        payload: Dict[str, Any] = {}
        if task_name:
            payload["task_name"] = task_name
        resp = await self._client.post(f"{self.base_url}/grade", json=payload)
        resp.raise_for_status()
        return resp.json()["score"]

    async def close(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()
