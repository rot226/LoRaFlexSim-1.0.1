import asyncio
import time

import anyio
from fastapi.routing import APIRoute

from loraflexsim.launcher import web_api


def test_rest_api_status_endpoint_lifecycle() -> None:
    routes = {
        route.path
        for route in web_api.app.routes
        if isinstance(route, APIRoute)
    }
    assert "/simulations/status" in routes

    async def scenario() -> None:
        original_simulator = web_api.Simulator

        class DummySimulator:
            def __init__(self, **_: object) -> None:
                self.running = True
                self.metrics = {"pdr": 0.0, "ticks": 0}

            def run(self) -> None:
                while self.running:
                    self.metrics["ticks"] += 1
                    time.sleep(0.01)

            def stop(self) -> None:
                self.running = False

            def get_metrics(self) -> dict:
                return dict(self.metrics)

        web_api.Simulator = DummySimulator
        web_api._sim = None
        web_api._sim_task = None

        try:
            initial_payload = await web_api.simulation_status()
            assert initial_payload["status"] == "idle"
            assert initial_payload["metrics"] == {}

            start_payload = web_api.Command(command="start_sim", params={})
            start_response = await web_api.start_simulation(start_payload)
            assert start_response["status"] == "started"

            await asyncio.sleep(0)

            running_payload = await web_api.simulation_status()
            assert running_payload["status"] == "running"
            assert "pdr" in running_payload["metrics"]

            stop_response = await web_api.stop_simulation()
            assert stop_response["status"] == "stopped"

            final_payload = await web_api.simulation_status()
            assert final_payload["status"] == "stopped"
            assert "pdr" in final_payload["metrics"]

        finally:
            try:
                if web_api._sim is not None and web_api._sim.running:
                    web_api._sim.stop()
                if web_api._sim_task is not None:
                    await web_api._sim_task
            finally:
                web_api._sim = None
                web_api._sim_task = None
                web_api.Simulator = original_simulator

    anyio.run(scenario, backend="asyncio")
