from __future__ import annotations


def test_registry_does_not_load_v3_random_forest(monkeypatch):
    from app.models import registry as registry_module

    class _Loader:
        calls = []

        def load_json(self, module, name):
            self.calls.append((module, name))
            if (module, name) == ("forecaster_v2", "evaluation.json"):
                return {"winner": "tier3_sarima", "winner_artifact": "tier3_sarima.joblib"}
            if (module, name) == ("forecaster_v2", "metadata.json"):
                return {}
            raise FileNotFoundError

        def resolve(self, module):
            assert module == "forecaster_v2"
            from pathlib import Path

            return Path("models/forecaster_v2")

    monkeypatch.setattr(
        registry_module.ModelLoader,
        "load_joblib",
        lambda _self, _module, _artifact: {"kind": "sarima"},
    )
    loader = _Loader()
    registry_module.ModelRegistry(loader=loader)._load_forecaster()

    assert all(module != "forecaster_v3" for module, _ in loader.calls)
