from scripts import setup_cpu


def test_setup_preserves_data_and_requirements(tmp_path, monkeypatch):
    cached = tmp_path / "training_data/features.pkl"
    cached.parent.mkdir()
    cached.write_bytes(b"research cache")
    requirements = tmp_path / "requirements-cpu.txt"
    requirements.write_text("example==1.0\n")
    created, commands = [], []
    monkeypatch.setattr(setup_cpu, "ROOT", tmp_path)
    monkeypatch.setattr(setup_cpu.venv.EnvBuilder, "create", lambda self, path: created.append(path))
    monkeypatch.setattr(setup_cpu.subprocess, "run", lambda cmd, **kw: commands.append((cmd, kw)))
    python = setup_cpu.create_environment(tmp_path / ".venv")
    assert created == [(tmp_path / ".venv").resolve()]
    assert commands == [([str(python), "-m", "pip", "install", "-r", str(requirements)], {"check": True})]
    assert cached.read_bytes() == b"research cache"
    assert requirements.read_text() == "example==1.0\n"
