from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


ROOT = Path(__file__).resolve().parents[1]


def _requirements_from_pyproject() -> dict[str, Requirement]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]
    return {item.name: item for item in map(Requirement, project["dependencies"])}


def _requirements_from_file() -> dict[str, Requirement]:
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
    return {
        item.name: item
        for item in map(Requirement, (line for line in lines if line.strip()))
    }


def test_runtime_dependency_files_stay_consistent() -> None:
    project = _requirements_from_pyproject()
    requirements = _requirements_from_file()
    for name, dependency in project.items():
        assert name in requirements
        assert requirements[name].specifier == dependency.specifier


def test_torch_213_is_supported_for_current_kaggle_images() -> None:
    torch_range = _requirements_from_pyproject()["torch"].specifier
    assert isinstance(torch_range, SpecifierSet)
    assert "2.13.0" in torch_range
    assert "2.14.0" not in torch_range
