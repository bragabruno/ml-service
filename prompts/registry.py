from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts"


@dataclass(frozen=True)
class PromptMetadata:
    name: str
    version: str
    model: str = "default"
    temperature: float = 0.0
    description: str = ""


@dataclass
class RenderedPrompt:
    system: str
    user: str
    metadata: PromptMetadata


_env: Environment | None = None


def _get_env() -> Environment:
    global _env
    if _env is None:
        _env = Environment(
            loader=FileSystemLoader(str(_PROMPTS_DIR)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
        )
    return _env


def load_prompt(name: str, version: str = "v1", **kwargs: Any) -> RenderedPrompt:
    base_path = f"{name}/{version}"
    meta_path = _PROMPTS_DIR / name / f"{version}.meta.json"

    metadata = PromptMetadata(name=name, version=version)
    if meta_path.exists():
        meta_dict = json.loads(meta_path.read_text())
        metadata = PromptMetadata(
            name=meta_dict.get("name", name),
            version=meta_dict.get("version", version),
            model=meta_dict.get("model", "default"),
            temperature=meta_dict.get("temperature", 0.0),
            description=meta_dict.get("description", ""),
        )

    env = _get_env()

    system_template = env.get_template(f"{base_path}.system.md")
    user_template = env.get_template(f"{base_path}.user.md")

    system = system_template.render(**kwargs)
    user = user_template.render(**kwargs)

    return RenderedPrompt(system=system, user=user, metadata=metadata)


def list_prompts() -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    if not _PROMPTS_DIR.exists():
        return results
    for category_dir in sorted(_PROMPTS_DIR.iterdir()):
        if not category_dir.is_dir() or category_dir.name.startswith("_"):
            continue
        for f in sorted(category_dir.iterdir()):
            if f.suffix == ".md" and ".system." in f.name:
                version = f.stem.replace(".system", "")
                results.append({"category": category_dir.name, "version": version})
    return results
