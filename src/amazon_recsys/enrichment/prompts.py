"""Load and render prompt files; define the extractor output schema and judge question spec."""

from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Annotated

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from amazon_recsys.utils.hashing import hash_obj, hash_text

USER_MARKER = "---USER---"


class PromptError(ValueError):
    """Raised for missing/invalid prompt files or unrendered variables."""


# --- extractor ----------------------------------------------------------------------------

Sentiment = Annotated[float, Field(ge=-1.0, le=1.0)]


class ExtractedAspect(BaseModel):
    aspect: str = Field(min_length=1)
    sentiment: Sentiment
    evidence: str = ""


class ExtractionOutput(BaseModel):
    """What the extractor must return for one review (extra keys are ignored)."""

    overall_sentiment: Sentiment
    aspects: list[ExtractedAspect] = []


def load_extractor_template(prompt_dir: str | Path, version: str) -> tuple[str, str]:
    """Return (template_text, hash) for prompts/<domain>/extractor_<version>.txt."""
    path = Path(prompt_dir) / f"extractor_{version}.txt"
    if not path.is_file():
        raise PromptError(f"Extractor prompt not found: {path}")
    text = path.read_text(encoding="utf-8")
    if USER_MARKER not in text:
        raise PromptError(f"{path} must contain a {USER_MARKER!r} line separating system and user")
    return text, hash_text(text)


def render_extractor(template: str, **variables: str) -> list[dict[str, str]]:
    """Render the template into chat messages. Every ${variable} must be supplied."""
    system, _, user = template.partition(USER_MARKER)
    try:
        user_text = Template(user.strip()).substitute(variables)
        system_text = Template(system.strip()).substitute(variables)
    except KeyError as err:
        raise PromptError(f"Missing prompt variable: {err.args[0]}") from err
    except ValueError as err:
        raise PromptError(f"Invalid prompt template: {err}") from err
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_text},
    ]


# --- judge questions ----------------------------------------------------------------------


class NoulSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: str = Field(min_length=1)


class ChoiceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: str = Field(min_length=1)
    options: list[str]

    @field_validator("options")
    @classmethod
    def _unique_options(cls, value: list[str]) -> list[str]:
        if len(value) < 2:
            raise ValueError("a Choice needs at least 2 options")
        if len(set(value)) != len(value):
            raise ValueError("Choice options must be unique")
        return value


class ScoreSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instructions: str = Field(min_length=1)
    levels: list[str]  # ordered worst first

    @field_validator("levels")
    @classmethod
    def _level_count(cls, value: list[str]) -> list[str]:
        if not 2 <= len(value) <= 10:
            raise ValueError("a Score needs between 2 and 10 levels")
        return value


class QuestionsSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: str
    aspect_supported: NoulSpec
    aspect_polarity: ChoiceSpec
    completeness: ScoreSpec
    overall_sentiment: ScoreSpec


def load_judge_questions(prompt_dir: str | Path, version: str) -> tuple[QuestionsSpec, str]:
    """Return (spec, hash) for prompts/<domain>/judge_questions_<version>.yaml.

    The hash covers the parsed content, so comments and whitespace do not change it.
    """
    path = Path(prompt_dir) / f"judge_questions_{version}.yaml"
    if not path.is_file():
        raise PromptError(f"Judge questions not found: {path}")
    try:
        spec = QuestionsSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    except ValidationError as err:
        raise PromptError(f"Invalid judge questions in {path}: {err}") from err
    return spec, hash_obj(spec.model_dump())


# --- aspect synonym map -------------------------------------------------------------------


def load_aspect_map(prompt_dir: str | Path) -> dict[str, str]:
    """Optional synonym -> canonical map; returns {} when the file does not exist."""
    path = Path(prompt_dir) / "aspect_map.yaml"
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise PromptError(f"{path} must be a mapping of synonym -> canonical aspect")
    return {str(k).strip().lower(): str(v).strip().lower() for k, v in raw.items()}
