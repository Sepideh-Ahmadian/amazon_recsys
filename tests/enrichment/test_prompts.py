from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from amazon_recsys.enrichment.prompts import (
    ExtractionOutput,
    PromptError,
    QuestionsSpec,
    load_aspect_map,
    load_extractor_template,
    load_judge_questions,
    render_extractor,
)

REAL_PROMPTS = Path(__file__).resolve().parents[2] / "prompts" / "all_beauty"
VARS = {"product_title": "Lotion", "review_title": "Nice", "review_text": "Smells great"}


@pytest.fixture
def prompt_dir(tmp_path):
    (tmp_path / "extractor_v1.txt").write_text("Be careful.\n---USER---\nReview: ${review_text}")
    return tmp_path


def test_renders_all_variables_into_system_and_user_messages(prompt_dir):
    template, _ = load_extractor_template(prompt_dir, "v1")
    messages = render_extractor(template, review_text="hello")
    assert messages == [
        {"role": "system", "content": "Be careful."},
        {"role": "user", "content": "Review: hello"},
    ]


def test_missing_variable_raises(prompt_dir):
    template, _ = load_extractor_template(prompt_dir, "v1")
    with pytest.raises(PromptError, match="review_text"):
        render_extractor(template)


def test_hash_stable_and_changes_with_template_text(prompt_dir):
    _, first = load_extractor_template(prompt_dir, "v1")
    _, again = load_extractor_template(prompt_dir, "v1")
    (prompt_dir / "extractor_v1.txt").write_text("Be careful!\n---USER---\nReview: ${review_text}")
    _, changed = load_extractor_template(prompt_dir, "v1")
    assert first == again and first != changed


def test_missing_file_and_missing_marker(tmp_path):
    with pytest.raises(PromptError, match="not found"):
        load_extractor_template(tmp_path, "v9")
    (tmp_path / "extractor_v1.txt").write_text("no marker here")
    with pytest.raises(PromptError, match="---USER---"):
        load_extractor_template(tmp_path, "v1")


def test_real_extractor_prompt_renders_with_json_braces_intact():
    template, digest = load_extractor_template(REAL_PROMPTS, "v1")
    messages = render_extractor(template, **VARS)
    assert "Smells great" in messages[1]["content"] and "Lotion" in messages[1]["content"]
    assert '"overall_sentiment"' in messages[1]["content"]
    assert "${" not in messages[1]["content"] and len(digest) == 64


def test_real_judge_questions_load_and_are_valid():
    spec, digest = load_judge_questions(REAL_PROMPTS, "v1")
    assert spec.aspect_polarity.options == ["positive", "negative", "neutral"]
    assert 2 <= len(spec.completeness.levels) <= 10
    assert len(spec.overall_sentiment.levels) == 5
    assert len(digest) == 64


def _questions(**overrides):
    base = yaml.safe_load((REAL_PROMPTS / "judge_questions_v1.yaml").read_text())
    base.update(overrides)
    return base


def test_judge_question_hash_ignores_comments_but_tracks_content(tmp_path):
    base = _questions()
    path = tmp_path / "judge_questions_v1.yaml"
    path.write_text(yaml.safe_dump(base))
    _, h1 = load_judge_questions(tmp_path, "v1")
    path.write_text("# a comment\n" + yaml.safe_dump(base))
    _, h2 = load_judge_questions(tmp_path, "v1")
    base["aspect_supported"]["instructions"] = "Different wording."
    path.write_text(yaml.safe_dump(base))
    _, h3 = load_judge_questions(tmp_path, "v1")
    assert h1 == h2 and h1 != h3


@pytest.mark.parametrize(
    "override",
    [
        {"aspect_polarity": {"instructions": "x", "options": ["a", "a"]}},  # duplicate options
        {"aspect_polarity": {"instructions": "x", "options": ["only"]}},  # < 2 options
        {"completeness": {"instructions": "x", "levels": ["one"]}},  # < 2 levels
        {"completeness": {"instructions": "x", "levels": [str(i) for i in range(11)]}},  # > 10
        {"surprise": 1},  # unknown key
    ],
)
def test_invalid_question_specs_rejected(tmp_path, override):
    (tmp_path / "judge_questions_v1.yaml").write_text(yaml.safe_dump(_questions(**override)))
    with pytest.raises(PromptError):
        load_judge_questions(tmp_path, "v1")
    with pytest.raises(ValidationError):
        QuestionsSpec.model_validate(_questions(**override))


def test_extraction_schema_validation():
    ok = ExtractionOutput.model_validate(
        {
            "overall_sentiment": 0.5,
            "aspects": [{"aspect": "Scent", "sentiment": -1, "evidence": "x"}],
        }
    )
    assert ok.aspects[0].sentiment == -1.0
    assert ExtractionOutput.model_validate({"overall_sentiment": 0.0}).aspects == []
    with pytest.raises(ValidationError):
        ExtractionOutput.model_validate({"overall_sentiment": 1.5, "aspects": []})
    with pytest.raises(ValidationError):
        ExtractionOutput.model_validate(
            {"overall_sentiment": 0, "aspects": [{"aspect": "A", "sentiment": 2}]}
        )
    extra = ExtractionOutput.model_validate({"overall_sentiment": 0, "note": "ignored"})
    assert not hasattr(extra, "note")


def test_aspect_map(tmp_path):
    assert load_aspect_map(tmp_path) == {}
    (tmp_path / "aspect_map.yaml").write_text("Smell: Scent\nFragrance: scent\n")
    assert load_aspect_map(tmp_path) == {"smell": "scent", "fragrance": "scent"}
    assert load_aspect_map(REAL_PROMPTS)["smell"] == "scent"


def test_real_questions_cover_mention_and_sentiment_association():
    spec, _ = load_judge_questions(REAL_PROMPTS, "v1")
    assert "really mentioned" in spec.aspect_supported.instructions
    assert "specific aspect" in spec.aspect_polarity.instructions
    assert spec.aspect_polarity.options == ["positive", "negative", "neutral"]
