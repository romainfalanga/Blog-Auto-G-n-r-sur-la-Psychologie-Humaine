#!/usr/bin/env python3
"""Read-only structure signals, complemented by hash-bound editorial reviews.

This does not assess clinical validity, example
relevance or whether three instructions are meaningfully different. A current
manual review can cover prose equivalents; a changed body invalidates that review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

H2 = re.compile(r"^##[ \t]+(.+)$", re.M)
H3 = re.compile(r"^###[ \t]+(.+)$", re.M)
ITEM = re.compile(r"^( *)(?:[-*+]|\d+[.)])[ \t]+(.+)$")
NOTE = re.compile(r"^(?:notes?\b|sources?\b|références?\b|précautions?\b|point de prudence\b|repères? scientifiques?\b|pour aller plus loin\b)", re.I)
EXERCISE = re.compile(r"^(?:exercice(?: pratique| concret)?|pour essayer\b.*|en pratique|mise en pratique|étapes?)[ \t]*[:：]?$", re.I)
EXAMPLE = re.compile(r"^(?:(?:autre|second|deuxième|premier|1er|2e)\s+)?exemple(?:s)?(?:\s+(?:concret|situé)s?)?\s*[:：]", re.I)
THIRD = re.compile(r"^(?:troisième|3e|3ème)\s+(?:étape|action|temps)\s*[:：]", re.I)
NUMBERED = re.compile(r"^([123])[.)][ \t]+")


def clean_line(line: str) -> str:
    return line.strip().strip("*_ ").strip()


def parts(text: str, heading: re.Pattern) -> list[dict]:
    marks = list(heading.finditer(text))
    return [dict(title=m.group(1).strip(), text=text[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(text)].strip()) for i, m in enumerate(marks)]


def list_groups(text: str) -> list[int]:
    """Count peer instructions, never nested examples as extra instructions."""
    groups, current = [], []
    for line in text.splitlines() + ["END"]:
        match = ITEM.match(line)
        if match:
            current.append(len(match.group(1)))
        elif line.strip() and current:
            base = min(current)
            groups.append(sum(indent == base for indent in current))
            current = []
    return groups


def unit_metrics(unit: dict, technique: bool) -> dict:
    body = unit["text"]
    lines = [clean_line(line) for line in body.splitlines()]
    exercise_at = next((i for i, line in enumerate(lines) if EXERCISE.match(line)), None)
    exercise = "\n".join(body.splitlines()[exercise_at + 1:]) if exercise_at is not None else body
    list_count = max(list_groups(exercise), default=0)
    # An explicitly named third step may follow two peer bullets in prose.
    third_in_prose = any(THIRD.match(line) for line in lines)
    action_count = max(list_count, 3 if list_count >= 2 and third_in_prose else 0)
    example_labels = sum(bool(EXAMPLE.match(line)) for line in lines)
    example_count = max(max(list_groups(body), default=0), example_labels)
    return {
        "heading": unit["title"],
        "nonempty": bool(body),
        "words": len(re.findall(r"\b[\wÀ-ÿ’'-]+\b", body)),
        "peer_action_markers": action_count if technique else None,
        "example_markers": example_count if not technique else None,
        "exercise_label": exercise_at is not None if technique else None,
        "goal_label_or_phrase": bool(re.search(r"\b(?:objectif|le but|il s.agit|l.idée est)\b", body, re.I)) if technique else None,
    }


def read_article(path: Path) -> tuple[str, str]:
    raw = path.read_text(encoding="utf-8")
    split = raw.split("---", 2)
    if len(split) != 3 or split[0].strip():
        raise ValueError("Missing YAML frontmatter delimiters")
    return split[1], split[2].strip()


def scalar(frontmatter: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:[ \t]*(.+?)[ \t]*$", frontmatter, re.M)
    return match.group(1).strip().strip("\"'") if match else None


def load_review_rows(path: Path) -> list[dict]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(doc, list):
        return doc
    for key in ("articles", "files", "reviews", "rows", "entries"):
        value = doc.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return [dict(row, path=name) for name, row in value.items()]
    return []


def read_mirror(root: Path, character_id: str, article_path: str) -> tuple[str | None, str | None]:
    path = root / "data/article_mirrors" / (character_id + ".json")
    key = Path(article_path).relative_to("content").with_suffix("").as_posix()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("expected an object keyed by article path")
        paragraphs = document[key]
        if not isinstance(paragraphs, list) or not paragraphs or not all(isinstance(p, str) and p.strip() for p in paragraphs):
            raise ValueError("expected a nonempty list of paragraph strings")
        serialized = json.dumps(paragraphs, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest(), None
    except (OSError, ValueError, KeyError) as exc:
        return None, f"Mirror missing or invalid at {path.relative_to(root)} [{key}]: {exc}"


def review_proof(row: dict | None, digest: str, mirror_digest: str | None = None) -> dict:
    proof = (row or {}).get("owner_review") or row or {}
    status = proof.get("status") or (row or {}).get("owner_status")
    recorded = (proof.get("body_sha256") or proof.get("sha256")) if status else None
    techniques = proof.get("techniques", [])
    def valid_technique(t: dict) -> bool:
        if not isinstance(t, dict):
            return False
        objective = t.get("objective") or t.get("objectif")
        actions = t.get("actions")
        return bool(isinstance(objective, str) and objective.strip()
                    and isinstance(actions, list) and len(actions) >= 3
                    and all(isinstance(action, str) and action.strip() for action in actions))
    techniques_complete = isinstance(techniques, list) and len(techniques) == 3 and all(valid_technique(t) for t in techniques)
    body_current = bool(recorded == digest and status in ("revised", "already_complete") and techniques_complete)
    mirror_current = bool(mirror_digest and proof.get("mirror_reviewed") is True and proof.get("mirror_sha256") == mirror_digest)
    return dict(
        status=status,
        current=bool(body_current and mirror_current),
        body_current=body_current,
        mirror_current=mirror_current,
        stale=bool(recorded and recorded != digest),
        mirror_stale=bool(proof.get("mirror_sha256") and proof["mirror_sha256"] != mirror_digest),
        three_techniques_with_three_action_summaries=techniques_complete,
    )


def analyze(body: str, review: dict | None = None, mirror_digest: str | None = None) -> dict:
    digest = hashlib.sha256(body.strip().encode("utf-8")).hexdigest()
    sections = parts(body, H2)
    problems, needs_review = [], []
    if len(sections) != 4:
        problems.append(f"Expected four H2 sections, found {len(sections)}")
    if any(not s["text"] for s in sections):
        problems.append("An H2 section is empty")
    manifests = parts(sections[1]["text"], H3) if len(sections) > 1 else []
    all_techniques = parts(sections[2]["text"], H3) if len(sections) > 2 else []
    numbered = [t for t in all_techniques if NUMBERED.match(t["title"])]
    # Scientific H3 notes may accompany the three numbered practical techniques.
    techniques = numbered if numbered else [t for t in all_techniques if not NOTE.match(t["title"])]
    mm = [unit_metrics(t, False) for t in manifests]
    tm = [unit_metrics(t, True) for t in techniques]
    if len(manifests) < 3:
        needs_review.append("Fewer than three manifestation H3 units; check prose equivalents")
    if len(techniques) != 3 or (numbered and sorted(int(NUMBERED.match(t["title"]).group(1)) for t in numbered) != [1, 2, 3]):
        needs_review.append("Three practical techniques could not be identified unambiguously")
    for unit in mm:
        if not unit["nonempty"]:
            problems.append("Empty manifestation: " + unit["heading"])
        if unit["example_markers"] < 2:
            needs_review.append("Two example markers not found: " + unit["heading"])
    for unit in tm:
        if not unit["nonempty"]:
            problems.append("Empty technique: " + unit["heading"])
        if unit["peer_action_markers"] < 3:
            needs_review.append("Three action markers not found: " + unit["heading"])
    if sections and not re.search(r"(?m)^---[ \t]*\n+\S", sections[-1]["text"]):
        problems.append("Missing conclusion after horizontal divider")
    proof = review_proof(review, digest, mirror_digest)
    # A current hash binds the editorial declaration to this exact body. It does not
    # turn regex signals into a second semantic or clinical certification.
    return dict(body_sha256=digest, h2_titles=[s["title"] for s in sections], manifestations=mm, techniques=tm,
                additional_technique_notes=[t["title"] for t in all_techniques if t not in techniques],
                mechanical_problems=problems, prose_signals_requiring_review=needs_review,
                manual_review=proof, review_required=bool(needs_review and not proof["body_current"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--audit", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-reviewed", action="store_true", help="Fail missing/stale manual reviews, including prose equivalents")
    args = parser.parse_args()
    root = args.root.resolve()
    registry = json.loads((root / "scripts/personnages.json").read_text(encoding="utf-8"))["personnages"]
    original_names = {r["prenom"]: r["id"] for r in registry}
    reviews = {}
    for audit in args.audit:
        for row in load_review_rows(audit):
            if row.get("path") in reviews:
                raise ValueError("Duplicate reviewed path: " + row["path"])
            reviews[row["path"]] = row
    articles = []
    for path in sorted((root / "content/posts").rglob("*.md")):
        fm, body = read_article(path)
        name = scalar(fm, "personnage")
        if name not in original_names:
            continue
        relative = path.relative_to(root).as_posix()
        character_id = original_names[name]
        mirror_digest, mirror_error = read_mirror(root, character_id, relative)
        result = analyze(body, reviews.get(relative), mirror_digest)
        if mirror_error:
            result["mechanical_problems"].append(mirror_error)
        articles.append(dict(path=relative, character_id=character_id, mirror_sha256=mirror_digest, **result))
    found_paths = {r["path"] for r in articles}
    unmatched_review_paths = sorted(set(reviews) - found_paths)
    dataset_errors = []
    if len(registry) != 33:
        dataset_errors.append(f"Frozen original registry must contain 33 characters, found {len(registry)}")
    if len(articles) != 387:
        dataset_errors.append(f"Frozen original pack must contain 387 articles, found {len(articles)}")
    found_characters = len({r["character_id"] for r in articles})
    if found_characters != 33:
        dataset_errors.append(f"Frozen original pack must contain 33 characters, found {found_characters}")
    for unmatched in unmatched_review_paths:
        dataset_errors.append("Review path not found in original corpus: " + unmatched)
    summary = dict(articles=len(articles), characters=len({r["character_id"] for r in articles}),
                   mechanical_failures=sum(bool(r["mechanical_problems"]) for r in articles),
                   current_manual_reviews=sum(r["manual_review"]["current"] for r in articles),
                   current_body_reviews=sum(r["manual_review"]["body_current"] for r in articles),
                   current_mirror_reviews=sum(r["manual_review"]["mirror_current"] for r in articles),
                   stale_manual_reviews=sum(r["manual_review"]["stale"] for r in articles),
                   stale_mirror_reviews=sum(r["manual_review"]["mirror_stale"] for r in articles),
                   prose_signals_without_current_review=sum(r["review_required"] for r in articles),
                   unmatched_review_paths=len(unmatched_review_paths), dataset_errors=len(dataset_errors))
    report = dict(schema_version=1, method="Mechanical signals and exact-body manual review binding; no semantic or clinical certification by regex.", summary=summary, dataset_errors=dataset_errors, articles=articles)
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    for error in dataset_errors:
        print(error)
    for row in articles:
        failures = row["mechanical_problems"][:]
        if args.require_reviewed and not row["manual_review"]["body_current"]:
            failures.append("Body manual review missing, incomplete or stale")
        if args.require_reviewed and not row["manual_review"]["mirror_current"]:
            failures.append("Mirror manual review missing or stale")
        if failures:
            print(row["path"] + ": " + "; ".join(failures))
    return int(bool(dataset_errors) or summary["mechanical_failures"] > 0 or (args.require_reviewed and (summary["current_body_reviews"] != len(articles) or summary["current_mirror_reviews"] != len(articles))))


if __name__ == "__main__":
    raise SystemExit(main())
