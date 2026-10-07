#!/usr/bin/env python3
"""Generate, independently review, validate, and package content-edit mappings with GPT."""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import subprocess
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "data/matched_full_5369_snr0_batch4_generated/manifest.tsv"
OUT = ROOT / "data/content_edit_full_gpt_20261001"
GEN_SCHEMA = ROOT / "scripts/content_edit_generation_schema.json"
REV_SCHEMA = ROOT / "scripts/content_edit_review_schema.json"
MODEL = "gpt-5.6-sol"
WORD_RE = re.compile(r"^[A-Za-z]+$")
ALLOWED_POS = {
    "noun", "verb", "main verb", "adjective", "adverb", "ordinal",
    "temporal", "temporal word", "temporal noun", "temporal adverb",
    "gerund", "past participle", "noun modifier",
}


GEN_RULES = """You are the final human-level curator for an English speech content-editing dataset. For every input sentence, decide whether there is a HIGH-QUALITY replacement of exactly one written word by exactly one written word. Internally consider multiple candidates and critically reject awkward choices before answering.

Hard requirements for eligible=true:
1. old_word occurs exactly once in source_text, with identical spelling and case.
2. Replace exactly that one word; every other character, including punctuation and contractions, remains unchanged.
3. The result is fully grammatical, idiomatic, semantically plausible natural English, not merely syntactically possible.
4. Old and new fill the same grammatical slot and require no article, agreement, tense, preposition, or surrounding-word change.
5. Edit a content-bearing noun, main verb, adjective, adverb, temporal word, or ordinal. Never edit a proper name, auxiliary, determiner, pronoun, conjunction, preposition, negation, contraction fragment, or repeated token.
6. The edit must cause a genuine semantic change. Do not use a near-synonym, spelling variant, singular/plural variant, or derivational variant.
7. Prefer similar spoken length or syllable count when several equally natural choices exist, but naturalness is more important.
8. Avoid offensive, sexual, medical, or otherwise sensitive substitutions unless already essential to the source.
9. route is B_semantic_contrast for a clear opposition and C_natural_context for another natural alternative.
10. If no clearly safe edit exists, set eligible=false and use empty strings for old_word/new_word/edited_text/part_of_speech, route=none. Do not force an edit.

Return exactly one concise item per input, in the same order and with the same text_group_id. Re-check each edited sentence before returning it. Keep reason under 18 words."""


REVIEW_RULES = """You are the independent senior reviewer for an English speech content-editing dataset. Inspect every proposed one-word replacement. Do not trust the proposal or its confidence. Return accept only when it is unquestionably natural and obeys every rule. Return revise with your own better one-word replacement when possible. Otherwise return reject.

Hard requirements for accept/revise:
1. old_word occurs exactly once in source_text with identical spelling and case.
2. edited_text differs from source_text only by replacing old_word with exactly one alphabetic word new_word.
3. The result sounds like something a native English speaker could naturally say in a realistic context.
4. It is grammatical and needs no changes to articles, agreement, tense, prepositions, punctuation, or surrounding words.
5. Edit only a content-bearing noun, main verb, adjective, adverb, temporal word, or ordinal; not a name or function word.
6. The edit makes a genuine semantic change, not a near-synonym or morphological variant.
7. Reject merely technically possible, pragmatically bizarre, misleading, or contextually incoherent sentences.
8. Prefer similar spoken length when equally natural alternatives exist.
9. route is B_semantic_contrast for clear opposition and C_natural_context otherwise.
10. For reject, use empty strings for old_word/new_word/edited_text/part_of_speech and route=none.

Return one concise result per input in the same order. Keep reason under 18 words."""


def load_source() -> tuple[list[dict], list[dict]]:
    with SOURCE_MANIFEST.open(encoding="utf-8", newline="") as f:
        pairs = list(csv.DictReader(f, delimiter="\t"))
    by_text: dict[str, list[dict]] = defaultdict(list)
    order = []
    for row in pairs:
        text = row["text"]
        if text not in by_text:
            order.append(text)
        by_text[text].append(row)
    groups = []
    for i, text in enumerate(order, 1):
        members = by_text[text]
        groups.append(
            {
                "text_group_id": f"text_{i:04d}",
                "source_text": text,
                "num_pairs": len(members),
                "pair_ids": [r["pair_id"] for r in members],
            }
        )
    return pairs, groups


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, delimiter="\t", fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def codex_call(prompt: str, schema: Path, output_path: Path, attempts: int = 3) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix(".log")
    cmd = [
        "codex", "exec", "--ephemeral", "--skip-git-repo-check", "--ignore-rules",
        "--sandbox", "read-only", "--model", MODEL,
        "-c", 'model_reasoning_effort="low"',
        "--output-schema", str(schema), "--output-last-message", str(output_path), "-",
    ]
    last_error = ""
    for attempt in range(1, attempts + 1):
        started = time.time()
        proc = subprocess.run(cmd, input=prompt, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        log_path.write_text(proc.stdout, encoding="utf-8")
        try:
            raw_payload = output_path.read_text(encoding="utf-8") if output_path.exists() else ""
            # On transient gateway errors codex may print a valid final JSON payload
            # to stdout but fail before writing --output-last-message.
            if not raw_payload:
                candidates = re.findall(r"\{\"items\":.*\}", proc.stdout, flags=re.DOTALL)
                raw_payload = candidates[-1] if candidates else ""
            payload = json.loads(raw_payload)
            if proc.returncode == 0 and isinstance(payload.get("items"), list):
                print(f"  completed {output_path.stem} in {time.time()-started:.1f}s")
                return payload
            last_error = f"returncode={proc.returncode}, malformed payload"
        except Exception as exc:
            last_error = f"returncode={proc.returncode}, {type(exc).__name__}: {exc}"
        print(f"  retry {attempt}/{attempts}: {output_path.stem}: {last_error}")
        time.sleep(min(10, attempt * 2))
    raise RuntimeError(f"Codex call failed for {output_path}: {last_error}; see {log_path}")


def expected_ids(batch: list[dict]) -> list[str]:
    return [x["text_group_id"] for x in batch]


def normalize_payload(payload: dict, batch: list[dict]) -> list[dict]:
    items = payload["items"]
    wanted = expected_ids(batch)
    found = [str(x.get("text_group_id", "")) for x in items]
    if len(found) != len(wanted) or set(found) != set(wanted) or len(set(found)) != len(found):
        raise ValueError(f"ID set mismatch: expected {wanted[:3]}... got {found[:3]}...")
    # Models occasionally swap two JSON objects despite the ordering request.
    # Canonicalize by the stable group id before saving/validating downstream.
    by_id = {str(x["text_group_id"]): x for x in items}
    payload["items"] = [by_id[group_id] for group_id in wanted]
    return payload["items"]


def generation_prompt(batch: list[dict]) -> str:
    data = [{"text_group_id": x["text_group_id"], "source_text": x["source_text"]} for x in batch]
    return GEN_RULES + "\n\nINPUT JSON:\n" + json.dumps(data, ensure_ascii=False)


def review_prompt(batch: list[dict], generated: dict[str, dict]) -> str:
    data = []
    for x in batch:
        proposal = generated[x["text_group_id"]]
        data.append(
            {
                "text_group_id": x["text_group_id"],
                "source_text": x["source_text"],
                "proposal": proposal,
            }
        )
    return REVIEW_RULES + "\n\nINPUT JSON:\n" + json.dumps(data, ensure_ascii=False)


def run_generation(groups: list[dict], batch_size: int, workers: int = 1) -> None:
    batch_dir = OUT / "generation_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    total = (len(groups) + batch_size - 1) // batch_size

    def one(job: tuple[int, int]) -> str:
        n, start = job
        batch = groups[start : start + batch_size]
        path = batch_dir / f"batch_{n:04d}.json"
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                normalize_payload(payload, batch)
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                return f"[generate {n}/{total}] cached"
            except Exception:
                pass
        payload = codex_call(generation_prompt(batch), GEN_SCHEMA, path)
        normalize_payload(payload, batch)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return f"[generate {n}/{total}] completed {batch[0]['text_group_id']}..{batch[-1]['text_group_id']}"

    jobs = [(n, start) for n, start in enumerate(range(0, len(groups), batch_size), 1)]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, job): job[0] for job in jobs}
        for future in as_completed(futures):
            print(future.result(), flush=True)


def load_batches(directory: Path) -> dict[str, dict]:
    result = {}
    for path in sorted(directory.glob("batch_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload["items"]:
            result[item["text_group_id"]] = item
    return result


def run_review(groups: list[dict], batch_size: int, workers: int = 1) -> None:
    generated = load_batches(OUT / "generation_batches")
    if len(generated) != len(groups):
        raise RuntimeError(f"Need {len(groups)} generated items, found {len(generated)}")
    batch_dir = OUT / "review_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    total = (len(groups) + batch_size - 1) // batch_size
    def one(job: tuple[int, int]) -> str:
        n, start = job
        batch = groups[start : start + batch_size]
        path = batch_dir / f"batch_{n:04d}.json"
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                normalize_payload(payload, batch)
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                return f"[review {n}/{total}] cached"
            except Exception:
                pass
        last_error = None
        for retry in range(3):
            try:
                if path.exists():
                    path.unlink()
                payload = codex_call(review_prompt(batch, generated), REV_SCHEMA, path)
                normalize_payload(payload, batch)
                break
            except Exception as exc:
                last_error = exc
                if retry == 2:
                    raise
        if last_error is not None and not path.exists():
            raise last_error
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return f"[review {n}/{total}] completed {batch[0]['text_group_id']}..{batch[-1]['text_group_id']}"

    jobs = [(n, start) for n, start in enumerate(range(0, len(groups), batch_size), 1)]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {pool.submit(one, job): job[0] for job in jobs}
        for future in as_completed(futures):
            print(future.result(), flush=True)


def validate_edit(source: str, old: str, new: str, edited: str) -> tuple[bool, str]:
    if not WORD_RE.fullmatch(old or "") or not WORD_RE.fullmatch(new or ""):
        return False, "old/new must each be one alphabetic word"
    if old.casefold() == new.casefold():
        return False, "replacement is unchanged"
    pattern = re.compile(rf"(?<![A-Za-z]){re.escape(old)}(?![A-Za-z])")
    matches = list(pattern.finditer(source))
    if len(matches) != 1:
        return False, f"old_word exact occurrence count is {len(matches)}"
    match = matches[0]
    expected = source[: match.start()] + new + source[match.end() :]
    if edited != expected:
        return False, "edited_text is not an exact one-word replacement"
    return True, ""


def simple_syllables(word: str) -> int:
    value = re.sub(r"[^a-z]", "", word.lower())
    if not value:
        return 0
    groups = re.findall(r"[aeiouy]+", value)
    count = len(groups)
    if value.endswith("e") and not value.endswith(("le", "ye")) and count > 1:
        count -= 1
    return max(1, count)


def finalize(pairs: list[dict], groups: list[dict]) -> None:
    generated = load_batches(OUT / "generation_batches")
    reviewed = load_batches(OUT / "review_batches")
    if len(generated) != len(groups) or len(reviewed) != len(groups):
        raise RuntimeError("Generation/review batches are incomplete")

    group_rows = []
    group_by_text = {}
    for group in groups:
        gid = group["text_group_id"]
        gen = generated[gid]
        rev = reviewed[gid]
        decision = rev["decision"]
        old, new, edited = rev["old_word"], rev["new_word"], rev["edited_text"]
        valid, error = validate_edit(group["source_text"], old, new, edited) if decision != "reject" else (False, "reviewer rejected")
        confidence = float(rev["confidence"])
        pos_allowed = str(rev["part_of_speech"]).strip().casefold() in ALLOWED_POS
        if decision in ("accept", "revise") and valid and confidence >= 0.90 and pos_allowed:
            status = "accepted"
        elif decision == "reject":
            status = "rejected"
        else:
            status = "needs_review"
        row = {
            "text_group_id": gid,
            "source_text": group["source_text"],
            "num_pairs": group["num_pairs"],
            "status": status,
            "review_decision": decision,
            "old_word": old if status != "rejected" else "",
            "new_word": new if status != "rejected" else "",
            "edited_text": edited if status != "rejected" else "",
            "operation": "replace" if status != "rejected" else "",
            "route": rev["route"] if status != "rejected" else "none",
            "part_of_speech": rev["part_of_speech"] if status != "rejected" else "",
            "confidence": confidence,
            "auk_instruction": f"Replace '{old}' with '{new}'." if status == "accepted" else "",
            "old_syllables_est": simple_syllables(old) if old else "",
            "new_syllables_est": simple_syllables(new) if new else "",
            "syllable_delta_est": (simple_syllables(new) - simple_syllables(old)) if old and new else "",
            "generation_eligible": gen["eligible"],
            "generation_old_word": gen["old_word"],
            "generation_new_word": gen["new_word"],
            "generation_confidence": gen["confidence"],
            "review_reason": rev["reason"],
            "mechanical_validation": "pass" if valid else "fail",
            "validation_error": error if error else ("disallowed part_of_speech" if not pos_allowed and decision != "reject" else ""),
        }
        group_rows.append(row)
        group_by_text[group["source_text"]] = row

    pair_rows = []
    for pair in pairs:
        edit = group_by_text[pair["text"]]
        pair_rows.append(
            {
                "pair_id": pair["pair_id"],
                "text_group_id": edit["text_group_id"],
                "speaker": pair["speaker"],
                "gender": pair["gender"],
                "anchor_id": pair["anchor_id"],
                "anchor_vctk_id": pair["anchor_vctk_id"],
                "source_pair_dir": pair["pair_dir"],
                "source_text": pair["text"],
                "status": edit["status"],
                "old_word": edit["old_word"],
                "new_word": edit["new_word"],
                "edited_text": edit["edited_text"],
                "route": edit["route"],
                "part_of_speech": edit["part_of_speech"],
                "confidence": edit["confidence"],
                "auk_instruction": edit["auk_instruction"],
                "anchor_duration_seconds": pair["anchor_duration_seconds"],
            }
        )

    group_fields = list(group_rows[0])
    pair_fields = list(pair_rows[0])
    write_tsv(OUT / "content_edit_text_manifest.tsv", group_rows, group_fields)
    write_tsv(OUT / "content_edit_pair_manifest.tsv", pair_rows, pair_fields)
    (OUT / "content_edit_text_manifest.json").write_text(json.dumps(group_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    status_groups = Counter(r["status"] for r in group_rows)
    status_pairs = Counter(r["status"] for r in pair_rows)
    accepted = [r for r in group_rows if r["status"] == "accepted"]
    report = {
        "created_date": "2026-10-01",
        "generator_and_reviewer": MODEL,
        "source_pairs": len(pairs),
        "unique_source_texts": len(groups),
        "group_status_counts": dict(status_groups),
        "pair_status_counts": dict(status_pairs),
        "accepted_route_counts": dict(Counter(r["route"] for r in accepted)),
        "accepted_pos_counts": dict(Counter(r["part_of_speech"] for r in accepted)),
        "accepted_review_decisions": dict(Counter(r["review_decision"] for r in accepted)),
        "accepted_abs_syllable_delta": dict(Counter(str(abs(int(r["syllable_delta_est"]))) for r in accepted)),
        "policy": "single-word replace; GPT generation plus independent GPT review plus deterministic validation",
    }
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    table = []
    for r in group_rows:
        table.append(
            "<tr>"
            f"<td>{html.escape(r['text_group_id'])}</td><td>{html.escape(r['status'])}</td>"
            f"<td>{html.escape(r['source_text'])}</td><td>{html.escape(r['edited_text'])}</td>"
            f"<td>{html.escape(r['auk_instruction'])}</td><td>{html.escape(r['review_reason'])}</td>"
            "</tr>"
        )
    preview = """<!doctype html><html><head><meta charset='utf-8'><title>Content edit manifest</title>
<style>body{font-family:system-ui;margin:20px}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:6px;vertical-align:top}th{position:sticky;top:0;background:#eee}tr:nth-child(even){background:#fafafa}</style></head><body>"""
    preview += f"<h1>AuK Content Editing Text Manifest</h1><pre>{html.escape(json.dumps(report, ensure_ascii=False, indent=2))}</pre>"
    preview += "<table><thead><tr><th>ID</th><th>Status</th><th>Source</th><th>Edited</th><th>Instruction</th><th>Review</th></tr></thead><tbody>"
    preview += "".join(table) + "</tbody></table></body></html>"
    (OUT / "preview.html").write_text(preview, encoding="utf-8")

    readme = f"""# AuK Content Editing：GPT 筛选与生成清单

生成日期：2026-10-01。母体包含 {len(pairs)} 个 pair、{len(groups)} 个唯一原句。

流程：GPT 逐句生成候选 → 独立 GPT 复核并接受/改写/拒绝 → 程序验证精确的一词替换 → 传播到 pair。

只有 `status=accepted` 且 `auk_instruction` 非空的行可以进入 AuK 推理。`needs_review` 必须人工处理；`rejected` 不应生成。相同原句共享 `text_group_id` 和编辑方案，后续实验划分应按 `text_group_id` 隔离。

统计见 `report.json`。唯一文本级清单为 `content_edit_text_manifest.tsv`；实际 pair 级清单为 `content_edit_pair_manifest.tsv`；浏览器审阅用 `preview.html`。
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    zip_path = OUT.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT.rglob("*")):
            if path.is_file() and path.suffix != ".log":
                zf.write(path, path.relative_to(OUT.parent))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Package: {zip_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("generate", "review", "finalize", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=80)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    pairs, groups = load_source()
    OUT.mkdir(parents=True, exist_ok=True)
    write_tsv(OUT / "source_text_groups.tsv", groups, ["text_group_id", "source_text", "num_pairs", "pair_ids"])
    if args.stage in ("generate", "all"):
        run_generation(groups, args.batch_size, args.workers)
    if args.stage in ("review", "all"):
        run_review(groups, args.batch_size, args.workers)
    if args.stage in ("finalize", "all"):
        finalize(pairs, groups)


if __name__ == "__main__":
    main()
