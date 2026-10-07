#!/usr/bin/env python3
"""Build a manually reviewable 50-item AuK content-editing pilot manifest."""

from __future__ import annotations

import csv
import html
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "data/matched_full_5369_snr0_batch4_generated/manifest.tsv"
OUTPUT_DIR = ROOT / "data/content_edit_candidates_50_20261001"


# pair_id, original token, replacement token, route, POS, short rationale
EDITS = [
    ("pair_00005", "lawyer", "doctor", "C_natural_context", "noun", "common occupation substitution"),
    ("pair_00069", "first", "second", "C_natural_context", "ordinal", "same grammatical slot and plausible meaning"),
    ("pair_00138", "good", "close", "C_natural_context", "adjective", "natural collocation: close friends"),
    ("pair_00287", "accept", "refuse", "B_semantic_contrast", "verb", "direct semantic contrast"),
    ("pair_00361", "Safety", "Cost", "C_natural_context", "noun", "natural sentence-level topic substitution"),
    ("pair_00501", "superb", "awful", "B_semantic_contrast", "adjective", "clear evaluative contrast"),
    ("pair_00647", "poor", "strong", "B_semantic_contrast", "adjective", "clear performance contrast"),
    ("pair_00720", "play", "fight", "C_natural_context", "verb", "natural collocation: fight for each other"),
    ("pair_00788", "Immediate", "Further", "C_natural_context", "adjective", "natural policy-language substitution"),
    ("pair_00931", "bought", "sold", "B_semantic_contrast", "verb", "opposing transaction"),
    ("pair_00998", "night", "weekend", "C_natural_context", "noun", "natural duration expression"),
    ("pair_01064", "Special", "Additional", "C_natural_context", "adjective", "natural collocation: additional measures"),
    ("pair_01133", "good", "poor", "B_semantic_contrast", "adjective", "clear polarity change"),
    ("pair_01281", "river", "road", "C_natural_context", "noun", "same syntactic role and plausible image"),
    ("pair_01351", "protect", "warn", "C_natural_context", "verb", "same transitive frame with plural object"),
    ("pair_01493", "movie", "dinner", "C_natural_context", "noun", "natural object of paid for"),
    ("pair_01638", "school", "hospital", "C_natural_context", "noun", "natural building substitution"),
    ("pair_01781", "big", "ambitious", "C_natural_context", "adjective", "natural collocation: ambitious ideas"),
    ("pair_01855", "improve", "decline", "B_semantic_contrast", "verb", "opposing performance trajectory"),
    ("pair_01919", "relations", "cooperation", "C_natural_context", "noun", "natural government-context substitution"),
    ("pair_01991", "companies", "properties", "C_natural_context", "noun", "natural subject for not for sale"),
    ("pair_02134", "angry", "calm", "B_semantic_contrast", "adjective", "emotion contrast"),
    ("pair_02280", "new", "different", "C_natural_context", "adjective", "natural plan modifier"),
    ("pair_02426", "great", "challenging", "B_semantic_contrast", "adjective", "changes assessment while remaining natural"),
    ("pair_02493", "trophy", "match", "C_natural_context", "noun", "natural object of win"),
    ("pair_02564", "friend", "rival", "B_semantic_contrast", "noun", "social-role contrast"),
    ("pair_02713", "today", "tomorrow", "B_semantic_contrast", "adverb", "temporal contrast"),
    ("pair_02860", "tough", "crucial", "C_natural_context", "adjective", "natural sports-context modifier"),
    ("pair_02925", "working", "living", "C_natural_context", "verb", "natural progressive construction"),
    ("pair_03073", "second", "first", "C_natural_context", "ordinal", "natural ranking substitution"),
    ("pair_03215", "better", "worse", "B_semantic_contrast", "adjective", "direct outcome contrast"),
    ("pair_03365", "frustration", "excitement", "B_semantic_contrast", "noun", "affective-state contrast"),
    ("pair_03427", "straightforward", "complicated", "B_semantic_contrast", "adjective", "complexity contrast"),
    ("pair_03496", "work", "training", "C_natural_context", "noun", "natural destination/state after get people into"),
    ("pair_03647", "destiny", "fiction", "C_natural_context", "noun", "natural rhetorical contrast with history"),
    ("pair_03795", "creative", "learning", "C_natural_context", "modifier", "natural process modifier"),
    ("pair_03933", "fine", "difficult", "B_semantic_contrast", "adjective", "outcome polarity change"),
    ("pair_04009", "likely", "possible", "C_natural_context", "adjective", "natural predicative substitution"),
    ("pair_04156", "upset", "afraid", "C_natural_context", "adjective", "natural adjective in the too-to construction"),
    ("pair_04301", "conditions", "details", "C_natural_context", "noun", "natural subject for were not specified"),
    ("pair_04364", "equal", "different", "B_semantic_contrast", "adjective", "identity/contrast relation"),
    ("pair_04510", "fantastic", "demanding", "B_semantic_contrast", "adjective", "changes evaluation but remains idiomatic"),
    ("pair_04577", "difference", "contribution", "C_natural_context", "noun", "natural make-a-contribution collocation"),
    ("pair_04721", "believe", "discuss", "C_natural_context", "verb", "same infinitival frame and natural object"),
    ("pair_04874", "learned", "benefited", "C_natural_context", "verb", "same from-complement frame"),
    ("pair_05018", "happen", "matter", "C_natural_context", "verb", "natural intransitive substitution"),
    ("pair_05160", "tragic", "encouraging", "B_semantic_contrast", "adjective", "clear evaluative polarity change"),
    ("pair_05311", "decided", "confirmed", "C_natural_context", "verb", "same passive construction"),
    ("pair_00011", "home", "away", "B_semantic_contrast", "modifier", "standard sports contrast"),
    ("pair_00292", "denied", "confirmed", "B_semantic_contrast", "verb", "claim-status contrast"),
]


def replace_once(text: str, old: str, new: str) -> tuple[str, str, str]:
    pattern = re.compile(rf"(?<!\w){re.escape(old)}(?!\w)", flags=re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(f"Expected one occurrence of {old!r} in {text!r}, found {len(matches)}")
    matched = matches[0].group(0)
    rendered_new = new
    if matched[:1].isupper() and new[:1].islower():
        rendered_new = new[:1].upper() + new[1:]
    edited = pattern.sub(rendered_new, text, count=1)
    return edited, matched, rendered_new


def main() -> None:
    with SOURCE_MANIFEST.open(encoding="utf-8", newline="") as f:
        source_rows = {r["pair_id"]: r for r in csv.DictReader(f, delimiter="\t")}

    rows = []
    seen_pairs = set()
    for index, (pair_id, old, new, route, pos, rationale) in enumerate(EDITS, start=1):
        if pair_id in seen_pairs:
            raise ValueError(f"Duplicate pair: {pair_id}")
        seen_pairs.add(pair_id)
        source = source_rows[pair_id]
        edited, matched_old, rendered_new = replace_once(source["text"], old, new)
        prompt = f"Replace '{matched_old}' with '{rendered_new}'."
        rows.append(
            {
                "candidate_id": f"ce_{index:04d}",
                "pair_id": pair_id,
                "speaker": source["speaker"],
                "gender": source["gender"],
                "anchor_id": source["anchor_id"],
                "anchor_vctk_id": source["anchor_vctk_id"],
                "original_text": source["text"],
                "edited_text": edited,
                "original_span": matched_old,
                "replacement_span": rendered_new,
                "operation": "replace",
                "route": route,
                "part_of_speech": pos,
                "auk_instruction": prompt,
                "rationale": rationale,
                "source_pair_dir": source["pair_dir"],
                "anchor_duration_seconds": source["anchor_duration_seconds"],
                "review_status": "pending",
                "review_notes": "",
            }
        )

    if len(rows) != 50:
        raise ValueError(f"Expected 50 candidates, got {len(rows)}")
    if len({r["speaker"] for r in rows}) != 48:
        raise ValueError("Pilot should cover all 48 speakers")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    tsv_path = OUTPUT_DIR / "content_edit_candidates_50.tsv"
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    with (OUTPUT_DIR / "content_edit_candidates_50.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    route_counts = Counter(r["route"] for r in rows)
    pos_counts = Counter(r["part_of_speech"] for r in rows)
    report = {
        "created_date": "2026-10-01",
        "source_manifest": str(SOURCE_MANIFEST.relative_to(ROOT)),
        "num_candidates": len(rows),
        "num_speakers": len({r["speaker"] for r in rows}),
        "operation": "single-span replacement only",
        "route_counts": dict(route_counts),
        "part_of_speech_counts": dict(pos_counts),
        "validation": {
            "original_span_occurs_exactly_once": True,
            "one_replacement_per_candidate": True,
            "all_pair_ids_unique": True,
        },
    }
    with (OUTPUT_DIR / "pilot_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    table_rows = []
    for r in rows:
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(r['candidate_id'])}</td>"
            f"<td>{html.escape(r['pair_id'])}</td>"
            f"<td>{html.escape(r['speaker'])}</td>"
            f"<td>{html.escape(r['route'])}</td>"
            f"<td>{html.escape(r['original_text'])}</td>"
            f"<td>{html.escape(r['edited_text'])}</td>"
            f"<td><code>{html.escape(r['auk_instruction'])}</code></td>"
            "</tr>"
        )
    preview = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>AuK Content Editing Pilot 50</title>
<style>body{{font-family:system-ui,sans-serif;margin:24px;line-height:1.45}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:8px;vertical-align:top}}th{{position:sticky;top:0;background:#f4f4f4}}tr:nth-child(even){{background:#fafafa}}code{{white-space:nowrap}}</style>
</head><body><h1>AuK Content Editing：50 条自然单词替换候选</h1>
<p>覆盖 48 位 speaker；每条仅替换一个出现一次的实词。此包是文本审阅版，尚未执行 AuK 推理。</p>
<table><thead><tr><th>ID</th><th>Pair</th><th>Speaker</th><th>路线</th><th>原句</th><th>改后句</th><th>AuK instruction</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></body></html>"""
    (OUTPUT_DIR / "preview.html").write_text(preview, encoding="utf-8")

    readme = """# AuK Content Editing Pilot：50 条候选（文本审阅版）

本包用于在正式音频生成前审阅 content-editing 文本。50 条候选覆盖当前 matched set 的全部 48 位 speaker，其中两位 speaker 各有两条。

## 当前约束

- 每条只做一次 `replace`，不混入插入或删除。
- 被替换 span 在原句中恰好出现一次，避免定位歧义。
- 只替换一个实词或固定语法槽位中的一个词。
- 不要求修改冠词、单复数或周围语法。
- 优先保证改后句自然；包含自然上下文替换与明确语义对立两类。
- AuK instruction 使用官方普通语音 content-edit 模板：`Replace '{original}' with '{new}'.`

## 文件

- `content_edit_candidates_50.tsv`：便于筛选、批注；可填写 `review_status` 与 `review_notes`。
- `content_edit_candidates_50.json`：后续生成脚本直接读取。
- `preview.html`：浏览器直接查看。
- `pilot_report.json`：数量、覆盖与机械校验摘要。

## 人工审阅建议

重点拒绝以下样本：改后语义明显别扭、替换词在口语中极不自然、容易造成冠词/时态/数的一致性问题、专名或上下文依赖过强。通过文本审阅后再生成音频，并对输出分别核验目标编辑成功、未编辑内容保持、说话人保持、音质与边界连续性。
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
