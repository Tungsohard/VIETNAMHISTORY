"""Analyse the survey: how often each model is picked, with a participant-level bootstrap interval (one
person's answers are correlated), paired differences and a sign-flip test. Chance is 1/3.
Reads participant JSON files or the CSV exported from the Google Form.

Run: python scripts/evaluate/user_study/analyze_study.py build/user_study/key.json answers.csv
"""
import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

QUESTIONS = {"natural": "Màu tự nhiên và thật nhất"}
MODELS = ("historical", "vivid", "ddcolor")
DISPLAY = {"historical": "VietColorNet · Lịch sử", "vivid": "VietColorNet · Tươi", "ddcolor": "DDColor gốc"}
GROUPS = ("archive", "film", "modern")


def read_responses(path):
    """Yield (label, response dict) from a participant JSON file or a Google Forms CSV export."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        yield path.name, json.loads(path.read_text(encoding="utf-8"))
        return
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    for number, row in enumerate(rows[1:], start=2):
        cell = next((c for c in reversed(row) if c.lstrip().startswith("{")), None)
        if cell is None:
            print(f"skipping {path.name} row {number}: no JSON answer", file=sys.stderr)
            continue
        try:
            yield f"{path.name}:{number}", json.loads(cell)
        except json.JSONDecodeError:
            print(f"skipping {path.name} row {number}: unreadable JSON", file=sys.stderr)


def load(key_path, response_paths):
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    participants, seen = [], set()
    for path in response_paths:
        for label, data in read_responses(path):
            answers = data.get("answers", {})
            if set(answers) != set(key) or not all(a.get("natural") for a in answers.values()):
                print(f"skipping {label}: incomplete answers", file=sys.stderr)
                continue
            identity = data.get("participant", label)
            if identity in seen:
                print(f"skipping {label}: participant {identity} already counted", file=sys.stderr)
                continue
            seen.add(identity)
            participants.append(answers)
    return key, participants


def votes(key, participants, question, group=None):
    """Matrix [participant, model] with the share of that participant's answers choosing the model."""
    codes = [c for c in key if group is None or key[c]["group"] == group]
    matrix = np.zeros((len(participants), len(MODELS)))
    for i, answers in enumerate(participants):
        for code in codes:
            matrix[i, MODELS.index(key[code][answers[code][question]])] += 1
        matrix[i] /= len(codes)
    return matrix


def bootstrap_interval(values, resamples, rng):
    n = len(values)
    means = np.array([values[rng.integers(0, n, n)].mean() for _ in range(resamples)])
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def sign_flip_p(diff, resamples, rng):
    """Two-sided p-value for mean(diff) == 0 by randomly flipping each participant's sign."""
    observed = abs(diff.mean())
    flips = rng.choice([-1.0, 1.0], size=(resamples, len(diff)))
    return float(((np.abs((flips * diff).mean(axis=1)) >= observed - 1e-12).sum() + 1) / (resamples + 1))


def summarise(matrix, resamples, rng):
    result = {"share": {}, "pairs": []}
    for j, model in enumerate(MODELS):
        result["share"][model] = {"mean": float(matrix[:, j].mean()),
                                  "ci95": bootstrap_interval(matrix[:, j], resamples, rng)}
    for a, b in combinations(range(len(MODELS)), 2):
        diff = matrix[:, a] - matrix[:, b]
        result["pairs"].append({"a": MODELS[a], "b": MODELS[b], "mean_diff": float(diff.mean()),
                                "ci95": bootstrap_interval(diff, resamples, rng),
                                "p": sign_flip_p(diff, resamples, rng)})
    return result


def render(report):
    lines = [f"Số người tham gia: {report['participants']}; số ảnh: {report['images']}; mức ngẫu nhiên = 33,3%.", ""]
    for question, title in QUESTIONS.items():
        for group, entry in report["results"][question].items():
            lines += [f"### {title} ({'tất cả ảnh' if group == 'all' else group})", "",
                      "| Mô hình | Tỉ lệ được chọn | KTC 95% |", "|---|---|---|"]
            for model in MODELS:
                s = entry["share"][model]
                lines.append(f"| {DISPLAY[model]} | {s['mean'] * 100:.1f}% | [{s['ci95'][0] * 100:.1f}%, {s['ci95'][1] * 100:.1f}%] |")
            lines += ["", "| Cặp | Chênh lệch | KTC 95% | p (đảo dấu) |", "|---|---|---|---|"]
            for pair in entry["pairs"]:
                lines.append(f"| {DISPLAY[pair['a']]} − {DISPLAY[pair['b']]} | {pair['mean_diff'] * 100:+.1f} điểm % | "
                             f"[{pair['ci95'][0] * 100:+.1f}, {pair['ci95'][1] * 100:+.1f}] | {pair['p']:.3g} |")
            lines.append("")
    return "\n".join(lines)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("key", type=Path)
    parser.add_argument("responses", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/reports/user_study"))
    parser.add_argument("--resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    key, participants = load(args.key, args.responses)
    if len(participants) < 2:
        raise SystemExit("Need at least two complete responses")
    rng = np.random.default_rng(args.seed)
    results = {q: {} for q in QUESTIONS}
    for question in QUESTIONS:
        for group in (None, *GROUPS):
            if group and not any(k["group"] == group for k in key.values()):
                continue
            results[question][group or "all"] = summarise(votes(key, participants, question, group),
                                                          args.resamples, rng)
    report = {"participants": len(participants), "images": len(key), "resamples": args.resamples, "results": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = render(report)
    args.output.with_suffix(".md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
