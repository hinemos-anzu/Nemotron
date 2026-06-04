#!/usr/bin/env python3
"""Phase 3 全自動実行スクリプト

パス自動検出 → カテゴリ分類 → 推論 → logprob → 集計 → 分類 → レポート
を一括実行します。

使い方:
    python phase3_run_all.py                        # 全ステップ実行
    python phase3_run_all.py --skip-inference       # 既存predictions.jsonlを使う
    python phase3_run_all.py --skip-logprob         # logprob取得をスキップ
    python phase3_run_all.py --steps 1,4,5,6,7     # 特定ステップのみ実行
    python phase3_run_all.py --dry-run              # パス確認のみ（推論なし）

SAFETY: adapter_model.safetensors / adapter_config.json は一切変更しません。
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SEED = 42
SCRIPTS_DIR = Path(os.environ.get("PHASE3_SCRIPTS_DIR", Path(__file__).parent))


# ─────────────────────────────────────────────────────────────────────────────
# パス自動検出
# ─────────────────────────────────────────────────────────────────────────────

def _find_first(candidates: List[str]) -> Optional[Path]:
    for c in candidates:
        if not c:
            continue
        p = Path(c)
        if p.exists():
            return p
    return None


def detect_paths() -> Dict[str, Optional[Path]]:
    """Kaggle / ローカル双方で動くパス自動検出"""
    problems = _find_first([
        "/kaggle/input/nvidia-nemotron-model-reasoning-challenge/problems.jsonl",
        "/kaggle/input/nemotron-reasoning-challenge/problems.jsonl",
        "/kaggle/input/nvidia-nemotron/problems.jsonl",
        "data/raw/problems.jsonl",
        "problems.jsonl",
    ])
    train_csv = _find_first([
        "/kaggle/input/nvidia-nemotron-model-reasoning-challenge/train.csv",
        "/kaggle/input/nemotron-reasoning-challenge/train.csv",
        "data/raw/train.csv",
        "train.csv",
    ])
    adapter = _find_first([
        *[f"/kaggle/input/models/huikang/nemotron-adapter/transformers/default/{v}"
          for v in range(30, 0, -1)],
        "/kaggle/input/models/huikang/nemotron-adapter/transformers/default",
        os.environ.get("ADAPTER_PATH", ""),
    ])
    model = _find_first([
        "/kaggle/input/metric/nemotron-3-nano-30b-a3b-bf16/transformers/default",
        "/kaggle/input/nemotron-3-nano/transformers/default",
        os.environ.get("MODEL_PATH", ""),
    ])
    return {"problems": problems, "train_csv": train_csv, "adapter": adapter, "model": model}


# ─────────────────────────────────────────────────────────────────────────────
# ロガー
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, log_path: Path) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(log_path, "a", encoding="utf-8")
        self._steps: List[Dict[str, Any]] = []

    def log(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        self._f.write(line + "\n")
        self._f.flush()

    def step_start(self, step: int, name: str) -> float:
        self.log(f"{'='*60}")
        self.log(f"Step {step}: {name} — 開始")
        return time.time()

    def step_done(self, step: int, name: str, t0: float, ok: bool) -> None:
        elapsed = time.time() - t0
        status = "✓ 完了" if ok else "✗ 失敗"
        self.log(f"Step {step}: {name} — {status}  ({elapsed:.0f}s)")
        self._steps.append({"step": step, "name": name, "ok": ok, "elapsed": round(elapsed)})

    def summary(self) -> None:
        self.log("\n" + "="*60)
        self.log("実行サマリー")
        for s in self._steps:
            icon = "✓" if s["ok"] else "✗"
            self.log(f"  {icon} Step {s['step']}: {s['name']:40s} {s['elapsed']}s")
        ok_count = sum(1 for s in self._steps if s["ok"])
        self.log(f"\n{ok_count}/{len(self._steps)} ステップ成功")

    def close(self) -> None:
        self._f.close()


# ─────────────────────────────────────────────────────────────────────────────
# サブプロセス実行ヘルパー
# ─────────────────────────────────────────────────────────────────────────────

def run_step(cmd: List[str], logger: Logger, timeout: int = 7200) -> bool:
    logger.log(f"  CMD: {' '.join(str(c) for c in cmd)}")
    try:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in proc.stdout:
            logger.log(f"  | {line.rstrip()}")
        proc.wait(timeout=timeout)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        proc.kill()
        logger.log("  [TIMEOUT] プロセスがタイムアウトしました")
        return False
    except Exception as exc:
        logger.log(f"  [ERROR] {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 各ステップ
# ─────────────────────────────────────────────────────────────────────────────

def step1_category_map(paths: Dict[str, Optional[Path]], out: Path, logger: Logger) -> bool:
    inputs = [str(p) for p in (paths["problems"], paths["train_csv"]) if p]
    if not inputs:
        logger.log("  [SKIP] 問題ファイルが見つかりません")
        return False
    return run_step([
        sys.executable, SCRIPTS_DIR / "phase3_build_category_map.py",
        "--input", *inputs,
        "--output",         out / "category_map.csv",
        "--labeled-output", out / "validation_set_labeled.csv",
    ], logger)


def step2_inference(paths: Dict[str, Optional[Path]], out: Path, logger: Logger,
                    dry_run: bool = False) -> bool:
    if not paths["adapter"]:
        logger.log("  [SKIP] adapterが見つかりません")
        return False
    if not paths["model"]:
        logger.log("  [SKIP] base modelが見つかりません")
        return False
    problems_path = paths["problems"] or paths["train_csv"]
    if not problems_path:
        logger.log("  [SKIP] 問題ファイルが見つかりません")
        return False
    cmd = [
        sys.executable, SCRIPTS_DIR / "phase3_run_golden_validation.py",
        "--adapter",      paths["adapter"],
        "--model",        paths["model"],
        "--problems",     problems_path,
        "--category-map", out / "category_map.csv",
        "--output-dir",   out,
        "--seed",         str(SEED),
    ]
    if dry_run:
        cmd.append("--dry-run")
    return run_step(cmd, logger, timeout=10800)


def step3_logprob(paths: Dict[str, Optional[Path]], out: Path, logger: Logger) -> bool:
    pred_path = out / "golden_validation_predictions.jsonl"
    if not pred_path.exists():
        logger.log("  [SKIP] golden_validation_predictions.jsonl がありません (Step 2を先に実行)")
        return False

    # まずinlineモードを試す
    logger.log("  inlineモードを試行...")
    ok = run_step([
        sys.executable, SCRIPTS_DIR / "phase3_extract_logprob.py",
        "--predictions", pred_path,
        "--output",      out / "min_logprob_summary.csv",
        "--mode",        "inline",
    ], logger)

    if ok:
        lp_path = out / "min_logprob_summary.csv"
        if lp_path.exists():
            with open(lp_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if any(r.get("min_logprob") for r in rows):
                logger.log("  inline logprobデータ取得成功")
                return True

    logger.log("  inline logprobなし → rerunモードにフォールバック")
    if not paths["adapter"] or not paths["model"]:
        logger.log("  [SKIP] rerunにはadapter/modelが必要")
        return False
    return run_step([
        sys.executable, SCRIPTS_DIR / "phase3_extract_logprob.py",
        "--predictions", pred_path,
        "--adapter",     paths["adapter"],
        "--model",       paths["model"],
        "--output",      out / "min_logprob_summary.csv",
        "--mode",        "rerun",
        "--seed",        str(SEED),
    ], logger, timeout=10800)


def step4_aggregate(out: Path, logger: Logger) -> bool:
    return run_step([
        sys.executable, SCRIPTS_DIR / "phase3_analyze_category_failures.py",
        "--predictions", out / "golden_validation_predictions.jsonl",
        "--logprob",     out / "min_logprob_summary.csv",
        "--output",      out / "category_failure_summary.csv",
    ], logger)


def step5_cryptarithm(out: Path, logger: Logger) -> bool:
    # failure_type_summaryをリセット
    ft_path = out / "failure_type_summary.csv"
    if ft_path.exists():
        ft_path.unlink()
    ft_path.write_text("category,failure_type,count,pct\n", encoding="utf-8")

    return run_step([
        sys.executable, SCRIPTS_DIR / "phase3_classify_cryptarithm_failures.py",
        "--predictions",          out / "golden_validation_predictions.jsonl",
        "--logprob",              out / "min_logprob_summary.csv",
        "--output",               out / "failure_cases_cryptarithm.csv",
        "--failure-type-summary", ft_path,
        "--include-correct-low-logprob",
    ], logger)


def step6_bit_numeral(out: Path, logger: Logger) -> bool:
    return run_step([
        sys.executable, SCRIPTS_DIR / "phase3_classify_bit_numeral_failures.py",
        "--predictions",          out / "golden_validation_predictions.jsonl",
        "--logprob",              out / "min_logprob_summary.csv",
        "--output-bit",           out / "failure_cases_bit_manipulation.csv",
        "--output-numeral",       out / "failure_cases_numeral_conversion.csv",
        "--failure-type-summary", out / "failure_type_summary.csv",
    ], logger)


def step7_recommendation(out: Path, logger: Logger) -> bool:
    return run_step([
        sys.executable, SCRIPTS_DIR / "phase3_make_recommendation.py",
        "--category-failure", out / "category_failure_summary.csv",
        "--failure-type",     out / "failure_type_summary.csv",
        "--summary",          out / "golden_validation_summary.csv",
        "--output",           out / "phase3_recommendation.md",
    ], logger)


# ─────────────────────────────────────────────────────────────────────────────
# 最終結果表示
# ─────────────────────────────────────────────────────────────────────────────

def print_final_results(out: Path, logger: Logger) -> None:
    logger.log("\n" + "="*60)
    logger.log("Phase 3 分析結果サマリー")
    logger.log("="*60)

    summary_path = out / "golden_validation_summary.csv"
    if summary_path.exists():
        with open(summary_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        overall = next((r for r in rows if r.get("category") == "ALL"), None)
        if overall:
            logger.log(f"\n[overall]  n={overall.get('n')}  "
                       f"accuracy={overall.get('accuracy')}  "
                       f"parse_rate={overall.get('parse_success_rate')}")

    fail_path = out / "category_failure_summary.csv"
    if fail_path.exists():
        with open(fail_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        def _safe_float(v: Any, default: float = 1.0) -> float:
            try:
                return float(str(v).replace("ESTIMATED_", "").replace("PLACEHOLDER", str(default)))
            except Exception:
                return default

        rows.sort(key=lambda r: _safe_float(r.get("accuracy", 1)))
        logger.log("\n[弱いカテゴリ Top5]")
        for r in rows[:5]:
            logger.log(f"  {r.get('category','')}/{r.get('subcategory','')}: "
                       f"acc={r.get('accuracy','')}  n={r.get('n','')}  "
                       f"priority={r.get('priority_score','')}")

    ft_path = out / "failure_type_summary.csv"
    if ft_path.exists():
        with open(ft_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        logger.log("\n[cryptarithm 主な失敗タイプ Top5]")
        crypto = [r for r in rows if r.get("category") == "cryptarithm"]
        crypto.sort(key=lambda r: -_safe_float(r.get("count", "0")))
        for r in crypto[:5]:
            logger.log(f"  {r.get('failure_type',''):30s} count={r.get('count','')}  {r.get('pct','')}%")

    logger.log("\n[生成ファイル]")
    for f in sorted(out.iterdir()):
        size = f.stat().st_size
        logger.log(f"  {f.name:50s} {size/1024:.1f} KB")

    logger.log("\n[確認] adapter変更・training・submission作成は実行していません")


def _safe_float(v: Any, default: float = 1.0) -> float:
    try:
        return float(str(v).replace("ESTIMATED_", "").replace("PLACEHOLDER", str(default)))
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--steps", default="1,2,3,4,5,6,7",
                        help="実行ステップ番号 カンマ区切り (デフォルト: 全ステップ)")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Step 2をスキップ (既存predictions.jsonlを使用)")
    parser.add_argument("--skip-logprob", action="store_true",
                        help="Step 3をスキップ (logprobなしで集計)")
    parser.add_argument("--dry-run", action="store_true",
                        help="パス確認のみ。推論は実行しない")
    parser.add_argument("--output-dir",
                        default=os.environ.get("PHASE3_OUTPUT_DIR",
                                               "/kaggle/working/phase3_analysis"),
                        help="出力ディレクトリ")
    parser.add_argument("--problems", default="", help="問題ファイルパスを手動指定")
    parser.add_argument("--adapter",  default="", help="adapterパスを手動指定")
    parser.add_argument("--model",    default="", help="base modelパスを手動指定")
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    logger = Logger(out / "run_all.log")
    t_start = time.time()

    logger.log("="*60)
    logger.log(f"Phase 3 自動分析開始  {datetime.now(timezone.utc).isoformat()}")
    logger.log(f"Output dir: {out}")
    logger.log("="*60)

    paths = detect_paths()
    if args.problems: paths["problems"] = Path(args.problems)
    if args.adapter:  paths["adapter"]  = Path(args.adapter)
    if args.model:    paths["model"]    = Path(args.model)

    logger.log("\n[パス確認]")
    for k, v in paths.items():
        flag = "✓" if v else "✗"
        logger.log(f"  {flag} {k:12s}: {v or 'NOT FOUND'}")

    if args.dry_run:
        t0 = logger.step_start(2, "Dry-run: inference config確認")
        ok = step2_inference(paths, out, logger, dry_run=True)
        logger.step_done(2, "Dry-run", t0, ok)
        logger.log("\n[DRY-RUN完了] 実際の推論は実行していません。")
        logger.close()
        return

    steps_to_run = [int(x.strip()) for x in args.steps.split(",") if x.strip().isdigit()]
    if args.skip_inference and 2 in steps_to_run:
        steps_to_run.remove(2)
    if args.skip_logprob and 3 in steps_to_run:
        steps_to_run.remove(3)

    logger.log(f"\n実行ステップ: {steps_to_run}\n")
    results: Dict[int, bool] = {}

    if 1 in steps_to_run:
        t0 = logger.step_start(1, "カテゴリマップ作成")
        results[1] = step1_category_map(paths, out, logger)
        logger.step_done(1, "カテゴリマップ作成", t0, results[1])

    if 2 in steps_to_run:
        t0 = logger.step_start(2, "Golden validation 推論")
        results[2] = step2_inference(paths, out, logger)
        logger.step_done(2, "Golden validation 推論", t0, results[2])

    if 3 in steps_to_run:
        t0 = logger.step_start(3, "logprob抽出")
        results[3] = step3_logprob(paths, out, logger)
        logger.step_done(3, "logprob抽出", t0, results[3])

    pred_exists = (out / "golden_validation_predictions.jsonl").exists()
    if not pred_exists:
        logger.log("\n[WARN] golden_validation_predictions.jsonl がありません。"
                   "Step 4-7 をスキップします。")
    else:
        if 4 in steps_to_run:
            t0 = logger.step_start(4, "カテゴリ失敗集計")
            results[4] = step4_aggregate(out, logger)
            logger.step_done(4, "カテゴリ失敗集計", t0, results[4])

        if 5 in steps_to_run:
            t0 = logger.step_start(5, "cryptarithm失敗分類")
            results[5] = step5_cryptarithm(out, logger)
            logger.step_done(5, "cryptarithm失敗分類", t0, results[5])

        if 6 in steps_to_run:
            t0 = logger.step_start(6, "bit/numeral失敗分類")
            results[6] = step6_bit_numeral(out, logger)
            logger.step_done(6, "bit/numeral失敗分類", t0, results[6])

        if 7 in steps_to_run:
            t0 = logger.step_start(7, "最終レポート生成")
            results[7] = step7_recommendation(out, logger)
            logger.step_done(7, "最終レポート生成", t0, results[7])

    logger.summary()
    print_final_results(out, logger)
    logger.log(f"\n総実行時間: {(time.time() - t_start)/60:.1f}分")
    logger.close()


if __name__ == "__main__":
    main()
