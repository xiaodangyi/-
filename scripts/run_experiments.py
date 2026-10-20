"""依次运行预先登记的对照实验，保存可见终端的完整 UTF-8 日志。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="运行固定计划中的对照实验")
    parser.add_argument("--plan", default="configs/ablation-20261020.json")
    args = parser.parse_args()
    os.chdir(ROOT)
    plan_path = Path(args.plan).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    root = (ROOT / plan["output_root"]).resolve()
    if ROOT not in root.parents:
        raise ValueError("实验输出必须位于项目目录内")
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    frozen_plan = root / (plan_path.stem + "-frozen.json")
    if frozen_plan.exists() and hashlib.sha256(frozen_plan.read_bytes()).hexdigest() != digest:
        raise RuntimeError("同名冻结计划已存在但内容不同，禁止覆盖")
    if not frozen_plan.exists():
        frozen_plan.write_bytes(plan_path.read_bytes())
    completed = []
    for experiment in plan["experiments"]:
        name = experiment["name"]
        if not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in name):
            raise ValueError("实验名称不合法")
        output_dir = root / name
        parameters = {**plan["common"], **experiment.get("changes", {}), "output-dir": str(output_dir)}
        summary_path = output_dir / "summary.json"
        if summary_path.is_file():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            saved = summary["training_args"]
            if any(saved.get(key.replace("-", "_")) != value for key, value in parameters.items()):
                raise RuntimeError(f"已有实验参数与计划不同：{name}")
            print(f"已完成，核对参数后跳过：{name}", flush=True)
        else:
            command = [sys.executable, "-s", "-u", str(ROOT / "scripts" / "train.py")]
            for key, value in parameters.items():
                command += ["--" + key, str(value)]
            print("\n" + "=" * 60, flush=True)
            print(f"开始实验：{name}\n参数：{json.dumps(parameters, ensure_ascii=False)}", flush=True)
            log_path = root / f"{name}-terminal.log"
            with log_path.open("a", encoding="utf-8") as log:
                process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONNOUSERSITE": "1"}, bufsize=1)
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    returncode = process.wait()
                except KeyboardInterrupt:
                    # 原始结果保留，明确停止当前批次，避免误报完成。
                    process.wait()
                    raise
            if returncode:
                raise RuntimeError(f"实验 {name} 退出，代码 {returncode}；查看 {log_path}")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        completed.append({"name": name, "best_epoch": summary["best_epoch"], "val_accuracy": summary["best_val_accuracy"], "val_macro_f1": summary["best_val_macro_f1"], "checkpoint": str(output_dir / "best.pt"), **experiment.get("changes", {})})
        table_path = root / (plan_path.stem + "-results.csv")
        pd.DataFrame(completed).to_csv(table_path, index=False, encoding="utf-8-sig")
        print(f"已保存验证结果：{table_path}", flush=True)
    print("本批实验完成。尚未根据测试集挑选模型，请先比较验证指标。", flush=True)


if __name__ == "__main__":
    main()
