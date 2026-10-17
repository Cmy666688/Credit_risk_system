# scripts/compare_runs.py
import json
from pathlib import Path
import pandas as pd
# # 获取当前脚本文件所在目录
# SCRIPT_DIR = Path(__file__).parent
# # 项目根目录：脚本目录向上一层
# ROOT_DIR = SCRIPT_DIR.parent
# # 模型目录
# MODEL_DIR = ROOT_DIR / "data" / "models"
records = []
for run_dir in Path("C:\Users\cmy\credit_risk_system2\data\models\latest").iterdir():
    if run_dir.name == "latest":
        continue
    meta_path = run_dir / "metadata.json"
    eval_path = run_dir / "eval_result.json"
    if meta_path.exists() and eval_path.exists():
        meta = json.loads(meta_path.read_text())
        result = json.loads(eval_path.read_text())
        records.append({
            "run_id": run_dir.name,
            "auc": result["auc"],
            "ks": result["ks"],
            "f1": result["f1"],
            "with_tuning": meta.get("with_tuning", False),
            "duration": result["training_time_seconds"],
        })

df = pd.DataFrame(records).sort_values("auc", ascending=False)
print(df.to_string())