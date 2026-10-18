"""🧪 A/B 测试
==============
扩展点 5 的 Web 展示：在线对照实验的离线复现。

把两个已训练模型当作 A（对照）/ B（实验）两组，用 src/ab_test/router.py 的
确定性哈希分桶把每一笔申请稳定分到其中一组，各自预测违约概率；再以真实违约标签
作为结局，用两比例 z 检验判断两组违约率差异是否显著。

页面分区：
    1. 实验配置：分流比例 / 样本量 / A·B 模型 / 显著性水平
    2. 离线实验结果：指标卡片 + 违约率对比图 + 曝光明细表（可下载）
    3. 分流演示：验证"同一申请编号恒定落到同一组"

运行: streamlit run app/streamlit_app.py  →  左侧导航选择「A/B 测试」
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.components import ab_test_utils as abu
from app.components.data_utils import find_data_file
from app.components.sidebar import render_sidebar
from src.utils.config_loader import ConfigLoader

# ===== 页面配置 =====
st.set_page_config(page_title="A/B 测试", page_icon="🧪", layout="wide")
render_sidebar()

# ===== 角色权限门禁（管理员专属）=====
from app.components.auth_gate import require_admin
require_admin()

CONFIGS = ConfigLoader(str(ROOT / "configs"))
AB_CFG = abu.load_ab_config(ROOT)
MODEL_DIR = ROOT / "data" / "models" / "latest"
MODEL_OPTIONS = {"XGBoost": "xgboost", "LightGBM": "lightgbm"}


# ===== 模型与数据加载 =====
def _load_one_model(model_key: str):
    if model_key == "lightgbm":
        from src.models.lgb_model import LGBModel

        path = MODEL_DIR / "lgb_model.txt"
        if not path.exists():
            return None
        model = LGBModel(CONFIGS.get("model").lightgbm)
        model.load(str(path))
        return model

    from src.models.xgb_model import XGBModel

    path = MODEL_DIR / "xgb_model.json"
    if not path.exists():
        return None
    model = XGBModel(CONFIGS.get("model").xgboost)
    model.load(str(path))
    return model


@st.cache_resource(show_spinner="加载模型与数据…")
def load_resources(model_key_a: str, model_key_b: str):
    """加载两个模型 + 预处理后的特征矩阵与带标签的样本（供离线实验使用）。"""
    model_a, model_b = _load_one_model(model_key_a), _load_one_model(model_key_b)
    prep_path = MODEL_DIR / "preprocessor.pkl"
    if model_a is None or model_b is None or not prep_path.exists():
        return None

    data_path, source = find_data_file(ROOT, prefer=AB_CFG["data_source"])
    if data_path is None:
        return None

    df = pd.read_csv(data_path, nrows=20000)
    target, id_col = CONFIGS.get("data").target_column, CONFIGS.get("data").id_column
    if target not in df.columns:
        return None

    X = prep_process(df.drop(columns=[target, id_col], errors="ignore"))
    selector_path = MODEL_DIR / "feature_selector.pkl"
    if selector_path.exists():
        X = joblib.load(selector_path).transform(X)

    return {
        "model_a": model_a,
        "model_b": model_b,
        "X": X.reset_index(drop=True),
        "keys": df[id_col].to_numpy(),
        "outcomes": df[target].to_numpy(),
        "source": source,
    }


def prep_process(X: pd.DataFrame) -> pd.DataFrame:
    """走预处理器 + 特征工程，与「单笔预测 / 模型解释」页面保持一致。"""
    from src.features.builder import FeatureBuilder

    X = joblib.load(MODEL_DIR / "preprocessor.pkl").transform(X)
    return FeatureBuilder(CONFIGS.get("feature")).build(X)


# ===== 页面标题 =====
st.title("🧪 A/B 测试框架")
st.caption(
    "把两个模型作为 A/B 两组做对照实验：确定性哈希分桶分流 → 分组预测 → 曝光日志 → "
    "两比例 z 检验。逻辑见 `src/ab_test/router.py`，页面纯逻辑见 `app/components/ab_test_utils.py`。"
)

# ===== 实验配置区 =====
with st.container(border=True):
    st.markdown("#### ⚙️ 实验配置")

    model_labels = list(MODEL_OPTIONS.keys())
    default_a = next((k for k, v in MODEL_OPTIONS.items() if v == AB_CFG["group_a_model"]), model_labels[0])
    default_b = next((k for k, v in MODEL_OPTIONS.items() if v == AB_CFG["group_b_model"]), model_labels[-1])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        label_a = st.selectbox("A 组（对照）模型", model_labels, index=model_labels.index(default_a))
    with c2:
        label_b = st.selectbox("B 组（实验）模型", model_labels, index=model_labels.index(default_b))
    with c3:
        ratio = st.slider("分流到 A 组的比例", 0.1, 0.9, float(AB_CFG["ratio"]), 0.05)
    with c4:
        alpha = st.select_slider("显著性水平 α", options=[0.01, 0.05, 0.10], value=float(AB_CFG["alpha"]))

    sample_size = st.slider("离线样本量", 500, 10000, int(AB_CFG["sample_size"]), 500,
                            help="从可用带标签数据中随机抽样参与本次离线对比")
    run = st.button("▶️ 运行实验", type="primary", use_container_width=True)

res = load_resources(MODEL_OPTIONS[label_a], MODEL_OPTIONS[label_b])
if res is None:
    st.warning("⚠️ 模型或数据不可用：请先在 **🤖 模型训练** 页面完成训练（需 XGBoost 与 LightGBM）")
    st.stop()

st.caption(f"数据来源：{res['source'] or '未知'}　·　可用带标签样本：{len(res['X']):,} 行")

if run:
    n = min(sample_size, len(res["X"]))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(res["X"]), size=n, replace=False)

    records, result = abu.run_experiment(
        res["model_a"], res["model_b"],
        res["X"].iloc[idx].reset_index(drop=True),
        res["keys"][idx], res["outcomes"][idx],
        ratio=ratio, bucket_count=int(AB_CFG["bucket_count"]),
    )
    st.session_state["ab_result"] = result
    st.session_state["ab_records"] = records

result = st.session_state.get("ab_result")
records = st.session_state.get("ab_records")

# ===== 实验结果 =====
if result is None:
    st.info("点击上方「▶️ 运行实验」开始一次离线 A/B 对比。", icon="ℹ️")
    st.stop()

st.subheader("📊 实验结果")
m = st.columns(5)
m[0].metric("A 组违约率", f"{result['A']['bad_rate']:.2%}", help=f"A 组样本 {result['A']['n']}")
m[1].metric("B 组违约率", f"{result['B']['bad_rate']:.2%}", help=f"B 组样本 {result['B']['n']}")
m[2].metric("违约率差 (B-A)", f"{result['rate_diff']:+.2%}")
m[3].metric("p 值", "—" if result["p_value"] is None else f"{result['p_value']:.4f}")
m[4].metric("结论", "显著" if result["significant"] else "不显著",
            delta=f"{result['better_group']} 组更优" if result.get("better_group") else "无显著优劣",
            delta_color="normal" if result.get("better_group") else "off")

if result["significant"]:
    st.success(f"在 α={alpha} 下两组违约率差异**显著**，"
               f"**{result['better_group']} 组更优**（违约率更低）。")
else:
    st.warning(f"在 α={alpha} 下两组违约率差异**不显著**，尚不能判定优劣。")

st.plotly_chart(abu.bad_rate_figure(result, alpha=alpha), width="stretch", alt="A/B 违约率对比柱状图")

# ===== 曝光明细 =====
with st.expander("📄 曝光明细（离线实验记录）", expanded=False):
    rec_df = pd.DataFrame(records)
    st.caption(f"共 {len(rec_df)} 条记录；字段与 `ExperimentLogger` 落地的 JSONL 一致。")
    st.dataframe(rec_df.head(200), width="stretch", hide_index=True)
    st.download_button(
        "⬇️ 下载全量记录 CSV",
        rec_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="ab_test_records.csv",
        mime="text/csv",
    )

# ===== 分流演示 =====
st.subheader("🎯 分流演示")
st.caption("验证确定性哈希分桶：同一个申请编号永远落到同一组，与运行次数、进程无关。")
demo_key = st.text_input("申请编号 / 任意字符串", value=str(res["keys"][0]))
from src.ab_test.router import ABTestRouter

_demo_router = ABTestRouter(None, None, ratio=ratio, bucket_count=int(AB_CFG["bucket_count"]))
st.markdown(f"编号 `{demo_key}` → 分到 **{_demo_router.assign(demo_key)}** 组")

st.divider()
st.caption("🧪 A/B 测试 · 分流与统计逻辑见 `src/ab_test/router.py`；实验参数见 `configs/ab_test_config.yaml`")
