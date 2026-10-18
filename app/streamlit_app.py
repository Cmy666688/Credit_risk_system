"""
智能信贷风控分析系统 - Streamlit 主入口
======================================
运行: streamlit run app/streamlit_app.py

采用 st.navigation 编程式导航:
    1. 未登录: 侧边栏仅显示"首页"，页面内容为登录/注册卡片
    2. 已登录: 按角色权限矩阵（configs/auth_config.yaml）动态生成导航
       - admin: 全部页面
       - user:  仅 单笔预测 / 模型解释 / LLM智能助手
页面文件位于 app/views/（已脱离 pages/ 自动发现机制，避免双入口）。
"""
import sys
from pathlib import Path

import streamlit as st

# 添加项目根到 sys.path（页面脚本 import src/app 需要）
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# ===== 页面配置 =====
st.set_page_config(
    page_title="智能信贷风控分析系统",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "## 智能信贷风控分析系统\n机器学习课程综合实战项目"
    }
)

# ===== 加载样式 =====
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ===== 全局初始化（带缓存，配置文件改动后自动失效） =====
def _config_stamp() -> str:
    """把 configs/*.yaml 的修改时间拼成缓存键。

    这样只要改了 YAML，缓存键就会变，刷新页面即可生效，不必重启应用；
    否则 @st.cache_resource 会把启动时的旧配置一直复用下去。
    """
    files = sorted((ROOT / "configs").glob("*.yaml"))
    return "|".join(f"{p.name}:{p.stat().st_mtime_ns}" for p in files)


@st.cache_resource
def init_system(config_stamp: str):  # config_stamp 仅作为缓存键，配置未变则复用
    """初始化系统（配置未变化时复用同一个 ConfigLoader）"""
    from src.utils.config_loader import ConfigLoader
    return ConfigLoader(str(ROOT / "configs"))


configs = init_system(_config_stamp())
st.session_state["configs"] = configs


# ======================================================================
# 首页（函数式页面：登录门禁 + 系统概览）
# ======================================================================

def render_home():
    from app.components.sidebar import render_sidebar
    from app.components.auth_gate import require_login

    render_sidebar()
    user = require_login()

    role_label = "🛡️ 管理员" if user["role"] == "admin" else "👤 普通用户"
    st.title("💳 智能信贷风控分析系统")
    st.markdown(
        f"##### 基于 XGBoost + PyTorch MLP + SHAP + DeepSeek LLM 的端到端风控解决方案"
    )
    st.success(f"欢迎回来，**{user['username']}**（{role_label}）")

    # 系统状态指示
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    # 检查模型状态
    model_dir = ROOT / "data" / "models" / "latest"
    has_model = model_dir.exists() and any(model_dir.glob("*.json"))

    # 检查数据状态
    sample_data = ROOT / "data" / "sample" / "sample.csv"
    real_data = ROOT / "data" / "raw" / "train.csv"
    has_data = sample_data.exists() or real_data.exists()

    # 检查 LLM 配置
    import os
    has_api_key = os.environ.get("DEEPSEEK_API_KEY", "").startswith("sk-") and \
                  os.environ.get("DEEPSEEK_API_KEY") != "sk-your-api-key-here"

    with status_col1:
        st.metric(
            "📦 数据状态",
            "已就绪" if has_data else "未就绪",
            delta="✅ OK" if has_data else "⚠️ 需要数据",
            delta_color="normal" if has_data else "inverse",
        )
    with status_col2:
        st.metric(
            "🤖 模型状态",
            "已训练" if has_model else "未训练",
            delta="✅ OK" if has_model else "⚠️ 需训练",
            delta_color="normal" if has_model else "inverse",
        )
    with status_col3:
        st.metric(
            "🔑 LLM API",
            "已配置" if has_api_key else "未配置",
            delta="✅ OK" if has_api_key else "⚠️ 需 API Key",
            delta_color="normal" if has_api_key else "inverse",
        )
    with status_col4:
        st.metric("⚙️ 系统版本", "v1.1.0", delta="多用户版")

    st.divider()

    # ===== 角色专属导航提示 =====
    st.markdown("## 🚀 快速开始")

    if user["role"] == "admin":
        st.markdown("""
        您以**管理员**身份登录，可访问全部功能模块：
        - 📊 **数据探索** / 🔧 **特征工程**：理解与处理数据
        - 🤖 **模型训练**：交互式训练 XGBoost、LightGBM、MLP 并融合
        - 🧪 **A/B测试** / 📈 **模型监控**：模型对照实验与漂移监控
        - 👤 **用户管理**：账号增删改、角色分配、预测行为审计
        - ⚙️ **系统配置**：在线修改 YAML 配置
        """)
    else:
        st.markdown("""
        您以**普通用户**身份登录，可直接使用系统最优模型（Ensemble 融合，AUC 0.7346）：
        - 🎯 **单笔预测**：录入申请人信息，秒级获得违约概率 + SHAP 解释 + LLM 审批报告
        - 🔍 **模型解释**：查看模型全局特征重要性
        - 💬 **LLM智能助手**：风控领域智能问答
        """)

    st.divider()

    # ===== 项目简介 =====
    with st.expander("📚 项目简介与数据集"):
        st.markdown("""
        本系统是一个**完整的金融风控分析平台**，模拟互联网金融公司的贷款审批流程，
        集成了机器学习课程涵盖的全部核心技术。

        | 模块 | 技术栈 | 应用场景 |
        |------|--------|---------|
        | 数据探索 | Pandas + Plotly | 数据质量分析、特征分布可视化 |
        | 特征工程 | Sklearn | 缺失值/异常值/编码/标准化 |
        | 模型训练 | XGBoost + LightGBM + PyTorch MLP | 三模型融合 |
        | 超参优化 | Optuna | 自动搜索最优超参 |
        | 模型解释 | SHAP | 全局/局部特征贡献分析 |
        | 智能助手 | DeepSeek LLM | 风险报告 / 特征建议 / 案例诊断 |

        > **数据集**: 天池"零基础入门金融风控-贷款违约预测"
        > 80 万条贷款记录，47 列特征，预测用户违约概率。
        """)

    # ===== 页脚 =====
    st.markdown("""
    ---
    <div style='text-align: center; color: gray; font-size: 0.85em;'>
        💳 智能信贷风控分析系统 · 多用户版 v1.1.0 · 仅用于学习研究<br>
        本系统生成的报告和分析结果不构成任何金融建议
    </div>
    """, unsafe_allow_html=True)


# ======================================================================
# 按角色构建导航
# ======================================================================
from app.components.nav_registry import visible_page_defs

user = st.session_state.get("user")
auth_cfg = configs.get("auth")

# 首页：所有人可见（未登录时内容为登录卡片）
home_page = st.Page(
    render_home,
    title="🏠 首页",
    icon="💳",
    url_path="home",
    default=True,
)

# 业务/管理页面：仅登录后按角色显示
page_objects = [home_page]
if user is not None:
    for p in visible_page_defs(user, auth_cfg):
        page_objects.append(
            st.Page(
                ROOT / p.file,          # 绝对路径，避免相对路径歧义
                title=p.title,
                icon=p.icon,
                url_path=p.url_path,
            )
        )

nav = st.navigation(page_objects)
nav.run()
