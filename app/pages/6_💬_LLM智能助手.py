"""
💬 LLM 智能助手页面
====================
聊天界面，支持四种模式：
- 风险报告
- 特征建议
- 异常分析
- 自由对话
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from app.components.sidebar import render_sidebar

st.set_page_config(page_title="LLM 智能助手", page_icon="💬", layout="wide")
render_sidebar()

st.title("💬 LLM 智能助手")
st.caption("基于 DeepSeek 大模型的风控领域 AI 助手")

configs = st.session_state.get("configs")
if configs is None:
    from src.utils.config_loader import ConfigLoader
    configs = ConfigLoader(str(ROOT / "configs"))


# ===== 检查 API Key =====
import os
api_key = os.environ.get("DEEPSEEK_API_KEY", "")
if not api_key.startswith("sk-") or api_key == "sk-your-api-key-here":
    st.error("⚠️ 未配置 DeepSeek API Key")
    st.info("""
    **配置步骤：**
    1. 访问 https://platform.deepseek.com 注册账号
    2. 生成 API Key
    3. 在项目根目录的 `.env` 文件中填入：
       ```
       DEEPSEEK_API_KEY=sk-your-real-key-here
       ```
    4. 重启 Streamlit 应用
    """)
    st.stop()


# ===== 初始化客户端 =====
@st.cache_resource
def get_llm_client():
    from src.llm.client import DeepSeekClient
    from src.llm.prompts import PromptManager
    return DeepSeekClient(configs.get("llm")), PromptManager(configs.get("llm"))


client, prompt_mgr = get_llm_client()


# ===== 模式选择 =====
mode = st.radio(
    "选择对话模式",
    ["💬 自由对话", "📝 风险报告", "💡 特征建议", "🔬 异常案例分析"],
    horizontal=True,
)

st.divider()


# ===== 模式：自由对话 =====
if mode == "💬 自由对话":
    st.markdown("##### 💡 你可以问我关于数据、模型、风控业务的任何问题")

    # 初始化对话历史
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # 渲染历史
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 输入框
    if user_input := st.chat_input("请输入您的问题..."):
        # 显示用户消息
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # 准备上下文
        try:
            # 加载系统状态作为上下文
            latest_dir = ROOT / "data" / "models" / "latest"
            model_info = {}
            dataset_info = {}

            eval_path = latest_dir / "eval_results.json"
            if eval_path.exists():
                eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
                model_info = {
                    "xgb_auc": round(eval_data.get("xgboost", {}).get("auc", 0), 4),
                    "mlp_auc": round(eval_data.get("mlp", {}).get("auc", 0), 4),
                    "ensemble_auc": round(eval_data.get("ensemble", {}).get("auc", 0), 4),
                }

            meta_path = latest_dir / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                dataset_info = {
                    "n_samples": meta.get("train_samples", 0),
                    "n_features": len(meta.get("feature_names", [])),
                    "default_rate": "约 20%",
                }

            # 渲染 Prompt
            prompt = prompt_mgr.render(
                "chat_assistant",
                user_question=user_input,
                model_info=model_info if model_info else None,
                dataset_info=dataset_info if dataset_info else None,
            )
            system_prompt = prompt_mgr.get_system_role("chat_assistant")

            # 历史对话（最多保留最近 N 轮）
            max_history = configs.get("llm").agents.chat_agent.max_history_turns
            history = []
            for h in st.session_state.chat_messages[-max_history * 2:-1]:
                history.append({"role": h["role"], "content": h["content"]})

            # 流式输出
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                for chunk in client.stream_chat(prompt, system_prompt=system_prompt, history=history):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)

            st.session_state.chat_messages.append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            with st.chat_message("assistant"):
                st.error(f"调用失败: {e}")

    # 清空按钮
    if st.session_state.chat_messages:
        if st.button("🗑️ 清空对话"):
            st.session_state.chat_messages = []
            st.rerun()


# ===== 模式：风险报告 =====
elif mode == "📝 风险报告":
    st.info("ℹ️ 完整的风险报告功能在 **🎯 单笔预测** 页面，本处提供快速演示。")
    st.markdown("##### 输入贷款信息，让 LLM 生成风险评估报告")

    col1, col2, col3 = st.columns(3)
    with col1:
        loan_amnt = st.number_input("贷款金额", 1000, 500000, 30000)
        annual_income = st.number_input("年收入", 10000, 1000000, 80000)
    with col2:
        interest_rate = st.slider("利率(%)", 5.0, 30.0, 13.5)
        dti = st.slider("DTI(%)", 0.0, 60.0, 18.0)
    with col3:
        fico = st.slider("FICO 分数", 580, 850, 690)
        # 简单计算"违约概率"用于演示
        proba = min(max(0.05 + (interest_rate - 5) * 0.02 + dti * 0.005 - (fico - 580) * 0.001, 0.0), 0.95)
        st.metric("模拟违约概率", f"{proba:.2%}")

    if st.button("🤖 生成简化报告", type="primary"):
        prompt = f"""请基于以下信贷申请信息，生成一份简短的风险评估意见（300字以内）：

- 贷款金额: ¥{loan_amnt:,}
- 利率: {interest_rate}%
- 年收入: ¥{annual_income:,}
- 负债收入比: {dti}%
- FICO 分数: {fico}
- 模型预测违约概率: {proba:.2%}

请给出: 1) 风险等级 2) 主要担忧点 3) 审批建议
"""
        try:
            with st.spinner("生成中..."):
                placeholder = st.empty()
                full = ""
                for chunk in client.stream_chat(prompt):
                    full += chunk
                    placeholder.markdown(full + "▌")
                placeholder.markdown(full)
        except Exception as e:
            st.error(f"调用失败: {e}")


# ===== 模式：特征建议 =====
elif mode == "💡 特征建议":
    st.markdown("##### 让 AI 帮你提出特征工程建议")

    if st.button("🚀 生成建议", type="primary"):
        try:
            from src.llm.agents.feature_agent import FeatureAgent
            from app.components.data_utils import find_data_file

            # 优先用真实数据，回退到合成数据
            data_path, source = find_data_file(ROOT, prefer="auto")
            if data_path is None:
                st.error(
                    "❌ 未找到任何数据。请将真实数据放到 `data/raw/train.csv` "
                    "或运行 `python scripts/generate_sample_data.py` 生成合成数据。"
                )
                st.stop()

            # 给 LLM 的样本不需要太多，1000 条够看分布
            df = pd.read_csv(data_path, nrows=1000)
            st.caption(f"使用数据来源：{source}（{len(df):,} 行）")

            agent = FeatureAgent(client, prompt_mgr, configs.get("llm").agents.feature_agent)

            with st.spinner("AI 思考中..."):
                result = agent.suggest(df=df, current_auc=0.72)

            suggestions = result.get("suggestions", [])
            if suggestions:
                for i, sug in enumerate(suggestions):
                    with st.expander(f"💡 建议 {i+1}: {sug.get('name', '未命名')}"):
                        st.markdown(f"**业务含义**: {sug.get('business_meaning', '')}")
                        st.markdown(f"**预期效果**: {sug.get('expected_impact', '')}")
                        st.code(sug.get("code", ""), language="python")

            if "general_advice" in result:
                st.info(f"**整体建议**: {result['general_advice']}")
        except Exception as e:
            st.error(f"调用失败: {e}")


# ===== 模式：异常分析 =====
elif mode == "🔬 异常案例分析":
    st.markdown("##### 分析模型预测错误的高置信度案例")
    st.info("此功能需要先训练模型，并在测试集上评估。完整版可访问 src/llm/agents/case_agent.py")

    st.markdown("**示例：分析一个 False Negative（漏判违约）**")

    if st.button("🔬 演示分析", type="primary"):
        prompt = """请分析以下机器学习模型预测错误的案例：

【真实情况】客户最终违约（label=1）
【模型预测】违约概率仅 5%（高置信度的错误判断）

【客户特征】
- 贷款金额: ¥45,000
- 年收入: ¥120,000（属于较高收入）
- FICO 分数: 750（信用良好）
- DTI: 12%（负债比例低）
- 就业年限: 8 years
- 房屋: 自有
- 历史逾期: 0次

请从数据、特征、模型、业务四个角度分析：为什么模型预测错误？
并给出 3 条改进建议。
"""
        try:
            with st.spinner("分析中..."):
                placeholder = st.empty()
                full = ""
                for chunk in client.stream_chat(prompt):
                    full += chunk
                    placeholder.markdown(full + "▌")
                placeholder.markdown(full)
        except Exception as e:
            st.error(f"调用失败: {e}")
