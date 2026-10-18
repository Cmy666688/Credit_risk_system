"""
政策合规检查 Agent 单元测试
============================
独立测试文件（不并入 test_basic.py）。

覆盖内容:
    1. compliance_check.j2 模板渲染（验证 llm_config.yaml 的 prompts 注册）
    2. ComplianceAgent.check 的 JSON 解析（围栏 / 裸 JSON / 嵌套 JSON）
    3. system_prompt / temperature / max_tokens 正确下达给 client
    4. 自定义 rules 覆盖 DEFAULT_RULES

运行: pytest tests/test_compliance_agent.py -v
"""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def configs():
    """全局配置"""
    from src.utils.config_loader import ConfigLoader
    return ConfigLoader(str(ROOT / "configs"))


@pytest.fixture()
def prompt_manager(configs):
    from src.llm.prompts import PromptManager
    return PromptManager(configs.get("llm"))


@pytest.fixture()
def agent_config(configs):
    """agents.compliance 配置"""
    return configs.get("llm").agents.compliance


class FakeClient:
    """记录调用参数的假客户端，避免真实访问 DeepSeek API。"""

    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def chat(self, prompt, system_prompt=None, history=None, **kwargs):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt, **kwargs})
        return self.response


FENCED_JSON = """审查结论如下：

```json
{
  "verdict": "non_compliant",
  "summary": "利率超过 4 倍 LPR 上限",
  "violations": [{"rule": "贷款利率不得超过 LPR 的 4 倍", "reason": "年化利率 30% 超过上限"}],
  "warnings": ["用途字段缺失"],
  "recommendations": ["下调利率至合规区间"]
}
```
"""

NESTED_JSON = (
    '{"verdict": "non_compliant", '
    '"violations": [{"rule": "r1", "reason": "嵌套对象 {a} 与 {b}"}], '
    '"warnings": [], "recommendations": []}'
)


# ===== 模板与配置注册 =====
class TestCompliancePromptRegistration:
    def test_prompt_registered_and_renders(self, prompt_manager):
        """验证 llm_config.yaml 中 prompts.compliance_check 已注册且模板可用"""
        prompt = prompt_manager.render(
            "compliance_check",
            features={"annualIncome": 80000, "grade": "C"},
            decision="通过",
            interest_rate=13.5,
            rules=["规则A", "规则B"],
        )
        assert "合规审查员" in prompt
        assert "13.5" in prompt
        assert "通过" in prompt
        assert "规则A" in prompt and "规则B" in prompt
        assert "80000" in prompt

    def test_system_role_non_empty(self, prompt_manager):
        role = prompt_manager.get_system_role("compliance_check")
        assert isinstance(role, str) and len(role) > 0


# ===== JSON 提取 =====
class TestExtractJson:
    def test_extract_fenced_json(self):
        from src.llm.agents.compliance_agent import ComplianceAgent
        extracted = ComplianceAgent._extract_json(FENCED_JSON)
        assert extracted.strip().startswith("{")
        assert extracted.strip().endswith("}")

    def test_extract_keeps_nested_json_intact(self):
        """回归：非贪婪正则会把嵌套 JSON 截断，须用贪婪匹配"""
        from src.llm.agents.compliance_agent import ComplianceAgent
        extracted = ComplianceAgent._extract_json(f"```json\n{NESTED_JSON}\n```")
        parsed = json.loads(extracted)
        assert parsed["violations"][0]["reason"] == "嵌套对象 {a} 与 {b}"

    def test_extract_fallback_without_fence(self):
        from src.llm.agents.compliance_agent import ComplianceAgent
        assert ComplianceAgent._extract_json(NESTED_JSON) == NESTED_JSON


# ===== Agent 行为 =====
class TestComplianceAgent:
    def test_check_parses_fenced_json(self, prompt_manager, agent_config):
        from src.llm.agents.compliance_agent import ComplianceAgent
        client = FakeClient(FENCED_JSON)
        agent = ComplianceAgent(client, prompt_manager, agent_config)

        result = agent.check(
            features={"annualIncome": 80000},
            decision="通过",
            interest_rate=13.5,
        )
        assert result["verdict"] == "non_compliant"
        assert result["summary"] == "利率超过 4 倍 LPR 上限"
        assert result["violations"][0]["rule"] == "贷款利率不得超过 LPR 的 4 倍"

    def test_check_parses_bare_json(self, prompt_manager, agent_config):
        from src.llm.agents.compliance_agent import ComplianceAgent
        client = FakeClient(NESTED_JSON)
        agent = ComplianceAgent(client, prompt_manager, agent_config)

        result = agent.check(
            features={}, decision="拒绝", interest_rate=18.0
        )
        assert result["violations"][0]["reason"] == "嵌套对象 {a} 与 {b}"

    def test_system_and_params_passed_to_client(self, prompt_manager, agent_config):
        """system_prompt / temperature / max_tokens 必须正确下达（修正 system= 的错误写法）"""
        from src.llm.agents.compliance_agent import ComplianceAgent
        client = FakeClient(FENCED_JSON)
        agent = ComplianceAgent(client, prompt_manager, agent_config)

        agent.check(features={"x": 1}, decision="通过", interest_rate=10.0)

        call = client.calls[0]
        assert call["system_prompt"]  # 非空，来自配置的 system_role
        assert "合规" in call["system_prompt"]
        assert call["temperature"] == agent_config.temperature
        assert call["max_tokens"] == agent_config.max_tokens
        assert "system" not in call  # 不应出现指导书里的错误 kwarg

    def test_custom_rules_override_default(self, prompt_manager, agent_config):
        from src.llm.agents.compliance_agent import ComplianceAgent
        client = FakeClient(FENCED_JSON)
        agent = ComplianceAgent(client, prompt_manager, agent_config)

        agent.check(
            features={}, decision="通过", interest_rate=10.0,
            rules=["自定义规则X"],
        )
        assert "自定义规则X" in client.calls[0]["prompt"]
        # 默认规则不应出现在 prompt 中
        assert "贷款利率不得超过 LPR 的 4 倍" not in client.calls[0]["prompt"]

    def test_default_rules_used_when_none(self, prompt_manager, agent_config):
        from src.llm.agents.compliance_agent import ComplianceAgent
        client = FakeClient(FENCED_JSON)
        agent = ComplianceAgent(client, prompt_manager, agent_config)

        agent.check(features={}, decision="通过", interest_rate=10.0)
        assert "贷款利率不得超过 LPR 的 4 倍" in client.calls[0]["prompt"]

    def test_empty_response_raises_clear_error(self, prompt_manager, agent_config):
        """回归：推理模型耗尽 max_tokens 会返回空内容，须给出明确报错而非 JSONDecodeError"""
        from src.llm.agents.compliance_agent import ComplianceAgent
        client = FakeClient("")
        agent = ComplianceAgent(client, prompt_manager, agent_config)

        with pytest.raises(ValueError, match="为空"):
            agent.check(features={}, decision="通过", interest_rate=10.0)

    def test_whitespace_response_also_guarded(self, prompt_manager, agent_config):
        from src.llm.agents.compliance_agent import ComplianceAgent
        client = FakeClient("   \n  ")
        agent = ComplianceAgent(client, prompt_manager, agent_config)

        with pytest.raises(ValueError, match="为空"):
            agent.check(features={}, decision="通过", interest_rate=10.0)


# ===== 三态结论（compliant / non_compliant / insufficient_info）=====
class TestVerdictStates:
    @pytest.mark.parametrize(
        "verdict", ["compliant", "non_compliant", "insufficient_info"]
    )
    def test_verdict_passthrough(self, prompt_manager, agent_config, verdict):
        """三种结论都应被原样解析出来，而不是被压成布尔值"""
        from src.llm.agents.compliance_agent import ComplianceAgent
        payload = json.dumps(
            {
                "verdict": verdict,
                "summary": "结论说明",
                "violations": [],
                "warnings": [],
                "recommendations": [],
            },
            ensure_ascii=False,
        )
        agent = ComplianceAgent(FakeClient(payload), prompt_manager, agent_config)
        result = agent.check(features={}, decision="通过", interest_rate=10.0)
        assert result["verdict"] == verdict

    def test_template_declares_three_states(self, prompt_manager):
        """由旧描述改成新 JSON schema（is_compliant 布尔 → verdict 三态）"""
        prompt = prompt_manager.render(
            "compliance_check",
            features={},
            decision="通过",
            interest_rate=10.0,
            rules=["规则A"],
        )
        for state in ("compliant", "non_compliant", "insufficient_info"):
            assert state in prompt
        assert "is_compliant" not in prompt

    def test_insufficient_info_keeps_warnings(self, prompt_manager, agent_config):
        """信息不足时，缺失项应落在 warnings 里"""
        from src.llm.agents.compliance_agent import ComplianceAgent
        payload = json.dumps(
            {
                "verdict": "insufficient_info",
                "summary": "缺少年龄与用途信息",
                "violations": [],
                "warnings": ["未提供出生日期", "贷款用途 purpose=0 无说明"],
                "recommendations": ["补充年龄证明与用途声明"],
            },
            ensure_ascii=False,
        )
        agent = ComplianceAgent(FakeClient(payload), prompt_manager, agent_config)
        result = agent.check(features={}, decision="通过", interest_rate=10.0)
        assert result["verdict"] == "insufficient_info"
        assert result["violations"] == []
        assert len(result["warnings"]) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
