# src/llm/agents/compliance_agent.py
"""
政策合规检查 Agent
==================
基于贷款审批的申请人信息、模型决策与利率，调用 LLM 逐条核对监管规则，
输出结构化的合规审查结果。

与 ReportAgent 遵循同一"三段式"模式：
    Prompt 模板（compliance_check.j2） + Agent 类 + 配置注册
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.llm.client import DeepSeekClient
from src.llm.prompts import PromptManager
from src.utils.logger import get_logger

logger = get_logger("llm.compliance_agent")


class ComplianceAgent:
    """
    政策合规检查 Agent。

    使用示例:
        >>> agent = ComplianceAgent(client, prompt_mgr, llm_cfg.agents.compliance)
        >>> result = agent.check(features, decision="通过", interest_rate=13.5)
    """

    DEFAULT_RULES = [
        "贷款利率不得超过 LPR 的 4 倍",
        "对 60 岁以上申请人，需要额外的还款能力证明",
        "贷款用途禁止用于房地产投资",
    ]

    def __init__(
        self,
        client: DeepSeekClient,
        prompt_manager: PromptManager,
        config,
    ):
        self.client = client
        self.pm = prompt_manager
        self.config = config

    def check(
        self,
        features: Dict[str, Any],
        decision: str,
        interest_rate: float,
        rules: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """逐条审查贷款审批是否合规，返回结构化 JSON 结果。"""
        prompt = self.pm.render(
            "compliance_check",
            features=features,
            decision=decision,
            interest_rate=interest_rate,
            rules=rules or self.DEFAULT_RULES,
        )
        system_prompt = self.pm.get_system_role("compliance_check")

        logger.info(f"合规检查 - 决策={decision}, 利率={interest_rate}%")

        response = self.client.chat(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        # 推理型模型可能把 token 预算耗尽在思维链上，导致正文为空
        if not response or not response.strip():
            raise ValueError(
                "LLM 返回内容为空（通常因推理过程耗尽 max_tokens 被截断）。"
                "请在 configs/llm_config.yaml 中调大 agents.compliance.max_tokens 后重试。"
            )
        return json.loads(self._extract_json(response))

    @staticmethod
    def _extract_json(text: str) -> str:
        # 提取被 ```json ... ``` 包裹的内容（贪婪匹配以兼容嵌套 JSON）
        import re
        match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', text, re.DOTALL)
        return match.group(1) if match else text
