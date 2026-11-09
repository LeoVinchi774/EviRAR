"""Context assembly extracted from the evaluated dialogue runner.

The disabled reasoning-tree branch is omitted. Enabled module failures propagate
instead of silently degrading to Baseline. See resources/README.md for inputs.
"""
import logging
from typing import Any
from evirar.knowledge.pipeline import TCMKBRAGPipeline
from evirar.experience.pipeline import ExperienceRetrievalPipeline
from evirar.reflection.pipeline import SelfReflectionPipeline
from evirar.protocol import extract_thinking, parse_doctor_action
logger = logging.getLogger(__name__)

class ContextManager:
    """
    管理四个 RAE 模块的生命周期和调用。

    每个对话实例应创建一个 RAEPipelineManager 以保持辨证路径树等
    有状态模块的独立性。
    """

    def __init__(
        self,
        enable_kb_rag: bool = False,
        enable_experience: bool = False,
        enable_self_reflection: bool = False,
        experience_md_dir: str = "",
    ):
        self.enable_kb_rag = enable_kb_rag
        self.enable_experience = enable_experience
        self.enable_self_reflection = enable_self_reflection
        self.experience_md_dir = experience_md_dir

        # 按需初始化各模块
        self._kb_pipeline: TCMKBRAGPipeline | None = None
        self._exp_pipeline: ExperienceRetrievalPipeline | None = None
        self._reflection_pipeline: SelfReflectionPipeline | None = None

        if self.enable_kb_rag:
            self._kb_pipeline = TCMKBRAGPipeline()
        if self.enable_experience:
            md_dir = experience_md_dir if experience_md_dir else None
            self._exp_pipeline = ExperienceRetrievalPipeline(
                md_experience_dir=md_dir,
            )
        if self.enable_self_reflection:
            self._reflection_pipeline = SelfReflectionPipeline()

        # 缓存最近一次的中间结果
        self._last_clinical_features: dict[str, Any] = {}
        self._last_kb_results: list[dict[str, Any]] = []
        self._last_kb_text: str = ""
        self._last_experience_text: str = ""
        self._last_reasoning_result: dict[str, Any] = {}
        self._kb_gate_turns: list[dict[str, Any]] = []
        self.last_knowledge_details: dict[str, Any] = {}

    def _to_pipeline_dialogue(
        self, doctor_messages: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """
        将 OpenAI 格式的对话历史转换为 pipeline 所需格式。

        doctor_messages 的 role 有 system / user / assistant，
        pipeline 需要 patient/doctor 格式。
        """
        result: list[dict[str, str]] = []
        for msg in doctor_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                # 用户消息在此场景下可能是患者或助理的回复
                result.append({"role": "patient", "content": content})
            elif role == "assistant":
                # 医生模型的输出——去掉 <think> 标签和前缀
                clean, _ = extract_thinking(content, None)
                _, payload = parse_doctor_action(clean)
                result.append({"role": "doctor", "content": payload})
        return result

    # ------------------------------------------------------------------
    # 每轮问诊前的"幕后推理"
    # ------------------------------------------------------------------

    def run_behind_the_scenes(
        self,
        doctor_messages: list[dict[str, str]],
        current_turn: int,
    ) -> str:
        """
        在医生模型生成回复之前，执行幕后推理流水线。

        返回一段精简的增强 context 文本，将被附加到最近一条 user
        消息之后注入到医生模型的对话中。

        Args:
            doctor_messages: 医生视角的完整对话历史（OpenAI格式）
            current_turn: 当前对话轮次

        Returns:
            增强 context 文本，为空字符串表示无增强
        """
        # 至少需要 1 轮患者输入才开始推理
        user_msgs = [m for m in doctor_messages if m["role"] == "user"]
        if not user_msgs:
            return ""

        pipeline_dialogue = self._to_pipeline_dialogue(doctor_messages)
        if not pipeline_dialogue:
            return ""

        context_parts: list[str] = []

        # ── 5.1 知识库检索增强（从 turn 0 开始） ──
        kb_results: list[dict[str, Any]] = []
        kb_text = ""
        clinical_features: dict[str, Any] = {}

        if self.enable_kb_rag and self._kb_pipeline is not None:
            try:
                details = self._kb_pipeline.retrieve_with_details(pipeline_dialogue)
                self.last_knowledge_details = details
                clinical_features = details.get("step1_clinical_features", {})
                kb_results = details.get("generation_results", details.get("final_results", []))
                reference_results = details.get(
                    "reference_results", details.get("final_results", kb_results)
                )
                self._last_clinical_features = clinical_features
                self._last_kb_results = kb_results
                action = details.get("retrieval_action", {})
                selected = details.get("evidence_selection", {})
                self._kb_gate_turns.append({
                    "turn": current_turn,
                    "retrieved": details.get("step3_retrieved_count", 0),
                    "validated": details.get("step4_validated_count", 0),
                    "roles": details.get("document_gate", {}).get("role_counts", {}),
                    "selected": selected.get("selected_count", 0),
                    "injected": len(kb_results),
                    "referenced": len(reference_results),
                    "action": action.get("action", "generate"),
                    "reason": action.get("reason", ""),
                })

                action_name = action.get("action", "generate")
                if action_name == "generate" and kb_results:
                    kb_text = self._format_kb_results(kb_results)
                    self._last_kb_text = kb_text
                    context_parts.append(
                        f"[知识库参考]\n{kb_text}"
                    )
                elif action_name in {"ask_follow_up", "retrieve_more"}:
                    if action_name == "ask_follow_up":
                        context_parts.append(
                            "[文档门控动作]\n当前证据不足以诊断，请继续问诊，"
                            "暂不输出最终诊断。\n"
                            f"建议优先询问：{action.get('next_question') or '请继续询问关键症状、舌象、脉象或病程信息。'}"
                        )
                    else:
                        context_parts.append(
                            "[文档门控动作]\n当前证据尚无主支持卡，请继续问诊收集"
                            "鉴别信息，暂不输出最终诊断。"
                        )
                    if reference_results:
                        reference_text = self._format_kb_results(
                            reference_results, include_scores=False, reference_only=True
                        )
                        context_parts.append(
                            "[待核实知识参考]\n"
                            "以下内容是门控筛选出的候选知识，只用于决定下一步询问方向；"
                            "不代表患者已经满足这些条件，也不代表已经确诊。\n"
                            f"{reference_text}\n"
                            "请优先围绕门控建议追问，并把候选内容当作待核实线索。"
                        )
                if action_name == "ask_follow_up":
                    question = action.get("next_question") or "请继续询问关键症状、舌象、脉象或病程信息。"
                    # Retain the local variable for compatibility with
                    # downstream debugging and make the priority explicit.
                    _ = question
            except Exception as e:
                raise RuntimeError("Knowledge retrieval failed") from e

        # 如果没有通过 KB-RAG 提取临床特征，则复用缓存
        if not clinical_features:
            clinical_features = self._last_clinical_features

        # ── 5.2 经验库检索（从 turn 1 开始，等有更多信息再检索） ──
        experience_text = ""

        # 5.2a: 常规经验库检索
        if (
            self.enable_experience
            and self._exp_pipeline is not None
            and current_turn >= 1
        ):
            try:
                syndrome_hypotheses: list[str] = []
                if self._last_reasoning_result:
                    for h in self._last_reasoning_result.get("hypothesis_set", []):
                        syndrome_hypotheses.append(h.get("syndrome_name", ""))

                # 优先尝试 MD 经验语义检索
                md_exp_text = self._exp_pipeline.retrieve_md_experiences(
                    dialogue_history=pipeline_dialogue,
                    clinical_features=clinical_features or None,
                    current_turn=current_turn,
                )
                if md_exp_text:
                    experience_text = md_exp_text
                else:
                    # 回退到传统 JSON 经验库
                    experience_text = self._exp_pipeline.retrieve(
                        dialogue_history=pipeline_dialogue,
                        clinical_features=clinical_features or None,
                        syndrome_hypotheses=syndrome_hypotheses or None,
                    )

                self._last_experience_text = experience_text

                if experience_text:
                    context_parts.append(
                        f"[经验库参考]\n{experience_text}"
                    )
            except Exception as e:
                raise RuntimeError("Experience retrieval failed") from e

        # 组装最终增强 context
        if not context_parts:
            return ""

        return (
            "\n\n---\n【辨证辅助分析】请务必参考以下分析进行回复：\n\n"
            + "\n\n".join(context_parts)
            + "\n---"
        )

    # ------------------------------------------------------------------
    # 诊断阶段的自省精炼（5.4）
    # ------------------------------------------------------------------

    def refine_diagnosis(
        self,
        doctor_messages: list[dict[str, str]],
        raw_diagnosis: str,
    ) -> dict[str, Any]:
        """
        在医生给出诊断后，通过自省机制精炼诊断结果。

        仅在 enable_self_reflection=True 时生效。

        Args:
            doctor_messages: 完整对话历史
            raw_diagnosis: 医生模型的原始诊断文本

        Returns:
            包含精炼诊断结果的字典，若不启用则返回原始诊断
        """
        if not self.enable_self_reflection or self._reflection_pipeline is None:
            return {"final_diagnosis": raw_diagnosis, "refined": False}

        pipeline_dialogue = self._to_pipeline_dialogue(doctor_messages)
        clinical_features = self._last_clinical_features

        # 获取假设列表
        hypothesis_list: list[dict[str, Any]] = []
        if self._last_reasoning_result:
            hypothesis_list = self._last_reasoning_result.get("hypothesis_set", [])
        reasoning_tree_note = self._format_reasoning_tree_for_reflection(
            self._last_reasoning_result
        )
        kb_knowledge = self._last_kb_text
        if reasoning_tree_note:
            prefix = "\n\n" if kb_knowledge else ""
            kb_knowledge = f"{kb_knowledge}{prefix}[辨证路径摘要]\n{reasoning_tree_note}"

        try:
            result = self._reflection_pipeline.refine_with_details(
                dialogue_history=pipeline_dialogue,
                clinical_features=clinical_features,
                hypothesis_list=hypothesis_list,
                kb_knowledge=kb_knowledge,
                experience_knowledge=self._last_experience_text,
            )
            return {
                "final_diagnosis": result["final_diagnosis"],
                "refined": True,
                "initial_diagnosis": result.get("initial_diagnosis", ""),
                "total_iterations": result.get("total_iterations", 0),
                "final_scores": result.get("final_scores", {}),
            }
        except Exception as e:
            raise RuntimeError("Diagnosis refinement failed") from e

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    @staticmethod
    def _format_reasoning_tree_for_reflection(
        reasoning_result: dict[str, Any] | None,
    ) -> str:
        """将 5.3 辨证路径树压缩为 5.4 可用的证据摘要。"""
        if not reasoning_result:
            return ""

        lines: list[str] = []
        best_path = reasoning_result.get("best_path", [])
        path_labels = [
            str(node.get("label", ""))
            for node in best_path
            if isinstance(node, dict) and node.get("label")
        ]
        if path_labels:
            lines.append("辨证路径: " + " -> ".join(path_labels[:4]))

        readiness = reasoning_result.get("readiness_score")
        if readiness is not None:
            lines.append(f"诊断准备度: {readiness:.0%}")

        coverage = reasoning_result.get("coverage", {})
        slots = coverage.get("slots", []) if isinstance(coverage, dict) else []
        covered_labels = [
            slot.get("label", "")
            for slot in slots
            if isinstance(slot, dict) and slot.get("covered")
        ]
        if covered_labels:
            lines.append("关键证据: " + "、".join(covered_labels[:8]))

        missing_slots = reasoning_result.get("missing_slots", [])
        missing_labels = [
            slot.get("label", "")
            for slot in missing_slots
            if isinstance(slot, dict)
        ]
        if missing_labels:
            lines.append("路径缺口: " + "、".join(missing_labels[:6]))

        convergence = reasoning_result.get("convergence", {})
        if isinstance(convergence, dict) and convergence.get("recommendation"):
            lines.append("准备度判断: " + str(convergence["recommendation"]))

        strategy = reasoning_result.get("strategy", {})
        actions = strategy.get("recommended_actions", []) if isinstance(strategy, dict) else []
        action_text = [
            str(action.get("action", ""))
            for action in actions[:2]
            if isinstance(action, dict) and action.get("action")
        ]
        if action_text:
            lines.append("建议补证: " + "；".join(action_text))

        return "\n".join(lines)

    @staticmethod
    def _format_kb_results(
        kb_results: list[dict[str, Any]],
        include_scores: bool = True,
        reference_only: bool = False,
    ) -> str:
        """格式化知识库检索结果，提取核心鉴别要点。

        仅返回 Top 2 条最相关的知识，提取主症、舌脉特征和鉴别要点，
        而非简单截断，以提高注入信息的质量。
        """
        if not kb_results:
            return "（无匹配知识）"
        lines: list[str] = []
        for i, r in enumerate(kb_results[:2], 1):
            name = r.get("syndrome_name", r.get("name", "未知"))
            score = r.get("relevance_score", r.get("score", 0))
            content = r.get("content", r.get("text", ""))

            # 尝试提取结构化字段
            key_parts: list[str] = []
            for field_key in ["主症", "主要症状", "main_symptoms"]:
                if field_key in str(content):
                    key_parts.append(f"主症相关")
                    break
            for field_key in ["舌", "脉"]:
                idx = content.find(field_key)
                if idx >= 0:
                    # 提取该字段附近的上下文（前后各50字）
                    start = max(0, idx - 10)
                    end = min(len(content), idx + 60)
                    snippet = content[start:end].strip()
                    if snippet:
                        key_parts.append(snippet)

            # 如果没有提取到结构化信息，则使用精简摘要
            if not key_parts:
                summary = content[:300] + "…" if len(content) > 300 else content
                key_parts.append(summary)

            if reference_only:
                role = (r.get("document_gate", {}) or {}).get("role", "")
                role_label = {
                    "primary_support": "支持性线索",
                    "partial_support": "部分支持线索",
                    "differential": "鉴别线索",
                }.get(role, "待核实线索")
                support = (r.get("document_gate", {}) or {}).get("support_points", [])
                support_text = f"；已匹配事实：{'、'.join(map(str, support[:3]))}" if support else ""
                lines.append(
                    f"{i}. [{role_label}] {name}\n   {'；'.join(key_parts)}{support_text}"
                )
            elif include_scores:
                lines.append(
                    f"{i}. {name}（相关度:{score:.2f}）\n   {'；'.join(key_parts)}"
                )
            else:
                lines.append(f"{i}. {name}\n   {'；'.join(key_parts)}")
        return "\n".join(lines)

    def get_ablation_config(self) -> dict[str, bool | str]:
        """返回当前消融实验配置。"""
        return {
            "kb_rag": self.enable_kb_rag,
            "experience": self.enable_experience,
            "self_reflection": self.enable_self_reflection,
            "experience_md_dir": self.experience_md_dir,
        }
