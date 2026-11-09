"""
多维自省诊断精炼流水线（Self-Reflection Pipeline）

编排多维自省诊断精炼的完整流程：
- Step 1: 初步诊断生成
- Step 2: 三维评判（证据一致性 + 中医理论一致性 + 鉴别诊断）
- Step 3: 文本梯度生成与诊断更新
- Step 4: 迭代（最多3轮，或评分均超阈值时提前终止）

与前三个模块联动：
- 调用知识库（5.1）验证中医理论一致性
- 调用经验库（5.2）参考鉴别诊断案例
- 接收推理树（5.3）的假设状态作为鉴别诊断输入

对应 TCM_BianZheng_RAE_Proposal.md 5.4 节完整流程。
"""

import logging
from typing import Any

from evirar.reflection import config
from evirar.reflection.critics import (
    DifferentialDiagnosisCritic,
    EvidenceConsistencyCritic,
    TCMTheoryConsistencyCritic,
)
from evirar.reflection.diagnosis_generator import (
    generate_initial_diagnosis,
    refine_diagnosis,
)
from evirar.reflection.gradient_merger import (
    compute_aggregate_scores,
    merge_critic_results,
)

logger = logging.getLogger(__name__)


class SelfReflectionPipeline:
    """
    多维自省诊断精炼流水线。

    整合诊断生成、三维评判、文本梯度合并和迭代精炼，
    并与知识库（5.1）、经验库（5.2）和推理树（5.3）联动工作。
    """

    def __init__(self) -> None:
        self._evidence_critic = EvidenceConsistencyCritic()
        self._theory_critic = TCMTheoryConsistencyCritic()
        self._differential_critic = DifferentialDiagnosisCritic()

    # ------------------------------------------------------------------
    # 主接口：精炼诊断
    # ------------------------------------------------------------------

    def refine(
        self,
        dialogue_history: list[dict[str, str]],
        clinical_features: dict[str, Any],
        hypothesis_list: list[dict[str, Any]],
        kb_knowledge: str = "",
        experience_knowledge: str = "",
        differential_experiences: str = "",
    ) -> str:
        """
        执行完整的多维自省诊断精炼流程。

        这是最常用的接口——输入对话和辅助信息，输出精炼后的最终诊断。

        Args:
            dialogue_history: 完整的问诊对话历史
            clinical_features: 从对话中提取的临床特征
            hypothesis_list: 推理树的假设列表（含置信度），
                格式为 [{"syndrome_name": ..., "confidence": ..., ...}, ...]
            kb_knowledge: 知识库检索的证候知识卡片文本
            experience_knowledge: 经验库检索的参考文本
            differential_experiences: 用于鉴别的类似但诊断不同的经验案例文本

        Returns:
            精炼后的最终诊断文本 D_final
        """
        result = self.refine_with_details(
            dialogue_history=dialogue_history,
            clinical_features=clinical_features,
            hypothesis_list=hypothesis_list,
            kb_knowledge=kb_knowledge,
            experience_knowledge=experience_knowledge,
            differential_experiences=differential_experiences,
        )
        return result["final_diagnosis"]

    def refine_with_details(
        self,
        dialogue_history: list[dict[str, str]],
        clinical_features: dict[str, Any],
        hypothesis_list: list[dict[str, Any]],
        kb_knowledge: str = "",
        experience_knowledge: str = "",
        differential_experiences: str = "",
    ) -> dict[str, Any]:
        """
        执行完整精炼流程并返回各步骤的详细中间结果。

        Args:
            dialogue_history: 完整的问诊对话历史
            clinical_features: 从对话中提取的临床特征
            hypothesis_list: 推理树的假设列表
            kb_knowledge: 知识库检索文本
            experience_knowledge: 经验库检索文本
            differential_experiences: 鉴别诊断参考案例文本

        Returns:
            包含各步骤中间结果的详细字典
        """
        logger.info("=" * 60)
        logger.info("[多维自省] 开始诊断精炼流程")
        logger.info("=" * 60)

        iterations: list[dict[str, Any]] = []

        # ---- Step 1: 初步诊断生成 ----
        current_diagnosis = generate_initial_diagnosis(
            dialogue_history=dialogue_history,
            clinical_features=clinical_features,
            hypothesis_list=hypothesis_list,
            kb_knowledge=kb_knowledge,
            experience_knowledge=experience_knowledge,
        )

        # ---- Step 2-4: 迭代精炼 ----
        max_iter = config.MAX_REFINEMENT_ITERATIONS
        for iteration in range(1, max_iter + 1):
            logger.info("-" * 40)
            logger.info("[多维自省] 第 %d/%d 轮评判", iteration, max_iter)
            logger.info("-" * 40)

            # Step 2: 三维评判
            critic_results = self._run_critics(
                diagnosis=current_diagnosis,
                dialogue_history=dialogue_history,
                clinical_features=clinical_features,
                hypothesis_list=hypothesis_list,
                kb_knowledge=kb_knowledge,
                differential_experiences=differential_experiences,
            )

            # 计算汇总评分
            aggregate = compute_aggregate_scores(critic_results)

            iteration_record: dict[str, Any] = {
                "iteration": iteration,
                "diagnosis": current_diagnosis,
                "critic_results": critic_results,
                "scores": aggregate,
            }

            # 检查是否提前终止（三个维度均超过阈值）
            all_pass = all(
                r.get("score", 0) >= config.SCORE_THRESHOLD
                for r in critic_results
            )
            if all_pass:
                logger.info(
                    "[多维自省] 三个维度评分均达标（≥%.1f），提前终止",
                    config.SCORE_THRESHOLD,
                )
                iteration_record["early_stop"] = True
                iterations.append(iteration_record)
                break

            # Step 3: 生成文本梯度并精炼诊断
            text_gradient = merge_critic_results(critic_results)
            iteration_record["text_gradient"] = text_gradient

            if iteration < max_iter:
                current_diagnosis = refine_diagnosis(
                    original_diagnosis=current_diagnosis,
                    text_gradient=text_gradient,
                    dialogue_history=dialogue_history,
                    kb_knowledge=kb_knowledge,
                    experience_knowledge=experience_knowledge,
                    differential_experiences=differential_experiences,
                    hypothesis_list=hypothesis_list,
                )

            iterations.append(iteration_record)

        logger.info("=" * 60)
        logger.info(
            "[多维自省] 诊断精炼完成，共迭代 %d 轮", len(iterations)
        )
        logger.info("=" * 60)

        # 获取最终评分
        final_scores = iterations[-1]["scores"] if iterations else {}

        return {
            "initial_diagnosis": iterations[0]["diagnosis"] if iterations else "",
            "final_diagnosis": current_diagnosis,
            "total_iterations": len(iterations),
            "final_scores": final_scores,
            "iterations": iterations,
        }

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _run_critics(
        self,
        diagnosis: str,
        dialogue_history: list[dict[str, str]],
        clinical_features: dict[str, Any],
        hypothesis_list: list[dict[str, Any]],
        kb_knowledge: str,
        differential_experiences: str,
    ) -> list[dict[str, Any]]:
        """
        执行三维评判。

        Args:
            diagnosis: 当前诊断文本
            dialogue_history: 问诊对话记录
            clinical_features: 临床特征
            hypothesis_list: 假设列表
            kb_knowledge: 知识库文本
            differential_experiences: 鉴别诊断参考文本

        Returns:
            三个评判器的结果列表
        """
        results: list[dict[str, Any]] = []

        # 维度一：证据一致性
        evidence_result = self._evidence_critic.evaluate(
            diagnosis=diagnosis,
            dialogue_history=dialogue_history,
            clinical_features=clinical_features,
        )
        results.append(evidence_result)

        # 维度二：中医理论一致性（含八纲一致性校验）
        theory_result = self._theory_critic.evaluate(
            diagnosis=diagnosis,
            kb_knowledge=kb_knowledge,
            hypothesis_list=hypothesis_list,
        )
        results.append(theory_result)

        # 维度三：鉴别诊断
        differential_result = self._differential_critic.evaluate(
            diagnosis=diagnosis,
            hypothesis_list=hypothesis_list,
            differential_experiences=differential_experiences,
        )
        results.append(differential_result)

        return results
