"""
LLM客户端封装模块。

提供统一的LLM调用接口和Embedding生成接口，
支持OpenAI API兼容格式的各种模型服务。
"""

import json
import logging
from typing import Any

from evirar.clients import current_runtime

logger = logging.getLogger(__name__)


def call_llm(
    prompt: str,
    system_message: str = "你是一位专业的中医诊断专家。",
    temperature: float | None = None,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
) -> str:
    """
    调用LLM生成响应。

    Args:
        prompt: 用户提示词
        system_message: 系统消息
        temperature: 温度参数，默认使用配置值
        max_tokens: 最大token数，默认使用配置值
        enable_thinking: 是否启用思考模式，默认使用配置值
        top_p: Top-p采样参数，默认使用配置值
        min_p: Min-p采样参数（通过extra_body传递），默认使用配置值
        presence_penalty: 存在惩罚参数，默认使用配置值
        repetition_penalty: 重复惩罚参数（通过extra_body传递），默认使用配置值
        response_format: 可选的OpenAI兼容结构化响应格式

    Returns:
        LLM生成的文本响应
    """
    return current_runtime().chat(
        [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        label="reasoning", temperature=temperature, max_tokens=max_tokens,
        response_format=response_format,
    )


def call_llm_json(
    prompt: str,
    system_message: str = "你是一位专业的中医诊断专家。请以JSON格式输出。",
) -> dict[str, Any]:
    """
    调用LLM并解析JSON响应。

    Args:
        prompt: 用户提示词
        system_message: 系统消息

    Returns:
        解析后的JSON字典
    """
    response = call_llm(prompt, system_message)
    if not response:
        logger.warning("call_llm_json 收到空响应")
        return {}
    text = response.strip()

    # 尝试提取JSON块
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("Model response is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def get_embedding(text: str) -> list[float]:
    """
    获取文本的向量嵌入。

    Args:
        text: 输入文本

    Returns:
        向量嵌入列表
    """
    return current_runtime().embeddings([text])[0]


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """
    批量获取文本的向量嵌入。

    Args:
        texts: 输入文本列表

    Returns:
        向量嵌入列表的列表
    """
    if not texts:
        return []

    return current_runtime().embeddings(texts)
