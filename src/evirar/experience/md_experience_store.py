"""
医案经验 Markdown 文件存储与 Embedding 管理模块（Md Experience Store）

负责从指定文件夹加载 .md 格式的医案经验文件，生成并缓存 Embedding 向量。
支持增量更新：首次生成 Embedding，后续若文件未变更则直接读取缓存。
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

from evirar.knowledge import config as kb_config
from evirar.knowledge.llm_client import get_embedding, get_embeddings_batch
from evirar.cache import cache_file

logger = logging.getLogger(__name__)

# Embedding 缓存文件名
_MD_CACHE_FILENAME = ".md_exp_embedding_cache.json"


class MdExperienceStore:
    """
    医案经验 Markdown 文件存储管理器。

    从指定目录加载 .md 文件，为每个文件生成 Embedding 向量，
    并使用 SHA256 内容哈希进行缓存失效检测。
    """

    def __init__(self, md_dir: str):
        """
        初始化 Markdown 经验库。

        Args:
            md_dir: 包含 .md 医案经验文件的目录路径
        """
        self._md_dir = md_dir
        self._experiences: list[dict[str, Any]] = []
        self._loaded = False

    @property
    def md_dir(self) -> str:
        return self._md_dir

    @property
    def experiences(self) -> list[dict[str, Any]]:
        if not self._loaded:
            self.load()
        return self._experiences

    @property
    def size(self) -> int:
        return len(self._experiences)

    # ------------------------------------------------------------------
    # 加载 MD 文件
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        从目录加载所有 .md 文件，生成或加载缓存的 Embedding。

        每个经验记录结构：
        {
            "id": "1",              # 文件名（不含扩展名）
            "filename": "1.md",
            "content": "...",        # 完整 md 内容
            "content_hash": "...",   # SHA256 哈希
            "embedding": [...]       # 向量嵌入
        }
        """
        md_path = Path(self._md_dir)
        if not md_path.exists() or not md_path.is_dir():
            logger.warning("MD 经验目录不存在: %s，初始化空经验库", self._md_dir)
            self._experiences = []
            self._loaded = True
            return

        # 收集所有 md 文件
        md_files = sorted(md_path.glob("*.md"))
        if not md_files:
            logger.warning("MD 经验目录中无 .md 文件: %s", self._md_dir)
            self._experiences = []
            self._loaded = True
            return

        # 读取文件内容
        experiences: list[dict[str, Any]] = []
        for f in md_files:
            content = f.read_text(encoding="utf-8")
            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            experiences.append({
                "id": f.stem,
                "filename": f.name,
                "content": content,
                "content_hash": content_hash,
                "embedding": [],
            })

        logger.info(
            "从目录加载了 %d 个 MD 经验文件: %s", len(experiences), self._md_dir
        )

        # 加载或生成 Embedding
        self._load_or_generate_embeddings(experiences)

        self._experiences = experiences
        self._loaded = True

    def load_single(self, file_path: str) -> dict[str, Any] | None:
        """
        加载单个 .md 文件并生成 Embedding。

        Args:
            file_path: .md 文件的完整路径

        Returns:
            经验记录字典，文件不存在时返回 None
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning("MD 经验文件不存在: %s", file_path)
            return None

        content = path.read_text(encoding="utf-8")
        if not content.strip():
            logger.warning("MD 经验文件为空: %s", file_path)
            return None

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        # 尝试从缓存获取 embedding
        cached = self._load_cache()
        embedding = []
        if cached:
            entry = cached.get(path.stem)
            if entry and entry.get("content_hash") == content_hash:
                embedding = entry.get("embedding", [])
                logger.debug("从缓存加载 MD 经验 Embedding: %s", path.stem)

        # 缓存未命中，生成新的 embedding
        if not embedding:
            try:
                # 提取关键文本用于 embedding
                embed_text = self._extract_embed_text(content)
                embedding = get_embedding(embed_text)
                # 保存到缓存
                self._save_single_to_cache(path.stem, content_hash, embedding)
            except Exception:
                logger.exception("生成 MD 经验 Embedding 失败: %s", file_path)

        return {
            "id": path.stem,
            "filename": path.name,
            "content": content,
            "content_hash": content_hash,
            "embedding": embedding,
        }

    # ------------------------------------------------------------------
    # Embedding 缓存管理
    # ------------------------------------------------------------------

    def _get_cache_path(self) -> str:
        return str(cache_file(_MD_CACHE_FILENAME))

    def _load_cache(self) -> dict[str, Any]:
        """
        加载 Embedding 缓存文件。

        Returns:
            id -> {"content_hash": ..., "embedding": [...]} 的映射
        """
        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            return {}

        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("MD Embedding 缓存文件损坏，将重新生成")
            return {}

        # 检查 embedding 模型是否一致
        if cache_data.get("embedding_model") != kb_config.EMBEDDING_MODEL:
            logger.info("嵌入模型已变更，MD 经验缓存失效")
            return {}

        return cache_data.get("entries", {})

    def _save_cache(
        self, entries: dict[str, dict[str, Any]]
    ) -> None:
        """保存完整的 Embedding 缓存。"""
        cache_data = {
            "embedding_model": kb_config.EMBEDDING_MODEL,
            "entries": entries,
        }
        cache_path = self._get_cache_path()
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False)
            logger.info("MD Embedding 缓存已保存: %s", cache_path)
        except OSError:
            logger.warning("MD Embedding 缓存保存失败: %s", cache_path)

    def _save_single_to_cache(
        self, exp_id: str, content_hash: str, embedding: list[float]
    ) -> None:
        """将单条经验的 Embedding 追加保存到缓存。"""
        entries = self._load_cache()
        entries[exp_id] = {
            "content_hash": content_hash,
            "embedding": embedding,
        }
        self._save_cache(entries)

    def _load_or_generate_embeddings(
        self, experiences: list[dict[str, Any]]
    ) -> None:
        """
        为经验列表加载或生成 Embedding。

        首先尝试从缓存读取，缓存命中（哈希一致）则直接使用；
        否则批量生成新的 Embedding 并更新缓存。
        """
        cached_entries = self._load_cache()

        # 分离：缓存命中 vs 需要生成
        need_embed: list[int] = []  # 需要生成 embedding 的索引
        for idx, exp in enumerate(experiences):
            exp_id = exp["id"]
            cached = cached_entries.get(exp_id)
            if cached and cached.get("content_hash") == exp["content_hash"]:
                exp["embedding"] = cached.get("embedding", [])
            else:
                need_embed.append(idx)

        cache_hit_count = len(experiences) - len(need_embed)
        if cache_hit_count > 0:
            logger.info("从缓存加载了 %d 条 MD 经验的 Embedding", cache_hit_count)

        if not need_embed:
            logger.info("所有 MD 经验的 Embedding 均来自缓存")
            return

        # 批量生成 Embedding
        logger.info("正在为 %d 个 MD 经验文件生成 Embedding...", len(need_embed))
        texts = [
            self._extract_embed_text(experiences[i]["content"])
            for i in need_embed
        ]

        try:
            embeddings = get_embeddings_batch(texts)
            for i, emb in zip(need_embed, embeddings):
                experiences[i]["embedding"] = emb
        except Exception:
            logger.exception("批量生成 MD 经验 Embedding 失败")
            return

        # 更新缓存
        new_entries: dict[str, dict[str, Any]] = {}
        for exp in experiences:
            if exp.get("embedding"):
                new_entries[exp["id"]] = {
                    "content_hash": exp["content_hash"],
                    "embedding": exp["embedding"],
                }
        self._save_cache(new_entries)
        logger.info("MD 经验 Embedding 生成完成并已缓存")

    # ------------------------------------------------------------------
    # 文本提取
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_embed_text(md_content: str) -> str:
        """
        从 MD 内容中提取用于生成 Embedding 的关键文本。

        提取策略：
        1. 提取「医案特征提取」相关内容（这是最核心的临床特征摘要）
        2. 提取「辨证诊断逻辑链」中的核心病位、病性和辨证结论
        3. 如果上述内容不足，回退到使用完整内容的精简版本
        """
        sections: list[str] = []

        # 提取"医案特征提取"部分
        for line in md_content.split("\n"):
            stripped = line.strip()
            if "医案特征提取" in stripped:
                # 提取冒号后的内容
                if "：" in stripped:
                    feature_text = stripped.split("：", 1)[1].strip()
                    if feature_text:
                        sections.append(feature_text)
                elif ":" in stripped:
                    feature_text = stripped.split(":", 1)[1].strip()
                    if feature_text:
                        sections.append(feature_text)

        # 提取核心病位、病性、辨证结论
        for marker in ["核心病位", "核心病性", "辨证结论"]:
            for line in md_content.split("\n"):
                if marker in line:
                    # 去掉 markdown 标记
                    clean = line.strip().lstrip("-*# ")
                    if "：" in clean:
                        text = clean.split("：", 1)[1].strip()
                    elif ":" in clean:
                        text = clean.split(":", 1)[1].strip()
                    else:
                        text = clean
                    # 去掉 markdown 粗体标记
                    text = text.replace("**", "")
                    if text:
                        sections.append(text)

        if sections:
            return "；".join(sections)

        # 回退：使用前 1500 字符，仅移除 markdown 语法标记
        import re
        clean_text = re.sub(r"^#{1,6}\s+", "", md_content, flags=re.MULTILINE)
        clean_text = clean_text.replace("**", "").replace("__", "")
        clean_text = re.sub(r"^[-*]\s+", "", clean_text, flags=re.MULTILINE)
        return clean_text[:1500].strip()

    def get_experience_by_id(self, exp_id: str) -> dict[str, Any] | None:
        """
        根据 ID 获取特定经验。

        Args:
            exp_id: 经验 ID（即 md 文件名，不含扩展名）

        Returns:
            经验记录字典，未找到返回 None
        """
        if not self._loaded:
            self.load()

        for exp in self._experiences:
            if exp["id"] == exp_id:
                return exp
        return None

    def get_all_experiences(self) -> list[dict[str, Any]]:
        """获取所有经验记录。"""
        if not self._loaded:
            self.load()
        return list(self._experiences)
