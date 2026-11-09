"""Resolve core settings from the active consultation without global env state."""
from evirar.clients import current_runtime


def __getattr__(name):
    settings = current_runtime().settings
    retrieval = settings.values["retrieval"]
    mapping = {
        "RETRIEVAL_TOP_K": retrieval["top_k"],
        "LOGIC_VALIDATION_THRESHOLD": retrieval["validation_threshold"],
        "MAX_RETURN_RESULTS": retrieval["max_decision_cards"],
        "GATE_ENABLE_A": retrieval["metadata_prefilter"],
        "GATE_ENABLE_EVIDENCE_CLASSIFICATION_GATE": retrieval["evidence_classification"],
        "GATE_ENABLE_DYNAMIC_EVIDENCE_STRATEGY": retrieval["dynamic_evidence"],
        "LOGIC_VALIDATION_BATCH": retrieval["validation_batch"],
        "LOGIC_VALIDATION_BATCH_SIZE": retrieval["validation_batch_size"],
        "LOGIC_VALIDATION_MAX_WORKERS": retrieval["validation_workers"],
        "LOGIC_VALIDATION_GLOBAL_CONCURRENCY": retrieval["validation_workers"],
        "LOGIC_VALIDATION_BATCH_MAX_TOKENS": settings.values["generation"]["max_tokens"],
        "VALIDATION_CONTENT_MAX_LENGTH": 2000,
        "KNOWLEDGE_BASE_DIR": str(settings.resource_path("knowledge")),
        "EMBEDDING_MODEL": settings.embedding.model,
        "EMBEDDING_API_BASE": settings.embedding.base_url,
    }
    if name not in mapping:
        raise AttributeError(name)
    return mapping[name]

