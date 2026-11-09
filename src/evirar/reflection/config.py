"""Reflection defaults retained from the evaluated implementation."""
from evirar.clients import current_runtime

SCORE_MIN = 1
SCORE_MAX = 10
DEFAULT_CRITIC_SCORE = 5.0
CRITIC_TEMPERATURE = 0.1
DIAGNOSIS_TEMPERATURE = 0.3
CRITIC_MAX_TOKENS = 2048
DIAGNOSIS_MAX_TOKENS = 3072


def __getattr__(name):
    values = current_runtime().settings.values["reflection"]
    if name == "MAX_REFINEMENT_ITERATIONS":
        return values["max_iterations"]
    if name == "SCORE_THRESHOLD":
        return values["score_threshold"]
    raise AttributeError(name)

