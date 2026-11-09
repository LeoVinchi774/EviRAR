"""Experience-stage defaults retained from the evaluated implementation."""
from evirar.clients import current_runtime

SCORE_THRESHOLD = 7.5
MAX_EXPERIENCES_PER_SYNDROME = 20
STAGE1_TOP_N = 10
STAGE2_TOP_K = 5
STAGE3_TOP_M = 3
NOVELTY_MAX_SIMILARITY = 0.98
EARLY_STAGE_END_TURN = 3
MIDDLE_STAGE_END_TURN = 8
REWARD_WEIGHT_PROCESS = 0.3
REWARD_WEIGHT_RESULT = 0.4
REWARD_WEIGHT_REASONING = 0.3


def __getattr__(name):
    if name in {"EXPERIENCE_LIBRARY_DIR", "MD_EXPERIENCE_DIR", "RAW_TRAJECTORY_DIR"}:
        return str(current_runtime().settings.resource_path("experience"))
    raise AttributeError(name)

