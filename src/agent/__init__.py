from src.agent.shopping_agent import ShoppingGuideAgent
from src.agent.langgraph_engine import ShoppingGuideGraph, ShoppingState, classify_stage, extract_profile_signals
from src.agent.shopping_tools import (
    SHOPPING_TOOLS, init_shopping_tools,
    create_search_products, create_get_product_detail, create_get_reviews,
    create_compare_products, create_get_user_profile, create_update_user_profile,
)
from src.agent.shopping_prompts import (
    SHOPPING_SYSTEM_PROMPT, STAGE_CLASSIFIER_PROMPT,
    COMMON_STYLE_GUIDE,
    DISCOVERY_AGENT_PROMPT, SEARCH_AGENT_PROMPT, COMPARE_AGENT_PROMPT,
    PROFILE_AGENT_PROMPT, RECOMMEND_AGENT_PROMPT,
)

__all__ = [
    # Agent class
    "ShoppingGuideAgent",
    "ShoppingGuideGraph",
    # State
    "ShoppingState",
    # Helpers (standalone)
    "classify_stage",
    "extract_profile_signals",
    # Tools (old globals — deprecated)
    "SHOPPING_TOOLS",
    "init_shopping_tools",
    # Tool factories
    "create_search_products",
    "create_get_product_detail",
    "create_get_reviews",
    "create_compare_products",
    "create_get_user_profile",
    "create_update_user_profile",
    # Prompts
    "SHOPPING_SYSTEM_PROMPT",  # deprecated — kept for run_simple fallback
    "STAGE_CLASSIFIER_PROMPT",
    "COMMON_STYLE_GUIDE",
    "DISCOVERY_AGENT_PROMPT",
    "SEARCH_AGENT_PROMPT",
    "COMPARE_AGENT_PROMPT",
    "PROFILE_AGENT_PROMPT",
    "RECOMMEND_AGENT_PROMPT",
]
