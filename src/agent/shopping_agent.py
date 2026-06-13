"""Shopping Guide Agent — wraps LLM + tools + retrievers + profile store.

Four-node LangGraph pipeline (analyze → retrieve → agent ⇄ tools) with
per-stage prompt injection to keep each phase focused without the overhead
of multi-agent orchestration.
"""

from typing import Optional
from pathlib import Path

from langchain.schema import HumanMessage, AIMessage

from src.agent.langgraph_engine import ShoppingGuideGraph
from src.agent.shopping_tools import (
    create_search_products,
    create_get_product_detail,
    create_get_reviews,
    create_compare_products,
    create_get_user_profile,
    create_update_user_profile,
)
from src.agent.shopping_prompts import (
    SHOPPING_SYSTEM_PROMPT,
    STAGE_CLASSIFIER_PROMPT,
    DISCOVERY_AGENT_PROMPT,
    SEARCH_AGENT_PROMPT,
    COMPARE_AGENT_PROMPT,
    RECOMMEND_AGENT_PROMPT,
)
from src.config import config


class ShoppingGuideAgent:
    """Complete shopping guide Agent with per-stage prompt injection.

    Wraps: LLM, ProductRetriever, ProfileStore, ShoppingGuideGraph (4-node).
    """

    def __init__(
        self,
        llm,
        product_retriever,
        profile_store,
        catalog_db: Optional[str] = None,
        reviews_db: Optional[str] = None,
        system_prompt: str = SHOPPING_SYSTEM_PROMPT,
        max_tool_rounds: int = 3,
    ):
        self.llm = llm
        self.product_retriever = product_retriever
        self.profile_store = profile_store
        self.catalog_db = catalog_db or config.PRODUCT_DB_PATH
        self.reviews_db = reviews_db or str(Path(config.PRODUCT_DB_PATH).parent / "product_reviews.db")
        self.max_tool_rounds = max_tool_rounds
        self._system_prompt = system_prompt  # kept for run_simple fallback

        # Create tools via factory functions (no module globals)
        tools = [
            create_search_products(product_retriever),
            create_get_product_detail(self.catalog_db),
            create_get_reviews(self.reviews_db),
            create_compare_products(self.catalog_db),
            create_get_user_profile(profile_store),
            create_update_user_profile(profile_store),
        ]

        # Per-stage prompt mapping: the agent node dynamically selects
        # the right prompt based on the current conversation stage.
        stage_prompts = {
            "discovery": DISCOVERY_AGENT_PROMPT,
            "needs_elicitation": DISCOVERY_AGENT_PROMPT,
            "search": SEARCH_AGENT_PROMPT,
            "comparison": COMPARE_AGENT_PROMPT,
            "objection_handling": SEARCH_AGENT_PROMPT,
            "recommendation": RECOMMEND_AGENT_PROMPT,
            "summary": RECOMMEND_AGENT_PROMPT,
        }

        self.graph = ShoppingGuideGraph(
            llm=llm,
            tools=tools,
            product_retriever=product_retriever,
            profile_store=profile_store,
            system_prompt=system_prompt,
            stage_classifier_prompt=STAGE_CLASSIFIER_PROMPT,
            max_tool_rounds=max_tool_rounds,
            stage_prompts=stage_prompts,
        )

    def run(self, question: str, conv_id: str = "default",
            chat_history: list = None) -> dict:
        """Run the shopping guide Agent for one conversation turn.

        Args:
            question: The user's latest message.
            conv_id: Conversation ID for profile persistence.
            chat_history: Optional list of prior LangChain messages.

        Returns:
            dict with keys: answer, stage, product_context, user_profile,
                           messages, tool_rounds
        """
        result = self.graph.run(
            user_message=question,
            conv_id=conv_id,
            chat_history=chat_history,
        )

        # Extract final AI response
        answer = ""
        for m in reversed(result["messages"]):
            if isinstance(m, AIMessage) and m.content:
                answer = m.content
                break

        return {
            "answer": answer,
            "stage": result["stage"],
            "product_context": result["product_context"],
            "user_profile": result["user_profile"],
            "messages": result["messages"],
            "tool_rounds": result["tool_rounds"],
        }

    def run_simple(self, question: str, conv_id: str = "default") -> dict:
        """Simplified single-shot mode: no LangGraph, one LLM call with tools.

        Useful for quick testing or when LangGraph is unavailable.
        """
        from langchain.prompts import ChatPromptTemplate
        from langchain.schema.output_parser import StrOutputParser

        user_profile = self.profile_store.serialize_profile(conv_id)

        # Quick product search
        try:
            product_context = self.product_retriever.retrieve(question, top_k=5)
        except Exception:
            product_context = "(产品检索暂不可用)"

        prompt = ChatPromptTemplate.from_messages([
            ("system", SHOPPING_SYSTEM_PROMPT),
            ("user", "{question}"),
        ])

        chain = prompt | self.llm | StrOutputParser()
        answer = chain.invoke({
            "conv_id": conv_id,
            "stage": "search",
            "user_profile": user_profile,
            "product_context": product_context,
            "question": question,
        })

        return {
            "answer": answer,
            "stage": "search",
            "product_context": product_context,
            "user_profile": user_profile,
            "messages": [],
            "tool_rounds": 0,
        }
