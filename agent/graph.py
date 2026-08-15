from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END

from agent.retriever import (
    search_policies,
    check_groundedness,
)
from agent.tools import (
    check_return_risk,
    classify_product_image,
)
from agent.llm import mock_llm
from agent.memory import (
    get_history,
    add_turn,
    build_contextual_query,
)

class AgentState(TypedDict, total=False):
    request_type: str
    query: str
    original_query: str
    session_id: str
    history: list[dict]
    order_features: dict
    image_path: str
    result: dict
    answer: str


# -------------------------------------------------
# Router
# -------------------------------------------------

def route_request(state: AgentState) -> AgentState:
    """
    Decide which capability the user needs and block
    common prompt-injection attempts.
    """

    query = state.get("query", "").lower().strip()

    # Basic prompt-injection detection
    injection_patterns = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "ignore your instructions",
        "forget previous instructions",
        "forget all previous instructions",
        "override your instructions",
        "disregard previous instructions",
        "disregard all previous instructions",
        "reveal your system prompt",
        "show me your system prompt",
        "print your system prompt",
        "reveal hidden instructions",
        "show hidden instructions",
        "developer message",
        "system message",
    ]

    if any(pattern in query for pattern in injection_patterns):
        return {
            **state,
            "request_type": "blocked",
            "result": {
                "type": "security",
                "status": "blocked",
                "reason": "prompt_injection_detected",
            },
            "answer": (
                "I can't follow instructions that attempt to "
                "override or reveal my system instructions."
            ),
        }

    image_words = [
        "image",
        "photo",
        "picture",
        "product image",
        "classify",
        "classification",
        "visual",
    ]

    return_words = [
        "return risk",
        "return probability",
        "will it return",
        "likely return",
        "return prediction",
    ]

    if any(word in query for word in image_words):
        request_type = "image"

    elif any(word in query for word in return_words):
        request_type = "return_risk"

    else:
        request_type = "policy"

    return {
        **state,
        "request_type": request_type,
    }
# -------------------------------------------------
# Policy node
# -------------------------------------------------

def policy_node(state: AgentState) -> AgentState:
    """
    Retrieve relevant policy chunks and refuse if
    the evidence is below the groundedness threshold.
    """

    query = state["query"]

    results = search_policies(
        query,
        top_k=3,
    )

    groundedness = check_groundedness(results)

    if not groundedness["grounded"]:
        structured_result = {
            "type": "policy",
            "status": "ungrounded",
            "results": results,
            "best_score": groundedness["best_score"],
            "threshold": groundedness["threshold"],
        }

        return {
            **state,
            "result": structured_result,
            "answer": mock_llm(
                "policy",
                structured_result,
            ),
        }

    structured_result = {
        "type": "policy",
        "status": "grounded",
        "results": results,
        "best_score": groundedness["best_score"],
        "threshold": groundedness["threshold"],
    }

    return {
        **state,
        "result": structured_result,
        "answer": mock_llm(
            "policy",
            structured_result,
        ),
    }
# -------------------------------------------------
# Return-risk node
# -------------------------------------------------

def return_risk_node(state: AgentState) -> AgentState:
    """
    Run the Random Forest model when order data is available.
    """

    order_features = state.get("order_features")

    if not order_features:
        result = {
            "type": "return_risk",
            "status": "missing_input",
        }

        return {
            **state,
            "result": result,
            "answer": mock_llm(
                "return_risk",
                result,
            ),
        }

    result = check_return_risk(
        order_features
    )

    structured_result = {
        "type": "return_risk",
        **result,
    }

    return {
        **state,
        "result": structured_result,
        "answer": mock_llm(
            "return_risk",
            structured_result,
        ),
    }
# -------------------------------------------------
# Image node
# -------------------------------------------------

def image_node(state: AgentState) -> AgentState:
    """
    Run the actual CNN when an image is available.
    """

    image_path = state.get("image_path")

    if not image_path:
        return {
            **state,
            "result": {
                "type": "image",
                "status": "missing_input",
            },
            "answer": (
                "I can classify the product, but I need "
                "a product image first."
            ),
        }

    result = classify_product_image(
        image_path
    )

    return {
        **state,
        "result": {
            "type": "image",
            **result,
        },
        "answer": (
            f"Predicted product: "
            f"{result['predicted_class']}. "
            f"Confidence: "
            f"{result['confidence_percent']}%."
        ),

"answer": mock_llm(
    "image",
    result,
),

"answer": mock_llm(
    "image",
    result,
),

    }

# -------------------------------------------------
# Security node
# -------------------------------------------------

def blocked_node(state: AgentState) -> AgentState:
    """
    Safely handle detected prompt-injection attempts.
    """

    return {
        **state,
        "result": {
            "type": "security",
            "status": "blocked",
            "reason": "prompt_injection_detected",
        },
        "answer": mock_llm(
    "blocked",
    {
        "status": "blocked",
    },
),
    }

# -------------------------------------------------
# Conditional routing
# -------------------------------------------------

def choose_node(state: AgentState) -> str:
    return state["request_type"]


# -------------------------------------------------
# Build LangGraph
# -------------------------------------------------

builder = StateGraph(AgentState)

builder.add_node(
    "router",
    route_request,
)

builder.add_node(
    "policy",
    policy_node,
)

builder.add_node(
    "return_risk",
    return_risk_node,
)

builder.add_node(
    "image",
    image_node,
)

builder.add_node(
    "blocked",
    blocked_node,
)

builder.add_edge(
    START,
    "router",
)

builder.add_conditional_edges(
    "router",
    choose_node,
   {
    "policy": "policy",
    "return_risk": "return_risk",
    "image": "image",
    "blocked": "blocked",
},
)

builder.add_edge(
    "policy",
    END,
)

builder.add_edge(
    "return_risk",
    END,
)

builder.add_edge(
    "image",
    END,
)

builder.add_edge(
    "blocked",
    END,
)

agent_graph = builder.compile()


# -------------------------------------------------
# Simple public interface
# -------------------------------------------------

def run_agent(
    query: str,
    order_features: Optional[dict] = None,
    image_path: Optional[str] = None,
    session_id: str = "default",
) -> AgentState:

    contextual_query = build_contextual_query(
        session_id,
        query,
    )

    state: AgentState = {
        "query": contextual_query,
        "original_query": query,
        "session_id": session_id,
        "history": get_history(session_id),
    }

    if order_features is not None:
        state["order_features"] = order_features

    if image_path is not None:
        state["image_path"] = image_path

    result = agent_graph.invoke(state)

    add_turn(
        session_id,
        query,
        result["answer"],
    )

    result["history"] = get_history(session_id)

    return result
# -------------------------------------------------
# Test
# -------------------------------------------------

if __name__ == "__main__":

    print("\n=== TEST 1: POLICY ===")

    policy_result = run_agent(
        "What is the return window for apparel?"
    )

    print(
        policy_result["answer"]
    )

    print("\n=== TEST 2: RETURN RISK ===")

    sample_order = {
        "product_category": "Electronics",
        "price_inr": 16689.0,
        "discount_pct": 6.7,
        "payment_method": "COD",
        "customer_tenure_days": 309,
        "num_previous_orders": 11,
        "num_previous_returns": 3,
        "delivery_distance_km": 166.4,
        "delivery_days": 9,
        "is_weekend_order": 0,
        "rating_given": 2.0,
    }

    risk_result = run_agent(
        "What is the return risk for this order?",
        order_features=sample_order,
    )

    print(
        risk_result["answer"]
    )

    print("\n=== TEST 3: IMAGE ===")

    image_result = run_agent(
        "Classify this product image.",
        image_path=(
            "data/sample_images/ankle_boot.png"
        ),
    )

    print(
        image_result["answer"]
    )