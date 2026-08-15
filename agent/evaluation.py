"""
Evaluation suite for the Flipkart Order Intelligence agent.
"""

from agent.retriever import search_policies


# -------------------------------------------------
# Retrieval evaluation dataset
# -------------------------------------------------

EVALUATION_CASES = [
    {
        "query": "What is the return window for apparel?",
        "expected_titles": [
            "Apparel and Footwear Return Window",
        ],
    },
    {
        "query": "What is the return window for electronics?",
        "expected_titles": [
            "Electronics Return Window",
        ],
    },
    {
        "query": "What are the conditions for returning a damaged product?",
        "expected_titles": [
            "Damaged Product",
            "Return Condition",
        ],
    },
    {
        "query": "Can I cancel an order?",
        "expected_titles": [
            "Order Cancellation",
        ],
    },
    {
        "query": "What happens if delivery is delayed?",
        "expected_titles": [
            "Delayed Delivery",
            "Delivery SLA",
        ],
    },
]


def evaluate_retrieval(top_k: int = 3):
    """
    Calculate Precision@K and Recall@K.
    """

    precision_scores = []
    recall_scores = []

    print("\n=== RETRIEVAL EVALUATION ===")

    for case in EVALUATION_CASES:

        results = search_policies(
            case["query"],
            top_k=top_k,
        )

        retrieved_titles = [
            result["title"]
            for result in results
        ]

        expected = set(
            case["expected_titles"]
        )

        retrieved = set(
            retrieved_titles
        )

        relevant_retrieved = (
            expected & retrieved
        )

        precision = (
            len(relevant_retrieved)
            / len(retrieved)
            if retrieved
            else 0.0
        )

        recall = (
            len(relevant_retrieved)
            / len(expected)
            if expected
            else 0.0
        )

        precision_scores.append(
            precision
        )

        recall_scores.append(
            recall
        )

        print(
            f"\nQuery: {case['query']}"
        )

        print(
            f"Retrieved: {retrieved_titles}"
        )

        print(
            f"Precision@{top_k}: "
            f"{precision:.2f}"
        )

        print(
            f"Recall@{top_k}: "
            f"{recall:.2f}"
        )

    mean_precision = (
        sum(precision_scores)
        / len(precision_scores)
    )

    mean_recall = (
        sum(recall_scores)
        / len(recall_scores)
    )

    print("\n=== SUMMARY ===")

    print(
        f"Mean Precision@{top_k}: "
        f"{mean_precision:.2f}"
    )

    print(
        f"Mean Recall@{top_k}: "
        f"{mean_recall:.2f}"
    )

    return {
        "precision_at_k": mean_precision,
        "recall_at_k": mean_recall,
    }


if __name__ == "__main__":
    evaluate_retrieval()