"""
Deterministic mock LLM response layer.
"""


def mock_llm(
    request_type: str,
    result: dict,
) -> str:

    # ---------------------------------------------
    # Security
    # ---------------------------------------------

    if request_type == "blocked":
        return (
            "I can't follow instructions that attempt to "
            "override or reveal my system instructions."
        )

    # ---------------------------------------------
    # Policy
    # ---------------------------------------------

    if request_type == "policy":

        if result.get("status") == "ungrounded":
            return (
                "I don't have enough policy information to "
                "answer that question reliably."
            )

        results = result.get("results", [])

        if not results:
            return (
                "I could not find a relevant policy."
            )

        best = results[0]

        return (
            f"{best['text']} "
            f"(Source: {best['title']})"
        )

    # ---------------------------------------------
    # Return risk
    # ---------------------------------------------

    if request_type == "return_risk":

        if result.get("status") == "missing_input":
            return (
                "I can calculate the return risk, but I need "
                "the order details first."
            )

        probability = result.get(
            "return_probability_percent"
        )

        prediction = result.get(
            "prediction"
        )

        if probability is None or prediction is None:
            return (
                "I could not calculate the return risk."
            )

        return (
            f"Return probability: {probability}%. "
            f"Prediction: {prediction}."
        )

    # ---------------------------------------------
    # Image
    # ---------------------------------------------

    if request_type == "image":

        if result.get("status") == "missing_input":
            return (
                "I can classify the product, but I need "
                "a product image first."
            )

        predicted_class = result.get(
            "predicted_class"
        )

        confidence = result.get(
            "confidence_percent"
        )

        if predicted_class is None or confidence is None:
            return (
                "I could not classify the product image."
            )

        return (
            f"Predicted product: {predicted_class}. "
            f"Confidence: {confidence}%."
        )

    return (
        "I could not generate a response for this request."
    )