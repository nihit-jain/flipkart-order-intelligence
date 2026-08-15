"""
Simple in-memory conversation state for the agent.
"""

from typing import Optional


_sessions: dict[str, list[dict]] = {}


def get_history(session_id: str) -> list[dict]:
    """Return conversation history for a session."""
    return _sessions.get(session_id, [])


def add_turn(
    session_id: str,
    user_message: str,
    assistant_message: str,
) -> None:
    """Store one conversation turn."""

    if session_id not in _sessions:
        _sessions[session_id] = []

    _sessions[session_id].append(
        {
            "role": "user",
            "content": user_message,
        }
    )

    _sessions[session_id].append(
        {
            "role": "assistant",
            "content": assistant_message,
        }
    )


def clear_session(session_id: str) -> None:
    """Delete a conversation session."""
    _sessions.pop(session_id, None)


def build_contextual_query(
    session_id: str,
    query: str,
) -> str:
    """
    Combine a short follow-up with the previous user
    message so retrieval has conversational context.
    """

    history = get_history(session_id)

    if not history:
        return query

    previous_user_messages = [
        turn["content"]
        for turn in history
        if turn["role"] == "user"
    ]

    if not previous_user_messages:
        return query

    follow_up_words = [
        "that",
        "this",
        "it",
        "they",
        "them",
        "after that",
        "what about",
        "and after",
        "how about",
    ]

    query_lower = query.lower()

    is_follow_up = (
        len(query.split()) <= 10
        or any(
            phrase in query_lower
            for phrase in follow_up_words
        )
    )

    if not is_follow_up:
        return query

    previous = previous_user_messages[-1]

    return f"{previous} Follow-up question: {query}"