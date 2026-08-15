from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from agent.graph import run_agent


router = APIRouter(
    prefix="/agent",
    tags=["Support Agent"],
)


class AgentRequest(BaseModel):
    query: str
    order_features: Optional[dict] = None
    image_path: Optional[str] = None
    session_id: str = "default"


class AgentResponse(BaseModel):
    answer: str
    request_type: str
    result: dict


@router.post("/chat", response_model=AgentResponse)
def agent_chat(request: AgentRequest):

    state = run_agent(
         query=request.query,
           order_features=request.order_features,
              image_path=request.image_path,
                  session_id=request.session_id,
    )

    return AgentResponse(
        answer=state.get(
            "answer",
            "I could not generate a response.",
        ),
        request_type=state.get(
            "request_type",
            "unknown",
        ),
        result=state.get(
            "result",
            {},
        ),
    )