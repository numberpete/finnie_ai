from pydantic import BaseModel, Field
from typing import List

class ChartArtifact(BaseModel):
    title: str
    filename: str

class AgentResponse(BaseModel):
    agent: str = Field(..., description="Name of the responding agent")
    message: str = Field(..., description="Primary textual response")
    charts: List[ChartArtifact] = Field(default_factory=list)
