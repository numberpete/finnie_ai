from pydantic import BaseModel, Field
from typing import List, Optional, Dict

class ChartArtifact(BaseModel):
    title: str
    filename: str

class AgentResponse(BaseModel):
    agent: str = Field(..., description="Name of the responding agent")
    message: str = Field(..., description="Primary textual response")
    charts: List[ChartArtifact] = Field(default_factory=list, description="Chart Artifacts containing chart metadata")
    portfolio: Optional[Dict] = None 