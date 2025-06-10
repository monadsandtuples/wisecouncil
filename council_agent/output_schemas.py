from pydantic import BaseModel, Field
from typing import List, Optional

class ExpertProfile(BaseModel):
    prompt: str = Field(description="The role/title of the expert (e.g., 'Market Research Analyst')")
    assigned_section_title: str = Field(description="The section title this expert is assigned to write")

class OutlineExpertOutput(BaseModel):
    report_title: str = Field(description="A compelling title for the business report")
    outline_sections: List[str] = Field(description="List of exactly 6 section titles for the report outline")
    experts: List[ExpertProfile] = Field(description="List of exactly 6 expert prompts, each assigned to one section")

class ConsistencyIssue(BaseModel):
    section: str = Field(description="The section title where the issue was found, or 'Overall Report' if general")
    description: str = Field(description="Detailed description of the inconsistency")

class ConsistencyCheckOutput(BaseModel):
    consistent: bool = Field(description="Whether the report is consistent (True) or has issues (False)")
    issues: Optional[List[ConsistencyIssue]] = Field(default=None, description="List of consistency issues if any exist")

class SectionQualityOutput(BaseModel):
    status: str = Field(description="Quality status: 'reasonable' or 'needs_revision'")
    critique: Optional[str] = Field(default=None, description="Actionable feedback if revision is needed, or None if reasonable")
