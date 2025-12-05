from typing import List, Optional
from langchain_core.tools import tool
from app.schemas.hiring import SeniorityLevel, PriorityWeights
from app.schemas.job_description import (
    EducationLevel, WorkMode, EmploymentType, MilitaryServiceRequirement, 
    Language, SalaryRange
)
from enum import Enum

class Path(str, Enum):
    REVIEW = "REVIEW"
    WRITE = "WRITE"
    COMPARE = "COMPARE"

class AgentTools:
    
    @staticmethod
    @tool
    def router_tool(path: Path):
        """Call this tool ONLY Once you understand the user's intent (REVIEW, WRITE, COMPARE) and which of the paths it needs."""
        return "Route determined."

    @staticmethod
    @tool
    def submit_hiring_requirements(
        role_title: str, 
        seniority: SeniorityLevel, 
        essential_hard_skills: List[str], 
        military_service_required: bool, 
        min_experience_years: int, 
        education_level: str,
        weights: PriorityWeights,
        university_tier: int,
        nice_to_have_skills: Optional[List[str]] = None,
        language_proficiency: Optional[str] = None,
        **kwargs
    ):
        """Call this tool ONLY when ALL hiring requirements are gathered."""
        return "Requirements captured."

    @staticmethod
    @tool
    def submit_jd_requirements(
        job_title: str, 
        seniority_level: SeniorityLevel, 
        location: str, 
        education_level: EducationLevel,
        study_fields: List[str],
        work_mode: WorkMode,
        employment_type: EmploymentType,
        military_service: MilitaryServiceRequirement, 
        min_experience_years: int, 
        days_and_hours: str,
        hard_skills: List[str],
        soft_skills: List[str],
        responsibilities: List[str],
        target_language: Language,
        benefits: List[str],
        salary: Optional[SalaryRange] = None,
        advantage_skills: Optional[List[str]] = None,
        **kwargs
    ):
        """Call this tool ONLY when ALL JD requirements are gathered."""
        return "JD Requirements captured."