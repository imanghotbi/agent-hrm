from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum

# ==================================================
# ENUMS
# ==================================================
class SeniorityLevel(str, Enum):
    INTERN = "Intern"
    JUNIOR = "Junior"
    MID_LEVEL = "Mid-Level"
    SENIOR = "Senior"
    LEAD = "Lead"
    MANAGER = "Manager"

# ==================================================
# SUB-MODELS
# ==================================================
class PriorityWeights(BaseModel):
    hard_skills_weight: int = Field(..., ge=0, le=10)
    experience_weight: int = Field(..., ge=0, le=10)
    education_weight: int = Field(..., ge=0, le=10)
    soft_skills_weight: int = Field(..., ge=0, le=10)
    university_tier_weight: int = Field(..., ge=0, le=10)
    military_status_weight: int = Field(default=5, ge=0, le=10)

# ==================================================
# Recruitment requirements
# ==================================================
class HiringRequirements(BaseModel):
    role_title: str
    seniority: SeniorityLevel
    military_service_required: bool = True
    essential_hard_skills: List[str]
    nice_to_have_skills: List[str] = []
    soft_skills: Optional[List[str]] = []
    min_experience_years: int = 0
    education_level: Optional[str] = None
    university_tier: int
    weights: PriorityWeights