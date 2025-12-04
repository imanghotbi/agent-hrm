from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from src.schemas.hiring import SeniorityLevel
# --- Enums ---
class WorkMode(str, Enum):
    ON_SITE = "On-site"
    REMOTE = "Remote"
    HYBRID = "Hybrid"

class EmploymentType(str, Enum):
    FULL_TIME = "Full-time"
    PART_TIME = "Part-time"

class Currency(str, Enum):
    TOMAN = "Toman"
    RIAL = "Rial"
    USD = "USD"
    EUR = "Euro"

class MilitaryServiceRequirement(str, Enum):
    """Specific to the Iranian Labor Market for male candidates."""
    REQUIRED = "Finished Service or Permanent Exemption Required"
    NOT_IMPORTANT = "Not Important"
    Educational_Exemption = "Educational Exemption Accepted"

class Language(str, Enum):
    PERSIAN = "Persian"
    ENGLISH = "English"

class EducationLevel(str, Enum):
    DIPLOMA = "Diploma"
    ASSOCIATE = "Associate Degree"
    BACHELOR = "Bachelor's Degree"
    MASTER = "Master's Degree"
    PHD = "PhD"
    NOT_IMPORTANT = "Not Important"

# --- Sub-Models ---
class SalaryRange(BaseModel):
    min_amount: Optional[int] = Field(None, description="Minimum salary amount")
    max_amount: Optional[int] = Field(None, description="Maximum salary amount")

# --- Main Data Model ---
class JobDescriptionRequest(BaseModel):
    """
    Data model to capture all requirements for writing a job description
    tailored for the Iranian market.
    """
    # Core Info
    job_title: str = Field(..., min_length=2, description="The official title of the role")
    target_language: Language = Field(default=Language.PERSIAN, description="Language to write the JD in")
    
    # Field and education
    education_level: EducationLevel = Field(
        default=EducationLevel.BACHELOR, 
        description="Minimum education degree required"
    )
    study_fields: List[str] = Field(
        default=[], 
        description="Preferred fields of study/majors (e.g., 'Computer Science', 'Marketing')"
    )

    # Location & Work Type
    location: str = Field(..., description="City and neighborhood (e.g., Tehran, Vanak)")
    work_mode: WorkMode = Field(..., description="Remote, Hybrid, or On-site")
    employment_type: EmploymentType = Field(..., description="Full-time, Part-time, etc.")
    
    # Seniority & Schedule
    seniority_level: SeniorityLevel = Field(..., description="The required job level")
    min_experience_years: int = Field(..., ge=0, description="Minimum years of experience required")
    days_and_hours: str = Field(
        default="Saturday to Wednesday, 09:00 to 18:00", 
        description="Working schedule (Standard Iran week starts Saturday)"
    )
    
    # Skills & Duties
    hard_skills: List[str] = Field(..., min_items=1, description="Technical skills required")
    soft_skills: List[str] = Field(..., min_items=1, description="Behavioral traits")
    advantage_skills: List[str] = Field(default=[], description="Nice-to-have skills")
    responsibilities: List[str] = Field(..., min_items=1, description="Key daily duties")
    
    # Benefits & Salary
    benefits: List[str] = Field(
        default=["Bimeh Tamin Ejtemaei (Social Security)", "Bimeh Takmili (Supplementary Insurance)"], 
        description="Perks and benefits"
    )
    salary: Optional[SalaryRange] = Field(None, description="Salary expectations")
    
    # Local Compliance
    military_service: MilitaryServiceRequirement = Field(
        default=MilitaryServiceRequirement.REQUIRED,
        description="Military service status for male applicants"
    )
