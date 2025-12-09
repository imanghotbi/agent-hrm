from typing import List, Optional
from pydantic import BaseModel, HttpUrl, EmailStr, Field
from enum import Enum

# ==================================================
# ENUMS
# ==================================================
class MaritalStatus(str, Enum):
    SINGLE = "Single"
    MARRIED = "Married"

class MilitaryServiceStatus(str, Enum):
    """Military service status for male Iranian candidates."""
    COMPLETED = "Completed"  
    EXEMPTED = "Exempted"  
    EDUCATION_EXEMPTION = "Educational Exemption" 
    SUBJECT_TO_SERVICE = "Subject to Service" 

class TierRank(str, Enum):
    TIER_1 = "Tier 1: Elite (Ivy League Equivalent)"
    TIER_2 = "Tier 2: Top Provincial/Specialized"
    TIER_3 = "Tier 3: Standard State & Top Azad"
    TIER_4 = "Tier 4: Mass Education"
    UNKNOWN = "Unknown/Foreign"
# ==================================================
# SUB-MODELS
# ==================================================

class PersonalInfo(BaseModel):
    """Basic candidate contact and demographic details."""
    full_name: Optional[str] = Field(None, description="Candidate's full legal name.")
    email: Optional[EmailStr] = Field(None, description="Primary email address.")
    phone_number: Optional[str] = Field(None, description="Mobile or landline number.")
    location: Optional[str] = Field(None, description="City and Country (e.g., Tehran, Iran).")
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    website: Optional[str] = None
    telegram_handle: Optional[str] = Field(None, pattern=r"^@?[a-zA-Z0-9_]{5,}$", description="Telegram username starting with @.")
    date_of_birth: Optional[str] = Field(None, description="YYYY-MM-DD or Jalali date string.")
    age: Optional[int] = Field(None, description="Age in years.")
    professional_summary: Optional[str] = Field(None, description="Brief intro or objective statement.")
    marital_status: Optional[MaritalStatus] = None              
    military_service_status: Optional[MilitaryServiceStatus] = None      
    gender: Optional[str] = None                      

class JobPreferences(BaseModel):
    """Candidate's job expectations."""
    expected_salary: Optional[str] = Field(None, description="Numeric value or range (e.g., '30-40M Tomans').")

class LanguageSkill(BaseModel):
    language: Optional[str] = Field(None, description="e.g., Persian, English, German.")
    level: Optional[str] = Field(None, description="e.g., Native, Fluent, Intermediate.")

class Skills(BaseModel):
    """Technical and soft skills."""
    hard_skills: Optional[List[str]] = Field(None, description="Technical tools, frameworks, programming languages.")
    soft_skills: Optional[List[str]] = Field(None, description="Interpersonal skills like 'Leadership', 'Communication'.")
    languages: Optional[List[LanguageSkill]] = None

class EducationEntry(BaseModel):
    """Details of a single educational qualification."""
    degree: Optional[str] = Field(None, description="B.Sc, M.Sc, PhD, Diploma, etc.")
    major: Optional[str] = Field(None, description="Field of study (e.g., Software Engineering).")
    school: Optional[str] = Field(None, description="University or Institution name.")
    university_tier: Optional[int] = Field(None, description="The prestige rank of the Iranian university on a scale of 1 to 4.Return null if the university is foreign or unknown.")
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[str] = Field(None, description="Grade Point Average (e.g., 18.5/20 or 3.8/4).")

class Education(BaseModel):
    items: Optional[List[EducationEntry]] = None

class ExperienceEntry(BaseModel):
    """Details of a single job position."""
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    employment_type: Optional[str] = Field(None, description="Full-time, Part-time, Contract, Remote.")
    location: Optional[str] = None
    company_tier: Optional[str] = Field(None, description="Inferred: Big Tech, Startup, Enterprise, Gov.")
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    extracted_skills: List[str] = Field(
        default_factory=list,
        description="Hard skills explicitly mentioned in this specific role's description."
    )
    technologies_used: Optional[List[str]] = Field(None, description="Tech stack used in this role.")

class Experience(BaseModel):
    items: Optional[List[ExperienceEntry]] = None

class Certification(BaseModel):
    certificate_name: Optional[str] = None
    issuer: Optional[str] = None
    issue_date: Optional[str] = None

class Certifications(BaseModel):
    items: Optional[List[Certification]] = None

class Project(BaseModel):
    """Academic or personal projects."""
    project_name: Optional[str] = None
    description: Optional[str] = None
    technologies: Optional[List[str]] = None
    github_repo: Optional[HttpUrl] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None

class Projects(BaseModel):
    items: Optional[List[Project]] = None

class Publication(BaseModel):
    title: Optional[str] = None
    publication_date: Optional[str] = None
    journal_or_conference: Optional[str] = None

class Publications(BaseModel):
    items: Optional[List[Publication]] = None

# ==================================================
# MASTER ROOT
# ==================================================
class ResumeData(BaseModel):
    """Root object for the parsed resume data."""
    personal_info: Optional[PersonalInfo] = None
    job_preferences: Optional[JobPreferences] = None
    skills: Optional[Skills] = None
    education: Optional[Education] = None
    work_experience: Optional[Experience] = None
    certifications: Optional[Certifications] = None
    projects: Optional[Projects] = None
    publications: Optional[Publications] = None
    resume_language: Optional[str] = Field(None, description="Language of the original resume (e.g., 'Persian', 'English').")

# ==================================================