from typing import Optional
from pydantic import BaseModel, Field
from src.schemas.resume import ResumeData

# ==================================================
# SUB-MODELS
# ==================================================
class CategoryScore(BaseModel):
    score: int = Field(..., description="Score from 0 to 100")
    reasoning: str = Field(..., description="Short explanation")

class ResumeEvaluation(BaseModel):
    hard_skills_score: CategoryScore
    experience_score: CategoryScore
    education_score: CategoryScore
    university_tier_score: CategoryScore
    soft_skills_score: CategoryScore
    military_status_score: CategoryScore
    
    final_weighted_score: float = Field(..., description="Final calculated score")
    summary_explanation: str

# ==================================================
# Resume Evaluator
# ==================================================
class ScoredResume(BaseModel):
    # This is the final object we save to Mongo
    resume: ResumeData
    evaluation: ResumeEvaluation
    final_score: float