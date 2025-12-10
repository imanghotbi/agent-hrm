import asyncio
from langchain_core.messages import HumanMessage

from app.config.config import config
from app.config.logger import logger
from app.services.llm_factory import LLMFactory
from app.schemas.resume import ResumeData
from app.schemas.evaluation import ResumeEvaluation
from app.schemas.hiring import HiringRequirements
from utils.prompt import STRUCTURE_PROMPT_TEMPLATE, SCORING_PROMPT
from utils.helper import save_token_cost

class ResumeAnalyzerService:
    
    def __init__(self):
        self.struct_sem = asyncio.Semaphore(config.structure_workers)
        self.eval_sem = asyncio.Semaphore(config.eval_workers)

    async def structure_text(self, file_key: str, text: str , session_id:str) -> dict | None:
        if not text:
            return None
            
        async with self.struct_sem:
            prompt = STRUCTURE_PROMPT_TEMPLATE.format(raw_text=text)
            llm = LLMFactory.get_model(structured_output=ResumeData)
            
            for attempt in range(1, config.structure_max_retries+1):
                try:
                    response = await llm.ainvoke([HumanMessage(content=prompt)])
                    asyncio.create_task(save_token_cost('batch_structure_node', session_id , response))
                    data = response.model_dump(mode='json')
                    data["_source_file"] = file_key
                    logger.info(f"✅ [STRUCT DONE] {file_key}")
                    return data
                except Exception as e:
                    if attempt == 3:
                        logger.error(f"❌ [STRUCT FAILED] {file_key}: {e}")
                    else:
                        await asyncio.sleep(attempt)
            return None

    async def evaluate_resume(self, resume_dict: dict, reqs: HiringRequirements, session_id:str) -> dict | None:
        async with self.eval_sem:
            try:
                resume_obj = ResumeData(**resume_dict)
                prompt = SCORING_PROMPT.format(
                    requirements_json=reqs.model_dump_json(),
                    resume_json=resume_obj.model_dump_json()
                )
                
                llm = LLMFactory.get_model(structured_output=ResumeEvaluation)
                eval_result = await llm.ainvoke([HumanMessage(content=prompt)])
                asyncio.create_task(save_token_cost('batch_evaluate_node', session_id , eval_result))
                # Logic: Weighted Calculation
                w = reqs.weights
                s = eval_result
                
                weighted_sum = (
                    (s.hard_skills_score.score * w.hard_skills_weight) +
                    (s.experience_score.score * w.experience_weight) +
                    (s.education_score.score * w.education_weight) +
                    (s.soft_skills_score.score * w.soft_skills_weight) +
                    (s.military_status_score.score * w.military_status_weight) + 
                    (s.university_tier_score.score * w.university_tier_weight)
                )
                
                total_weight = sum([
                    w.hard_skills_weight, w.experience_weight, w.education_weight,
                    w.soft_skills_weight, w.military_status_weight, w.university_tier_weight
                ])
                
                final_score = round(weighted_sum / total_weight, 2) if total_weight > 0 else 0
                eval_result.final_weighted_score = final_score
                
                logger.info(f"⚖️ [EVAL DONE] {resume_dict.get('_source_file')} -> {final_score}")
                
                return {
                    "resume": resume_dict,
                    "evaluation": eval_result.model_dump(),
                    "final_score": final_score
                }
            except Exception as e:
                logger.error(f"❌ [EVAL ERROR] {resume_dict.get('_source_file')}: {e}")
                return None