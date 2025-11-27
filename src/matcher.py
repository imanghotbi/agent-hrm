import json
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from src.config import config, logger
from src.schema import HiringRequirements, ResumeData, ResumeEvaluation, ScoredResume
from src.database import MongoHandler
from utils.prompt import SCORING_PROMPT , MONGO_QA_PROMPT


class MatcherEngine:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key,
            temperature=0
        )
        self.mongo = MongoHandler()

    async def evaluate_resume(self, resume: Dict, reqs: HiringRequirements) -> Dict:
        """
        1. Asks LLM for category scores (0-100).
        2. Calculates weighted average using User's weights.
        """
        # Prepare context
        req_json = reqs.model_dump_json()
        res_json = json.dumps(resume, default=str) # handle dates
        
        # 1. Get Category Scores from LLM
        prompt = SCORING_PROMPT.format(requirements_json=req_json, resume_json=res_json)
        
        # to ensure the LLM gives us the components.
        structured_llm = self.llm.with_structured_output(ResumeEvaluation) 
        
        # Note: We rely on the LLM to fill the sub-models. 
        # For the 'final_weighted_score', we will overwrite it mathematically to be safe.
        eval_result: ResumeEvaluation = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        
        # 2. Mathematical Calculation (Weighted Average)
        weights = reqs.weights
        
        w_skills = weights.hard_skills_weight
        w_exp = weights.experience_weight
        w_edu = weights.education_weight
        w_soft = weights.soft_skills_weight
        w_mil = weights.military_status_weight
        
        total_weight = w_skills + w_exp + w_edu + w_soft + w_mil
        
        # Calculate weighted sum
        weighted_sum = (
            (eval_result.hard_skills_score.score * w_skills) +
            (eval_result.experience_score.score * w_exp) +
            (eval_result.education_score.score * w_edu) +
            (eval_result.soft_skills_score.score * w_soft) +
            (eval_result.military_status_score.score * w_mil)
        )
        
        final_score = weighted_sum / total_weight
        
        # Update the model
        eval_result.final_weighted_score = round(final_score, 2)
        
        return eval_result.model_dump()

    async def run_qa(self, user_question: str) -> str:
        """
        Text-to-Query -> Execute -> Text-Answer
        """
        # Step 1: Generate Query
        try:
            query_parser = JsonOutputParser()
            query_msg = MONGO_QA_PROMPT.format(question=user_question)
            
            # Helper to get raw JSON
            raw_response = await self.llm.ainvoke([HumanMessage(content=query_msg)])
            query_dict = query_parser.parse(raw_response.content)
            
            logger.info(f"🔍 Generated Mongo Query: {query_dict}")
            
            # Step 2: Execute
            results = await self.mongo.execute_raw_query(query_dict)
            
            if not results:
                return "I searched the database but found no matching candidates."
            
            # Step 3: Synthesize Answer
            # We summarize the findings for the LLM to describe
            summary = [
                f"- Name: {r['resume']['personal_info'].get('full_name', 'Unknown')}, "
                f"Score: {r.get('final_score')}, "
                f"Location: {r['resume']['personal_info'].get('location')}"
                for r in results
            ]
            
            ans_prompt = (
                f"User Question: {user_question}\n\n"
                f"Database Results:\n{str(summary)}\n\n"
                "Answer the user's question based on these results in Persian."
            )
            
            final_ans = await self.llm.ainvoke([HumanMessage(content=ans_prompt)])
            return final_ans.content

        except Exception as e:
            logger.error(f"QA Failed: {e}")
            return "Sorry, I could not process that question."