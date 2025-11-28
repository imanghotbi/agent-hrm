from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import JsonOutputParser

from src.config import config, logger
from src.database import MongoHandler
from utils.prompt import  MONGO_QA_PROMPT


class MongoRetriver:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=config.model_name,
            google_api_key=config.google_api_key,
            temperature=0
        )
        self.mongo = MongoHandler()

    async def run_qa(self, user_question: str , db_structure) -> str:
        """
        Text-to-Query -> Execute -> Text-Answer
        """
        # Step 1: Generate Query
        try:
            query_parser = JsonOutputParser()
            query_msg = MONGO_QA_PROMPT.format(question=user_question  , structure=db_structure)
            
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