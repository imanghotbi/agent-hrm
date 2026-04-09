from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import DESCENDING
from app.config.config import config
from app.config.logger import logger
from utils.process_stracutre import enrich_resume_with_durations , fix_age_field


class MongoHandler:
    def __init__(self):
        self.client = AsyncIOMotorClient(config.mongo_uri)
        self.db = self.client[config.mongo_db_name]
        self.collection = self.db[config.mongo_collection]
        self.usage_logs = self.db[config.mongo_db_usage]

    async def save_candidate(self, resume_data: dict):
        """Saves or updates a candidate."""
        if not isinstance(resume_data, dict):
            logger.warning("Skipping candidate save: payload is not a dict.")
            return False

        resume = resume_data.get("resume")
        if not isinstance(resume, dict):
            logger.warning("Skipping candidate save: missing or invalid 'resume' object.")
            return False

        personal_info = resume.get("personal_info")
        if personal_info is None:
            personal_info = {}
        if not isinstance(personal_info, dict):
            logger.warning("Skipping candidate save: 'personal_info' is not a dict.")
            personal_info = {}

        # We use email or a hash as a unique identifier to avoid duplicates
        email = personal_info.get("email")
        if not email:
            # Fallback if no email: use filename or full name
            source_file = resume.get("_source_file")
            if not source_file:
                logger.warning("Skipping candidate save: neither email nor _source_file is available.")
                return False
            query = {"_source_file": source_file}
        else:
            query = {"resume.personal_info.email": email}
        resume_data = enrich_resume_with_durations(resume_data)
        resume_data = fix_age_field(resume_data)
        await self.collection.update_one(query, {"$set": resume_data}, upsert=True)
        score = resume_data.get("final_score")
        if isinstance(score, (int, float)):
            logger.info(f"💾 Saved candidate to DB: {score:.1f}/100")
        else:
            logger.info("💾 Saved candidate to DB.")
        return True

    async def get_top_candidates(self, limit: int = 5):
        """Retrieves top N candidates sorted by final_score."""
        cursor = self.collection.find().sort("final_score", DESCENDING).limit(limit)
        return await cursor.to_list(length=limit)

    async def save_doc(self,collection_name,data):
        result = await collection_name.insert_one(data)
        return result

    async def execute_raw_query(self, query: dict , projection: dict = None):
        """Executes a generated query (for the Q&A feature)."""

        cursor = self.collection.find(query , projection)
        return await cursor.to_list(length=10)
