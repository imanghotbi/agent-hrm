import logging
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import DESCENDING
from src.config import config

logger = logging.getLogger(__name__)

class MongoHandler:
    def __init__(self):
        self.client = AsyncIOMotorClient(config.mongo_uri)
        self.db = self.client[config.mongo_db_name]
        self.collection = self.db[config.mongo_collection]

    async def save_candidate(self, resume_data: dict, evaluation_data: dict):
        """Saves or updates a candidate."""
        # We use email or a hash as a unique identifier to avoid duplicates
        email = resume_data.get("personal_info", {}).get("email")
        if not email:
            # Fallback if no email: use filename or full name
            query = {"_source_file": resume_data.get("_source_file")}
        else:
            query = {"resume.personal_info.email": email}

        document = {
            "resume": resume_data,
            "evaluation": evaluation_data,
            "final_score": evaluation_data["final_weighted_score"] # Lifted to top level for easy sorting
        }
        
        await self.collection.update_one(query, {"$set": document}, upsert=True)
        logger.info(f"💾 Saved candidate to DB: {evaluation_data['final_weighted_score']:.1f}/100")

    async def get_top_candidates(self, limit: int = 5):
        """Retrieves top N candidates sorted by final_score."""
        cursor = self.collection.find().sort("final_score", DESCENDING).limit(limit)
        return await cursor.to_list(length=limit)

    async def execute_raw_query(self, query_dict: dict):
        """Executes a generated query (for the Q&A feature)."""

        cursor = self.collection.find(query_dict)
        return await cursor.to_list(length=10)