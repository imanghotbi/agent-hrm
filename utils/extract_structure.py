import pymongo
from typing import Dict
from bson import ObjectId
from datetime import datetime
from app.services.mongo_service import MongoHandler

mongo_db = MongoHandler()

class ExtractSchema:
    def __init__(self , mongo_uri , db_name , collection_name):
        self.client = pymongo.MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def get_random_doc(self):
        return self.collection.find_one()
    
    def generate_schema(self , doc):
        """
        Recursively traverses a document to create a schema representation
        containing data types instead of values.
        """

        if isinstance(doc, dict):
            return {k: self.generate_schema(v) for k, v in doc.items()}
        
        elif isinstance(doc, list):
            # If list is not empty, assume all items match the first item's structure
            if len(doc) > 0:
                return [self.generate_schema(doc[0])]
            else:
                return ["Array (Empty)"]
                
        elif isinstance(doc, ObjectId):
            return "ObjectId"
            
        elif isinstance(doc, datetime):
            return "Date (ISO)"
            
        else:
            # Return the type name (e.g., 'str', 'int', 'bool', 'float')
            return type(doc).__name__

async def save_token_cost(node_name:str , session_id:str , response) -> Dict:
    token_usage = response.response_metadata['token_usage']
    del token_usage['is_byok']
    data = {
        'node_name': node_name,
        'session_id': session_id,
        **token_usage
    }
    result = await mongo_db.save_doc(mongo_db.usage_logs , data)
    return result