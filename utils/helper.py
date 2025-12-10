from app.services.mongo_service import MongoHandler
from typing import Dict

mongo_db = MongoHandler()

def candidate_summary(top_candidate_resume) -> str:
    text = ""
    for index , candidate in enumerate(top_candidate_resume):
        text += f"Candidate {index + 1}:\n"
        text += f"Name: {candidate['resume']['personal_info'].get('full_name')}\n"
        text += f"email: {candidate['resume']['personal_info'].get('email')}\n"
        for keys in candidate["evaluation"].keys():
            if keys == "summary_explanation":
                text += f"{keys}: {candidate['evaluation'][keys]}\n"
            elif keys != "final_weighted_score":
                text += f"{keys}: {candidate['evaluation'][keys]["reasoning"]}\n"
        text += "=" * 10 + "\n"
    return text

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