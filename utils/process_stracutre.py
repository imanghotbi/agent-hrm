from utils.date_calulator import DateCalculator
from typing import Dict

data_calculater  = DateCalculator()

def enrich_resume_with_durations(resume: Dict) -> Dict:
    """
    Iterates over the Resume Data , calculates durations for 
    Education, Experience, and  and updates the object in place.
    """
    fields = ['education' , 'work_experience']
    
    for field in fields:
        if resume['resume'][field] and resume['resume'][field]['items']:
            for item in resume['resume'][field]['items']:
                if item['start_date']:
                    item['duration_months'] = data_calculater.calculate_duration(item['start_date'], item['end_date'])

    return resume

def fix_age_field(resume:Dict) -> Dict:
    data_of_birth = resume['resume']['personal_info']['date_of_birth']
    del resume['resume']['personal_info']['date_of_birth']
    if resume['resume']['personal_info']['age']:
        return resume
    else:
        if data_of_birth:
            resume['resume']['personal_info']['age'] = data_calculater.calculate_age(data_of_birth)
            return resume
        return resume