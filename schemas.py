from pydantic import BaseModel
from typing import List, Dict, Optional

class PIIDetection(BaseModel):
    position: List[int]
    classification: str
    entity: str

class ClassificationRequest(BaseModel):
    email: str

class ClassificationResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[PIIDetection]
    masked_email: str
    category_of_the_email: str
