from fastapi import FastAPI, HTTPException
from .pii_detector import PIIDetector
from .classifier import EmailClassifier
from .schemas import ClassificationRequest, ClassificationResponse, PIIDetection
import os

app = FastAPI(
    title="Email Classification API",
    description="API for classifying support emails with PII masking",
    version="1.0.0"
)

# Initialize components
pii_detector = PIIDetector()
classifier = EmailClassifier(os.path.join("models", "bert_classifier"))

@app.post("/classify", response_model=ClassificationResponse)
async def classify_email(request: ClassificationRequest):
    try:
        # Step 1: Detect PII
        detected_entities = pii_detector.detect_pii(request.email)
        
        # Step 2: Mask PII
        masked_email, processed_entities = pii_detector.mask_text(
            request.email, detected_entities
        )
        
        # Step 3: Classify email
        classification_result = classifier.classify(masked_email)
        
        # Prepare response
        response = {
            "input_email_body": request.email,
            "list_of_masked_entities": [
                {
                    "position": entity["position"],
                    "classification": entity["classification"],
                    "entity": entity["entity"]
                }
                for entity in processed_entities
            ],
            "masked_email": masked_email,
            "category_of_the_email": classification_result["category"]
        }
        
        return response
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing email: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
