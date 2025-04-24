# Email Classification System with PII Masking

## Overview
This system classifies support emails while masking personally identifiable information (PII). It uses:
- Regex and NLP for PII detection
- Fine-tuned BERT model for classification
- FastAPI for the REST interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/email-classification-system.git
cd email-classification-system

2. Install dependencies:
pip install -r requirements.txt
python -m spacy download en_core_web_sm
