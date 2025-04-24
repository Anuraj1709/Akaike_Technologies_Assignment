import re
from typing import List, Dict, Tuple
import spacy

class PIIDetector:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.patterns = {
            "full_name": [
                r"((?:[A-Z][a-z]+\s?){2,3})"
            ],
            "email": [
                r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})"
            ],
            "phone_number": [
                r"(\+?\d{1,3}[-.\s]?\(?\d{1,3}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,4})"
            ],
            "dob": [
                r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                r"([A-Z][a-z]+\s\d{1,2},\s\d{4})"
            ],
            "credit_debit_no": [
                r"(\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4})"
            ],
            "cvv_no": [
                r"(\b\d{3}\b)"
            ],
            "expiry_no": [
                r"(\d{2}/\d{2,4})"
            ]
        }

    def detect_pii(self, text: str) -> List[Dict]:
        detected_entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text):
                    start, end = match.span()
                    detected_entities.append({
                        "position": [start, end],
                        "classification": entity_type,
                        "entity": match.group()
                    })
        
        # Remove overlapping entities (keep the first detected)
        detected_entities = self._remove_overlaps(detected_entities)
        
        # Validate names using spaCy (reduce false positives)
        detected_entities = self._validate_names(text, detected_entities)
        
        return detected_entities

    def _remove_overlaps(self, entities: List[Dict]) -> List[Dict]:
        entities.sort(key=lambda x: x["position"][0])
        filtered = []
        prev_end = 0
        
        for entity in entities:
            start, end = entity["position"]
            if start >= prev_end:
                filtered.append(entity)
                prev_end = end
        
        return filtered

    def _validate_names(self, text: str, entities: List[Dict]) -> List[Dict]:
        doc = self.nlp(text)
        valid_names = {ent.text for ent in doc.ents if ent.label_ == "PERSON"}
        
        filtered_entities = []
        for entity in entities:
            if entity["classification"] != "full_name":
                filtered_entities.append(entity)
            elif entity["entity"] in valid_names:
                filtered_entities.append(entity)
        
        return filtered_entities

    def mask_text(self, text: str, entities: List[Dict]) -> Tuple[str, List[Dict]]:
        masked_text = text
        offset = 0
        processed_entities = []
        
        for entity in sorted(entities, key=lambda x: x["position"][0]):
            start, end = entity["position"]
            original_entity = entity["entity"]
            entity_type = entity["classification"]
            
            # Adjust positions based on previous masking
            adj_start = start + offset
            adj_end = end + offset
            
            # Create mask
            mask = f"[{entity_type}]"
            
            # Apply mask
            masked_text = masked_text[:adj_start] + mask + masked_text[adj_end:]
            
            # Update offset
            offset += len(mask) - (end - start)
            
            # Store processed entity with original positions
            processed_entities.append({
                "position": [start, end],
                "classification": entity_type,
                "entity": original_entity
            })
        
        return masked_text, processed_entities
