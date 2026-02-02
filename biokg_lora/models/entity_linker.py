"""
Entity linker for identifying biological entities in text.

Used to:
1. Annotate training data with entity mentions
2. Link entities to KG IDs for augmentation
"""

import logging
import re
from typing import Dict, List, Optional, Set, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional spacy import
try:
    import spacy
    from spacy.matcher import PhraseMatcher
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    logger.warning("spaCy not installed. Entity linking will use pattern matching only. "
                   "Install with: pip install spacy && python -m spacy download en_core_web_sm")


class EntityLinker:
    """
    Links biological entities in text to KG entity IDs.
    
    Uses:
    - Dictionary matching for gene names, phenotypes, GO terms
    - Pattern matching for IDs (MP:XXXXXXX, GO:XXXXXXX)
    - SpaCy NER for general biomedical entities
    """
    
    def __init__(
        self,
        entity2id: Dict[str, int],
        use_scispacy: bool = False,
    ):
        """
        Args:
            entity2id: Mapping from entity names to KG IDs
            use_scispacy: Whether to use scispacy models (slower but more accurate)
        """
        self.entity2id = entity2id
        self.id2entity = {v: k for k, v in entity2id.items()}
        
        # Load spaCy model
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                if use_scispacy:
                    self.nlp = spacy.load("en_core_sci_md")
                    logger.info("Loaded scispacy model: en_core_sci_md")
                else:
                    self.nlp = spacy.load("en_core_web_sm")
                    logger.info("Loaded spaCy model: en_core_web_sm")
            except OSError:
                logger.warning("SpaCy model not found. Using pattern matching only.")
        else:
            logger.info("SpaCy not installed. Using pattern matching only.")
        
        # Build phrase matcher for entity names
        if self.nlp:
            self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            patterns = [self.nlp.make_doc(name.lower()) for name in entity2id.keys()]
            self.matcher.add("ENTITIES", patterns)
        
        # Compile regex patterns for IDs
        self.patterns = {
            "mp_id": re.compile(r"MP:\d{7}"),
            "go_id": re.compile(r"GO:\d{7}"),
            "kegg_id": re.compile(r"[a-z]{2,4}\d{5}"),
        }
        
        logger.info(f"Initialized EntityLinker with {len(entity2id)} entities")
    
    def link_entities(
        self,
        text: str,
        return_char_spans: bool = True,
    ) -> List[Tuple[str, int, int, Optional[int]]]:
        """
        Find and link entities in text.
        
        Args:
            text: Input text
            return_char_spans: If True, return character spans
        
        Returns:
            List of (entity_name, start, end, entity_id) tuples
            If entity not in KG, entity_id is None
        """
        entities = []
        
        # 1. Pattern matching for IDs
        for pattern_name, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                entity_name = match.group(0)
                if entity_name in self.entity2id:
                    entities.append((
                        entity_name,
                        match.start(),
                        match.end(),
                        self.entity2id[entity_name]
                    ))
        
        # 2. Dictionary matching via PhraseMatcher
        if self.nlp and self.matcher:
            doc = self.nlp(text)
            matches = self.matcher(doc)
            
            for match_id, start, end in matches:
                span = doc[start:end]
                entity_name_lower = span.text.lower()
                
                # Find matching entity (case-insensitive)
                entity_id = None
                for name, eid in self.entity2id.items():
                    if name.lower() == entity_name_lower:
                        entity_id = eid
                        entity_name = name
                        break
                
                if entity_id is not None:
                    entities.append((
                        entity_name,
                        span.start_char,
                        span.end_char,
                        entity_id
                    ))
        
        # Remove duplicates (keep longest span)
        entities = self._remove_overlapping(entities)
        
        return entities
    
    def _remove_overlapping(
        self,
        entities: List[Tuple[str, int, int, Optional[int]]]
    ) -> List[Tuple[str, int, int, Optional[int]]]:
        """
        Remove overlapping entity spans, keeping longer ones.
        """
        if not entities:
            return []
        
        # Sort by start position, then by length (descending)
        entities = sorted(entities, key=lambda x: (x[1], -(x[2] - x[1])))
        
        non_overlapping = []
        last_end = -1
        
        for entity in entities:
            start = entity[1]
            end = entity[2]
            
            if start >= last_end:
                non_overlapping.append(entity)
                last_end = end
        
        return non_overlapping
    
    def annotate_qa_pair(
        self,
        question: str,
        answer: str,
    ) -> Dict[str, any]:
        """
        Annotate a QA pair with entity mentions.
        
        Args:
            question: Question text
            answer: Answer text
        
        Returns:
            Dict with:
                - question: Original question
                - answer: Original answer
                - question_entities: List of entities in question
                - answer_entities: List of entities in answer
                - all_entities: Set of all unique entity IDs
        """
        question_entities = self.link_entities(question)
        answer_entities = self.link_entities(answer)
        
        # Get unique entity IDs
        all_entity_ids = set()
        for entities in [question_entities, answer_entities]:
            for _, _, _, eid in entities:
                if eid is not None:
                    all_entity_ids.add(eid)
        
        return {
            "question": question,
            "answer": answer,
            "question_entities": question_entities,
            "answer_entities": answer_entities,
            "all_entity_ids": list(all_entity_ids),
        }


def create_simple_linker(entity_names: List[str]) -> EntityLinker:
    """
    Create a simple entity linker from a list of entity names.
    
    Useful for testing without loading full KG.
    
    Args:
        entity_names: List of entity names
    
    Returns:
        EntityLinker instance
    """
    entity2id = {name: i for i, name in enumerate(entity_names)}
    return EntityLinker(entity2id, use_scispacy=False)


if __name__ == "__main__":
    # Test entity linker
    entity_names = ["Thbd", "Bmp4", "Fgfr2", "MP:0003350", "GO:0007596"]
    entity2id = {name: i for i, name in enumerate(entity_names)}
    
    linker = EntityLinker(entity2id, use_scispacy=False)
    
    # Test text
    text = "The gene Thbd causes phenotype MP:0003350 (renal infarct) and is involved in GO:0007596."
    
    entities = linker.link_entities(text)
    
    print(f"Text: {text}")
    print(f"\nFound {len(entities)} entities:")
    for name, start, end, eid in entities:
        print(f"  {name:15s} [{start:3d}:{end:3d}] -> ID {eid}")
    
    # Test QA annotation
    question = "What phenotype does Thbd cause?"
    answer = "Thbd causes MP:0003350 (renal infarct)."
    
    annotated = linker.annotate_qa_pair(question, answer)
    print(f"\nQuestion entities: {len(annotated['question_entities'])}")
    print(f"Answer entities: {len(annotated['answer_entities'])}")
    print(f"Unique entity IDs: {annotated['all_entity_ids']}")
