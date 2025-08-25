"""
Knowledge Graph for Enhanced RAG System.

This module implements a knowledge graph to track entities and their relationships
across the book content, enabling more contextual and coherent retrieval.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json
import pickle
from pathlib import Path
import re
from enum import Enum
import networkx as nx

# Optional imports
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class EntityType(Enum):
    """Types of entities in the knowledge graph."""
    PERSON = "person"
    LOCATION = "location"
    ORGANIZATION = "organization"
    EVENT = "event"
    CONCEPT = "concept"
    TIME = "time"
    OBJECT = "object"
    CHAPTER = "chapter"
    SCENE = "scene"


class RelationType(Enum):
    """Types of relationships between entities."""
    # Character relationships
    KNOWS = "knows"
    LOVES = "loves"
    HATES = "hates"
    FAMILY = "family"
    FRIEND = "friend"
    ENEMY = "enemy"
    WORKS_WITH = "works_with"
    
    # Location relationships
    LOCATED_IN = "located_in"
    TRAVELS_TO = "travels_to"
    FROM = "from"
    LIVES_IN = "lives_in"
    
    # Event relationships
    PARTICIPATES_IN = "participates_in"
    CAUSES = "causes"
    RESULTS_IN = "results_in"
    HAPPENS_BEFORE = "happens_before"
    HAPPENS_AFTER = "happens_after"
    HAPPENS_DURING = "happens_during"
    
    # Structural relationships
    APPEARS_IN = "appears_in"
    MENTIONED_IN = "mentioned_in"
    PART_OF = "part_of"
    CONTAINS = "contains"
    
    # Semantic relationships
    RELATED_TO = "related_to"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    IS_A = "is_a"


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    name: str
    type: EntityType
    description: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[np.ndarray] = None
    mentions: List[Dict[str, Any]] = field(default_factory=list)
    importance: float = 0.0


@dataclass
class Relationship:
    """Represents a relationship between entities."""
    source_id: str
    target_id: str
    type: RelationType
    weight: float = 1.0
    attributes: Dict[str, Any] = field(default_factory=dict)
    contexts: List[str] = field(default_factory=list)


@dataclass
class KnowledgeGraphConfig:
    """Configuration for the knowledge graph."""
    # Entity extraction
    use_ner: bool = True
    ner_model: str = "en_core_web_sm"
    entity_threshold: float = 0.7
    
    # Relationship extraction
    extract_relationships: bool = True
    relationship_patterns: bool = True
    min_relationship_confidence: float = 0.5
    
    # Graph settings
    max_entities: int = 10000
    max_relationships: int = 50000
    prune_threshold: float = 0.1
    
    # Embeddings
    use_embeddings: bool = True
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Persistence
    auto_save: bool = True
    save_path: str = ".rag/knowledge_graph"


class KnowledgeGraphBuilder:
    """
    Builds and maintains a knowledge graph from text content.
    
    Features:
    - Named Entity Recognition (NER) for entity extraction
    - Pattern-based relationship extraction
    - Entity resolution and merging
    - Graph-based context retrieval
    - Entity importance scoring
    """
    
    def __init__(self, config: Optional[KnowledgeGraphConfig] = None):
        self.config = config or KnowledgeGraphConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize graph
        self.graph = nx.MultiDiGraph()
        self.entities: Dict[str, Entity] = {}
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # name -> entity_ids
        
        # NLP components
        self.nlp = None
        if SPACY_AVAILABLE and self.config.use_ner:
            try:
                import spacy
                self.nlp = spacy.load(self.config.ner_model)
            except:
                try:
                    # Try to download the model if not available
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", self.config.ner_model])
                    self.nlp = spacy.load(self.config.ner_model)
                except:
                    self.logger.warning(f"Failed to load spaCy model {self.config.ner_model}")
        
        # Embeddings
        self.encoder = None
        if SENTENCE_TRANSFORMERS_AVAILABLE and self.config.use_embeddings:
            try:
                self.encoder = SentenceTransformer(self.config.embedding_model)
            except Exception as e:
                self.logger.warning(f"Failed to initialize encoder: {e}")
        
        # Statistics
        self.stats = {
            "entities_extracted": 0,
            "relationships_extracted": 0,
            "entities_merged": 0,
            "graph_queries": 0
        }
    
    def extract_entities_from_text(self, text: str, context: Optional[Dict] = None) -> List[Entity]:
        """
        Extract entities from text using NER.
        
        Args:
            text: Text to extract entities from
            context: Optional context information (e.g., chapter, scene)
        
        Returns:
            List of extracted entities
        """
        entities = []
        
        if not self.nlp:
            return entities
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                # Map spaCy entity types to our EntityType
                entity_type = self._map_entity_type(ent.label_)
                if not entity_type:
                    continue
                
                # Create entity
                entity_id = f"{entity_type.value}_{len(self.entities)}"
                entity = Entity(
                    id=entity_id,
                    name=ent.text,
                    type=entity_type,
                    description=f"Extracted from: {text[:100]}..."
                )
                
                # Add mention context
                mention = {
                    "text": text[max(0, ent.start_char - 50):min(len(text), ent.end_char + 50)],
                    "position": {"start": ent.start_char, "end": ent.end_char}
                }
                if context:
                    mention.update(context)
                entity.mentions.append(mention)
                
                # Generate embeddings if available
                if self.encoder:
                    entity.embeddings = self.encoder.encode(ent.text, convert_to_numpy=True)
                
                entities.append(entity)
                self.stats["entities_extracted"] += 1
            
            # Also extract custom patterns (e.g., character names, locations)
            entities.extend(self._extract_pattern_entities(text, context))
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
        
        return entities
    
    def _map_entity_type(self, spacy_type: str) -> Optional[EntityType]:
        """Map spaCy entity types to our EntityType enum."""
        mapping = {
            "PERSON": EntityType.PERSON,
            "PER": EntityType.PERSON,
            "LOC": EntityType.LOCATION,
            "GPE": EntityType.LOCATION,
            "ORG": EntityType.ORGANIZATION,
            "EVENT": EntityType.EVENT,
            "DATE": EntityType.TIME,
            "TIME": EntityType.TIME,
            "PRODUCT": EntityType.OBJECT,
            "WORK_OF_ART": EntityType.CONCEPT
        }
        return mapping.get(spacy_type)
    
    def _extract_pattern_entities(self, text: str, context: Optional[Dict] = None) -> List[Entity]:
        """Extract entities using custom patterns."""
        entities = []
        
        # Pattern for quoted dialogue attribution
        dialogue_pattern = r'"[^"]+"\s+(?:said|asked|replied|whispered|shouted)\s+(\w+)'
        for match in re.finditer(dialogue_pattern, text):
            name = match.group(1)
            if name and len(name) > 2:
                entity_id = f"person_{len(self.entities)}"
                entity = Entity(
                    id=entity_id,
                    name=name,
                    type=EntityType.PERSON,
                    description="Character (from dialogue)"
                )
                entities.append(entity)
        
        # Pattern for chapter references
        chapter_pattern = r'(?:Chapter|chapter)\s+(\d+|\w+)'
        for match in re.finditer(chapter_pattern, text):
            chapter = match.group(0)
            entity_id = f"chapter_{len(self.entities)}"
            entity = Entity(
                id=entity_id,
                name=chapter,
                type=EntityType.CHAPTER,
                description="Chapter reference"
            )
            entities.append(entity)
        
        return entities
    
    def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """
        Extract relationships between entities from text.
        
        Args:
            text: Text containing the entities
            entities: List of entities found in the text
        
        Returns:
            List of extracted relationships
        """
        relationships = []
        
        if len(entities) < 2:
            return relationships
        
        # Create entity position map
        entity_positions = {}
        for entity in entities:
            if entity.mentions:
                pos = entity.mentions[0].get("position", {})
                if "start" in pos:
                    entity_positions[entity.id] = pos["start"]
        
        # Pattern-based relationship extraction
        relationship_patterns = [
            (r"(\w+)\s+(?:loves?|adores?|cherishes?)\s+(\w+)", RelationType.LOVES),
            (r"(\w+)\s+(?:hates?|despises?|loathes?)\s+(\w+)", RelationType.HATES),
            (r"(\w+)\s+(?:knows?|meets?|encounters?)\s+(\w+)", RelationType.KNOWS),
            (r"(\w+)\s+(?:travels?|goes?|journeys?)\s+to\s+(\w+)", RelationType.TRAVELS_TO),
            (r"(\w+)\s+(?:lives?|resides?|dwells?)\s+in\s+(\w+)", RelationType.LIVES_IN),
            (r"(\w+)\s+(?:works?|collaborates?)\s+with\s+(\w+)", RelationType.WORKS_WITH),
            (r"(\w+)\s+(?:causes?|triggers?|leads?\s+to)\s+(\w+)", RelationType.CAUSES),
            (r"(\w+)\s+(?:and|&)\s+(\w+)", RelationType.RELATED_TO),
        ]
        
        for pattern, rel_type in relationship_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                source_text = match.group(1)
                target_text = match.group(2)
                
                # Find matching entities
                source_entity = self._find_entity_by_name(source_text, entities)
                target_entity = self._find_entity_by_name(target_text, entities)
                
                if source_entity and target_entity and source_entity.id != target_entity.id:
                    rel = Relationship(
                        source_id=source_entity.id,
                        target_id=target_entity.id,
                        type=rel_type,
                        contexts=[match.group(0)]
                    )
                    relationships.append(rel)
                    self.stats["relationships_extracted"] += 1
        
        # Proximity-based relationships
        if entity_positions:
            sorted_entities = sorted(entity_positions.items(), key=lambda x: x[1])
            for i in range(len(sorted_entities) - 1):
                entity1_id = sorted_entities[i][0]
                entity2_id = sorted_entities[i + 1][0]
                pos1 = sorted_entities[i][1]
                pos2 = sorted_entities[i + 1][1]
                
                # If entities are close in text, they're likely related
                if pos2 - pos1 < 100:  # Within 100 characters
                    rel = Relationship(
                        source_id=entity1_id,
                        target_id=entity2_id,
                        type=RelationType.RELATED_TO,
                        weight=0.5,
                        attributes={"proximity": pos2 - pos1}
                    )
                    relationships.append(rel)
        
        return relationships
    
    def _find_entity_by_name(self, name: str, entities: List[Entity]) -> Optional[Entity]:
        """Find an entity by name (case-insensitive)."""
        name_lower = name.lower()
        for entity in entities:
            if entity.name.lower() == name_lower or name_lower in entity.name.lower():
                return entity
        return None
    
    def add_to_graph(self, entities: List[Entity], relationships: List[Relationship]):
        """
        Add entities and relationships to the knowledge graph.
        
        Args:
            entities: List of entities to add
            relationships: List of relationships to add
        """
        # Add or merge entities
        for entity in entities:
            existing_id = self._find_existing_entity(entity)
            if existing_id:
                # Merge with existing entity
                self._merge_entities(existing_id, entity)
                self.stats["entities_merged"] += 1
            else:
                # Add new entity
                self.entities[entity.id] = entity
                self.entity_index[entity.name.lower()].add(entity.id)
                self.graph.add_node(entity.id, **entity.__dict__)
        
        # Add relationships
        for rel in relationships:
            if rel.source_id in self.entities and rel.target_id in self.entities:
                self.graph.add_edge(
                    rel.source_id,
                    rel.target_id,
                    type=rel.type.value,
                    weight=rel.weight,
                    **rel.attributes
                )
        
        # Update entity importance scores
        self._update_importance_scores()
        
        # Prune if needed
        if len(self.entities) > self.config.max_entities:
            self._prune_graph()
    
    def _find_existing_entity(self, entity: Entity) -> Optional[str]:
        """Find an existing entity that matches the given entity."""
        # Check exact name match
        name_lower = entity.name.lower()
        if name_lower in self.entity_index:
            candidates = self.entity_index[name_lower]
            for candidate_id in candidates:
                candidate = self.entities[candidate_id]
                if candidate.type == entity.type:
                    return candidate_id
        
        # Check similarity if embeddings available
        if entity.embeddings is not None and self.encoder:
            max_similarity = 0.0
            best_match = None
            
            for existing_id, existing in self.entities.items():
                if existing.type != entity.type:
                    continue
                    
                if existing.embeddings is not None:
                    similarity = np.dot(entity.embeddings, existing.embeddings) / (
                        np.linalg.norm(entity.embeddings) * np.linalg.norm(existing.embeddings)
                    )
                    if similarity > max_similarity and similarity > self.config.entity_threshold:
                        max_similarity = similarity
                        best_match = existing_id
            
            if best_match:
                return best_match
        
        return None
    
    def _merge_entities(self, existing_id: str, new_entity: Entity):
        """Merge a new entity with an existing one."""
        existing = self.entities[existing_id]
        
        # Merge mentions
        existing.mentions.extend(new_entity.mentions)
        
        # Update description if longer
        if new_entity.description and len(new_entity.description) > len(existing.description or ""):
            existing.description = new_entity.description
        
        # Merge attributes
        existing.attributes.update(new_entity.attributes)
        
        # Update embeddings (average)
        if new_entity.embeddings is not None and existing.embeddings is not None:
            existing.embeddings = (existing.embeddings + new_entity.embeddings) / 2
    
    def _update_importance_scores(self):
        """Update importance scores for entities based on graph structure."""
        if not self.graph.nodes():
            return
        
        try:
            # Use PageRank for importance scoring
            pagerank_scores = nx.pagerank(self.graph, weight='weight')
            
            for entity_id, score in pagerank_scores.items():
                if entity_id in self.entities:
                    # Combine with mention frequency
                    mention_score = len(self.entities[entity_id].mentions) / 100
                    self.entities[entity_id].importance = (score + mention_score) / 2
        except:
            # Fallback to degree centrality
            for entity_id in self.entities:
                if entity_id in self.graph:
                    degree = self.graph.degree(entity_id)
                    mentions = len(self.entities[entity_id].mentions)
                    self.entities[entity_id].importance = (degree + mentions) / 100
    
    def _prune_graph(self):
        """Remove low-importance entities and relationships."""
        # Find entities to remove
        entities_to_remove = []
        for entity_id, entity in self.entities.items():
            if entity.importance < self.config.prune_threshold:
                entities_to_remove.append(entity_id)
        
        # Remove from graph
        for entity_id in entities_to_remove:
            if entity_id in self.graph:
                self.graph.remove_node(entity_id)
            
            # Remove from indices
            entity = self.entities[entity_id]
            self.entity_index[entity.name.lower()].discard(entity_id)
            del self.entities[entity_id]
        
        self.logger.info(f"Pruned {len(entities_to_remove)} low-importance entities")
    
    def query_graph(self, query: str, max_hops: int = 2) -> Dict[str, Any]:
        """
        Query the knowledge graph for relevant context.
        
        Args:
            query: Query text
            max_hops: Maximum graph traversal depth
        
        Returns:
            Dictionary containing relevant entities and relationships
        """
        self.stats["graph_queries"] += 1
        
        # Extract entities from query
        query_entities = self.extract_entities_from_text(query)
        
        # Find matching entities in graph
        matched_entities = set()
        for query_entity in query_entities:
            existing_id = self._find_existing_entity(query_entity)
            if existing_id:
                matched_entities.add(existing_id)
        
        # If no direct matches, try fuzzy matching
        if not matched_entities and query:
            query_lower = query.lower()
            for name, entity_ids in self.entity_index.items():
                if name in query_lower or query_lower in name:
                    matched_entities.update(entity_ids)
        
        # Perform graph traversal
        context = {
            "entities": {},
            "relationships": [],
            "subgraph": None
        }
        
        if matched_entities:
            # Get subgraph around matched entities
            subgraph_nodes = set()
            for entity_id in matched_entities:
                if entity_id in self.graph:
                    # Add entity and its neighbors up to max_hops
                    subgraph_nodes.add(entity_id)
                    for hop in range(1, max_hops + 1):
                        neighbors = set()
                        for node in subgraph_nodes:
                            neighbors.update(self.graph.neighbors(node))
                            neighbors.update(self.graph.predecessors(node))
                        subgraph_nodes.update(neighbors)
            
            # Extract subgraph
            if subgraph_nodes:
                context["subgraph"] = self.graph.subgraph(subgraph_nodes)
                
                # Add entities
                for node in subgraph_nodes:
                    if node in self.entities:
                        context["entities"][node] = self.entities[node]
                
                # Add relationships
                for edge in context["subgraph"].edges(data=True):
                    rel_data = {
                        "source": edge[0],
                        "target": edge[1],
                        "type": edge[2].get("type", "related_to"),
                        "weight": edge[2].get("weight", 1.0)
                    }
                    context["relationships"].append(rel_data)
        
        return context
    
    def get_entity_context(self, entity_id: str, include_relationships: bool = True) -> Dict[str, Any]:
        """
        Get detailed context for a specific entity.
        
        Args:
            entity_id: ID of the entity
            include_relationships: Whether to include relationships
        
        Returns:
            Dictionary containing entity details and relationships
        """
        if entity_id not in self.entities:
            return {}
        
        entity = self.entities[entity_id]
        context = {
            "entity": entity,
            "mentions": entity.mentions,
            "importance": entity.importance
        }
        
        if include_relationships and entity_id in self.graph:
            # Get incoming relationships
            incoming = []
            for source, target, data in self.graph.in_edges(entity_id, data=True):
                incoming.append({
                    "source": source,
                    "type": data.get("type", "related_to"),
                    "weight": data.get("weight", 1.0)
                })
            
            # Get outgoing relationships
            outgoing = []
            for source, target, data in self.graph.out_edges(entity_id, data=True):
                outgoing.append({
                    "target": target,
                    "type": data.get("type", "related_to"),
                    "weight": data.get("weight", 1.0)
                })
            
            context["relationships"] = {
                "incoming": incoming,
                "outgoing": outgoing
            }
        
        return context
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics."""
        stats = self.stats.copy()
        stats.update({
            "total_entities": len(self.entities),
            "total_relationships": self.graph.number_of_edges(),
            "entity_types": defaultdict(int),
            "relationship_types": defaultdict(int),
            "avg_entity_importance": 0.0
        })
        
        # Count entity types
        for entity in self.entities.values():
            stats["entity_types"][entity.type.value] += 1
        
        # Count relationship types
        for _, _, data in self.graph.edges(data=True):
            rel_type = data.get("type", "unknown")
            stats["relationship_types"][rel_type] += 1
        
        # Average importance
        if self.entities:
            total_importance = sum(e.importance for e in self.entities.values())
            stats["avg_entity_importance"] = total_importance / len(self.entities)
        
        return stats
    
    def save(self, path: Optional[str] = None):
        """Save the knowledge graph to disk."""
        save_path = Path(path or self.config.save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save graph
        nx.write_gpickle(self.graph, save_path / "graph.gpickle")
        
        # Save entities
        with open(save_path / "entities.pkl", "wb") as f:
            pickle.dump(self.entities, f)
        
        # Save indices
        with open(save_path / "indices.pkl", "wb") as f:
            pickle.dump(dict(self.entity_index), f)
        
        # Save config and stats
        with open(save_path / "metadata.json", "w") as f:
            json.dump({
                "config": self.config.__dict__,
                "stats": self.get_stats()
            }, f, indent=2)
        
        self.logger.info(f"Knowledge graph saved to {save_path}")
    
    def load(self, path: Optional[str] = None) -> bool:
        """Load the knowledge graph from disk."""
        load_path = Path(path or self.config.save_path)
        if not load_path.exists():
            return False
        
        try:
            # Load graph
            graph_path = load_path / "graph.gpickle"
            if graph_path.exists():
                self.graph = nx.read_gpickle(graph_path)
            
            # Load entities
            entities_path = load_path / "entities.pkl"
            if entities_path.exists():
                with open(entities_path, "rb") as f:
                    self.entities = pickle.load(f)
            
            # Load indices
            indices_path = load_path / "indices.pkl"
            if indices_path.exists():
                with open(indices_path, "rb") as f:
                    loaded_index = pickle.load(f)
                    self.entity_index = defaultdict(set, loaded_index)
            
            self.logger.info(f"Knowledge graph loaded from {load_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load knowledge graph: {e}")
            return False