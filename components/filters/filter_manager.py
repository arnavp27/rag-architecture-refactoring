"""
FilterManager - Manages query filters and conversation state

This class manages filter extraction and state across conversation turns.
It uses an LLM to extract filters from user queries and maintains
conversation history.

Ported from: RAG_v2/retrieval/filter_manager.py
Updated to use: LLMProvider interface instead of direct Gemini client

Design Pattern: State Management
SOLID Principles:
- Single Responsibility: Only manages filter state
- Dependency Inversion: Depends on LLMProvider interface
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

from core.interfaces.llm_provider import LLMProvider


@dataclass
class ConversationTurn:
    """
    Represents a single conversation turn.
    
    Tracks what filters were detected, what filters were active,
    and whether a reset was triggered.
    """
    query: str
    detected_filters: Dict[str, Any]
    active_filters_after: Dict[str, Any]
    timestamp: datetime
    reset_triggered: bool = False


class FilterManager:
    """
    Manages filter state across conversation turns.
    
    Responsibilities:
    - Extract filters from queries using LLM
    - Maintain active filter state
    - Track conversation history
    - Detect reset triggers
    - Provide conversation context for LLM
    
    Example usage:
        manager = FilterManager(llm_provider)
        
        # Extract and update filters
        filters = manager.extract_and_update_filters(
            query="Show me positive economic statements",
            conversation_history=[]
        )
        
        # Get conversation context for next query
        context = manager.get_conversation_context(last_n_turns=3)
    """
    
    def __init__(self, llm: LLMProvider):
        """
        Initialize the filter manager.
        
        Args:
            llm: LLM provider for filter extraction
        """
        self._llm = llm
        self._conversation_history: List[ConversationTurn] = []
        self._active_filters: Dict[str, Any] = {}
        self._logger = logging.getLogger(__name__)
        
        # Reset trigger phrases
        self._reset_phrases = [
            "start over",
            "clear filters",
            "reset",
            "new search",
            "forget everything",
            "clear everything"
        ]
        
        self._logger.debug("FilterManager initialized")
    
    def extract_and_update_filters(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Extract filters from query and update active filters.
        
        This is the main method that combines extraction and update logic.
        
        Args:
            query: User's current query
            conversation_history: Previous conversation turns (for context)
            
        Returns:
            Updated active filters
        """
        # Get conversation context
        context = self.get_conversation_context(last_n_turns=3)
        
        # Extract filters using LLM
        detected_filters = self._extract_filters_with_llm(
            query=query,
            current_filters=self._active_filters,
            conversation_context=context
        )
        
        # Update filters
        updated_filters, reset_triggered = self._update_filters(
            query=query,
            detected_filters=detected_filters
        )
        
        return updated_filters
    
    def _extract_filters_with_llm(
        self,
        query: str,
        current_filters: Dict[str, Any],
        conversation_context: str
    ) -> Dict[str, Any]:
        """
        Extract filters from query using LLM.
        
        Args:
            query: User's current query
            current_filters: Currently active filters
            conversation_context: Previous conversation turns
            
        Returns:
            Extracted filters as dictionary
        """
        prompt = self._build_filter_extraction_prompt(
            query=query,
            current_filters=current_filters,
            conversation_context=conversation_context
        )
        
        try:
            # Use LLM to extract filters
            # Request structured output (JSON)
            response_text = self._llm.generate(prompt)
            
            # Parse JSON response
            filters = self._parse_filter_response(response_text)
            
            self._logger.debug(f"Extracted filters from query '{query[:50]}...': {filters}")
            return filters
            
        except Exception as e:
            self._logger.error(f"Filter extraction failed: {e}")
            return {}
    
    def _build_filter_extraction_prompt(
        self,
        query: str,
        current_filters: Dict[str, Any],
        conversation_context: str
    ) -> str:
        """
        Build prompt for LLM filter extraction.
        
        Args:
            query: User's query
            current_filters: Currently active filters
            conversation_context: Previous turns
            
        Returns:
            Formatted prompt string
        """
        return f"""You are a filter extraction system for a political statement database.

Your task: Extract filters from the user's query.

Available filter types:
- theme: List of themes (e.g., ["Economy", "Healthcare", "Education"])
- sentiment: "Positive", "Negative", "Neutral", or "Mixed"
- politician: Politician name
- source: Source/location name
- classification: "Policy", "Rhetoric", "Attack", "Defense"
- temporal_focus: "Forward-looking", "Retrospective", "Present"
- perspective: "By Politician", "About Politician"

Current active filters:
{json.dumps(current_filters, indent=2)}

Conversation context:
{conversation_context}

Current query: "{query}"

Instructions:
1. Extract ONLY the filters mentioned in the current query
2. If the query says "reset" or "clear filters", return: {{"reset": true}}
3. Return ONLY valid JSON, no other text
4. Use exact values from the schema above

Return your response as JSON:
{{
  "theme": [...],
  "sentiment": "...",
  "reset": false
}}"""
    
    def _parse_filter_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse LLM response into filter dictionary.
        
        Handles common formatting issues like markdown code blocks.
        
        Args:
            response_text: Raw LLM response
            
        Returns:
            Parsed filter dictionary
        """
        # Handle empty response
        if not response_text or not response_text.strip():
            self._logger.debug("Empty LLM response, returning empty filters")
            return {}
        
        # Clean up response
        response_text = response_text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith("```json"):
            response_text = response_text.split("```json")[1].split("```")[0]
        elif response_text.startswith("```"):
            response_text = response_text.split("```")[1].split("```")[0]
        
        response_text = response_text.strip()
        
        # Handle empty after cleanup
        if not response_text:
            self._logger.debug("Empty response after cleanup, returning empty filters")
            return {}
        
        # Parse JSON
        try:
            filters = json.loads(response_text)
            return filters
        except json.JSONDecodeError as e:
            self._logger.warning(f"Failed to parse filter JSON: {e}")
            self._logger.debug(f"Response text was: {response_text[:200]}")
            return {}
    
    def _update_filters(
        self,
        query: str,
        detected_filters: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Update active filters based on detected filters.
        
        Args:
            query: User's query
            detected_filters: Filters extracted from query
            
        Returns:
            Tuple of (updated_filters, reset_triggered)
        """
        # Check for reset triggers
        reset_triggered = self._check_reset_triggers(query, detected_filters)
        
        if reset_triggered:
            self._active_filters = {}
            self._logger.info("Filters reset due to reset trigger")
        
        # Store conversation turn
        turn = ConversationTurn(
            query=query,
            detected_filters=detected_filters,
            active_filters_after=self._active_filters.copy(),
            timestamp=datetime.now(),
            reset_triggered=reset_triggered
        )
        
        # Update active filters if not reset
        if not reset_triggered and detected_filters:
            # Remove 'reset' key if present
            filters_to_add = {k: v for k, v in detected_filters.items() if k != "reset"}
            self._active_filters.update(filters_to_add)
        
        # Update turn with final filters
        turn.active_filters_after = self._active_filters.copy()
        self._conversation_history.append(turn)
        
        self._logger.info(
            f"Updated filters - Query: '{query[:50]}...' "
            f"Active filters: {self._active_filters}"
        )
        
        return self._active_filters.copy(), reset_triggered
    
    def _check_reset_triggers(
        self,
        query: str,
        detected_filters: Dict[str, Any]
    ) -> bool:
        """
        Check if query contains reset triggers.
        
        Args:
            query: User's query
            detected_filters: Detected filters
            
        Returns:
            True if reset should be triggered
        """
        query_lower = query.lower()
        
        # Check for explicit reset phrases
        for phrase in self._reset_phrases:
            if phrase in query_lower:
                return True
        
        # Check for reset in detected filters
        return detected_filters.get("reset", False)
    
    def get_conversation_context(self, last_n_turns: int = 3) -> str:
        """
        Get formatted conversation context for LLM prompts.
        
        Args:
            last_n_turns: Number of recent turns to include
            
        Returns:
            Formatted conversation context string
        """
        if not self._conversation_history:
            return "No previous conversation history."
        
        recent_turns = self._conversation_history[-last_n_turns:]
        
        context_parts = []
        for i, turn in enumerate(recent_turns, 1):
            context_parts.append(
                f"Turn {i}: User asked: '{turn.query}'\n"
                f"Detected filters: {turn.detected_filters}\n"
                f"Active filters after: {turn.active_filters_after}"
            )
        
        return "\n\n".join(context_parts)
    
    def get_active_filters(self) -> Dict[str, Any]:
        """
        Get current active filters.
        
        Returns:
            Dictionary of active filters
        """
        return self._active_filters.copy()
    
    def get_active_filters_summary(self) -> str:
        """
        Get human-readable summary of active filters.
        
        Returns:
            Formatted summary string
        """
        if not self._active_filters:
            return "No active filters - searching all statements"
        
        summary_parts = []
        for key, value in self._active_filters.items():
            if isinstance(value, list):
                summary_parts.append(f"{key}: {', '.join(map(str, value))}")
            else:
                summary_parts.append(f"{key}: {value}")
        
        return "Active filters: " + " | ".join(summary_parts)
    
    def clear_history(self) -> None:
        """Clear conversation history and filters."""
        self._conversation_history = []
        self._active_filters = {}
        self._logger.info("Cleared conversation history and filters")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Export filter state to dictionary.
        
        Returns:
            Dictionary with filter state
        """
        return {
            "active_filters": self._active_filters,
            "conversation_history": [
                {
                    **asdict(turn),
                    "timestamp": turn.timestamp.isoformat()
                }
                for turn in self._conversation_history
            ]
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Import filter state from dictionary.
        
        Args:
            data: Dictionary with filter state
        """
        self._active_filters = data.get("active_filters", {})
        
        history_data = data.get("conversation_history", [])
        self._conversation_history = []
        
        for turn_data in history_data:
            turn_data["timestamp"] = datetime.fromisoformat(turn_data["timestamp"])
            self._conversation_history.append(ConversationTurn(**turn_data))
        
        self._logger.info("Imported filter state from dictionary")