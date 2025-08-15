"""
Event handler setup for monitoring book generation progress.
"""

import logging

from events import Event, EventType, event_manager


class EventSetup:
    """Setup and manage event handlers for book generation monitoring."""

    def __init__(self) -> None:
        """Initialize the event setup handler."""
        self.logger = logging.getLogger(__name__)

    def setup_event_handlers(self) -> None:
        """Setup custom event handlers for monitoring generation progress."""
        # Subscribe handlers
        event_manager.subscribe(EventType.CHAPTER_COMPLETED, self._log_progress)
        event_manager.subscribe(EventType.SECTION_COMPLETED, self._log_progress)
        event_manager.subscribe(EventType.GENERATION_FAILED, self._handle_errors)
        event_manager.subscribe(EventType.API_CALL_FAILED, self._handle_errors)

        self.logger.debug("Event handlers configured")

    def _log_progress(self, event: Event) -> None:
        """
        Log progress events for chapters and sections.

        Args:
            event: Event containing progress information
        """
        if event.type == EventType.CHAPTER_COMPLETED:
            chapter_num = event.data.get('chapter_number')
            chapter_title = event.data.get('chapter_title')
            self.logger.info(f"✓ Chapter {chapter_num}: {chapter_title} completed")

        elif event.type == EventType.SECTION_COMPLETED:
            section_num = event.data.get('section_number')
            section_title = event.data.get('section_title')
            self.logger.info(f"  ✓ Section {section_num}: {section_title} completed")

    def _handle_errors(self, event: Event) -> None:
        """
        Handle error events from generation and API calls.

        Args:
            event: Event containing error information
        """
        if event.type == EventType.GENERATION_FAILED:
            stage = event.data.get('stage')
            error = event.data.get('error')
            self.logger.error(f"Generation failed at {stage}: {error}")

        elif event.type == EventType.API_CALL_FAILED:
            provider = event.data.get('provider')
            error = event.data.get('error')
            self.logger.error(f"API call failed for {provider}: {error}")