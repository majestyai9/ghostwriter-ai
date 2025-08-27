#!/usr/bin/env python3
"""
Verify and test event system integration in GradioHandlers.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from events import Event, EventType, UIEventType, event_manager


def test_event_system():
    """Test that event system is properly configured."""
    
    print("Testing Event System Integration...")
    print("=" * 50)
    
    # Test 1: Check UIEventType exists
    print("\n1. Checking UIEventType enum...")
    ui_events = [e for e in dir(UIEventType) if not e.startswith('_')]
    print(f"   Found {len(ui_events)} UI event types")
    print(f"   Sample events: {ui_events[:5]}")
    
    # Test 2: Check event manager
    print("\n2. Checking event_manager...")
    print(f"   Event manager type: {type(event_manager)}")
    print(f"   Has emit method: {hasattr(event_manager, 'emit')}")
    print(f"   Has subscribe method: {hasattr(event_manager, 'subscribe')}")
    
    # Test 3: Test event emission
    print("\n3. Testing event emission...")
    test_events_emitted = []
    
    def test_listener(event):
        test_events_emitted.append(event)
        print(f"   Received event: {event.type.value if hasattr(event.type, 'value') else event.type}")
    
    # Subscribe to UI events
    event_manager.subscribe(UIEventType.PROJECT_CREATED, test_listener)
    event_manager.subscribe(UIEventType.CACHE_CLEARED, test_listener)
    
    # Emit test events
    event_manager.emit(Event(UIEventType.PROJECT_CREATED, {"test": "data"}))
    event_manager.emit(Event(UIEventType.CACHE_CLEARED, {"cleared": 10}))
    
    print(f"   Events emitted: {len(test_events_emitted)}")
    
    # Test 4: Check gradio_handlers integration
    print("\n4. Checking gradio_handlers integration...")
    try:
        from gradio_handlers import GradioHandlers
        print("   GradioHandlers import: OK")
        
        # Check if methods exist
        handler_methods = [
            'create_project', 'delete_project', 'export_book',
            'batch_export_books', 'clear_cache'
        ]
        
        for method in handler_methods:
            if hasattr(GradioHandlers, method):
                print(f"   Method {method}: OK")
            else:
                print(f"   Method {method}: MISSING")
                
    except ImportError as e:
        print(f"   ERROR: Could not import GradioHandlers: {e}")
    
    # Test 5: Verify event types used in handlers
    print("\n5. Checking event usage in handlers...")
    events_used = {
        'UIEventType.PROJECT_CREATED': 'Project creation',
        'UIEventType.PROJECT_DELETED': 'Project deletion',
        'UIEventType.CHARACTER_CREATED': 'Character creation',
        'UIEventType.EXPORT_STARTED': 'Export started',
        'UIEventType.CACHE_CLEARED': 'Cache cleared',
        'UIEventType.BATCH_OPERATION_STARTED': 'Batch operation',
    }
    
    gradio_file = Path(__file__).parent / 'gradio_handlers.py'
    if gradio_file.exists():
        content = gradio_file.read_text(encoding='utf-8')
        for event_type, description in events_used.items():
            if event_type in content:
                print(f"   {description}: FOUND")
            else:
                print(f"   {description}: NOT FOUND (may need implementation)")
    
    print("\n" + "=" * 50)
    print("Event System Verification Complete!")
    
    return len(test_events_emitted) == 2


if __name__ == "__main__":
    success = test_event_system()
    sys.exit(0 if success else 1)