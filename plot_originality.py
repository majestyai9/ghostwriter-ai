"""
Plot Originality Validator for detecting and preventing repetitive plot elements.

This module ensures plot originality by tracking used plot devices, suggesting
alternatives, and maintaining a diverse story structure.
"""

import re
import random
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class PlotDevice(Enum):
    """Common plot devices to track."""
    BETRAYAL = "betrayal"
    DOUBLE_CROSS = "double_cross"
    HIDDEN_IDENTITY = "hidden_identity"
    FAKE_DEATH = "fake_death"
    POISONING = "poisoning"
    CAR_CHASE = "car_chase"
    ROOFTOP_CHASE = "rooftop_chase"
    EXPLOSION = "explosion"
    HOSTAGE_SITUATION = "hostage"
    INFILTRATION = "infiltration"
    HACK_COMPUTER = "hacking"
    TORTURE_SCENE = "torture"
    ROMANTIC_SUBPLOT = "romance"
    MENTOR_DEATH = "mentor_death"
    MACGUFFIN = "macguffin"
    TICKING_CLOCK = "ticking_clock"
    RED_HERRING = "red_herring"
    PLOT_TWIST = "plot_twist"
    FLASHBACK = "flashback"
    DREAM_SEQUENCE = "dream"


@dataclass
class PlotElement:
    """Represents a single plot element in the story."""
    device_type: PlotDevice
    chapter: int
    description: str
    characters_involved: List[str]
    location: Optional[str] = None
    outcome: Optional[str] = None
    uniqueness_score: float = 1.0


@dataclass
class OriginalityReport:
    """Report on plot originality."""
    repetition_count: Dict[PlotDevice, int]
    similarity_warnings: List[str]
    overused_devices: List[PlotDevice]
    suggestions: List[str]
    originality_score: float
    unique_elements: List[str]


class PlotOriginalityValidator:
    """
    Validates plot originality and suggests alternatives to overused devices.
    
    This validator tracks plot devices, detects repetitions, and ensures
    narrative variety throughout the book.
    """
    
    # Maximum uses before a device is considered overused
    DEVICE_LIMITS = {
        PlotDevice.POISONING: 1,
        PlotDevice.FAKE_DEATH: 1,
        PlotDevice.MENTOR_DEATH: 1,
        PlotDevice.BETRAYAL: 2,
        PlotDevice.DOUBLE_CROSS: 2,
        PlotDevice.CAR_CHASE: 2,
        PlotDevice.EXPLOSION: 3,
        PlotDevice.HOSTAGE_SITUATION: 2,
        PlotDevice.INFILTRATION: 3,
        PlotDevice.HACK_COMPUTER: 3,
        PlotDevice.PLOT_TWIST: 3,
        PlotDevice.FLASHBACK: 4,
        PlotDevice.RED_HERRING: 3,
    }
    
    # Alternative plot devices for each category
    PLOT_ALTERNATIVES = {
        PlotDevice.POISONING: [
            "biological weapon threat",
            "nerve gas attack",
            "radiation exposure",
            "viral outbreak",
            "drugged unconscious"
        ],
        PlotDevice.CAR_CHASE: [
            "motorcycle pursuit",
            "boat chase",
            "helicopter chase",
            "parkour foot chase",
            "subway tunnel chase",
            "drone pursuit"
        ],
        PlotDevice.BETRAYAL: [
            "loyalty test",
            "false allegiance",
            "undercover revelation",
            "sleeper agent activation",
            "blackmail coercion"
        ],
        PlotDevice.EXPLOSION: [
            "building collapse",
            "EMP attack",
            "cyberattack",
            "chemical spill",
            "avalanche trap",
            "flood escape"
        ],
        PlotDevice.INFILTRATION: [
            "social engineering",
            "insider recruitment",
            "digital penetration",
            "supply chain compromise",
            "diplomatic cover operation"
        ],
        PlotDevice.HOSTAGE_SITUATION: [
            "prisoner exchange",
            "kidnapping prevention",
            "extraction mission",
            "witness protection",
            "safe house compromise"
        ],
        PlotDevice.TORTURE_SCENE: [
            "psychological manipulation",
            "truth serum interrogation",
            "virtual reality interrogation",
            "sensory deprivation",
            "memory extraction technology"
        ]
    }
    
    # Unique plot twists bank
    UNIQUE_TWISTS = [
        "The mission was a training simulation all along",
        "The protagonist has been working for the enemy unknowingly",
        "The MacGuffin was a decoy; the real objective was different",
        "The villain and hero are related by blood",
        "The entire operation was orchestrated by an AI",
        "The protagonist has multiple personality disorder",
        "Time loop - events keep repeating with variations",
        "The supporting character is the real mastermind",
        "The agency itself is the true enemy",
        "Parallel universe/timeline interference",
        "The protagonist's memories have been altered",
        "The villain's plan was to be captured",
        "Everyone except the protagonist is an actor",
        "The death was staged using advanced hologram technology",
        "Quantum entanglement enables instantaneous communication"
    ]
    
    def __init__(self, save_path: Optional[Path] = None):
        """
        Initialize the Plot Originality Validator.
        
        Args:
            save_path: Optional path to save plot tracking data
        """
        self.save_path = save_path or Path("projects/plot_tracker.json")
        
        self.plot_elements: List[PlotElement] = []
        self.device_usage: Dict[PlotDevice, int] = defaultdict(int)
        self.used_twists: Set[str] = set()
        self.location_usage: Counter = Counter()
        self.action_sequences: List[Tuple[int, str]] = []
        self.emotional_beats: List[Tuple[int, str]] = []
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load existing plot tracking data if available."""
        if self.save_path.exists():
            try:
                with open(self.save_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Restore plot elements
                    self.plot_elements = [
                        PlotElement(
                            device_type=PlotDevice(elem['device_type']),
                            chapter=elem['chapter'],
                            description=elem['description'],
                            characters_involved=elem['characters_involved'],
                            location=elem.get('location'),
                            outcome=elem.get('outcome'),
                            uniqueness_score=elem.get('uniqueness_score', 1.0)
                        )
                        for elem in data.get('plot_elements', [])
                    ]
                    
                    # Restore usage tracking
                    self.device_usage = defaultdict(int, {
                        PlotDevice(k): v for k, v in data.get('device_usage', {}).items()
                    })
                    self.used_twists = set(data.get('used_twists', []))
                    self.location_usage = Counter(data.get('location_usage', {}))
                    
                logger.info(f"Loaded plot tracking data from {self.save_path}")
            except Exception as e:
                logger.warning(f"Could not load plot tracking data: {e}")
    
    def _save_data(self) -> None:
        """Save plot tracking data for persistence."""
        try:
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'plot_elements': [
                    {
                        'device_type': elem.device_type.value,
                        'chapter': elem.chapter,
                        'description': elem.description,
                        'characters_involved': elem.characters_involved,
                        'location': elem.location,
                        'outcome': elem.outcome,
                        'uniqueness_score': elem.uniqueness_score
                    }
                    for elem in self.plot_elements
                ],
                'device_usage': {k.value: v for k, v in self.device_usage.items()},
                'used_twists': list(self.used_twists),
                'location_usage': dict(self.location_usage),
                'action_sequences': self.action_sequences,
                'emotional_beats': self.emotional_beats
            }
            
            with open(self.save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.debug(f"Saved plot tracking data to {self.save_path}")
        except Exception as e:
            logger.error(f"Failed to save plot tracking data: {e}")
    
    def detect_repetitions(self, chapter: int, chapter_text: str) -> List[str]:
        """
        Detect repetitive plot elements in a chapter.
        
        Args:
            chapter: Chapter number
            chapter_text: Chapter content
            
        Returns:
            List of detected repetitions
        """
        repetitions = []
        detected_devices = self._detect_plot_devices(chapter_text)
        
        for device in detected_devices:
            if device in self.device_usage:
                current_usage = self.device_usage[device]
                limit = self.DEVICE_LIMITS.get(device, 3)
                
                if current_usage >= limit:
                    repetitions.append(
                        f"{device.value} used {current_usage} times (limit: {limit})"
                    )
        
        # Check for similar action sequences
        action_type = self._classify_action_sequence(chapter_text)
        if action_type:
            recent_actions = [a[1] for a in self.action_sequences[-3:]]
            if action_type in recent_actions:
                repetitions.append(f"Similar action sequence: {action_type}")
        
        # Check location repetition
        location = self._extract_location(chapter_text)
        if location and self.location_usage[location] > 2:
            repetitions.append(f"Location '{location}' overused ({self.location_usage[location]} times)")
        
        return repetitions
    
    def _detect_plot_devices(self, text: str) -> List[PlotDevice]:
        """Detect plot devices used in the text."""
        detected = []
        text_lower = text.lower()
        
        # Detection patterns for each device
        patterns = {
            PlotDevice.POISONING: [
                r'poison', r'toxic', r'venom', r'paralyz', r'antidote'
            ],
            PlotDevice.CAR_CHASE: [
                r'car chase', r'vehicle pursuit', r'sped through traffic',
                r'weaving between cars', r'tires screeched'
            ],
            PlotDevice.BETRAYAL: [
                r'betray', r'double.?cross', r'stabbed in the back',
                r'traitor', r'turncoat'
            ],
            PlotDevice.EXPLOSION: [
                r'explod', r'detonate', r'blast', r'bomb', r'C-4', r'TNT'
            ],
            PlotDevice.INFILTRATION: [
                r'infiltrat', r'sneak in', r'break into', r'penetrate',
                r'undercover', r'pose as'
            ],
            PlotDevice.HOSTAGE_SITUATION: [
                r'hostage', r'held at gunpoint', r'ransom', r'kidnap'
            ],
            PlotDevice.FAKE_DEATH: [
                r'faked?.death', r'staged?.death', r'presumed dead',
                r'death was?.staged'
            ],
            PlotDevice.HACK_COMPUTER: [
                r'hack', r'cyber', r'firewall', r'encrypt', r'malware',
                r'backdoor', r'breach'
            ],
            PlotDevice.ROOFTOP_CHASE: [
                r'rooftop', r'jumped between buildings', r'across the roof'
            ],
            PlotDevice.TORTURE_SCENE: [
                r'tortur', r'interrogat', r'extract information'
            ],
            PlotDevice.PLOT_TWIST: [
                r'wasn\'t who', r'all along', r'revealed to be',
                r'true identity', r'real plan was'
            ],
            PlotDevice.FLASHBACK: [
                r'years ago', r'remembered when', r'flashback',
                r'memory of', r'in the past'
            ]
        }
        
        for device, device_patterns in patterns.items():
            for pattern in device_patterns:
                if re.search(pattern, text_lower):
                    detected.append(device)
                    break
        
        return detected
    
    def _classify_action_sequence(self, text: str) -> Optional[str]:
        """Classify the type of action sequence in the text."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['chase', 'pursuit', 'fled', 'ran']):
            if 'car' in text_lower or 'vehicle' in text_lower:
                return "vehicle_chase"
            elif 'foot' in text_lower or 'alley' in text_lower:
                return "foot_chase"
            elif 'boat' in text_lower or 'water' in text_lower:
                return "water_chase"
            else:
                return "general_chase"
        
        if any(word in text_lower for word in ['fight', 'combat', 'battle']):
            if 'gun' in text_lower or 'shoot' in text_lower:
                return "gunfight"
            elif 'knife' in text_lower or 'blade' in text_lower:
                return "knife_fight"
            else:
                return "hand_combat"
        
        if any(word in text_lower for word in ['sneak', 'stealth', 'silent']):
            return "stealth_sequence"
        
        return None
    
    def _extract_location(self, text: str) -> Optional[str]:
        """Extract primary location from text."""
        # Simple location extraction - could be enhanced
        location_keywords = [
            'hotel', 'casino', 'airport', 'embassy', 'warehouse',
            'office', 'laboratory', 'mansion', 'yacht', 'train',
            'subway', 'bridge', 'tower', 'bunker', 'safehouse'
        ]
        
        text_lower = text.lower()
        for location in location_keywords:
            if location in text_lower:
                return location
        
        return None
    
    def suggest_alternatives(self, overused_device: PlotDevice) -> List[str]:
        """
        Suggest alternative plot devices.
        
        Args:
            overused_device: The overused plot device
            
        Returns:
            List of alternative suggestions
        """
        if overused_device in self.PLOT_ALTERNATIVES:
            # Filter out already used alternatives
            available = [
                alt for alt in self.PLOT_ALTERNATIVES[overused_device]
                if alt not in [elem.description for elem in self.plot_elements]
            ]
            
            if available:
                return random.sample(available, min(3, len(available)))
        
        # Generic alternatives
        return [
            "Psychological manipulation instead of physical action",
            "Technology-based solution instead of violence",
            "Diplomatic negotiation instead of confrontation",
            "Information warfare instead of direct conflict"
        ]
    
    def validate_uniqueness(self, plot_description: str, chapter: int) -> Tuple[bool, List[str]]:
        """
        Validate if a plot element is unique.
        
        Args:
            plot_description: Description of the plot element
            chapter: Chapter number
            
        Returns:
            Tuple of (is_unique, list_of_similar_plots)
        """
        similar_plots = []
        
        # Check against existing plot elements
        for elem in self.plot_elements:
            similarity = self._calculate_plot_similarity(plot_description, elem.description)
            if similarity > 0.7:  # 70% similarity threshold
                similar_plots.append(
                    f"Chapter {elem.chapter}: {elem.description[:50]}... (similarity: {similarity:.0%})"
                )
        
        is_unique = len(similar_plots) == 0
        return is_unique, similar_plots
    
    def _calculate_plot_similarity(self, plot1: str, plot2: str) -> float:
        """Calculate similarity between two plot descriptions."""
        # Simple word overlap similarity
        words1 = set(plot1.lower().split())
        words2 = set(plot2.lower().split())
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 
                       'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is'}
        words1 -= common_words
        words2 -= common_words
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def track_plot_element(self, device: PlotDevice, chapter: int, 
                          description: str, characters: List[str],
                          location: Optional[str] = None) -> None:
        """
        Track a new plot element.
        
        Args:
            device: Type of plot device
            chapter: Chapter number
            description: Description of the plot element
            characters: Characters involved
            location: Location of the event
        """
        # Calculate uniqueness score
        is_unique, similar = self.validate_uniqueness(description, chapter)
        uniqueness_score = 1.0 if is_unique else max(0.3, 1.0 - len(similar) * 0.2)
        
        element = PlotElement(
            device_type=device,
            chapter=chapter,
            description=description,
            characters_involved=characters,
            location=location,
            uniqueness_score=uniqueness_score
        )
        
        self.plot_elements.append(element)
        self.device_usage[device] += 1
        
        if location:
            self.location_usage[location] += 1
        
        # Track action sequence if applicable
        action_type = self._classify_action_sequence(description)
        if action_type:
            self.action_sequences.append((chapter, action_type))
        
        self._save_data()
        logger.debug(f"Tracked plot element: {device.value} in chapter {chapter}")
    
    def generate_unique_twist(self) -> str:
        """
        Generate a unique plot twist suggestion.
        
        Returns:
            A unique twist suggestion
        """
        available_twists = [
            twist for twist in self.UNIQUE_TWISTS 
            if twist not in self.used_twists
        ]
        
        if available_twists:
            twist = random.choice(available_twists)
            self.used_twists.add(twist)
            self._save_data()
            return twist
        
        # Generate a procedural twist if all predefined ones are used
        templates = [
            "The {character} was actually {revelation}",
            "The {object} contained {surprise}",
            "The real enemy was {unexpected}",
            "{event} was orchestrated by {mastermind}",
            "The protagonist discovers they are {shocking_truth}"
        ]
        
        revelations = [
            "a deep cover agent", "an AI construct", "from the future",
            "protecting the villain", "the heir to a criminal empire"
        ]
        
        template = random.choice(templates)
        if '{revelation}' in template:
            return template.format(character="trusted ally", revelation=random.choice(revelations))
        
        return "Create an unexpected revelation about a main character's true identity or motivation"
    
    def generate_originality_report(self) -> OriginalityReport:
        """
        Generate a comprehensive originality report.
        
        Returns:
            OriginalityReport with analysis
        """
        report = OriginalityReport(
            repetition_count=dict(self.device_usage),
            similarity_warnings=[],
            overused_devices=[],
            suggestions=[],
            originality_score=1.0,
            unique_elements=[]
        )
        
        # Find overused devices
        for device, count in self.device_usage.items():
            limit = self.DEVICE_LIMITS.get(device, 3)
            if count > limit:
                report.overused_devices.append(device)
                report.similarity_warnings.append(
                    f"{device.value} overused: {count} times (limit: {limit})"
                )
        
        # Check for repetitive patterns
        if len(self.action_sequences) >= 3:
            recent_actions = [a[1] for a in self.action_sequences[-5:]]
            action_counts = Counter(recent_actions)
            for action, count in action_counts.items():
                if count >= 3:
                    report.similarity_warnings.append(
                        f"Repetitive action type: {action} ({count} times recently)"
                    )
        
        # Find unique elements
        unique_elements = [
            elem for elem in self.plot_elements 
            if elem.uniqueness_score >= 0.8
        ]
        report.unique_elements = [
            f"Chapter {elem.chapter}: {elem.description[:50]}..."
            for elem in unique_elements[:5]
        ]
        
        # Calculate overall originality score
        if self.plot_elements:
            avg_uniqueness = sum(e.uniqueness_score for e in self.plot_elements) / len(self.plot_elements)
            repetition_penalty = len(report.overused_devices) * 0.1
            report.originality_score = max(0.0, min(1.0, avg_uniqueness - repetition_penalty))
        
        # Generate suggestions
        if report.overused_devices:
            for device in report.overused_devices[:3]:
                alternatives = self.suggest_alternatives(device)
                report.suggestions.append(
                    f"Replace {device.value} with: {', '.join(alternatives[:2])}"
                )
        
        if report.originality_score < 0.7:
            report.suggestions.append("Add more unique plot twists and unexpected developments")
            report.suggestions.append("Vary action sequences and locations")
            report.suggestions.append("Introduce subplots with different tones and pacing")
        
        return report
    
    def generate_originality_prompt(self) -> str:
        """
        Generate a prompt to ensure plot originality.
        
        Returns:
            Prompt for maintaining originality
        """
        # Get devices to avoid
        overused = [
            device.value for device, count in self.device_usage.items()
            if count >= self.DEVICE_LIMITS.get(device, 3)
        ]
        
        # Get fresh alternatives
        fresh_ideas = []
        if overused:
            for device_str in overused[:3]:
                try:
                    device = PlotDevice(device_str)
                    if device in self.PLOT_ALTERNATIVES:
                        fresh_ideas.extend(self.PLOT_ALTERNATIVES[device][:2])
                except ValueError:
                    pass
        
        prompt = f"""
        PLOT ORIGINALITY REQUIREMENTS:
        
        Avoid Overused Elements:
        {chr(10).join(f'- {device}' for device in overused) if overused else '- None currently overused'}
        
        Fresh Plot Elements to Consider:
        {chr(10).join(f'- {idea}' for idea in fresh_ideas) if fresh_ideas else '- Focus on unique, unexpected developments'}
        
        Unused Locations to Explore:
        - Underwater facility
        - Arctic research station
        - Space station
        - Underground bunker
        - Abandoned theme park
        - Cargo ship at sea
        - Mountain monastery
        - Desert compound
        
        Creative Guidelines:
        1. Subvert reader expectations with unexpected twists
        2. Use psychological tension instead of repetitive action
        3. Develop character-driven conflicts, not just external threats
        4. Introduce moral dilemmas and grey areas
        5. Create unique combinations of familiar elements
        6. Focus on consequences of actions, not just the actions themselves
        
        Remember: Originality comes from HOW events unfold, not just WHAT happens.
        Make familiar situations fresh through unique character reactions and unexpected outcomes.
        """
        
        return prompt
    
    def track_chapter(self, chapter_num: int, content: str) -> None:
        """
        Track plot elements from a chapter.
        
        Args:
            chapter_num: Chapter number
            content: Chapter content
        """
        import re
        from collections import Counter
        
        # Detect plot devices in the content
        content_lower = content.lower()
        
        # Check for common plot devices
        device_patterns = {
            PlotDevice.BETRAYAL: r'betray|betrayed|betrayal|double.?cross|backstab',
            PlotDevice.CAR_CHASE: r'car chase|pursuit|high.?speed|sped away|tires screeching',
            PlotDevice.EXPLOSION: r'explod|detonat|blast|bomb',
            PlotDevice.POISONING: r'poison|toxic|venom|drugged',
            PlotDevice.INFILTRATION: r'infiltrat|sneak|stealth|break.?in|penetrat',
            PlotDevice.HOSTAGE_SITUATION: r'hostage|captive|ransom|kidnap',
            PlotDevice.HACK_COMPUTER: r'hack|decrypt|firewall|cyber|malware',
            PlotDevice.FLASHBACK: r'remembered|years ago|flashback|recalled|memory of',
        }
        
        for device, pattern in device_patterns.items():
            if re.search(pattern, content_lower):
                # Extract context around the match
                matches = re.finditer(pattern, content_lower)
                for match in matches:
                    start = max(0, match.start() - 50)
                    end = min(len(content), match.end() + 50)
                    context = content[start:end]
                    
                    # Track this plot element
                    self.track_plot_element(
                        device=device,
                        chapter=chapter_num,
                        description=context[:100],
                        characters=[],  # Would need character extraction
                        location=None
                    )
                    break  # Only track once per device per chapter
        
        self._save_data()
    
    def validate_plot_diversity(self, chapter_num: int) -> List[str]:
        """
        Validate plot diversity up to this chapter.
        
        Args:
            chapter_num: Current chapter number
            
        Returns:
            List of diversity warnings
        """
        warnings = []
        
        # Check for repetitive devices in recent chapters
        recent_elements = [
            elem for elem in self.plot_elements 
            if elem.chapter >= chapter_num - 3
        ]
        
        if len(recent_elements) >= 3:
            recent_devices = [elem.device_type for elem in recent_elements]
            device_counts = Counter(recent_devices)
            
            for device, count in device_counts.items():
                if count >= 2:
                    warnings.append(
                        f"Plot device '{device.value}' used {count} times in recent chapters"
                    )
        
        # Check for lack of variety
        if len(set(elem.device_type for elem in self.plot_elements)) < 5 and len(self.plot_elements) >= 10:
            warnings.append("Limited plot device variety - consider introducing new elements")
        
        return warnings
    
    def validate_chapter_originality(self, chapter: int, chapter_text: str) -> Tuple[bool, List[str]]:
        """
        Validate chapter originality and provide feedback.
        
        Args:
            chapter: Chapter number
            chapter_text: Chapter content
            
        Returns:
            Tuple of (is_original, list_of_issues)
        """
        issues = []
        
        # Detect plot devices
        detected = self._detect_plot_devices(chapter_text)
        
        for device in detected:
            current_usage = self.device_usage.get(device, 0)
            limit = self.DEVICE_LIMITS.get(device, 3)
            
            if current_usage >= limit:
                issues.append(
                    f"Overused plot device: {device.value} "
                    f"(already used {current_usage} times, limit is {limit})"
                )
        
        # Check for repetitive action
        action_type = self._classify_action_sequence(chapter_text)
        if action_type:
            recent_count = sum(1 for _, a in self.action_sequences[-5:] if a == action_type)
            if recent_count >= 2:
                issues.append(f"Repetitive action sequence: {action_type}")
        
        # Check for clichéd phrases
        cliches = [
            "it was all a dream",
            "chosen one",
            "ancient prophecy",
            "evil twin",
            "amnesia",
            "love triangle"
        ]
        
        text_lower = chapter_text.lower()
        for cliche in cliches:
            if cliche in text_lower:
                issues.append(f"Clichéd plot element: {cliche}")
        
        is_original = len(issues) == 0
        return is_original, issues