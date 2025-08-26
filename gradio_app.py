"""
GhostWriter AI - Gradio Web Interface
Professional web UI for AI-powered book generation system
"""

import gradio as gr
import logging
import os
import sys
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime
import traceback
import json
from pathlib import Path

# Import core modules
from containers import get_container, init_container
from events import event_manager, EventType, Event
from exceptions import GhostwriterException
from project_manager import ProjectManager
from app_config import settings

# Import services
from services.generation_service import GenerationService
from providers.factory import ProviderFactory
from style_templates import StyleManager
from character_tracker import CharacterDatabase
from export_formats import BookExporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GradioInterface:
    """Main Gradio interface for GhostWriter AI"""
    
    def __init__(self):
        """Initialize Gradio interface with dependency injection"""
        self.container = None
        self.project_manager = None
        self.generation_service = None
        self.style_manager = StyleManager()
        self.character_db = None
        self.current_project_id = None
        self.generation_active = False
        self.setup_container()
        
    def setup_container(self):
        """Initialize DI container and services"""
        try:
            self.container = get_container()
            self.project_manager = ProjectManager()
            logger.info("✅ Container and services initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize container: {e}")
            raise
    
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface with all tabs"""
        
        with gr.Blocks(
            title="📚 GhostWriter AI - Advanced Book Generation System",
            theme=gr.themes.Soft(
                primary_hue="indigo",
                secondary_hue="purple",
                neutral_hue="slate",
                font=["Inter", "system-ui", "sans-serif"]
            ),
            css=self.get_custom_css()
        ) as app:
            
            # State management
            state = gr.State({
                "current_project_id": None,
                "generation_active": False,
                "selected_provider": settings.LLM_PROVIDER,
                "selected_style": "thriller",
                "current_chapter": 0,
                "total_chapters": 0
            })
            
            # Header
            with gr.Row():
                gr.Markdown(
                    """
                    # 📚 GhostWriter AI
                    ### Advanced AI-Powered Book Generation System
                    *Powered by GPT-5, Claude 4, Gemini 2.5, and more*
                    """,
                    elem_classes="header-title"
                )
            
            # Main tabs
            with gr.Tabs() as tabs:
                
                # Tab 1: Project Management
                with gr.Tab("📁 Projects", id="tab_projects"):
                    self.create_projects_tab(state)
                
                # Tab 2: Book Generation
                with gr.Tab("✍️ Generate Book", id="tab_generate"):
                    self.create_generation_tab(state)
                
                # Tab 3: Character Management
                with gr.Tab("👥 Characters", id="tab_characters"):
                    self.create_characters_tab(state)
                
                # Tab 4: Styles & Templates
                with gr.Tab("🎨 Styles", id="tab_styles"):
                    self.create_styles_tab(state)
                
                # Tab 5: Monitoring & Analytics
                with gr.Tab("📊 Analytics", id="tab_analytics"):
                    self.create_analytics_tab(state)
                
                # Tab 6: Export
                with gr.Tab("📤 Export", id="tab_export"):
                    self.create_export_tab(state)
                
                # Tab 7: Settings
                with gr.Tab("⚙️ Settings", id="tab_settings"):
                    self.create_settings_tab(state)
            
            # Footer
            with gr.Row():
                gr.Markdown(
                    """
                    ---
                    *GhostWriter AI v2.0 | © 2025 | [GitHub](https://github.com/yourusername/ghostwriter-ai) | [Documentation](docs/)*
                    """,
                    elem_classes="footer"
                )
            
            # Setup event handlers
            self.setup_event_handlers(app, state)
            
        return app
    
    def create_projects_tab(self, state):
        """Create project management tab"""
        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("## 📋 Your Book Projects")
                
                # Project list
                project_list = gr.Dataframe(
                    headers=["ID", "Title", "Status", "Language", "Chapters", "Words", "Created", "Modified"],
                    datatype=["str", "str", "str", "str", "number", "number", "date", "date"],
                    interactive=False,
                    elem_id="project_list"
                )
                
                # Action buttons
                with gr.Row():
                    btn_refresh = gr.Button("🔄 Refresh", variant="secondary")
                    btn_new = gr.Button("➕ New Project", variant="primary")
                    btn_open = gr.Button("📖 Open", variant="secondary")
                    btn_delete = gr.Button("🗑️ Delete", variant="stop")
                    btn_archive = gr.Button("📦 Archive", variant="secondary")
            
            with gr.Column(scale=2):
                gr.Markdown("## 📝 Project Details")
                
                # New project form (initially hidden)
                with gr.Group(visible=False) as new_project_form:
                    gr.Markdown("### Create New Project")
                    new_title = gr.Textbox(
                        label="Book Title",
                        placeholder="Enter your book title...",
                        max_lines=1
                    )
                    new_language = gr.Dropdown(
                        label="Language",
                        choices=["English", "Spanish", "French", "German", "Italian", "Portuguese", 
                                "Russian", "Chinese", "Japanese", "Korean", "Arabic", "Hindi", "Polish"],
                        value="English"
                    )
                    new_style = gr.Dropdown(
                        label="Writing Style",
                        choices=self.get_available_styles(),
                        value="thriller"
                    )
                    new_instructions = gr.Textbox(
                        label="Book Instructions",
                        placeholder="Describe what your book should be about...",
                        lines=5
                    )
                    new_chapters = gr.Slider(
                        label="Number of Chapters",
                        minimum=5,
                        maximum=100,
                        value=20,
                        step=1
                    )
                    
                    with gr.Row():
                        btn_create = gr.Button("✅ Create Project", variant="primary")
                        btn_cancel = gr.Button("❌ Cancel", variant="secondary")
                
                # Project info display
                with gr.Group(visible=True) as project_info:
                    project_details = gr.JSON(label="Selected Project Details", visible=False)
                    project_stats = gr.Markdown("*Select a project to view details*")
        
        # Event handlers for project tab
        def refresh_projects():
            try:
                projects = self.project_manager.list_projects()
                data = []
                for p in projects:
                    data.append([
                        p.get('id', ''),
                        p.get('title', ''),
                        p.get('status', 'draft'),
                        p.get('language', 'English'),
                        p.get('chapters_count', 0),
                        p.get('word_count', 0),
                        p.get('created_at', ''),
                        p.get('modified_at', '')
                    ])
                return data
            except Exception as e:
                logger.error(f"Failed to refresh projects: {e}")
                return []
        
        def show_new_project_form():
            return gr.update(visible=True), gr.update(visible=False)
        
        def hide_new_project_form():
            return gr.update(visible=False), gr.update(visible=True)
        
        def create_new_project(title, language, style, instructions, chapters):
            try:
                if not title:
                    return gr.update(), "❌ Please enter a book title"
                
                project_id = self.project_manager.create_project(
                    title=title,
                    language=language,
                    metadata={
                        'style': style,
                        'instructions': instructions,
                        'target_chapters': chapters
                    }
                )
                
                return refresh_projects(), f"✅ Project '{title}' created successfully!"
            except Exception as e:
                return gr.update(), f"❌ Failed to create project: {str(e)}"
        
        # Connect events
        btn_refresh.click(refresh_projects, outputs=[project_list])
        btn_new.click(show_new_project_form, outputs=[new_project_form, project_info])
        btn_cancel.click(hide_new_project_form, outputs=[new_project_form, project_info])
        btn_create.click(
            create_new_project,
            inputs=[new_title, new_language, new_style, new_instructions, new_chapters],
            outputs=[project_list, project_stats]
        )
        
        # Load projects on start
        project_list.value = refresh_projects()
    
    def create_generation_tab(self, state):
        """Create book generation tab"""
        with gr.Row():
            # Left column - Controls
            with gr.Column(scale=2):
                gr.Markdown("## 🎮 Generation Controls")
                
                # Project selector
                with gr.Group():
                    current_project = gr.Dropdown(
                        label="Select Project",
                        choices=self.get_project_choices(),
                        interactive=True
                    )
                    
                    # Generation parameters
                    gr.Markdown("### 📝 Book Parameters")
                    
                    gen_provider = gr.Dropdown(
                        label="AI Provider",
                        choices=["openai", "anthropic", "gemini", "cohere", "openrouter"],
                        value=settings.LLM_PROVIDER
                    )
                    
                    gen_model = gr.Dropdown(
                        label="Model",
                        choices=self.get_model_choices("openai"),
                        value="gpt-5"
                    )
                    
                    gen_temperature = gr.Slider(
                        label="Creativity (Temperature)",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1
                    )
                    
                    # Advanced options
                    with gr.Accordion("🔧 Advanced Options", open=False):
                        enable_rag = gr.Checkbox(label="Enable RAG System", value=True)
                        enable_quality = gr.Checkbox(label="Enable Quality Validators", value=True)
                        max_tokens = gr.Number(label="Max Tokens per Chapter", value=16000)
                        enable_cache = gr.Checkbox(label="Enable Response Cache", value=True)
                    
                    # Control buttons
                    with gr.Row():
                        btn_start = gr.Button("▶️ Start Generation", variant="primary", size="lg")
                        btn_pause = gr.Button("⏸️ Pause", variant="secondary", size="lg")
                        btn_stop = gr.Button("⏹️ Stop", variant="stop", size="lg")
                    
                    with gr.Row():
                        btn_resume = gr.Button("⏮️ Resume", variant="secondary")
                        btn_regenerate = gr.Button("🔄 Regenerate Chapter", variant="secondary")
            
            # Right column - Progress & Monitoring
            with gr.Column(scale=3):
                gr.Markdown("## 📊 Generation Progress")
                
                # Progress indicators
                with gr.Group():
                    progress_bar = gr.Progress()
                    progress_text = gr.Markdown("Ready to generate...")
                    
                    with gr.Row():
                        current_chapter_text = gr.Textbox(
                            label="Current Chapter",
                            value="0 / 0",
                            interactive=False
                        )
                        eta_text = gr.Textbox(
                            label="Estimated Time",
                            value="--:--",
                            interactive=False
                        )
                        tokens_used = gr.Textbox(
                            label="Tokens Used",
                            value="0",
                            interactive=False
                        )
                    
                    # Live logs
                    generation_logs = gr.Textbox(
                        label="📜 Generation Logs",
                        lines=15,
                        max_lines=30,
                        interactive=False,
                        placeholder="Generation logs will appear here...",
                        autoscroll=True
                    )
                    
                    # Chapter preview
                    with gr.Accordion("📖 Chapter Preview", open=False):
                        chapter_preview = gr.Markdown("*Chapter content will appear here during generation*")
        
        # Event handlers for generation
        def update_model_choices(provider):
            return gr.update(choices=self.get_model_choices(provider))
        
        def start_generation(project_id, provider, model, temperature, enable_rag, enable_quality):
            if not project_id:
                return "❌ Please select a project first", "", "", ""
            
            try:
                # Initialize generation service with selected provider
                self.setup_generation_service(provider, model, temperature)
                
                # Start generation (this would be async in production)
                self.generation_active = True
                return "🚀 Generation started!", "1 / 20", "~2 hours", "0"
                
            except Exception as e:
                return f"❌ Failed to start generation: {str(e)}", "", "", ""
        
        # Connect events
        gen_provider.change(update_model_choices, inputs=[gen_provider], outputs=[gen_model])
        btn_start.click(
            start_generation,
            inputs=[current_project, gen_provider, gen_model, gen_temperature, enable_rag, enable_quality],
            outputs=[progress_text, current_chapter_text, eta_text, tokens_used]
        )
    
    def create_characters_tab(self, state):
        """Create character management tab"""
        with gr.Row():
            # Character list
            with gr.Column(scale=2):
                gr.Markdown("## 👥 Character Database")
                
                character_list = gr.Dataframe(
                    headers=["Name", "Role", "Personality", "Chapters"],
                    interactive=False
                )
                
                with gr.Row():
                    btn_add_character = gr.Button("➕ Add Character", variant="primary")
                    btn_edit_character = gr.Button("✏️ Edit", variant="secondary")
                    btn_delete_character = gr.Button("🗑️ Delete", variant="stop")
            
            # Character editor
            with gr.Column(scale=3):
                gr.Markdown("## ✏️ Character Editor")
                
                with gr.Group():
                    char_name = gr.Textbox(label="Character Name", placeholder="John Doe")
                    char_role = gr.Dropdown(
                        label="Role",
                        choices=["Protagonist", "Antagonist", "Supporting", "Minor"],
                        value="Supporting"
                    )
                    char_description = gr.Textbox(
                        label="Description",
                        lines=3,
                        placeholder="Physical appearance, background, etc."
                    )
                    
                    # OCEAN personality model
                    gr.Markdown("### 🧠 Personality Traits (OCEAN Model)")
                    with gr.Row():
                        ocean_o = gr.Slider(label="Openness", minimum=0, maximum=100, value=50)
                        ocean_c = gr.Slider(label="Conscientiousness", minimum=0, maximum=100, value=50)
                    with gr.Row():
                        ocean_e = gr.Slider(label="Extraversion", minimum=0, maximum=100, value=50)
                        ocean_a = gr.Slider(label="Agreeableness", minimum=0, maximum=100, value=50)
                    ocean_n = gr.Slider(label="Neuroticism", minimum=0, maximum=100, value=50)
                    
                    # Relationships
                    gr.Markdown("### 🤝 Relationships")
                    relationships = gr.Dataframe(
                        headers=["Character", "Relationship", "Strength"],
                        datatype=["str", "str", "number"],
                        interactive=True
                    )
                    
                    with gr.Row():
                        btn_save_character = gr.Button("💾 Save Character", variant="primary")
                        btn_cancel_character = gr.Button("❌ Cancel", variant="secondary")
    
    def create_styles_tab(self, state):
        """Create styles and templates tab"""
        gr.Markdown("## 🎨 Writing Styles Gallery")
        
        with gr.Row():
            # Style cards grid
            with gr.Column(scale=3):
                style_gallery = gr.Gallery(
                    label="Available Styles",
                    show_label=False,
                    elem_id="style_gallery",
                    columns=3,
                    rows=3,
                    height="auto"
                )
                
                # Style filter
                with gr.Row():
                    style_category = gr.Dropdown(
                        label="Category",
                        choices=["All", "Fiction", "Non-Fiction", "Academic", "Creative"],
                        value="All"
                    )
                    style_rating = gr.Dropdown(
                        label="Content Rating",
                        choices=["All", "E - Everyone", "T - Teen", "M - Mature", "A - Adult"],
                        value="All"
                    )
            
            # Style details
            with gr.Column(scale=2):
                gr.Markdown("## 📋 Style Details")
                
                with gr.Group():
                    selected_style_name = gr.Textbox(label="Style Name", interactive=False)
                    selected_style_description = gr.Textbox(
                        label="Description",
                        lines=3,
                        interactive=False
                    )
                    
                    # Style parameters
                    gr.Markdown("### Parameters")
                    style_tone = gr.Textbox(label="Tone", interactive=False)
                    style_vocabulary = gr.Textbox(label="Vocabulary Level", interactive=False)
                    style_pacing = gr.Textbox(label="Pacing", interactive=False)
                    
                    # Example text
                    gr.Markdown("### 📖 Example Output")
                    style_example = gr.Textbox(
                        label="",
                        lines=5,
                        interactive=False,
                        placeholder="Select a style to see example..."
                    )
                    
                    with gr.Row():
                        btn_use_style = gr.Button("✅ Use This Style", variant="primary")
                        btn_customize = gr.Button("🔧 Customize", variant="secondary")
    
    def create_analytics_tab(self, state):
        """Create analytics and monitoring tab"""
        gr.Markdown("## 📊 Generation Analytics & Metrics")
        
        with gr.Row():
            # Real-time metrics
            with gr.Column():
                gr.Markdown("### ⚡ Real-Time Metrics")
                
                with gr.Group():
                    metric_status = gr.Textbox(label="Status", value="Idle", interactive=False)
                    metric_speed = gr.Textbox(label="Generation Speed", value="0 words/min", interactive=False)
                    metric_tokens = gr.Number(label="Total Tokens Used", value=0, interactive=False)
                    metric_cost = gr.Textbox(label="Estimated Cost", value="$0.00", interactive=False)
            
            # Quality metrics
            with gr.Column():
                gr.Markdown("### ✨ Quality Metrics")
                
                with gr.Group():
                    quality_narrative = gr.Slider(
                        label="Narrative Consistency",
                        minimum=0,
                        maximum=100,
                        value=0,
                        interactive=False
                    )
                    quality_character = gr.Slider(
                        label="Character Consistency",
                        minimum=0,
                        maximum=100,
                        value=0,
                        interactive=False
                    )
                    quality_dialogue = gr.Slider(
                        label="Dialogue Quality",
                        minimum=0,
                        maximum=100,
                        value=0,
                        interactive=False
                    )
                    quality_originality = gr.Slider(
                        label="Plot Originality",
                        minimum=0,
                        maximum=100,
                        value=0,
                        interactive=False
                    )
        
        # Charts
        with gr.Row():
            gr.Markdown("### 📈 Performance Charts")
            # Placeholder for Plotly charts
            performance_chart = gr.Plot(label="Generation Timeline")
            token_usage_chart = gr.Plot(label="Token Usage Distribution")
    
    def create_export_tab(self, state):
        """Create export tab"""
        gr.Markdown("## 📤 Export Your Book")
        
        with gr.Row():
            # Export options
            with gr.Column(scale=2):
                gr.Markdown("### 📚 Select Project to Export")
                
                export_project = gr.Dropdown(
                    label="Project",
                    choices=self.get_project_choices(),
                    interactive=True
                )
                
                gr.Markdown("### 📄 Export Formats")
                
                with gr.Group():
                    export_epub = gr.Checkbox(label="📖 EPUB (E-readers)", value=True)
                    export_pdf = gr.Checkbox(label="📄 PDF (Print)", value=False)
                    export_docx = gr.Checkbox(label="📝 DOCX (Word)", value=False)
                    export_html = gr.Checkbox(label="🌐 HTML (Web)", value=False)
                    export_txt = gr.Checkbox(label="📃 TXT (Plain Text)", value=False)
                
                gr.Markdown("### 📝 Metadata")
                
                meta_author = gr.Textbox(label="Author Name", placeholder="Your name")
                meta_publisher = gr.Textbox(label="Publisher", placeholder="Self-published")
                meta_isbn = gr.Textbox(label="ISBN (optional)", placeholder="978-...")
                meta_cover = gr.File(label="Cover Image (optional)", file_types=["image"])
                
                btn_export = gr.Button("🚀 Start Export", variant="primary", size="lg")
            
            # Export status
            with gr.Column(scale=3):
                gr.Markdown("### 📊 Export Status")
                
                export_status = gr.Textbox(
                    label="Status",
                    value="Ready to export...",
                    interactive=False
                )
                
                export_progress = gr.Progress()
                
                gr.Markdown("### 📥 Download Links")
                
                download_area = gr.HTML(
                    value="<p style='color: gray;'>Export links will appear here...</p>"
                )
                
                # Preview area
                with gr.Accordion("👁️ Preview", open=False):
                    export_preview = gr.HTML(label="Preview")
    
    def create_settings_tab(self, state):
        """Create settings tab"""
        gr.Markdown("## ⚙️ Application Settings")
        
        with gr.Tabs():
            # API Configuration
            with gr.Tab("🔑 API Keys"):
                gr.Markdown("### Configure API Keys")
                gr.Markdown("⚠️ **Note:** API keys are stored securely and never displayed after saving.")
                
                with gr.Column():
                    api_openai = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        placeholder="sk-...",
                        value=""
                    )
                    api_anthropic = gr.Textbox(
                        label="Anthropic API Key",
                        type="password",
                        placeholder="sk-ant-...",
                        value=""
                    )
                    api_gemini = gr.Textbox(
                        label="Google Gemini API Key",
                        type="password",
                        placeholder="...",
                        value=""
                    )
                    api_cohere = gr.Textbox(
                        label="Cohere API Key",
                        type="password",
                        placeholder="...",
                        value=""
                    )
                    api_openrouter = gr.Textbox(
                        label="OpenRouter API Key",
                        type="password",
                        placeholder="sk-or-...",
                        value=""
                    )
                    
                    btn_save_api = gr.Button("💾 Save API Keys", variant="primary")
                    api_status = gr.Markdown("")
            
            # General Settings
            with gr.Tab("🎛️ General"):
                gr.Markdown("### General Settings")
                
                with gr.Column():
                    setting_theme = gr.Radio(
                        label="Theme",
                        choices=["Light", "Dark", "Auto"],
                        value="Light"
                    )
                    setting_language = gr.Dropdown(
                        label="Interface Language",
                        choices=["English", "Spanish", "French", "German", "Polish"],
                        value="English"
                    )
                    setting_autosave = gr.Slider(
                        label="Auto-save Interval (minutes)",
                        minimum=1,
                        maximum=60,
                        value=5,
                        step=1
                    )
                    setting_debug = gr.Checkbox(
                        label="Enable Debug Mode",
                        value=False
                    )
            
            # Advanced Settings
            with gr.Tab("🔧 Advanced"):
                gr.Markdown("### Advanced Configuration")
                
                with gr.Column():
                    adv_cache = gr.Dropdown(
                        label="Cache Backend",
                        choices=["memory", "redis", "file"],
                        value="memory"
                    )
                    adv_rag_mode = gr.Dropdown(
                        label="RAG Mode",
                        choices=["disabled", "basic", "hybrid", "full"],
                        value="hybrid"
                    )
                    adv_chunk_size = gr.Number(
                        label="RAG Chunk Size",
                        value=512,
                        minimum=128,
                        maximum=2048
                    )
                    adv_log_level = gr.Dropdown(
                        label="Log Level",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        value="INFO"
                    )
                    
                    btn_save_advanced = gr.Button("💾 Save Advanced Settings", variant="primary")
    
    def get_available_styles(self) -> List[str]:
        """Get list of available writing styles"""
        return [
            "thriller", "romance", "fantasy", "sci-fi", "mystery",
            "horror", "literary", "adventure", "historical", "comedy",
            "drama", "young_adult", "children", "academic", "technical"
        ]
    
    def get_project_choices(self) -> List[Tuple[str, str]]:
        """Get list of projects for dropdown"""
        try:
            projects = self.project_manager.list_projects()
            return [(f"{p['title']} ({p['id'][:8]}...)", p['id']) for p in projects]
        except:
            return []
    
    def get_model_choices(self, provider: str) -> List[str]:
        """Get available models for a provider"""
        models = {
            "openai": ["gpt-5", "gpt-5-mini", "gpt-4-turbo", "gpt-4"],
            "anthropic": ["claude-opus-4.1", "claude-sonnet-4", "claude-haiku-4"],
            "gemini": ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-pro"],
            "cohere": ["command-r-plus", "command-r", "command"],
            "openrouter": ["best", "auto", "anthropic/claude-opus", "openai/gpt-5"]
        }
        return models.get(provider, ["default"])
    
    def setup_generation_service(self, provider: str, model: str, temperature: float):
        """Initialize generation service with selected provider"""
        try:
            # Get provider configuration
            provider_config = {
                'provider': provider,
                'model': model,
                'temperature': temperature
            }
            
            # Create provider instance
            provider_instance = ProviderFactory.create(provider, provider_config)
            
            # Initialize generation service
            cache_manager = self.container.cache_manager()
            self.generation_service = GenerationService(
                provider=provider_instance,
                cache_manager=cache_manager
            )
            
            logger.info(f"Generation service initialized with {provider}/{model}")
            
        except Exception as e:
            logger.error(f"Failed to setup generation service: {e}")
            raise
    
    def setup_event_handlers(self, app, state):
        """Setup global event handlers for the interface"""
        
        # Subscribe to generation events
        def on_chapter_completed(event):
            logger.info(f"Chapter {event.data.get('chapter_number')} completed")
            # Update UI state
            state["current_chapter"] = event.data.get('chapter_number', 0)
        
        def on_generation_error(event):
            logger.error(f"Generation error: {event.data.get('error')}")
            state["generation_active"] = False
        
        event_manager.subscribe(EventType.CHAPTER_COMPLETED, on_chapter_completed)
        event_manager.subscribe(EventType.GENERATION_ERROR, on_generation_error)
    
    def get_custom_css(self) -> str:
        """Get custom CSS for the interface"""
        return """
        .header-title {
            background: linear-gradient(90deg, #4F46E5, #7C3AED);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 20px;
        }
        
        #project_list {
            max-height: 400px;
            overflow-y: auto;
        }
        
        #style_gallery {
            gap: 10px;
        }
        
        .footer {
            text-align: center;
            padding: 20px;
            color: #666;
        }
        
        /* Custom button styles */
        .gr-button-primary {
            background: linear-gradient(90deg, #4F46E5, #7C3AED) !important;
        }
        
        .gr-button-stop {
            background: #EF4444 !important;
        }
        
        /* Tab styling */
        .gr-tab {
            font-weight: 500;
        }
        
        .gr-tab-selected {
            border-bottom: 3px solid #4F46E5;
        }
        """


def launch_gradio_app(
    server_name: str = "0.0.0.0",
    server_port: int = 7860,
    share: bool = False,
    debug: bool = False
):
    """Launch the Gradio application"""
    
    logger.info("🚀 Starting GhostWriter AI Gradio Interface...")
    
    try:
        # Initialize interface
        interface = GradioInterface()
        app = interface.create_interface()
        
        # Launch configuration
        launch_config = {
            "server_name": server_name,
            "server_port": server_port,
            "share": share,
            "debug": debug,
            "show_error": True,
            "quiet": not debug,
            "show_api": False,
            "inbrowser": not share,  # Open browser only if not sharing
        }
        
        logger.info(f"📡 Launching server on http://{server_name}:{server_port}")
        
        # Launch the app
        app.launch(**launch_config)
        
    except KeyboardInterrupt:
        logger.info("\n👋 Shutting down Gradio interface...")
    except Exception as e:
        logger.error(f"Failed to launch Gradio app: {e}")
        logger.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GhostWriter AI - Gradio Web Interface")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    parser.add_argument("--share", action="store_true", help="Create public share link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Launch the application
    launch_gradio_app(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug
    )