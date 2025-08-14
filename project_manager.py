"""
Project Management System for Book Isolation and Organization
"""
import os
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class ProjectMetadata:
    """Project metadata structure"""
    project_id: str
    title: str
    created_at: str
    modified_at: str
    language: str
    style: str
    status: str  # draft, in_progress, completed, archived
    word_count: int = 0
    chapter_count: int = 0
    format_exports: List[str] = None
    tags: List[str] = None
    
    def to_dict(self):
        return asdict(self)

class ProjectManager:
    """Manages book projects with complete isolation"""
    
    def __init__(self, base_dir: str = "projects"):
        """
        Initialize project manager
        
        Args:
            base_dir: Base directory for all projects
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        self.config_file = self.base_dir / "projects.json"
        self.current_project_file = self.base_dir / ".current_project"
        self.logger = logging.getLogger(__name__)
        
        self.projects = self._load_projects()
        self.current_project = self._load_current_project()
        
    def _load_projects(self) -> Dict[str, ProjectMetadata]:
        """Load project registry"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return {
                        pid: ProjectMetadata(**pdata) 
                        for pid, pdata in data.items()
                    }
            except Exception as e:
                self.logger.error(f"Failed to load projects: {e}")
        return {}
        
    def _save_projects(self):
        """Save project registry"""
        data = {
            pid: meta.to_dict() 
            for pid, meta in self.projects.items()
        }
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
    def _load_current_project(self) -> Optional[str]:
        """Load current active project"""
        if self.current_project_file.exists():
            return self.current_project_file.read_text().strip()
        return None
        
    def _save_current_project(self, project_id: str = None):
        """Save current active project"""
        if project_id:
            self.current_project_file.write_text(project_id)
        elif self.current_project_file.exists():
            self.current_project_file.unlink()
            
    def create_project(self, 
                       title: str,
                       language: str = "English",
                       style: str = "general",
                       **kwargs) -> str:
        """
        Create a new isolated project
        
        Args:
            title: Book title
            language: Book language
            style: Writing style
            
        Returns:
            Project ID
        """
        # Generate unique project ID
        project_id = hashlib.md5(
            f"{title}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Create project directory
        project_dir = self.base_dir / project_id
        project_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (project_dir / "content").mkdir(exist_ok=True)
        (project_dir / "exports").mkdir(exist_ok=True)
        (project_dir / "cache").mkdir(exist_ok=True)
        (project_dir / "characters").mkdir(exist_ok=True)
        (project_dir / "assets").mkdir(exist_ok=True)
        
        # Create metadata
        metadata = ProjectMetadata(
            project_id=project_id,
            title=title,
            created_at=datetime.now().isoformat(),
            modified_at=datetime.now().isoformat(),
            language=language,
            style=style,
            status="draft",
            format_exports=[],
            tags=kwargs.get('tags', [])
        )
        
        # Save project info
        self.projects[project_id] = metadata
        self._save_projects()
        
        # Create initial project config
        project_config = {
            'project_id': project_id,
            'title': title,
            'language': language,
            'style': style,
            'instructions': kwargs.get('instructions', ''),
            'settings': {
                'auto_save': True,
                'backup_enabled': True,
                'cache_enabled': True,
                'isolation_mode': 'strict'
            }
        }
        
        config_path = project_dir / "project.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(project_config, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Created project {project_id}: {title}")
        
        # Set as current project
        self.switch_project(project_id)
        
        return project_id
        
    def switch_project(self, project_id: str):
        """
        Switch to a different project
        
        Args:
            project_id: Project to switch to
        """
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
            
        self.current_project = project_id
        self._save_current_project(project_id)
        
        # Clear any cached data from previous project
        self._clear_runtime_cache()
        
        self.logger.info(f"Switched to project {project_id}")
        
    def get_current_project(self) -> Optional[ProjectMetadata]:
        """Get current project metadata"""
        if self.current_project:
            return self.projects.get(self.current_project)
        return None
        
    def get_project_dir(self, project_id: str = None) -> Path:
        """
        Get project directory path
        
        Args:
            project_id: Project ID (uses current if None)
            
        Returns:
            Project directory path
        """
        pid = project_id or self.current_project
        if not pid:
            raise ValueError("No project selected")
        return self.base_dir / pid
        
    def list_projects(self, 
                     status: str = None,
                     style: str = None) -> List[ProjectMetadata]:
        """
        List all projects with optional filtering
        
        Args:
            status: Filter by status
            style: Filter by style
            
        Returns:
            List of project metadata
        """
        projects = list(self.projects.values())
        
        if status:
            projects = [p for p in projects if p.status == status]
        if style:
            projects = [p for p in projects if p.style == style]
            
        # Sort by modified date (newest first)
        projects.sort(key=lambda x: x.modified_at, reverse=True)
        
        return projects
        
    def delete_project(self, 
                      project_id: str,
                      confirm: bool = False) -> bool:
        """
        Delete a project and all its data
        
        Args:
            project_id: Project to delete
            confirm: Safety confirmation
            
        Returns:
            Success status
        """
        if not confirm:
            self.logger.warning("Delete cancelled - confirmation required")
            return False
            
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
            
        # Can't delete current project
        if project_id == self.current_project:
            self.current_project = None
            self._save_current_project()
            
        # Delete directory
        project_dir = self.base_dir / project_id
        if project_dir.exists():
            shutil.rmtree(project_dir)
            
        # Remove from registry
        del self.projects[project_id]
        self._save_projects()
        
        self.logger.info(f"Deleted project {project_id}")
        return True
        
    def archive_project(self, 
                       project_id: str,
                       archive_dir: str = None) -> str:
        """
        Archive a project to save space
        
        Args:
            project_id: Project to archive
            archive_dir: Custom archive directory
            
        Returns:
            Archive file path
        """
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
            
        project_dir = self.base_dir / project_id
        archive_dir = Path(archive_dir or self.base_dir / "archives")
        archive_dir.mkdir(exist_ok=True)
        
        # Create archive
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{project_id}_{timestamp}"
        archive_path = archive_dir / archive_name
        
        # Create zip archive
        shutil.make_archive(str(archive_path), 'zip', project_dir)
        
        # Update project status
        self.projects[project_id].status = "archived"
        self.projects[project_id].modified_at = datetime.now().isoformat()
        self._save_projects()
        
        # Optionally delete original
        # shutil.rmtree(project_dir)
        
        self.logger.info(f"Archived project {project_id} to {archive_path}.zip")
        return f"{archive_path}.zip"
        
    def cleanup_old_projects(self, 
                           days: int = 30,
                           status: str = "draft",
                           dry_run: bool = True) -> List[str]:
        """
        Clean up old projects
        
        Args:
            days: Age threshold in days
            status: Only cleanup projects with this status
            dry_run: If True, only report what would be deleted
            
        Returns:
            List of cleaned project IDs
        """
        from datetime import timedelta
        
        threshold = datetime.now() - timedelta(days=days)
        to_clean = []
        
        for pid, meta in self.projects.items():
            modified = datetime.fromisoformat(meta.modified_at)
            
            if modified < threshold and meta.status == status:
                to_clean.append(pid)
                
                if not dry_run:
                    self.delete_project(pid, confirm=True)
                else:
                    self.logger.info(f"Would delete: {pid} ({meta.title})")
                    
        return to_clean
        
    def export_project_list(self, output_file: str = "project_list.json"):
        """Export project list for backup"""
        data = {
            'exported_at': datetime.now().isoformat(),
            'projects': {
                pid: meta.to_dict() 
                for pid, meta in self.projects.items()
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        self.logger.info(f"Exported project list to {output_file}")
        
    def get_project_stats(self, project_id: str = None) -> Dict[str, Any]:
        """
        Get project statistics
        
        Args:
            project_id: Project ID (uses current if None)
            
        Returns:
            Project statistics
        """
        pid = project_id or self.current_project
        if not pid:
            raise ValueError("No project selected")
            
        project_dir = self.base_dir / pid
        meta = self.projects[pid]
        
        # Calculate sizes
        def get_dir_size(path):
            total = 0
            for entry in path.rglob('*'):
                if entry.is_file():
                    total += entry.stat().st_size
            return total
            
        content_size = get_dir_size(project_dir / "content")
        export_size = get_dir_size(project_dir / "exports")
        cache_size = get_dir_size(project_dir / "cache")
        total_size = get_dir_size(project_dir)
        
        # Count files
        export_formats = []
        if (project_dir / "exports").exists():
            export_formats = [
                f.suffix for f in (project_dir / "exports").iterdir()
                if f.is_file()
            ]
            
        return {
            'project_id': pid,
            'title': meta.title,
            'status': meta.status,
            'created': meta.created_at,
            'modified': meta.modified_at,
            'sizes': {
                'content': content_size,
                'exports': export_size,
                'cache': cache_size,
                'total': total_size
            },
            'counts': {
                'chapters': meta.chapter_count,
                'words': meta.word_count,
                'exports': len(export_formats)
            },
            'formats': list(set(export_formats))
        }
        
    def _clear_runtime_cache(self):
        """Clear runtime cache when switching projects"""
        # Clear any in-memory caches
        try:
            from cache_manager import get_cache
            cache = get_cache()
            cache.clear()
        except:
            pass
            
        # Clear token optimizer state
        try:
            from token_optimizer import get_optimizer
            optimizer = get_optimizer()
            optimizer.book_manager = None
        except:
            pass
            
    def update_project_metadata(self,
                               project_id: str = None,
                               **updates):
        """Update project metadata"""
        pid = project_id or self.current_project
        if not pid or pid not in self.projects:
            raise ValueError(f"Project {pid} not found")
            
        meta = self.projects[pid]
        for key, value in updates.items():
            if hasattr(meta, key):
                setattr(meta, key, value)
                
        meta.modified_at = datetime.now().isoformat()
        self._save_projects()

# Global project manager instance
_project_manager = None

def get_project_manager() -> ProjectManager:
    """Get global project manager instance"""
    global _project_manager
    if _project_manager is None:
        _project_manager = ProjectManager()
    return _project_manager

def get_current_project_dir() -> Path:
    """Get current project directory"""
    pm = get_project_manager()
    return pm.get_project_dir()

def ensure_project_isolation(func):
    """Decorator to ensure project isolation"""
    def wrapper(*args, **kwargs):
        pm = get_project_manager()
        if not pm.current_project:
            raise ValueError("No project selected. Create or switch to a project first.")
        
        # Set working directory context
        original_cwd = os.getcwd()
        project_dir = pm.get_project_dir()
        
        try:
            os.chdir(project_dir)
            return func(*args, **kwargs)
        finally:
            os.chdir(original_cwd)
            
    return wrapper