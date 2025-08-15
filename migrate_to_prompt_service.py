"""
Migration script to transition from old prompt system to new PromptService.

This script helps migrate existing code to use the new comprehensive
PromptService while maintaining backward compatibility.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import click
import yaml

from services.prompt_service import PromptService, get_prompt_service
from services.prompt_wrapper import (
    chapter,
    chapter_topics,
    section,
    section_topics,
    summary,
    table_of_contents,
    title,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PromptMigrator:
    """Handles migration from old prompt system to new PromptService."""
    
    def __init__(self, dry_run: bool = True):
        """
        Initialize the migrator.
        
        Args:
            dry_run: If True, don't make actual changes
        """
        self.dry_run = dry_run
        self.old_imports = {
            "from prompts_templated import",
            "import prompts_templated",
            "from prompts import",
            "import prompts",
        }
        self.function_mapping = {
            "prompts.title": "title",
            "prompts.table_of_contents": "table_of_contents",
            "prompts.summary": "summary",
            "prompts.chapter_topics": "chapter_topics",
            "prompts.chapter": "chapter",
            "prompts.section_topics": "section_topics",
            "prompts.section": "section",
            "prompts_templated.title": "title",
            "prompts_templated.table_of_contents": "table_of_contents",
            "prompts_templated.summary": "summary",
            "prompts_templated.chapter_topics": "chapter_topics",
            "prompts_templated.chapter": "chapter",
            "prompts_templated.section_topics": "section_topics",
            "prompts_templated.section": "section",
        }
    
    def scan_codebase(self, root_dir: str) -> List[Path]:
        """
        Scan codebase for files using old prompt system.
        
        Args:
            root_dir: Root directory to scan
            
        Returns:
            List of files that need migration
        """
        root_path = Path(root_dir)
        files_to_migrate = []
        
        for py_file in root_path.rglob("*.py"):
            # Skip migration script itself
            if py_file.name == "migrate_to_prompt_service.py":
                continue
            
            # Skip test files for now
            if "test" in py_file.name:
                continue
            
            try:
                content = py_file.read_text(encoding="utf-8")
                
                # Check for old imports
                for old_import in self.old_imports:
                    if old_import in content:
                        files_to_migrate.append(py_file)
                        break
                
                # Check for old function calls
                for old_func in self.function_mapping:
                    if old_func in content:
                        if py_file not in files_to_migrate:
                            files_to_migrate.append(py_file)
                        break
            
            except Exception as e:
                logger.error(f"Error scanning {py_file}: {e}")
        
        return files_to_migrate
    
    def migrate_file(self, file_path: Path) -> Tuple[bool, str]:
        """
        Migrate a single file to use new PromptService.
        
        Args:
            file_path: Path to file to migrate
            
        Returns:
            Tuple of (success, message)
        """
        try:
            content = file_path.read_text(encoding="utf-8")
            original_content = content
            
            # Replace imports
            content = self._replace_imports(content)
            
            # Replace function calls
            content = self._replace_function_calls(content)
            
            # Only write if content changed
            if content != original_content:
                if not self.dry_run:
                    # Create backup
                    backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
                    file_path.rename(backup_path)
                    
                    # Write migrated content
                    file_path.write_text(content, encoding="utf-8")
                    
                    return True, f"Migrated {file_path} (backup: {backup_path})"
                else:
                    return True, f"Would migrate {file_path}"
            else:
                return True, f"No changes needed for {file_path}"
        
        except Exception as e:
            return False, f"Failed to migrate {file_path}: {e}"
    
    def _replace_imports(self, content: str) -> str:
        """Replace old imports with new ones."""
        # Replace old imports
        patterns = [
            (r"from prompts_templated import .*", 
             "from services.prompt_wrapper import (title, table_of_contents, summary, "
             "chapter_topics, chapter, section_topics, section)"),
            (r"import prompts_templated\n", 
             "from services import prompt_wrapper\n"),
            (r"from prompts import .*", 
             "from services.prompt_wrapper import (title, table_of_contents, summary, "
             "chapter_topics, chapter, section_topics, section)"),
            (r"import prompts\n", 
             "from services import prompt_wrapper\n"),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        return content
    
    def _replace_function_calls(self, content: str) -> str:
        """Replace old function calls with new ones."""
        for old_call, new_call in self.function_mapping.items():
            # Replace module.function with just function
            if "prompts." in old_call:
                content = content.replace(old_call, f"prompt_wrapper.{new_call}")
            elif "prompts_templated." in old_call:
                content = content.replace(old_call, new_call)
        
        return content
    
    def migrate_templates(
        self,
        old_template_path: str,
        new_template_path: str
    ) -> Tuple[bool, str]:
        """
        Migrate old template files to new format.
        
        Args:
            old_template_path: Path to old template file
            new_template_path: Path to new template file
            
        Returns:
            Tuple of (success, message)
        """
        try:
            old_path = Path(old_template_path)
            new_path = Path(new_template_path)
            
            if not old_path.exists():
                return False, f"Old template file not found: {old_path}"
            
            # Load old templates
            with open(old_path, "r", encoding="utf-8") as f:
                old_templates = yaml.safe_load(f)
            
            # Convert to new format
            new_templates = {}
            for name, template_data in old_templates.items():
                if isinstance(template_data, dict) and "template" in template_data:
                    # Already in correct format
                    new_templates[name] = template_data
                else:
                    # Convert simple string to new format
                    new_templates[name] = {
                        "template": template_data,
                        "description": f"Migrated {name} template",
                        "version": "1.0.0"
                    }
            
            # Save new templates
            if not self.dry_run:
                new_path.parent.mkdir(parents=True, exist_ok=True)
                with open(new_path, "w", encoding="utf-8") as f:
                    yaml.dump(new_templates, f, default_flow_style=False, allow_unicode=True)
                
                return True, f"Migrated templates to {new_path}"
            else:
                return True, f"Would migrate templates to {new_path}"
        
        except Exception as e:
            return False, f"Failed to migrate templates: {e}"


@click.command()
@click.option(
    "--root-dir",
    default=".",
    help="Root directory to scan for migration"
)
@click.option(
    "--dry-run/--execute",
    default=True,
    help="Perform dry run or execute migration"
)
@click.option(
    "--migrate-templates/--skip-templates",
    default=True,
    help="Migrate template files"
)
def main(root_dir: str, dry_run: bool, migrate_templates: bool):
    """
    Migrate codebase to use new PromptService.
    
    This script will:
    1. Scan for files using old prompt system
    2. Update imports to use new system
    3. Update function calls
    4. Migrate template files to new format
    5. Create backups of modified files
    """
    click.echo(f"Starting migration (dry_run={dry_run})")
    
    migrator = PromptMigrator(dry_run=dry_run)
    
    # Scan for files to migrate
    click.echo(f"Scanning {root_dir} for files to migrate...")
    files = migrator.scan_codebase(root_dir)
    
    if not files:
        click.echo("No files found that need migration.")
    else:
        click.echo(f"Found {len(files)} files to migrate:")
        for file in files:
            click.echo(f"  - {file}")
        
        # Migrate each file
        click.echo("\nMigrating files...")
        success_count = 0
        for file in files:
            success, message = migrator.migrate_file(file)
            if success:
                success_count += 1
                click.echo(f"  ✓ {message}")
            else:
                click.echo(f"  ✗ {message}")
        
        click.echo(f"\nMigrated {success_count}/{len(files)} files successfully")
    
    # Migrate templates
    if migrate_templates:
        click.echo("\nMigrating template files...")
        success, message = migrator.migrate_templates(
            "templates/prompts.yaml",
            "templates/prompts_migrated.yaml"
        )
        if success:
            click.echo(f"  ✓ {message}")
        else:
            click.echo(f"  ✗ {message}")
    
    # Print next steps
    if dry_run:
        click.echo("\n" + "="*50)
        click.echo("DRY RUN COMPLETE")
        click.echo("To execute the migration, run with --execute flag")
        click.echo("="*50)
    else:
        click.echo("\n" + "="*50)
        click.echo("MIGRATION COMPLETE")
        click.echo("Next steps:")
        click.echo("1. Review the changes in migrated files")
        click.echo("2. Run tests to ensure everything works")
        click.echo("3. Update any custom prompt templates as needed")
        click.echo("4. Remove backup files once verified")
        click.echo("="*50)


def verify_migration():
    """
    Verify that the migration was successful.
    
    This function runs basic tests to ensure the new system works.
    """
    try:
        # Test basic imports
        from services.prompt_wrapper import (
            chapter,
            chapter_topics,
            section,
            section_topics,
            summary,
            table_of_contents,
            title,
        )
        
        # Test basic function calls
        test_title = title("Test Book")
        assert test_title is not None
        
        test_toc = table_of_contents("Test instructions")
        assert test_toc is not None
        
        # Test service initialization
        service = get_prompt_service()
        assert service is not None
        
        # Test template rendering
        templates = service.list_templates()
        assert len(templates) > 0
        
        logger.info("Migration verification successful!")
        return True
    
    except Exception as e:
        logger.error(f"Migration verification failed: {e}")
        return False


if __name__ == "__main__":
    main()