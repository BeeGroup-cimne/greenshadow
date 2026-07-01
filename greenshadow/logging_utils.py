# -*- coding: utf-8 -*-
import sys
from typing import Optional, Any
from contextlib import contextmanager


class StructuredLogger:
    """
    Structured logger for hierarchical output with sections and indentation.
    """
    
    def __init__(self, indent_char: str = "  ", section_char: str = "="):
        self.indent_level = 0
        self.indent_char = indent_char
        self.section_char = section_char
        self.verbose = True
    
    def _write(self, message: str, end: str = "\n"):
        """Write message with current indentation."""
        indent = self.indent_char * self.indent_level
        sys.stderr.write(f"{indent}{message}{end}")
        sys.stderr.flush()
    
    def section(self, title: str, level: int = 3):
        """Print a section header with === formatting."""
        if level < 1:
            level = 1
        separator = self.section_char * level
        self._write(f"\n{separator} {title} {separator}")
    
    def subsection(self, title: str):
        """Print a subsection header with --- formatting."""
        self._write(f"\n--- {title} ---")
    
    def info(self, message: str):
        """Print an info message."""
        self._write(message)
    
    def success(self, message: str):
        """Print a success message."""
        self._write(f"✓ {message}")
    
    def warning(self, message: str):
        """Print a warning message."""
        self._write(f"⚠ Warning: {message}")
    
    def error(self, message: str):
        """Print an error message."""
        self._write(f"✗ Error: {message}")
    
    def progress(self, current: int, total: int, description: str = ""):
        """Print progress information."""
        if total > 0:
            percentage = (current / total) * 100
            desc_part = f" {description}" if description else ""
            self._write(f"Progress: {current}/{total} ({percentage:.1f}%){desc_part}")
    
    def summary(self, items: dict):
        """Print a summary of key-value pairs."""
        for key, value in items.items():
            self._write(f"{key}: {value}")
    
    @contextmanager
    def indent(self, description: Optional[str] = None):
        """Context manager for indented logging."""
        if description:
            self._write(f"{description}...")
        self.indent_level += 1
        try:
            yield self
        finally:
            self.indent_level -= 1
    
    @contextmanager
    def section_context(self, title: str, level: int = 3):
        """Context manager for section with auto-indentation."""
        self.section(title, level)
        self.indent_level += 1
        try:
            yield self
        finally:
            self.indent_level -= 1
    
    def set_verbose(self, verbose: bool):
        """Enable/disable verbose output."""
        self.verbose = verbose
    
    def debug(self, message: str):
        """Print debug message only in verbose mode."""
        if self.verbose:
            self._write(f"DEBUG: {message}")


# Global logger instance
logger = StructuredLogger()

# Convenience functions for common patterns
def section(title: str, level: int = 3):
    """Create a section header."""
    logger.section(title, level)

def subsection(title: str):
    """Create a subsection header."""
    logger.subsection(title)

def info(message: str):
    """Print info message."""
    logger.info(message)

def success(message: str):
    """Print success message."""
    logger.success(message)

def warning(message: str):
    """Print warning message."""
    logger.warning(message)

def error(message: str):
    """Print error message."""
    logger.error(message)

@contextmanager
def indent(description: Optional[str] = None):
    """Context manager for indented output."""
    with logger.indent(description):
        yield

@contextmanager
def section_context(title: str, level: int = 3):
    """Context manager for section with indentation."""
    with logger.section_context(title, level):
        yield

def set_verbose(verbose: bool):
    """Set verbose mode."""
    logger.set_verbose(verbose)

def progress(current: int, total: int, description: str = ""):
    """Print progress."""
    logger.progress(current, total, description)

def summary(items: dict):
    """Print summary."""
    logger.summary(items)