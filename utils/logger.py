"""
Logger module alias for AcademicLogger, ExperimentLogger, and get_logger.
"""

from utils.logging import ExperimentLogger, ExperimentLogger as AcademicLogger, get_logger

__all__ = ['ExperimentLogger', 'AcademicLogger', 'get_logger']
