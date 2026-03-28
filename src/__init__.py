"""
YOLOv8s 智能安防系统
"""

from src.detector import YOLODetector, Detection
from src.tracker import TrackerManager, Track
from src.zone_manager import ZoneManager, ZoneConfig, ZoneEvent
from src.behavior_analyzer import BehaviorAnalyzer, BehaviorEvent
from src.alarm_manager import AlarmManager, AlarmEvent
from src.annotator import Annotator, AnnotatorConfig
from src.video_source import VideoSource
from src.pipeline import SecurityPipeline

__version__ = "1.0.0"
__all__ = [
    "YOLODetector",
    "Detection",
    "TrackerManager",
    "Track",
    "ZoneManager",
    "ZoneConfig",
    "ZoneEvent",
    "BehaviorAnalyzer",
    "BehaviorEvent",
    "AlarmManager",
    "AlarmEvent",
    "Annotator",
    "AnnotatorConfig",
    "VideoSource",
    "SecurityPipeline",
]
