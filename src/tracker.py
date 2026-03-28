"""
多目标跟踪模块
使用 ByteTrack 或 BoT-SORT
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False

from src.detector import Detection


@dataclass
class Track:
    """目标轨迹"""
    track_id: int
    class_id: int
    class_name: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    history: List[List[float]] = field(default_factory=list)  # 历史中心点
    frame_count: int = 0
    state: str = "active"  # active, lost, removed

    def get_center(self) -> Tuple[float, float]:
        """获取边界框中心点"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def get_area(self) -> float:
        """获取边界框面积"""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def update_position(self, bbox: List[float], confidence: float):
        """更新位置"""
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = time.time()
        self.frame_count += 1

        # 记录历史轨迹
        center = self.get_center()
        self.history.append(list(center))

        # 限制历史长度
        if len(self.history) > 300:
            self.history = self.history[-300:]

    def get_total_distance(self) -> float:
        """计算总移动距离"""
        if len(self.history) < 2:
            return 0.0

        total = 0.0
        for i in range(1, len(self.history)):
            dx = self.history[i][0] - self.history[i-1][0]
            dy = self.history[i][1] - self.history[i-1][1]
            total += np.sqrt(dx*dx + dy*dy)

        return total

    def get_track_duration(self) -> float:
        """获取轨迹持续时间（秒）"""
        return self.last_seen - self.first_seen

    def is_loitering(self, distance_threshold: float = 100) -> bool:
        """判断是否在徘徊"""
        if self.get_track_duration() < 10:
            return False

        # 计算最近N帧的移动距离
        recent_history = self.history[-30:] if len(self.history) >= 30 else self.history
        if len(recent_history) < 2:
            return False

        total_distance = 0.0
        for i in range(1, len(recent_history)):
            dx = recent_history[i][0] - recent_history[i-1][0]
            dy = recent_history[i][1] - recent_history[i-1][1]
            total_distance += np.sqrt(dx*dx + dy*dy)

        # 平均每帧移动距离很小，认为在徘徊
        avg_distance = total_distance / len(recent_history)
        return avg_distance < distance_threshold / 30  # 转换为每帧阈值


class TrackerManager:
    """跟踪器管理类"""

    def __init__(self, tracker_type: str = "bytetrack",
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 min_box_area: int = 10):
        """
        初始化跟踪器

        Args:
            tracker_type: 跟踪器类型 ("bytetrack" 或 "botsort")
            track_buffer: 轨迹缓冲帧数
            match_thresh: 匹配阈值
            min_box_area: 最小框面积
        """
        self.tracker_type = tracker_type
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.min_box_area = min_box_area

        self.tracker = None
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.frame_count = 0

        # 初始化跟踪器
        self._init_tracker()

    def _init_tracker(self):
        """初始化跟踪器"""
        if not SUPERVISION_AVAILABLE:
            raise ImportError("请安装 supervision: pip install supervision")

        if self.tracker_type == "bytetrack":
            self.tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=self.match_thresh,
                frame_rate=30
            )
        else:
            # BoT-SORT
            self.tracker = sv.BOTSort(
                track_activation_threshold=0.25,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=self.match_thresh,
                frame_rate=30
            )

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        更新跟踪器

        Args:
            detections: 当前帧的检测结果

        Returns:
            当前活跃的轨迹列表
        """
        self.frame_count += 1

        # 转换为 supervision 格式
        if len(detections) > 0:
            xyxy = np.array([d.bbox for d in detections])
            confidence = np.array([d.confidence for d in detections])
            class_id = np.array([d.class_id for d in detections])
        else:
            xyxy = np.empty((0, 4))
            confidence = np.empty(0)
            class_id = np.empty(0, dtype=int)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        # 更新跟踪器
        tracked = self.tracker.update_with_detections(sv_detections)

        # 更新轨迹字典
        current_track_ids = set()
        result_tracks = []

        for i in range(len(tracked)):
            track_id = int(tracked.tracker_id[i])
            current_track_ids.add(track_id)

            bbox = tracked.xyxy[i].tolist()
            conf = float(tracked.confidence[i])
            cls_id = int(tracked.class_id[i])

            if track_id in self.tracks:
                # 更新现有轨迹
                self.tracks[track_id].update_position(bbox, conf)
            else:
                # 创建新轨迹
                from src.detector import YOLODetector
                class_name = YOLODetector.COCO_CLASSES.get(cls_id, f"class_{cls_id}")
                self.tracks[track_id] = Track(
                    track_id=track_id,
                    class_id=cls_id,
                    class_name=class_name,
                    bbox=bbox,
                    confidence=conf
                )
                self.tracks[track_id].history.append(list(self.tracks[track_id].get_center()))

            result_tracks.append(self.tracks[track_id])

        # 标记丢失的轨迹
        for track_id, track in self.tracks.items():
            if track_id not in current_track_ids:
                track.state = "lost"

        # 清理过期的轨迹
        self._cleanup_tracks()

        return result_tracks

    def _cleanup_tracks(self):
        """清理过期的轨迹"""
        current_time = time.time()
        expired_ids = []

        for track_id, track in self.tracks.items():
            # 超过缓冲时间的丢失轨迹被移除
            if track.state == "lost" and (current_time - track.last_seen) > self.track_buffer / 30:
                expired_ids.append(track_id)

        for track_id in expired_ids:
            del self.tracks[track_id]

    def get_active_tracks(self) -> List[Track]:
        """获取所有活跃轨迹"""
        return [t for t in self.tracks.values() if t.state == "active"]

    def get_tracks_by_class(self, class_id: int) -> List[Track]:
        """按类别获取轨迹"""
        return [t for t in self.tracks.values() if t.class_id == class_id and t.state == "active"]

    def get_person_tracks(self) -> List[Track]:
        """获取人员轨迹"""
        return self.get_tracks_by_class(0)

    def get_vehicle_tracks(self) -> List[Track]:
        """获取车辆轨迹"""
        return [t for t in self.tracks.values()
                if t.class_id in [2, 3, 5, 7] and t.state == "active"]

    def get_track_count(self) -> Dict[int, int]:
        """获取各类别的轨迹数量"""
        counts = defaultdict(int)
        for track in self.tracks.values():
            if track.state == "active":
                counts[track.class_id] += 1
        return dict(counts)
