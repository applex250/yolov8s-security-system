"""
行为分析模块
实现徘徊检测、人群聚集检测等功能
"""

import time
import numpy as np
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict

from src.tracker import Track


@dataclass
class BehaviorEvent:
    """行为事件"""
    event_type: str           # "loitering", "crowd", "intrusion"
    track_ids: List[int]
    class_ids: List[int]
    confidence: float
    location: List[float]     # [x, y] 事件中心位置
    timestamp: float
    details: Dict = field(default_factory=dict)


class LoiteringDetector:
    """徘徊检测器"""

    def __init__(self,
                 time_threshold: float = 30.0,
                 distance_threshold: float = 100.0,
                 min_frames: int = 30):
        """
        初始化徘徊检测器

        Args:
            time_threshold: 徘徊时间阈值(秒)
            distance_threshold: 移动距离阈值(像素)，小于此值视为徘徊
            min_frames: 最小连续帧数
        """
        self.time_threshold = time_threshold
        self.distance_threshold = distance_threshold
        self.min_frames = min_frames

        # 记录已报警的轨迹
        self.alerted_tracks: Set[int] = set()

    def detect(self, tracks: List[Track]) -> List[BehaviorEvent]:
        """
        检测徘徊行为

        Args:
            tracks: 轨迹列表

        Returns:
            行为事件列表
        """
        events = []

        for track in tracks:
            # 只检测人员
            if track.class_id != 0:
                continue

            # 检查是否已经报警
            if track.track_id in self.alerted_tracks:
                continue

            # 检查持续时间
            duration = track.get_track_duration()
            if duration < self.time_threshold:
                continue

            # 检查移动距离
            if track.is_loitering(self.distance_threshold):
                self.alerted_tracks.add(track.track_id)

                events.append(BehaviorEvent(
                    event_type="loitering",
                    track_ids=[track.track_id],
                    class_ids=[track.class_id],
                    confidence=0.9,
                    location=list(track.get_center()),
                    timestamp=time.time(),
                    details={
                        "duration": duration,
                        "avg_distance": track.get_total_distance() / max(1, len(track.history))
                    }
                ))

        return events

    def reset(self):
        """重置状态"""
        self.alerted_tracks.clear()


class CrowdDetector:
    """人群聚集检测器"""

    def __init__(self,
                 min_people: int = 3,
                 distance_threshold: float = 150.0,
                 time_threshold: float = 10.0):
        """
        初始化人群聚集检测器

        Args:
            min_people: 触发报警的最少人数
            distance_threshold: 人与人之间的距离阈值(像素)
            time_threshold: 持续时间阈值(秒)
        """
        self.min_people = min_people
        self.distance_threshold = distance_threshold
        self.time_threshold = time_threshold

        # 记录人群聚集区域
        self.crowd_regions: Dict[tuple, dict] = {}  # (grid_x, grid_y) -> info

        # 已报警的聚集区域
        self.alerted_regions: Set[tuple] = set()

    def detect(self, tracks: List[Track]) -> List[BehaviorEvent]:
        """
        检测人群聚集

        Args:
            tracks: 轨迹列表

        Returns:
            行为事件列表
        """
        events = []

        # 只考虑人员
        person_tracks = [t for t in tracks if t.class_id == 0]
        if len(person_tracks) < self.min_people:
            return events

        # 使用聚类方法检测聚集
        clusters = self._cluster_people(person_tracks)

        for cluster in clusters:
            if len(cluster) >= self.min_people:
                # 计算聚集中心
                centers = [t.get_center() for t in cluster]
                avg_x = np.mean([c[0] for c in centers])
                avg_y = np.mean([c[1] for c in centers])

                # 网格化位置
                grid_pos = (int(avg_x / 100), int(avg_y / 100))

                # 检查是否已报警
                if grid_pos in self.alerted_regions:
                    continue

                # 更新聚集信息
                if grid_pos not in self.crowd_regions:
                    self.crowd_regions[grid_pos] = {
                        "first_seen": time.time(),
                        "track_ids": set()
                    }

                self.crowd_regions[grid_pos]["track_ids"].update(t.track_id for t in cluster)

                # 检查持续时间
                duration = time.time() - self.crowd_regions[grid_pos]["first_seen"]
                if duration >= self.time_threshold:
                    self.alerted_regions.add(grid_pos)

                    events.append(BehaviorEvent(
                        event_type="crowd",
                        track_ids=list(t.track_id for t in cluster),
                        class_ids=[0] * len(cluster),
                        confidence=0.85,
                        location=[avg_x, avg_y],
                        timestamp=time.time(),
                        details={
                            "count": len(cluster),
                            "duration": duration
                        }
                    ))

        return events

    def _cluster_people(self, tracks: List[Track]) -> List[List[Track]]:
        """
        简单的距离聚类

        Args:
            tracks: 人员轨迹列表

        Returns:
            聚类结果，每个聚类是一个轨迹列表
        """
        if not tracks:
            return []

        # 使用简单的连通分量方法
        n = len(tracks)
        visited = [False] * n
        clusters = []

        def get_distance(t1: Track, t2: Track) -> float:
            c1 = t1.get_center()
            c2 = t2.get_center()
            return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

        def dfs(idx: int, cluster: List[Track]):
            visited[idx] = True
            cluster.append(tracks[idx])

            for j in range(n):
                if not visited[j]:
                    if get_distance(tracks[idx], tracks[j]) <= self.distance_threshold:
                        dfs(j, cluster)

        for i in range(n):
            if not visited[i]:
                cluster = []
                dfs(i, cluster)
                clusters.append(cluster)

        return clusters

    def reset(self):
        """重置状态"""
        self.crowd_regions.clear()
        self.alerted_regions.clear()


class BehaviorAnalyzer:
    """行为分析管理器"""

    def __init__(self,
                 enable_loitering: bool = True,
                 enable_crowd: bool = True,
                 loitering_config: Dict = None,
                 crowd_config: Dict = None):
        """
        初始化行为分析器

        Args:
            enable_loitering: 是否启用徘徊检测
            enable_crowd: 是否启用聚集检测
            loitering_config: 徘徊检测配置
            crowd_config: 聚集检测配置
        """
        self.enable_loitering = enable_loitering
        self.enable_crowd = enable_crowd

        # 初始化检测器
        if enable_loitering:
            config = loitering_config or {}
            self.loitering_detector = LoiteringDetector(
                time_threshold=config.get("time_threshold", 30.0),
                distance_threshold=config.get("distance_threshold", 100.0),
                min_frames=config.get("min_frames", 30)
            )
        else:
            self.loitering_detector = None

        if enable_crowd:
            config = crowd_config or {}
            self.crowd_detector = CrowdDetector(
                min_people=config.get("min_people", 3),
                distance_threshold=config.get("distance_threshold", 150.0),
                time_threshold=config.get("time_threshold", 10.0)
            )
        else:
            self.crowd_detector = None

    def analyze(self, tracks: List[Track]) -> List[BehaviorEvent]:
        """
        执行行为分析

        Args:
            tracks: 轨迹列表

        Returns:
            行为事件列表
        """
        events = []

        if self.loitering_detector:
            events.extend(self.loitering_detector.detect(tracks))

        if self.crowd_detector:
            events.extend(self.crowd_detector.detect(tracks))

        return events

    def reset(self):
        """重置所有检测器"""
        if self.loitering_detector:
            self.loitering_detector.reset()
        if self.crowd_detector:
            self.crowd_detector.reset()

    def get_statistics(self, tracks: List[Track]) -> Dict:
        """
        获取统计信息

        Args:
            tracks: 轨迹列表

        Returns:
            统计信息字典
        """
        person_tracks = [t for t in tracks if t.class_id == 0]
        vehicle_tracks = [t for t in tracks if t.class_id in [2, 3, 5, 7]]

        return {
            "total_tracks": len(tracks),
            "person_count": len(person_tracks),
            "vehicle_count": len(vehicle_tracks),
            "avg_person_speed": self._calculate_avg_speed(person_tracks),
            "avg_vehicle_speed": self._calculate_avg_speed(vehicle_tracks)
        }

    def _calculate_avg_speed(self, tracks: List[Track]) -> float:
        """计算平均速度（像素/帧）"""
        if not tracks:
            return 0.0

        speeds = []
        for track in tracks:
            if len(track.history) >= 2:
                total_dist = track.get_total_distance()
                avg_speed = total_dist / len(track.history)
                speeds.append(avg_speed)

        return np.mean(speeds) if speeds else 0.0
