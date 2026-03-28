"""
电子围栏区域管理模块
使用 supervision.PolygonZone 实现多边形区域检测
"""

import json
import numpy as np
from typing import List, Dict, Optional, Set
from dataclasses import dataclass
from pathlib import Path

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False

from src.tracker import Track


@dataclass
class ZoneConfig:
    """区域配置"""
    id: int
    name: str
    polygon: List[List[int]]
    enabled: bool = True
    detect_classes: List[int] = None
    description: str = ""

    def __post_init__(self):
        if self.detect_classes is None:
            self.detect_classes = []


@dataclass
class ZoneEvent:
    """区域事件"""
    zone_id: int
    zone_name: str
    track_id: int
    class_id: int
    class_name: str
    event_type: str  # "enter", "leave", "inside"
    bbox: List[float]


class ZoneManager:
    """电子围栏管理类"""

    def __init__(self, config_path: str = "configs/zones.json"):
        """
        初始化区域管理器

        Args:
            config_path: 区域配置文件路径
        """
        self.config_path = config_path
        self.zones: Dict[int, ZoneConfig] = {}
        self.polygon_zones: Dict[int, sv.PolygonZone] = {}
        self.zone_annotators: Dict[int, sv.PolygonZoneAnnotator] = {}

        # 跟踪每个区域内的目标
        self.tracks_in_zone: Dict[int, Set[int]] = {}  # zone_id -> set of track_ids

        # 加载配置
        self.load_config()

    def load_config(self):
        """加载区域配置"""
        config_file = Path(self.config_path)

        if not config_file.exists():
            print(f"区域配置文件不存在: {self.config_path}")
            return

        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        for zone_data in config.get("zones", []):
            zone = ZoneConfig(
                id=zone_data["id"],
                name=zone_data["name"],
                polygon=zone_data["polygon"],
                enabled=zone_data.get("enabled", True),
                detect_classes=zone_data.get("detect_classes", []),
                description=zone_data.get("description", "")
            )
            self.add_zone(zone)

    def add_zone(self, zone: ZoneConfig):
        """添加区域"""
        if not SUPERVISION_AVAILABLE:
            raise ImportError("请安装 supervision: pip install supervision")

        self.zones[zone.id] = zone
        self.tracks_in_zone[zone.id] = set()

        # 创建 PolygonZone
        polygon = np.array(zone.polygon, dtype=np.int32)
        self.polygon_zones[zone.id] = sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=(sv.Position.CENTER,)
        )

        # 创建标注器
        self.zone_annotators[zone.id] = sv.PolygonZoneAnnotator(
            zone=self.polygon_zones[zone.id],
            color=sv.Color.RED,
            thickness=2,
            text_thickness=1,
            text_scale=0.5
        )

    def remove_zone(self, zone_id: int):
        """移除区域"""
        if zone_id in self.zones:
            del self.zones[zone_id]
            del self.polygon_zones[zone_id]
            del self.zone_annotators[zone_id]
            del self.tracks_in_zone[zone_id]

    def update(self, tracks: List[Track]) -> List[ZoneEvent]:
        """
        更新区域状态

        Args:
            tracks: 当前帧的轨迹列表

        Returns:
            区域事件列表
        """
        events = []

        # 构建 supervision Detections
        if len(tracks) > 0:
            xyxy = np.array([t.bbox for t in tracks])
            tracker_ids = np.array([t.track_id for t in tracks])
            class_ids = np.array([t.class_id for t in tracks])
        else:
            xyxy = np.empty((0, 4))
            tracker_ids = np.empty(0, dtype=int)
            class_ids = np.empty(0, dtype=int)

        detections = sv.Detections(
            xyxy=xyxy,
            tracker_id=tracker_ids,
            class_id=class_ids
        )

        # 检查每个区域
        for zone_id, zone in self.zones.items():
            if not zone.enabled:
                continue

            # 过滤类别
            if zone.detect_classes:
                class_mask = np.isin(detections.class_id, zone.detect_classes)
                filtered_detections = detections[class_mask]
            else:
                filtered_detections = detections

            # 检测在区域内的目标
            mask = self.polygon_zones[zone_id].trigger(filtered_detections)

            current_tracks: Set[int] = set()
            for i, is_inside in enumerate(mask):
                if is_inside:
                    track_id = int(filtered_detections.tracker_id[i])
                    current_tracks.add(track_id)

            # 生成事件
            prev_tracks = self.tracks_in_zone[zone_id]

            # 进入事件
            entered = current_tracks - prev_tracks
            for track_id in entered:
                track = self._find_track(tracks, track_id)
                if track:
                    events.append(ZoneEvent(
                        zone_id=zone_id,
                        zone_name=zone.name,
                        track_id=track_id,
                        class_id=track.class_id,
                        class_name=track.class_name,
                        event_type="enter",
                        bbox=track.bbox
                    ))

            # 离开事件
            left = prev_tracks - current_tracks
            for track_id in left:
                track = self._find_track(tracks, track_id)
                if track:
                    events.append(ZoneEvent(
                        zone_id=zone_id,
                        zone_name=zone.name,
                        track_id=track_id,
                        class_id=track.class_id,
                        class_name=track.class_name,
                        event_type="leave",
                        bbox=track.bbox
                    ))

            # 更新状态
            self.tracks_in_zone[zone_id] = current_tracks

        return events

    def _find_track(self, tracks: List[Track], track_id: int) -> Optional[Track]:
        """查找轨迹"""
        for track in tracks:
            if track.track_id == track_id:
                return track
        return None

    def get_tracks_in_zone(self, zone_id: int) -> Set[int]:
        """获取区域内的轨迹ID"""
        return self.tracks_in_zone.get(zone_id, set())

    def get_zone_count(self, zone_id: int) -> int:
        """获取区域内的目标数量"""
        return len(self.tracks_in_zone.get(zone_id, set()))

    def is_track_in_zone(self, track_id: int, zone_id: int) -> bool:
        """判断轨迹是否在区域内"""
        return track_id in self.tracks_in_zone.get(zone_id, set())

    def get_all_zones_info(self) -> List[Dict]:
        """获取所有区域信息"""
        info = []
        for zone_id, zone in self.zones.items():
            info.append({
                "id": zone_id,
                "name": zone.name,
                "enabled": zone.enabled,
                "count": self.get_zone_count(zone_id),
                "detect_classes": zone.detect_classes
            })
        return info

    def save_config(self, path: Optional[str] = None):
        """保存区域配置"""
        path = path or self.config_path

        config = {
            "zones": [
                {
                    "id": zone.id,
                    "name": zone.name,
                    "polygon": zone.polygon,
                    "enabled": zone.enabled,
                    "detect_classes": zone.detect_classes,
                    "description": zone.description
                }
                for zone in self.zones.values()
            ],
            "version": "1.0"
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"区域配置已保存到: {path}")

    def draw_zones(self, frame, draw_labels: bool = True):
        """
        在帧上绘制区域

        Args:
            frame: 输入帧
            draw_labels: 是否绘制标签

        Returns:
            绘制后的帧
        """
        annotated_frame = frame.copy()

        for zone_id, zone in self.zones.items():
            if not zone.enabled:
                continue

            # 绘制多边形
            polygon = np.array(zone.polygon, dtype=np.int32)

            # 半透明填充
            overlay = annotated_frame.copy()
            cv2.fillPoly(overlay, [polygon], (0, 0, 255))
            annotated_frame = cv2.addWeighted(overlay, 0.2, annotated_frame, 0.8, 0)

            # 绘制边框
            cv2.polylines(annotated_frame, [polygon], True, (0, 0, 255), 2)

            # 绘制标签
            if draw_labels:
                count = self.get_zone_count(zone_id)
                label = f"{zone.name}: {count}"

                # 计算文本位置（多边形顶部中心）
                top_center = np.mean(polygon[:2], axis=0).astype(int)

                cv2.putText(
                    annotated_frame,
                    label,
                    (top_center[0] - 50, top_center[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        return annotated_frame


# 导入 cv2（用于绘制）
import cv2
