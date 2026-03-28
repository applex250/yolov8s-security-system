"""
可视化标注模块
统一绘制检测框、轨迹、区域、报警等元素
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

try:
    import supervision as sv
    SUPERVISION_AVAILABLE = True
except ImportError:
    SUPERVISION_AVAILABLE = False

from src.detector import Detection
from src.tracker import Track
from src.zone_manager import ZoneManager
from src.alarm_manager import AlarmManager


@dataclass
class AnnotatorConfig:
    """标注器配置"""
    # 检测框
    box_thickness: int = 2
    box_color_person: Tuple[int, int, int] = (0, 255, 0)    # 绿色
    box_color_vehicle: Tuple[int, int, int] = (255, 0, 0)   # 蓝色
    box_color_other: Tuple[int, int, int] = (255, 255, 0)   # 青色

    # 轨迹
    show_tracks: bool = True
    track_thickness: int = 2
    track_color: Tuple[int, int, int] = (0, 255, 255)       # 黄色
    track_history_length: int = 30

    # 区域
    show_zones: bool = True
    zone_fill_alpha: float = 0.2
    zone_border_thickness: int = 2

    # 标签
    show_labels: bool = True
    show_confidence: bool = True
    label_font_scale: float = 0.5
    label_thickness: int = 1

    # FPS
    show_fps: bool = True
    fps_position: Tuple[int, int] = (10, 30)


class Annotator:
    """统一标注器"""

    # 类别颜色映射
    CLASS_COLORS = {
        0: (0, 255, 0),      # person - 绿色
        1: (255, 0, 255),    # bicycle - 紫色
        2: (255, 0, 0),      # car - 蓝色
        3: (255, 128, 0),    # motorcycle - 橙色
        5: (0, 0, 255),      # bus - 红色
        7: (0, 128, 255),    # truck - 浅红色
    }

    def __init__(self, config: AnnotatorConfig = None):
        """
        初始化标注器

        Args:
            config: 标注器配置
        """
        self.config = config or AnnotatorConfig()

        # supervision 标注器
        if SUPERVISION_AVAILABLE:
            self.box_annotator = sv.BoxAnnotator(
                thickness=self.config.box_thickness
            )
            self.label_annotator = sv.LabelAnnotator()
            self.trace_annotator = sv.TraceAnnotator(
                thickness=self.config.track_thickness,
                trace_length=self.config.track_history_length
            )

    def annotate_frame(self,
                       frame: np.ndarray,
                       tracks: List[Track] = None,
                       detections: List[Detection] = None,
                       zone_manager: ZoneManager = None,
                       alarm_manager: AlarmManager = None,
                       fps: float = 0.0,
                       additional_info: Dict = None) -> np.ndarray:
        """
        标注帧

        Args:
            frame: 输入帧
            tracks: 轨迹列表
            detections: 检测结果列表
            zone_manager: 区域管理器
            alarm_manager: 报警管理器
            fps: 当前FPS
            additional_info: 额外信息

        Returns:
            标注后的帧
        """
        annotated = frame.copy()

        # 1. 绘制区域
        if zone_manager and self.config.show_zones:
            annotated = self._draw_zones(annotated, zone_manager)

        # 2. 绘制轨迹
        if tracks and self.config.show_tracks:
            annotated = self._draw_tracks(annotated, tracks)

        # 3. 绘制检测框
        if detections:
            annotated = self._draw_detections(annotated, detections)

        # 4. 绘制报警效果
        if alarm_manager:
            annotated = alarm_manager.apply_visual_alarm(annotated)

        # 5. 绘制FPS和信息
        annotated = self._draw_info(annotated, fps, additional_info)

        return annotated

    def _draw_zones(self, frame: np.ndarray, zone_manager: ZoneManager) -> np.ndarray:
        """绘制区域"""
        annotated = frame.copy()

        for zone_id, zone in zone_manager.zones.items():
            if not zone.enabled:
                continue

            polygon = np.array(zone.polygon, dtype=np.int32)

            # 半透明填充
            overlay = annotated.copy()
            cv2.fillPoly(overlay, [polygon], (0, 0, 255))
            annotated = cv2.addWeighted(
                overlay, self.config.zone_fill_alpha,
                annotated, 1 - self.config.zone_fill_alpha, 0
            )

            # 边框
            cv2.polylines(
                annotated, [polygon], True,
                (0, 0, 255), self.config.zone_border_thickness
            )

            # 标签
            count = zone_manager.get_zone_count(zone_id)
            label = f"{zone.name}: {count}"

            # 计算文本位置
            x, y = polygon[0]
            cv2.putText(
                annotated, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 255), 2
            )

        return annotated

    def _draw_tracks(self, frame: np.ndarray, tracks: List[Track]) -> np.ndarray:
        """绘制轨迹"""
        annotated = frame.copy()

        for track in tracks:
            # 绘制历史轨迹
            if len(track.history) >= 2:
                points = np.array(track.history[-self.config.track_history_length:], dtype=np.int32)
                cv2.polylines(
                    annotated, [points], False,
                    self.config.track_color, self.config.track_thickness
                )

            # 绘制边界框
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            color = self.CLASS_COLORS.get(track.class_id, self.config.box_color_other)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.config.box_thickness)

            # 绘制标签
            if self.config.show_labels:
                label = f"#{track.track_id} {track.class_name}"
                if self.config.show_confidence:
                    label += f" {track.confidence:.2f}"

                # 背景
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.label_font_scale, self.config.label_thickness
                )
                cv2.rectangle(
                    annotated,
                    (x1, y1 - text_h - 8),
                    (x1 + text_w + 4, y1),
                    color, -1
                )

                # 文本
                cv2.putText(
                    annotated, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.label_font_scale,
                    (255, 255, 255), self.config.label_thickness
                )

            # 绘制中心点
            center = track.get_center()
            cv2.circle(annotated, (int(center[0]), int(center[1])), 4, color, -1)

        return annotated

    def _draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """绘制检测结果"""
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = self.CLASS_COLORS.get(det.class_id, self.config.box_color_other)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.config.box_thickness)

            if self.config.show_labels:
                label = det.class_name
                if self.config.show_confidence:
                    label += f" {det.confidence:.2f}"

                cv2.putText(
                    annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.label_font_scale,
                    color, self.config.label_thickness
                )

        return annotated

    def _draw_info(self, frame: np.ndarray, fps: float, additional_info: Dict = None) -> np.ndarray:
        """绘制信息"""
        annotated = frame.copy()

        # FPS
        if self.config.show_fps:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(
                annotated, fps_text, self.config.fps_position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        # 额外信息
        if additional_info:
            y_offset = 60
            for key, value in additional_info.items():
                text = f"{key}: {value}"
                cv2.putText(
                    annotated, text, (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
                y_offset += 25

        return annotated

    def draw_alarm_info(self, frame: np.ndarray, alarm_events: List) -> np.ndarray:
        """绘制报警信息"""
        annotated = frame.copy()

        if not alarm_events:
            return annotated

        # 显示最近的报警
        y_offset = frame.shape[0] - 30
        for event in alarm_events[-3:]:  # 最多显示3条
            text = f"[{event.event_type}] {event.message}"
            cv2.putText(
                annotated, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )
            y_offset -= 25

        return annotated


def create_legend(frame_size: Tuple[int, int]) -> np.ndarray:
    """创建图例"""
    legend = np.zeros((150, 200, 3), dtype=np.uint8)

    items = [
        ("Person", (0, 255, 0)),
        ("Car", (255, 0, 0)),
        ("Bus", (0, 0, 255)),
        ("Truck", (0, 128, 255)),
        ("Track", (0, 255, 255)),
    ]

    y = 25
    for name, color in items:
        cv2.rectangle(legend, (10, y - 10), (30, y + 5), color, -1)
        cv2.putText(legend, name, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y += 25

    return legend
