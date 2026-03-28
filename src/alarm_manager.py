"""
报警管理模块
实现视觉报警、音频报警、截图保存
"""

import os
import time
import threading
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from src.zone_manager import ZoneEvent
from src.behavior_analyzer import BehaviorEvent


@dataclass
class AlarmEvent:
    """报警事件"""
    event_type: str
    source: str           # "zone" 或 "behavior"
    timestamp: float
    message: str
    details: Dict
    snapshot_path: Optional[str] = None


class VisualAlarm:
    """视觉报警器"""

    def __init__(self,
                 border_color: List[int] = None,
                 border_thickness: int = 5,
                 flash_interval: float = 0.5,
                 alert_text: str = "ALERT!"):
        """
        初始化视觉报警器

        Args:
            border_color: 边框颜色 (BGR)
            border_thickness: 边框厚度
            flash_interval: 闪烁间隔(秒)
            alert_text: 报警文本
        """
        self.border_color = border_color or [0, 0, 255]
        self.border_thickness = border_thickness
        self.flash_interval = flash_interval
        self.alert_text = alert_text

        self.is_flashing = False
        self.flash_state = True
        self.last_flash_time = 0

    def update(self):
        """更新闪烁状态"""
        current_time = time.time()
        if self.is_flashing:
            if current_time - self.last_flash_time >= self.flash_interval:
                self.flash_state = not self.flash_state
                self.last_flash_time = current_time

    def apply(self, frame: np.ndarray, active: bool) -> np.ndarray:
        """
        应用视觉效果

        Args:
            frame: 输入帧
            active: 是否激活报警

        Returns:
            处理后的帧
        """
        self.is_flashing = active
        self.update()

        result = frame.copy()

        if active and self.flash_state:
            h, w = frame.shape[:2]

            # 绘制闪烁边框
            color = tuple(self.border_color)
            cv2.rectangle(result, (0, 0), (w, h), color, self.border_thickness)
            cv2.rectangle(result, (self.border_thickness, self.border_thickness),
                         (w - self.border_thickness, h - self.border_thickness),
                         color, self.border_thickness)

            # 绘制报警文本
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.5
            thickness = 3
            text_size = cv2.getTextSize(self.alert_text, font, font_scale, thickness)[0]

            # 文本位置（顶部居中）
            text_x = (w - text_size[0]) // 2
            text_y = 50

            # 绘制背景
            cv2.rectangle(result,
                         (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 0), -1)

            # 绘制文本
            cv2.putText(result, self.alert_text, (text_x, text_y),
                       font, font_scale, color, thickness)

        return result


class AudioAlarm:
    """音频报警器"""

    def __init__(self,
                 alarm_type: str = "beep",
                 duration: float = 0.3,
                 frequency: int = 1000,
                 custom_file: str = ""):
        """
        初始化音频报警器

        Args:
            alarm_type: 报警类型 ("beep" 或 "custom")
            duration: 蜂鸣持续时间(秒)
            frequency: 蜂鸣频率(Hz)
            custom_file: 自定义音频文件路径
        """
        self.alarm_type = alarm_type
        self.duration = duration
        self.frequency = frequency
        self.custom_file = custom_file

        self.is_playing = False
        self.last_play_time = 0
        self.min_interval = 1.0  # 最小播放间隔

        # 尝试导入音频库
        self.audio_available = False
        try:
            import winsound
            self.winsound = winsound
            self.audio_available = True
        except ImportError:
            # 非 Windows 系统
            try:
                import simpleaudio as sa
                self.sa = sa
                self.audio_available = True
            except ImportError:
                print("警告: 音频库不可用，音频报警将被禁用")

    def play(self):
        """播放报警音"""
        if not self.audio_available:
            return

        current_time = time.time()
        if current_time - self.last_play_time < self.min_interval:
            return

        self.last_play_time = current_time

        if self.alarm_type == "beep":
            self._play_beep()
        elif self.alarm_type == "custom" and self.custom_file:
            self._play_custom()

    def _play_beep(self):
        """播放蜂鸣声"""
        try:
            if hasattr(self, 'winsound'):
                # Windows 系统
                self.winsound.Beep(self.frequency, int(self.duration * 1000))
            else:
                # 生成简单的蜂鸣波形
                import numpy as np
                sample_rate = 44100
                t = np.linspace(0, self.duration, int(sample_rate * self.duration), False)
                wave = np.sin(2 * np.pi * self.frequency * t) * 0.5
                audio = (wave * 32767).astype(np.int16)

                play_obj = self.sa.play_buffer(audio, 1, 2, sample_rate)
        except Exception as e:
            print(f"播放蜂鸣声失败: {e}")

    def _play_custom(self):
        """播放自定义音频"""
        try:
            if hasattr(self, 'sa'):
                wave_obj = self.sa.WaveObject.from_wave_file(self.custom_file)
                play_obj = wave_obj.play()
        except Exception as e:
            print(f"播放自定义音频失败: {e}")


class SnapshotManager:
    """截图管理器"""

    def __init__(self,
                 save_path: str = "output/snapshots",
                 prefix: str = "alert_",
                 format: str = "jpg",
                 quality: int = 95):
        """
        初始化截图管理器

        Args:
            save_path: 保存路径
            prefix: 文件前缀
            format: 图像格式
            quality: 图像质量 (1-100)
        """
        self.save_path = Path(save_path)
        self.prefix = prefix
        self.format = format
        self.quality = quality

        # 创建保存目录
        self.save_path.mkdir(parents=True, exist_ok=True)

        # 截图计数
        self.snapshot_count = 0

    def save(self, frame: np.ndarray, event_type: str = "") -> Optional[str]:
        """
        保存截图

        Args:
            frame: 要保存的帧
            event_type: 事件类型

        Returns:
            保存的文件路径，失败返回 None
        """
        try:
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.prefix}{timestamp}_{event_type}_{self.snapshot_count:04d}.{self.format}"
            filepath = self.save_path / filename

            # 保存图像
            if self.format.lower() == "jpg":
                cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            else:
                cv2.imwrite(str(filepath), frame)

            self.snapshot_count += 1
            print(f"截图已保存: {filepath}")

            return str(filepath)

        except Exception as e:
            print(f"保存截图失败: {e}")
            return None


class AlarmManager:
    """报警管理器"""

    def __init__(self,
                 enable_visual: bool = True,
                 enable_audio: bool = True,
                 enable_snapshot: bool = True,
                 visual_config: Dict = None,
                 audio_config: Dict = None,
                 snapshot_config: Dict = None):
        """
        初始化报警管理器

        Args:
            enable_visual: 是否启用视觉报警
            enable_audio: 是否启用音频报警
            enable_snapshot: 是否启用截图保存
            visual_config: 视觉报警配置
            audio_config: 音频报警配置
            snapshot_config: 截图配置
        """
        self.enable_visual = enable_visual
        self.enable_audio = enable_audio
        self.enable_snapshot = enable_snapshot

        # 初始化各报警器
        if enable_visual:
            config = visual_config or {}
            self.visual_alarm = VisualAlarm(
                border_color=config.get("border_color", [0, 0, 255]),
                border_thickness=config.get("border_thickness", 5),
                flash_interval=config.get("flash_interval", 0.5),
                alert_text=config.get("alert_text", "ALERT!")
            )
        else:
            self.visual_alarm = None

        if enable_audio:
            config = audio_config or {}
            self.audio_alarm = AudioAlarm(
                alarm_type=config.get("type", "beep"),
                duration=config.get("duration", 0.3),
                frequency=config.get("frequency", 1000),
                custom_file=config.get("custom_file", "")
            )
        else:
            self.audio_alarm = None

        if enable_snapshot:
            config = snapshot_config or {}
            self.snapshot_manager = SnapshotManager(
                save_path=config.get("save_path", "output/snapshots"),
                prefix=config.get("prefix", "alert_"),
                format=config.get("format", "jpg"),
                quality=config.get("quality", 95)
            )
        else:
            self.snapshot_manager = None

        # 报警事件记录
        self.alarm_events: List[AlarmEvent] = []
        self.is_alarm_active = False

        # 事件回调
        self.callbacks: List[Callable] = []

    def process_zone_events(self, events: List[ZoneEvent], frame: np.ndarray) -> List[AlarmEvent]:
        """处理区域事件"""
        alarm_events = []

        for event in events:
            if event.event_type == "enter":
                alarm = AlarmEvent(
                    event_type="zone_intrusion",
                    source="zone",
                    timestamp=time.time(),
                    message=f"Target #{event.track_id} entered [{event.zone_name}]",
                    details={
                        "zone_id": event.zone_id,
                        "zone_name": event.zone_name,
                        "track_id": event.track_id,
                        "class_name": event.class_name
                    }
                )

                # 保存截图
                if self.snapshot_manager:
                    alarm.snapshot_path = self.snapshot_manager.save(frame, "intrusion")
                    print(f"[ALERT] Zone Intrusion: {alarm.message}")
                    print(f"        Snapshot saved: {alarm.snapshot_path}")

                alarm_events.append(alarm)
                self.alarm_events.append(alarm)

        if alarm_events:
            self._trigger_alarm(frame)

        return alarm_events

    def process_behavior_events(self, events: List[BehaviorEvent], frame: np.ndarray) -> List[AlarmEvent]:
        """处理行为事件"""
        alarm_events = []

        for event in events:
            if event.event_type == "loitering":
                alarm = AlarmEvent(
                    event_type="loitering_detected",
                    source="behavior",
                    timestamp=time.time(),
                    message=f"Loitering detected: ID {event.track_ids}",
                    details={
                        "track_ids": event.track_ids,
                        "duration": event.details.get("duration", 0)
                    }
                )
            elif event.event_type == "crowd":
                alarm = AlarmEvent(
                    event_type="crowd_detected",
                    source="behavior",
                    timestamp=time.time(),
                    message=f"Crowd detected: {event.details.get('count', 0)} people",
                    details={
                        "track_ids": event.track_ids,
                        "count": event.details.get("count", 0),
                        "location": event.location
                    }
                )
            else:
                continue

            # 保存截图
            if self.snapshot_manager:
                alarm.snapshot_path = self.snapshot_manager.save(frame, event.event_type)
                print(f"[ALERT] {alarm.event_type}: {alarm.message}")
                print(f"        Snapshot saved: {alarm.snapshot_path}")

            alarm_events.append(alarm)
            self.alarm_events.append(alarm)

        if alarm_events:
            self._trigger_alarm(frame)

        return alarm_events

    def _trigger_alarm(self, frame: np.ndarray):
        """触发报警"""
        self.is_alarm_active = True

        # 播放音频
        if self.audio_alarm:
            self.audio_alarm.play()

        # 调用回调
        for callback in self.callbacks:
            try:
                callback(self.alarm_events[-1] if self.alarm_events else None)
            except Exception as e:
                print(f"报警回调执行失败: {e}")

    def apply_visual_alarm(self, frame: np.ndarray) -> np.ndarray:
        """应用视觉报警效果"""
        if self.visual_alarm:
            return self.visual_alarm.apply(frame, self.is_alarm_active)
        return frame

    def update(self):
        """更新报警状态"""
        # 重置报警状态（可以根据需要调整超时时间）
        pass

    def add_callback(self, callback: Callable):
        """添加报警回调"""
        self.callbacks.append(callback)

    def get_recent_events(self, count: int = 10) -> List[AlarmEvent]:
        """获取最近的报警事件"""
        return self.alarm_events[-count:]

    def clear_events(self):
        """清除所有事件记录"""
        self.alarm_events.clear()

    def reset_alarm_state(self):
        """重置报警状态"""
        self.is_alarm_active = False
