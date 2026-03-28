"""
主处理管线
串联所有模块：检测 -> 跟踪 -> 区域检测 -> 行为分析 -> 报警 -> 可视化
"""

import time
import yaml
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass

from src.video_source import VideoSource
from src.detector import YOLODetector, Detection
from src.tracker import TrackerManager, Track
from src.zone_manager import ZoneManager, ZoneEvent
from src.behavior_analyzer import BehaviorAnalyzer, BehaviorEvent
from src.alarm_manager import AlarmManager, AlarmEvent
from src.annotator import Annotator, AnnotatorConfig


@dataclass
class PipelineStats:
    """管线统计"""
    fps: float = 0.0
    detection_time: float = 0.0
    tracking_time: float = 0.0
    zone_time: float = 0.0
    behavior_time: float = 0.0
    total_time: float = 0.0
    frame_count: int = 0
    track_count: int = 0
    alarm_count: int = 0


class SecurityPipeline:
    """智能安防处理管线"""

    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        初始化管线

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()

        # 初始化各模块
        self.video_source: Optional[VideoSource] = None
        self.detector: Optional[YOLODetector] = None
        self.tracker: Optional[TrackerManager] = None
        self.zone_manager: Optional[ZoneManager] = None
        self.behavior_analyzer: Optional[BehaviorAnalyzer] = None
        self.alarm_manager: Optional[AlarmManager] = None
        self.annotator: Optional[Annotator] = None

        # 视频输出
        self.video_writer: Optional[cv2.VideoWriter] = None

        # 统计信息
        self.stats = PipelineStats()

        # 运行状态
        self.is_running = False
        self.is_paused = False

        # 初始化模块
        self._init_modules()

    def _load_config(self) -> Dict:
        """加载配置"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _init_modules(self):
        """初始化所有模块"""
        print("初始化模块...")

        # 1. 检测器
        model_cfg = self.config.get("model", {})
        self.detector = YOLODetector(
            model_path=model_cfg.get("path", "models/yolov8s.pt"),
            confidence_threshold=model_cfg.get("confidence_threshold", 0.5),
            iou_threshold=model_cfg.get("iou_threshold", 0.45),
            device=model_cfg.get("device", "cuda"),
            classes=model_cfg.get("classes")
        )
        self.detector.load_model()
        print(f"  [OK] 检测器已加载 ({model_cfg.get('device', 'cuda')})")

        # 2. 跟踪器
        tracker_cfg = self.config.get("tracker", {})
        self.tracker = TrackerManager(
            tracker_type=tracker_cfg.get("type", "bytetrack"),
            track_buffer=tracker_cfg.get("track_buffer", 30),
            match_thresh=tracker_cfg.get("match_thresh", 0.8)
        )
        print("  [OK] 跟踪器已初始化")

        # 3. 区域管理器
        zones_cfg = self.config.get("zones", {})
        if zones_cfg.get("enable", True):
            self.zone_manager = ZoneManager(
                config_path=zones_cfg.get("config_path", "configs/zones.json")
            )
            print(f"  [OK] 区域管理器已加载 ({len(self.zone_manager.zones)} 个区域)")
        else:
            self.zone_manager = None
            print("  [--] 区域管理器已禁用")

        # 4. 行为分析器
        behavior_cfg = self.config.get("behavior", {})
        loitering_cfg = behavior_cfg.get("loitering", {})
        crowd_cfg = behavior_cfg.get("crowd", {})

        self.behavior_analyzer = BehaviorAnalyzer(
            enable_loitering=loitering_cfg.get("enable", True),
            enable_crowd=crowd_cfg.get("enable", True),
            loitering_config={
                "time_threshold": loitering_cfg.get("time_threshold", 30.0),
                "distance_threshold": loitering_cfg.get("distance_threshold", 100),
                "min_frames": loitering_cfg.get("min_frames", 30)
            },
            crowd_config={
                "min_people": crowd_cfg.get("min_people", 3),
                "distance_threshold": crowd_cfg.get("distance_threshold", 150),
                "time_threshold": crowd_cfg.get("time_threshold", 10.0)
            }
        )
        print("  [OK] 行为分析器已初始化")

        # 5. 报警管理器
        alarm_cfg = self.config.get("alarm", {})
        self.alarm_manager = AlarmManager(
            enable_visual=alarm_cfg.get("visual", {}).get("enable", True),
            enable_audio=alarm_cfg.get("audio", {}).get("enable", True),
            enable_snapshot=alarm_cfg.get("snapshot", {}).get("enable", True),
            visual_config=alarm_cfg.get("visual", {}),
            audio_config=alarm_cfg.get("audio", {}),
            snapshot_config=alarm_cfg.get("snapshot", {})
        )
        print("  [OK] 报警管理器已初始化")

        # 6. 标注器
        display_cfg = self.config.get("display", {})
        annotator_config = AnnotatorConfig(
            show_tracks=display_cfg.get("show_tracks", True),
            show_zones=display_cfg.get("show_zones", True),
            show_labels=display_cfg.get("show_labels", True),
            show_confidence=display_cfg.get("show_confidence", True),
            show_fps=display_cfg.get("show_fps", True)
        )
        self.annotator = Annotator(annotator_config)
        print("  [OK] 标注器已初始化")

        print("所有模块初始化完成!")

    def set_video_source(self, source_type: str = "file", source_path: str = "", webcam_id: int = 0):
        """设置视频源"""
        video_cfg = self.config.get("video", {})

        self.video_source = VideoSource(
            source_type=source_type or video_cfg.get("source_type", "file"),
            source_path=source_path or video_cfg.get("source_path", ""),
            webcam_id=webcam_id or video_cfg.get("webcam_id", 0)
        )

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        处理单帧

        Args:
            frame: 输入帧

        Returns:
            处理后的帧
        """
        start_time = time.time()

        # 1. 目标检测
        det_start = time.time()
        detections = self.detector.detect(frame)
        self.stats.detection_time = time.time() - det_start

        # 2. 多目标跟踪
        track_start = time.time()
        tracks = self.tracker.update(detections)
        self.stats.tracking_time = time.time() - track_start

        # 3. 区域检测
        zone_events = []
        if self.zone_manager:
            zone_start = time.time()
            zone_events = self.zone_manager.update(tracks)
            self.stats.zone_time = time.time() - zone_start

        # 4. 行为分析
        behavior_events = []
        if self.behavior_analyzer:
            behavior_start = time.time()
            behavior_events = self.behavior_analyzer.analyze(tracks)
            self.stats.behavior_time = time.time() - behavior_start

        # 5. 报警处理
        if self.alarm_manager:
            self.alarm_manager.process_zone_events(zone_events, frame)
            self.alarm_manager.process_behavior_events(behavior_events, frame)

        # 6. 可视化
        annotated_frame = self.annotator.annotate_frame(
            frame,
            tracks=tracks,
            detections=detections,
            zone_manager=self.zone_manager,
            alarm_manager=self.alarm_manager,
            fps=self.stats.fps,
            additional_info={
                "Tracks": len(tracks),
                "Detections": len(detections)
            }
        )

        # 更新统计
        self.stats.total_time = time.time() - start_time
        self.stats.frame_count += 1
        self.stats.track_count = len(tracks)
        self.stats.alarm_count = len(self.alarm_manager.alarm_events) if self.alarm_manager else 0

        # 计算FPS
        if self.stats.total_time > 0:
            self.stats.fps = 1.0 / self.stats.total_time

        return annotated_frame

    def run(self, save_output: bool = False, display: bool = True):
        """
        运行管线

        Args:
            save_output: 是否保存输出视频
            display: 是否显示窗口
        """
        if not self.video_source:
            raise RuntimeError("请先设置视频源")

        # 打开视频源
        if not self.video_source.open():
            raise RuntimeError("无法打开视频源")

        video_info = self.video_source.get_info()
        print(f"视频信息: {video_info['width']}x{video_info['height']} @ {video_info['fps']:.1f}fps")

        # 初始化视频输出
        if save_output:
            output_cfg = self.config.get("video", {})
            output_path = Path(output_cfg.get("output_path", "output/videos"))
            output_path.mkdir(parents=True, exist_ok=True)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_file = output_path / f"output_{int(time.time())}.mp4"
            self.video_writer = cv2.VideoWriter(
                str(output_file), fourcc,
                output_cfg.get("output_fps", 30),
                (video_info["width"], video_info["height"])
            )
            print(f"输出视频将保存到: {output_file}")

        self.is_running = True
        print("\n开始处理...")
        print("按 'q' 退出, 'p' 暂停/继续, 's' 保存截图")

        try:
            while self.is_running:
                if self.is_paused:
                    if display:
                        key = cv2.waitKey(100) & 0xFF
                        if key == ord('p'):
                            self.is_paused = False
                        elif key == ord('q'):
                            break
                    continue

                # 读取帧
                frame, frame_id = self.video_source.read()
                if frame is None:
                    print("视频结束")
                    break

                # 处理帧
                result_frame = self.process_frame(frame)

                # 保存输出
                if self.video_writer:
                    self.video_writer.write(result_frame)

                # 显示
                if display:
                    cv2.imshow("YOLOv8s Security System", result_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        self.is_paused = True
                        print("已暂停")
                    elif key == ord('s'):
                        # 保存截图
                        snapshot_path = Path("output/snapshots")
                        snapshot_path.mkdir(parents=True, exist_ok=True)
                        filename = snapshot_path / f"manual_{int(time.time())}.jpg"
                        cv2.imwrite(str(filename), result_frame)
                        print(f"截图已保存: {filename}")

        except KeyboardInterrupt:
            print("\n用户中断")

        finally:
            self.stop()

    def stop(self):
        """停止管线"""
        self.is_running = False

        if self.video_source:
            self.video_source.close()

        if self.video_writer:
            self.video_writer.release()

        cv2.destroyAllWindows()

        # 打印统计
        print("\n===== 处理统计 =====")
        print(f"总帧数: {self.stats.frame_count}")
        print(f"平均FPS: {self.stats.fps:.1f}")
        print(f"检测耗时: {self.stats.detection_time*1000:.1f}ms")
        print(f"跟踪耗时: {self.stats.tracking_time*1000:.1f}ms")
        print(f"区域检测耗时: {self.stats.zone_time*1000:.1f}ms")
        print(f"行为分析耗时: {self.stats.behavior_time*1000:.1f}ms")
        print(f"报警次数: {self.stats.alarm_count}")

    def get_stats(self) -> PipelineStats:
        """获取统计信息"""
        return self.stats


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv8s 智能安防系统")
    parser.add_argument("-c", "--config", default="configs/config.yaml", help="配置文件路径")
    parser.add_argument("-s", "--source", default="", help="视频源路径")
    parser.add_argument("-t", "--type", default="file", choices=["file", "webcam", "rtsp"], help="视频源类型")
    parser.add_argument("-w", "--webcam", type=int, default=0, help="摄像头ID")
    parser.add_argument("--save", action="store_true", help="保存输出视频")
    parser.add_argument("--no-display", action="store_true", help="不显示窗口")

    args = parser.parse_args()

    # 创建管线
    pipeline = SecurityPipeline(config_path=args.config)

    # 设置视频源
    pipeline.set_video_source(
        source_type=args.type,
        source_path=args.source,
        webcam_id=args.webcam
    )

    # 运行
    pipeline.run(
        save_output=args.save,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()
