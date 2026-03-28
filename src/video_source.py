"""
视频源管理模块
支持本地视频文件、RTSP流、摄像头
"""

import cv2
import threading
import queue
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class FrameInfo:
    """帧信息"""
    frame: any
    frame_id: int
    timestamp: float


class VideoSource:
    """视频源管理类"""

    def __init__(self, source_type: str = "file", source_path: str = "",
                 webcam_id: int = 0, buffer_size: int = 30):
        """
        初始化视频源

        Args:
            source_type: 源类型 ("file", "rtsp", "webcam")
            source_path: 视频文件路径或RTSP地址
            webcam_id: 摄像头ID
            buffer_size: 帧缓冲区大小
        """
        self.source_type = source_type
        self.source_path = source_path
        self.webcam_id = webcam_id
        self.buffer_size = buffer_size

        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.is_running = False
        self.read_thread: Optional[threading.Thread] = None

        self.frame_count = 0
        self.fps = 30.0
        self.width = 1920
        self.height = 1080

    def open(self) -> bool:
        """打开视频源"""
        if self.source_type == "file":
            if not self.source_path:
                raise ValueError("视频文件路径不能为空")
            self.cap = cv2.VideoCapture(self.source_path)
        elif self.source_type == "rtsp":
            if not self.source_path:
                raise ValueError("RTSP地址不能为空")
            self.cap = cv2.VideoCapture(self.source_path)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        elif self.source_type == "webcam":
            self.cap = cv2.VideoCapture(self.webcam_id)
        else:
            raise ValueError(f"不支持的源类型: {self.source_type}")

        if not self.cap.isOpened():
            return False

        # 获取视频信息
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 启动读取线程
        self.is_running = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()

        return True

    def _read_loop(self):
        """帧读取循环（后台线程）"""
        frame_id = 0
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                if self.source_type == "file":
                    # 视频结束，循环播放
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break

            try:
                self.frame_queue.put((frame, frame_id), timeout=1.0)
                frame_id += 1
            except queue.Full:
                # 缓冲区满，丢弃旧帧
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put((frame, frame_id), timeout=1.0)
                    frame_id += 1
                except:
                    pass

    def read(self) -> Tuple[Optional[any], int]:
        """
        读取一帧

        Returns:
            (frame, frame_id) 或 (None, -1) 如果视频结束
        """
        try:
            frame, frame_id = self.frame_queue.get(timeout=2.0)
            return frame, frame_id
        except queue.Empty:
            return None, -1

    def get_info(self) -> dict:
        """获取视频信息"""
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
            "source_type": self.source_type
        }

    def close(self):
        """关闭视频源"""
        self.is_running = False
        if self.read_thread:
            self.read_thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
