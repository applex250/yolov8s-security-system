"""
YOLOv8s 目标检测模块
"""

import torch
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from ultralytics import YOLO


@dataclass
class Detection:
    """检测结果"""
    bbox: List[float]      # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str


class YOLODetector:
    """YOLOv8s 检测器封装"""

    # COCO 类别名称
    COCO_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
        4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck',
        8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
        11: 'stop sign', 12: 'parking meter', 13: 'bench',
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse',
        18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear',
        22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
        26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee',
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
        34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard',
        37: 'surfboard', 38: 'tennis racket', 39: 'bottle',
        40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
        44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
        48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot',
        52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
        56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
        60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop',
        64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone',
        68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
        72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
        76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    }

    def __init__(self, model_path: str = "models/yolov8s.pt",
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 device: str = "cuda",
                 classes: Optional[List[int]] = None):
        """
        初始化检测器

        Args:
            model_path: 模型权重路径
            confidence_threshold: 置信度阈值
            iou_threshold: IOU阈值
            device: 运行设备 ("cuda" 或 "cpu")
            classes: 要检测的类别ID列表，None表示所有类别
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes

        self.model: Optional[YOLO] = None
        self.warmup_done = False

    def load_model(self):
        """加载模型"""
        self.model = YOLO(self.model_path)

        # 设置设备
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA不可用，切换到CPU")
            self.device = "cpu"

        # 预热模型
        self._warmup()

    def _warmup(self):
        """模型预热"""
        if self.warmup_done:
            return

        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        self.detect(dummy_input)
        self.warmup_done = True
        print(f"模型预热完成，设备: {self.device}")

    def detect(self, frame) -> List[Detection]:
        """
        执行目标检测

        Args:
            frame: BGR格式的图像 (numpy array)

        Returns:
            检测结果列表
        """
        if self.model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        # 执行推理
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            classes=self.classes,
            verbose=False
        )

        # 解析结果
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                class_name = self.COCO_CLASSES.get(class_id, f"class_{class_id}")

                detection = Detection(
                    bbox=bbox,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name
                )
                detections.append(detection)

        return detections

    def get_class_name(self, class_id: int) -> str:
        """获取类别名称"""
        return self.COCO_CLASSES.get(class_id, f"class_{class_id}")

    def is_person(self, class_id: int) -> bool:
        """判断是否为人员"""
        return class_id == 0

    def is_vehicle(self, class_id: int) -> bool:
        """判断是否为车辆"""
        return class_id in [2, 3, 5, 7]  # car, motorcycle, bus, truck

    def get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        info = {
            "device": self.device,
            "model_path": self.model_path
        }

        if self.device == "cuda" and torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / 1024**3

        return info


class RK3588Detector(YOLODetector):
    """
    RK3588 专用检测器
    使用 ONNX Runtime 进行推理
    """

    def __init__(self, onnx_path: str, **kwargs):
        """
        初始化RK3588检测器

        Args:
            onnx_path: ONNX模型路径
        """
        super().__init__(**kwargs)
        self.onnx_path = onnx_path
        self.session = None

    def load_model(self):
        """加载ONNX模型"""
        import onnxruntime as ort

        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)

        # 获取输入输出信息
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        print(f"ONNX模型加载完成: {self.onnx_path}")
        print(f"输入: {self.input_name}, 输出: {self.output_names}")

    def detect(self, frame) -> List[Detection]:
        """使用ONNX执行检测"""
        # 预处理
        input_tensor = self._preprocess(frame)

        # 推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # 后处理
        detections = self._postprocess(outputs, frame.shape)

        return detections

    def _preprocess(self, frame):
        """图像预处理"""
        import cv2

        # 调整大小
        img = cv2.resize(frame, (640, 640))
        img = img[:, :, ::-1]  # BGR -> RGB
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        return img

    def _postprocess(self, outputs, original_shape):
        """后处理，解析检测结果"""
        detections = []

        # 这里需要根据实际的ONNX输出格式进行调整
        # YOLOv8 ONNX 输出格式: (1, 84, 8400)
        output = outputs[0][0]  # (84, 8400)

        # 转置为 (8400, 84)
        output = output.transpose(1, 0)

        # 过滤低置信度
        scores = output[:, 4:].max(axis=1)
        mask = scores > self.confidence_threshold

        for i in range(len(mask)):
            if not mask[i]:
                continue

            x, y, w, h = output[i, :4]
            class_scores = output[i, 4:]
            class_id = class_scores.argmax()
            confidence = class_scores[class_id]

            # 转换为原始图像坐标
            h_orig, w_orig = original_shape[:2]
            scale_x = w_orig / 640
            scale_y = h_orig / 640

            x1 = (x - w / 2) * scale_x
            y1 = (y - h / 2) * scale_y
            x2 = (x + w / 2) * scale_x
            y2 = (y + h / 2) * scale_y

            detections.append(Detection(
                bbox=[x1, y1, x2, y2],
                confidence=float(confidence),
                class_id=int(class_id),
                class_name=self.COCO_CLASSES.get(int(class_id), f"class_{class_id}")
            ))

        return detections
