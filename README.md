# YOLOv8s 智能安防系统

基于 YOLOv8s 的智能视频监控系统，支持目标检测、多目标跟踪、电子围栏、行为分析和报警功能。

---

## 目录

- [功能特性](#功能特性)
- [环境要求](#环境要求)
- [安装部署](#安装部署)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [报警系统](#报警系统)
- [命令行参数](#命令行参数)
- [区域框选工具](#区域框选工具)
- [性能参数](#性能参数)
- [RK3588 部署](#rk3588-部署)
- [常见问题](#常见问题)

---

## 功能特性

| 功能 | 说明 |
|------|------|
| 目标检测 | 检测人员、车辆等 80 类目标 (COCO 数据集) |
| 多目标跟踪 | ByteTrack / BoT-SORT 实时跟踪 |
| 电子围栏 | 自定义多边形区域，检测入侵 |
| 徘徊检测 | 检测在区域内长时间停留的人员 |
| 人群聚集 | 检测多人聚集情况 |
| 报警系统 | 视觉闪烁、蜂鸣声、自动截图 |
| 视频输出 | 支持保存处理后的视频 |

---

## 环境要求

### 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| CPU | 4核 | 8核+ |
| 内存 | 4GB | 8GB+ |
| GPU | - | NVIDIA GTX 1060+ |
| 显存 | - | 4GB+ |

### 软件要求

| 软件 | 版本 |
|------|------|
| Python | 3.8+ |
| PyTorch | 2.0+ |
| CUDA | 11.8+ (GPU模式) |
| OpenCV | 4.5+ |

### 支持的视频格式

- **视频文件**: mp4, avi, mkv, mov, wmv, flv
- **网络流**: RTSP, RTMP
- **摄像头**: USB 摄像头

---

## 安装部署

### 1. 安装依赖

```bash
pip install torch torchvision ultralytics supervision lapx pyyaml opencv-python numpy
```

### 2. GPU 加速 (推荐)

```bash
# CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CUDA 13.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 3. 验证安装

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 4. 下载模型

首次运行会自动下载 YOLOv8s 模型，或手动下载：

```bash
# 模型会自动下载到 models/yolov8s.pt
wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt -O models/yolov8s.pt
```

---

## 快速开始

### 处理视频文件

```bash
# 基本用法
python main.py -s video.mp4

# 支持 .avi 格式
python main.py -s video.avi

# 保存输出视频
python main.py -s video.mp4 --save
```

### 使用摄像头

```bash
# 默认摄像头 (ID: 0)
python main.py -t webcam -w 0

# 指定摄像头
python main.py -t webcam -w 1
```

### RTSP 网络流

```bash
python main.py -t rtsp -s rtsp://admin:password@192.168.1.100:554/stream1
```

### 无头模式 (服务器运行)

```bash
python main.py -s video.mp4 --no-display --save
```

---

## 配置说明

### 主配置文件: `configs/config.yaml`

#### 模型配置

```yaml
model:
  path: "models/yolov8s.pt"       # 模型路径
  confidence_threshold: 0.5       # 置信度阈值 (0-1)
  iou_threshold: 0.45             # IOU 阈值 (0-1)
  device: "cuda"                  # cuda 或 cpu
  classes: [0, 1, 2, 3, 5, 7]     # 检测类别
```

#### 常用类别 ID (COCO)

| ID | 类别 | ID | 类别 | ID | 类别 |
|----|------|----|------|----|------|
| 0 | person (人) | 5 | bus (公交车) | 16 | dog (狗) |
| 1 | bicycle (自行车) | 7 | truck (卡车) | 17 | cat (猫) |
| 2 | car (小汽车) | 9 | traffic light | 24 | backpack |
| 3 | motorcycle (摩托车) | 11 | stop sign | 26 | handbag |
| 4 | airplane (飞机) | 15 | bird (鸟) | 67 | cell phone |

#### 跟踪配置

```yaml
tracker:
  type: "bytetrack"     # bytetrack 或 botsort
  track_buffer: 30      # 轨迹缓冲帧数
  match_thresh: 0.8     # 匹配阈值
```

#### 电子围栏配置

```yaml
zones:
  config_path: "configs/zones.json"
  enable: true          # 启用/禁用
```

#### 行为分析配置

```yaml
behavior:
  # 徘徊检测
  loitering:
    enable: true
    time_threshold: 30.0       # 停留时间阈值 (秒)
    distance_threshold: 100    # 移动距离阈值 (像素)
    min_frames: 30             # 最小连续帧数

  # 人群聚集检测
  crowd:
    enable: true
    min_people: 3              # 最少人数
    distance_threshold: 150    # 人与人距离 (像素)
    time_threshold: 10.0       # 持续时间 (秒)
```

#### 报警配置

```yaml
alarm:
  # 视觉报警
  visual:
    enable: true
    border_color: [0, 0, 255]  # BGR 颜色
    border_thickness: 5
    flash_interval: 0.5        # 闪烁间隔 (秒)
    alert_text: "ALERT!"

  # 音频报警
  audio:
    enable: true
    type: "beep"               # beep 或 custom
    frequency: 1000            # 频率 (Hz)
    duration: 0.3              # 持续时间 (秒)

  # 截图保存
  snapshot:
    enable: true
    save_path: "output/snapshots"
    prefix: "alert_"
    format: "jpg"
    quality: 95
```

### 区域配置文件: `configs/zones.json`

```json
{
  "zones": [
    {
      "id": 1,
      "name": "Zone-A-Entrance",
      "polygon": [[100, 100], [300, 100], [300, 400], [100, 400]],
      "enabled": true,
      "detect_classes": [0],
      "description": "Main entrance, detect people"
    },
    {
      "id": 2,
      "name": "Zone-B-Parking",
      "polygon": [[400, 100], [800, 100], [800, 500], [400, 500]],
      "enabled": true,
      "detect_classes": [2, 3, 5, 7],
      "description": "Parking area, detect vehicles"
    }
  ]
}
```

**字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| id | int | 区域唯一标识 |
| name | string | 区域名称 (英文) |
| polygon | array | 多边形顶点坐标 [[x1,y1], [x2,y2], ...] |
| enabled | bool | 是否启用 |
| detect_classes | array | 检测的类别 ID，空数组表示所有类别 |
| description | string | 区域描述 |

---

## 报警系统

### 报警类型

| 类型 | 触发条件 | 示例输出 |
|------|---------|---------|
| Zone Intrusion | 目标进入电子围栏区域 | `[ALERT] Zone Intrusion: Target #5 entered [Zone-A-Entrance]` |
| Loitering | 人员长时间停留 | `[ALERT] loitering_detected: Loitering detected: ID [3]` |
| Crowd | 多人聚集 | `[ALERT] crowd_detected: Crowd detected: 7 people` |

### 报警输出

1. **视觉报警**: 红色闪烁边框 + "ALERT!" 文字
2. **音频报警**: 蜂鸣声 (Windows)
3. **截图保存**: 自动保存到 `output/snapshots/`

**截图命名格式**:
```
alert_YYYYMMDD_HHMMSS_类型_序号.jpg

示例:
alert_20260328_154518_intrusion_0000.jpg
alert_20260328_154520_loitering_0001.jpg
alert_20260328_154525_crowd_0002.jpg
```

### 关闭报警

编辑 `configs/config.yaml`:

```yaml
# 关闭所有报警
alarm:
  visual:
    enable: false
  audio:
    enable: false
  snapshot:
    enable: false

# 关闭特定检测
behavior:
  loitering:
    enable: false
  crowd:
    enable: false

zones:
  enable: false
```

---

## 命令行参数

```
usage: main.py [-h] [-c CONFIG] [-s SOURCE] [-t TYPE] [-w WEBCAM] [--save] [--no-display]

选项:
  -h, --help            显示帮助信息
  -c, --config          配置文件路径 (默认: configs/config.yaml)
  -s, --source          视频源路径 (文件/RTSP地址)
  -t, --type            源类型: file, webcam, rtsp (默认: file)
  -w, --webcam          摄像头 ID (默认: 0)
  --save                保存输出视频
  --no-display          不显示窗口 (无头模式)
```

### 使用示例

```bash
# 基本使用
python main.py -s video.mp4

# 使用自定义配置
python main.py -c my_config.yaml -s video.mp4

# 使用第二个摄像头
python main.py -t webcam -w 1

# RTSP 流并保存
python main.py -t rtsp -s rtsp://192.168.1.100/stream --save

# 服务器无头运行
python main.py -s video.mp4 --no-display --save
```

---

## 区域框选工具

使用 `utils/zone_selector.py` 可视化定义监控区域。

### 使用方法

```bash
python utils/zone_selector.py -s video.mp4 -o configs/zones.json
```

### 参数说明

| 参数 | 说明 |
|------|------|
| -s, --source | 视频文件或图片路径 |
| -o, --output | 输出配置文件路径 |
| -f, --frame | 使用的帧号 (默认: 0) |

### 操作说明

| 操作 | 功能 |
|------|------|
| 鼠标左键 | 添加多边形顶点 |
| c | 完成当前多边形 |
| r | 重置当前多边形 |
| d | 删除最后一个区域 |
| s | 保存配置文件 |
| q | 退出 |

---

## 性能参数

### RTX 5060 性能参考

| 模式 | FPS | 显存占用 |
|------|-----|---------|
| 纯检测 | 150-200+ | ~1.5 GB |
| 检测+跟踪 | 100-150 | ~1.8 GB |
| 完整管线 | 50-80 | ~2.0 GB |

### 性能优化建议

1. **使用 GPU**: 设置 `device: "cuda"`
2. **减少检测类别**: 只检测需要的类别
3. **降低置信度阈值**: 可提高召回率但增加误检
4. **调整图像分辨率**: 较小的输入尺寸可提高速度

### CPU 模式

如果没有 GPU，使用 CPU 模式：

```yaml
model:
  device: "cpu"
```

预计 FPS: 5-15 (取决于 CPU)

---

## RK3588 部署

### 1. 导出 ONNX 模型

```bash
python export/export_onnx.py -m models/yolov8s.pt --opset 12 --simplify
```

### 2. 转换为 RKNN

```python
from rknn.api import RKNN

rknn = RKNN()
rknn.config(target_platform='rk3588')
rknn.load_onnx(model='yolov8s.onnx')
rknn.build(do_quantization=True, dataset='dataset.txt')
rknn.export_rknn('yolov8s.rknn')
```

### 3. 修改检测器

在 RK3588 上使用 `RK3588Detector`:

```python
from src.detector import RK3588Detector

detector = RK3588Detector(onnx_path="yolov8s.rknn")
detector.load_model()
```

---

## 常见问题

### Q: CUDA out of memory

**解决方案**:
- 减少检测类别
- 使用更小的模型 (yolov8n)
- 降低输入分辨率

### Q: 找不到模块

**解决方案**:
```bash
# 确保在项目根目录运行
cd E:\yolo_test\v1
python main.py -s video.mp4
```

### Q: 视频无法打开

**解决方案**:
```bash
# 安装完整版 OpenCV
pip install opencv-python opencv-contrib-python
```

### Q: 中文显示乱码

OpenCV 不支持中文，区域名称请使用英文。

### Q: 报警声音不响

- Windows: 默认使用 winsound
- Linux/Mac: 需要安装 simpleaudio
  ```bash
  pip install simpleaudio
  ```

### Q: FPS 很低

**检查项**:
1. 确认使用 GPU: `device: "cuda"`
2. 检查 CUDA 是否可用: `torch.cuda.is_available()`
3. 减少 `classes` 列表中的类别数量

---

## 项目结构

```
E:\yolo_test\v1\
├── configs/
│   ├── config.yaml          # 主配置文件
│   └── zones.json           # 电子围栏区域配置
├── models/
│   └── yolov8s.pt           # YOLOv8s 模型权重
├── src/
│   ├── __init__.py
│   ├── detector.py          # 目标检测模块
│   ├── tracker.py           # 多目标跟踪模块
│   ├── zone_manager.py      # 电子围栏模块
│   ├── behavior_analyzer.py # 行为分析模块
│   ├── alarm_manager.py     # 报警管理模块
│   ├── annotator.py         # 可视化标注模块
│   ├── video_source.py      # 视频源模块
│   └── pipeline.py          # 主管线模块
├── utils/
│   └── zone_selector.py     # 区域框选工具
├── export/
│   └── export_onnx.py       # ONNX 导出脚本
├── output/
│   ├── snapshots/           # 报警截图
│   └── videos/              # 输出视频
├── main.py                  # 主入口
├── README.md                # 使用说明
└── requirements.txt         # 依赖列表
```

---

## 快捷键

| 按键 | 功能 |
|------|------|
| q | 退出程序 |
| p | 暂停/继续 |
| s | 手动保存截图 |

---

## 版本历史

- **v1.0.0** - 初始版本
  - YOLOv8s 目标检测
  - ByteTrack 多目标跟踪
  - 电子围栏区域检测
  - 徘徊/人群聚集检测
  - 视觉/音频/截图报警
  - RTSP 流支持
  - RK3588 部署支持

---

## License

MIT License
