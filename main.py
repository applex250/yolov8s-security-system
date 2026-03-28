#!/usr/bin/env python3
"""
YOLOv8s 智能安防系统 - 主入口

功能:
    - 目标检测: YOLOv8s 检测人员/车辆
    - 多目标跟踪: ByteTrack/BoT-SORT
    - 区域入侵: 电子围栏多边形区域检测
    - 行为分析: 徘徊检测、人群聚集检测
    - 报警系统: 视觉(闪烁边框)、音频(Beep)、截图保存

支持的视频格式: mp4, avi, mkv, mov, wmv 等

使用方法:
    python main.py -s video.mp4              # 处理视频文件
    python main.py -s video.avi              # 支持 .avi 格式
    python main.py -t webcam -w 0            # 使用摄像头
    python main.py -s rtsp://... -t rtsp     # RTSP流
    python main.py -s video.mp4 --save       # 保存输出视频
    python main.py --help                    # 显示帮助
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import SecurityPipeline


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="YOLOv8s 智能安防系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s -s video.mp4                    处理视频文件 (.mp4/.avi/.mkv)
  %(prog)s -s video.avi                    支持 .avi 格式
  %(prog)s -t webcam -w 0                  使用默认摄像头
  %(prog)s -s rtsp://192.168.1.100/stream  RTSP流
  %(prog)s -s video.mp4 --save             保存输出视频
  %(prog)s -s video.mp4 --no-display       无头模式运行

快捷键:
  q - 退出
  p - 暂停/继续
  s - 手动保存截图
        """
    )

    parser.add_argument(
        "-c", "--config",
        default="configs/config.yaml",
        help="配置文件路径 (默认: configs/config.yaml)"
    )

    parser.add_argument(
        "-s", "--source",
        default="",
        help="视频源路径 (文件路径/RTSP地址)"
    )

    parser.add_argument(
        "-t", "--type",
        default="file",
        choices=["file", "webcam", "rtsp"],
        help="视频源类型 (默认: file)"
    )

    parser.add_argument(
        "-w", "--webcam",
        type=int,
        default=0,
        help="摄像头ID (默认: 0)"
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="保存输出视频"
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="不显示窗口 (无头模式)"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    print("=" * 50)
    print("  YOLOv8s 智能安防系统 v1.0.0")
    print("=" * 50)
    print()

    # 验证视频源
    if args.type == "file" and not args.source:
        print("错误: 文件模式需要指定视频源路径 (-s)")
        print("示例: python main.py -s video.mp4  或  python main.py -s video.avi")
        sys.exit(1)

    if args.type == "rtsp" and not args.source:
        print("错误: RTSP模式需要指定RTSP地址 (-s)")
        print("示例: python main.py -t rtsp -s rtsp://192.168.1.100/stream")
        sys.exit(1)

    # 验证配置文件
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)

    try:
        # 创建管线
        print(f"加载配置: {config_path}")
        pipeline = SecurityPipeline(config_path=str(config_path))

        # 设置视频源
        print(f"视频源类型: {args.type}")
        if args.source:
            print(f"视频源: {args.source}")

        pipeline.set_video_source(
            source_type=args.type,
            source_path=args.source,
            webcam_id=args.webcam
        )

        # 运行
        print()
        pipeline.run(
            save_output=args.save,
            display=not args.no_display
        )

    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n程序结束")


if __name__ == "__main__":
    main()
