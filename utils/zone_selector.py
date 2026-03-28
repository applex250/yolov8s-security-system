#!/usr/bin/env python3
"""
多边形区域框选工具
用于生成 zones.json 配置文件

使用方法:
    python utils/zone_selector.py -s video.mp4
    python utils/zone_selector.py -s video.mp4 -o configs/zones.json

操作说明:
    - 鼠标左键点击添加顶点
    - 按 'c' 完成当前多边形
    - 按 'r' 重置当前多边形
    - 按 'd' 删除最后一个区域
    - 按 's' 保存配置
    - 按 'q' 退出
"""

import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import argparse


class ZoneSelector:
    """多边形区域选择器"""

    def __init__(self, frame: np.ndarray, output_path: str = "configs/zones.json"):
        """
        初始化选择器

        Args:
            frame: 参考帧
            output_path: 输出配置文件路径
        """
        self.frame = frame.copy()
        self.display_frame = frame.copy()
        self.output_path = output_path

        # 当前多边形顶点
        self.current_points: List[List[int]] = []

        # 已完成的区域
        self.zones: List[Dict] = []

        # 区域配置
        self.zone_id = 1
        self.current_class_filter = [0]  # 默认检测人员

        # 窗口名称
        self.window_name = "Zone Selector"

        # 颜色
        self.color_current = (0, 255, 0)   # 绿色 - 当前绘制
        self.color_completed = (0, 0, 255)  # 红色 - 已完成
        self.color_fill = (0, 0, 255)

    def draw(self):
        """绘制当前状态"""
        self.display_frame = self.frame.copy()

        # 绘制已完成的区域
        for zone in self.zones:
            polygon = np.array(zone["polygon"], dtype=np.int32)

            # 半透明填充
            overlay = self.display_frame.copy()
            cv2.fillPoly(overlay, [polygon], self.color_fill)
            cv2.addWeighted(overlay, 0.2, self.display_frame, 0.8, 0, self.display_frame)

            # 边框
            cv2.polylines(self.display_frame, [polygon], True, self.color_completed, 2)

            # 标签
            x, y = polygon[0]
            label = f"{zone['name']} (ID:{zone['id']})"
            cv2.putText(self.display_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color_completed, 2)

        # 绘制当前正在绘制的多边形
        if len(self.current_points) >= 1:
            # 绘制顶点
            for pt in self.current_points:
                cv2.circle(self.display_frame, tuple(pt), 5, self.color_current, -1)

            # 绘制线段
            if len(self.current_points) >= 2:
                pts = np.array(self.current_points, dtype=np.int32)
                cv2.polylines(self.display_frame, [pts], False, self.color_current, 2)

        # 绘制帮助信息
        self._draw_help()

    def _draw_help(self):
        """绘制帮助信息"""
        help_texts = [
            "Left Click: Add point",
            "c: Complete polygon",
            "r: Reset current",
            "d: Delete last zone",
            "s: Save config",
            "q: Quit",
            f"Zones: {len(self.zones)}"
        ]

        y = 30
        for text in help_texts:
            cv2.putText(self.display_frame, text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y += 20

    def add_point(self, x: int, y: int):
        """添加顶点"""
        self.current_points.append([x, y])
        self.draw()

    def complete_polygon(self):
        """完成当前多边形"""
        if len(self.current_points) < 3:
            print("需要至少3个顶点")
            return

        # 获取区域名称
        zone_name = f"Zone_{self.zone_id}"

        zone = {
            "id": self.zone_id,
            "name": zone_name,
            "polygon": self.current_points.copy(),
            "enabled": True,
            "detect_classes": self.current_class_filter.copy(),
            "description": f"区域 {self.zone_id}"
        }

        self.zones.append(zone)
        self.zone_id += 1
        self.current_points.clear()

        print(f"区域已添加: {zone_name}")
        self.draw()

    def reset_current(self):
        """重置当前多边形"""
        self.current_points.clear()
        self.draw()
        print("当前多边形已重置")

    def delete_last(self):
        """删除最后一个区域"""
        if self.zones:
            zone = self.zones.pop()
            print(f"已删除区域: {zone['name']}")
            self.draw()
        else:
            print("没有可删除的区域")

    def save(self):
        """保存配置"""
        config = {
            "zones": self.zones,
            "version": "1.0",
            "description": "电子围栏区域配置文件"
        }

        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        print(f"配置已保存到: {output_path}")
        print(f"共 {len(self.zones)} 个区域")

    def run(self):
        """运行选择器"""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        self.draw()

        print("\n===== 区域框选工具 =====")
        print("鼠标左键点击添加顶点")
        print("按 'c' 完成当前多边形")
        print("按 'r' 重置当前多边形")
        print("按 'd' 删除最后一个区域")
        print("按 's' 保存配置")
        print("按 'q' 退出")
        print("=" * 25)

        while True:
            cv2.imshow(self.window_name, self.display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('c'):
                self.complete_polygon()
            elif key == ord('r'):
                self.reset_current()
            elif key == ord('d'):
                self.delete_last()
            elif key == ord('s'):
                self.save()

        cv2.destroyAllWindows()

        # 退出前询问是否保存
        if self.zones:
            response = input("是否保存配置? (y/n): ")
            if response.lower() == 'y':
                self.save()

    def _mouse_callback(self, event, x, y, flags, param):
        """鼠标回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.add_point(x, y)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多边形区域框选工具")
    parser.add_argument("-s", "--source", required=True, help="视频文件或图片路径")
    parser.add_argument("-o", "--output", default="configs/zones.json", help="输出配置文件路径")
    parser.add_argument("-f", "--frame", type=int, default=0, help="使用的帧号 (默认: 0)")

    args = parser.parse_args()

    # 读取参考图像
    source_path = Path(args.source)

    if source_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # 图片文件
        frame = cv2.imread(str(source_path))
        if frame is None:
            print(f"无法读取图片: {source_path}")
            return
    else:
        # 视频文件
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            print(f"无法打开视频: {source_path}")
            return

        # 跳到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("无法读取视频帧")
            return

    print(f"参考帧尺寸: {frame.shape[1]}x{frame.shape[0]}")

    # 运行选择器
    selector = ZoneSelector(frame, output_path=args.output)
    selector.run()


if __name__ == "__main__":
    main()
