#!/usr/bin/env python3
"""
ONNX 模型导出脚本
用于将 YOLOv8s 模型导出为 ONNX 格式，以便在 RK3588 上部署

使用方法:
    python export/export_onnx.py -m models/yolov8s.pt
    python export/export_onnx.py -m models/yolov8s.pt --opset 12 --simplify
"""

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("请安装 ultralytics: pip install ultralytics")
    exit(1)


def export_onnx(
    model_path: str,
    output_dir: str = None,
    opset: int = 12,
    simplify: bool = True,
    dynamic: bool = False,
    imgsz: int = 640,
    half: bool = False
):
    """
    导出 ONNX 模型

    Args:
        model_path: PyTorch 模型路径
        output_dir: 输出目录
        opset: ONNX opset 版本
        simplify: 是否简化模型
        dynamic: 是否使用动态输入尺寸
        imgsz: 输入图像尺寸
        half: 是否使用 FP16 半精度
    """
    print("=" * 50)
    print("  YOLOv8s ONNX 导出工具")
    print("=" * 50)

    # 加载模型
    print(f"\n加载模型: {model_path}")
    model = YOLO(model_path)

    # 导出参数
    export_args = {
        "format": "onnx",
        "opset": opset,
        "simplify": simplify,
        "dynamic": dynamic,
        "imgsz": imgsz,
    }

    # RK3588 推荐配置
    print("\n导出配置:")
    print(f"  - ONNX Opset: {opset}")
    print(f"  - 简化模型: {simplify}")
    print(f"  - 动态尺寸: {dynamic}")
    print(f"  - 输入尺寸: {imgsz}")
    print(f"  - 半精度: {half}")

    # 执行导出
    print("\n开始导出...")
    try:
        output_path = model.export(**export_args)
        print(f"\n导出成功: {output_path}")

        # 显示模型信息
        import onnx
        onnx_model = onnx.load(output_path)

        print("\n模型信息:")
        print(f"  - IR版本: {onnx_model.ir_version}")
        print(f"  - Opset版本: {onnx_model.opset_import[0].version}")

        # 输入信息
        print("\n输入:")
        for input in onnx_model.graph.input:
            name = input.name
            dims = [d.dim_value if d.dim_value > 0 else "dynamic" for d in input.type.tensor_type.shape.dim]
            print(f"  - {name}: {dims}")

        # 输出信息
        print("\n输出:")
        for output in onnx_model.graph.output:
            name = output.name
            dims = [d.dim_value if d.dim_value > 0 else "dynamic" for d in output.type.tensor_type.shape.dim]
            print(f"  - {name}: {dims}")

        # 文件大小
        file_size = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"\n文件大小: {file_size:.2f} MB")

        print("\n" + "=" * 50)
        print("  RK3588 部署提示:")
        print("  1. 安装 rknn-toolkit2")
        print("  2. 使用 rknn-toolkit2 将 ONNX 转换为 RKNN")
        print("  3. 启用 INT8 量化以提高性能")
        print("=" * 50)

        return output_path

    except Exception as e:
        print(f"\n导出失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def verify_onnx(onnx_path: str):
    """
    验证 ONNX 模型

    Args:
        onnx_path: ONNX 模型路径
    """
    import onnx
    import onnxruntime as ort
    import numpy as np

    print("\n验证 ONNX 模型...")

    # 加载模型
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  [OK] 模型结构验证通过")

    # 创建推理会话
    session = ort.InferenceSession(onnx_path)

    # 获取输入信息
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape

    print(f"  输入名称: {input_name}")
    print(f"  输入形状: {input_shape}")

    # 测试推理
    batch_size = 1
    channels = 3
    height = input_shape[2] if isinstance(input_shape[2], int) else 640
    width = input_shape[3] if isinstance(input_shape[3], int) else 640

    dummy_input = np.random.randn(batch_size, channels, height, width).astype(np.float32)

    print(f"\n  执行测试推理 ({height}x{width})...")
    outputs = session.run(None, {input_name: dummy_input})

    print(f"  [OK] 推理成功")
    print(f"  输出数量: {len(outputs)}")
    for i, output in enumerate(outputs):
        print(f"    输出 {i}: shape={output.shape}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="YOLOv8s ONNX 导出工具")
    parser.add_argument("-m", "--model", default="models/yolov8s.pt", help="PyTorch 模型路径")
    parser.add_argument("-o", "--output", default=None, help="输出目录")
    parser.add_argument("--opset", type=int, default=12, help="ONNX opset 版本 (默认: 12)")
    parser.add_argument("--simplify", action="store_true", help="简化 ONNX 模型")
    parser.add_argument("--dynamic", action="store_true", help="使用动态输入尺寸")
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸 (默认: 640)")
    parser.add_argument("--half", action="store_true", help="使用 FP16 半精度")
    parser.add_argument("--verify", action="store_true", help="导出后验证模型")

    args = parser.parse_args()

    # 导出
    output_path = export_onnx(
        model_path=args.model,
        output_dir=args.output,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=args.dynamic,
        imgsz=args.imgsz,
        half=args.half
    )

    # 验证
    if output_path and args.verify:
        verify_onnx(output_path)


if __name__ == "__main__":
    main()
