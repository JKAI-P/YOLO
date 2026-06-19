"""
下载无人机目标检测数据集
尝试多个可用来源.
"""

import os

DATA_DIR = "C:/Users/soldier/Desktop/ultralytics-main/datasets"


def try_download():
    """尝试下载."""
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 50)
    print("无人机目标检测数据集下载")
    print("=" * 50)
    print()
    print("推荐以下数据集来源:")
    print()
    print("1. VisDrone (推荐)")
    print("   下载地址: https://github.com/VisDrone/VisDrone-Dataset")
    print("   说明: 包含多种复杂场景，类别丰富")
    print()
    print("2. 使用Python直接下载VisDrone:")
    print("   pip install gdown")
    print("   gdown --fuzzy https://drive.google.com/file/d/1m1qBHkAqlLVCf6xhbN3bhXK3mR3hX8g/view")
    print()
    print("3. Roboflow Universe搜索 'uav' 或 'drone'")
    print("   https://universe.roboflow.com/")
    print()

    # 创建示例配置
    create_sample_config()


def create_sample_config():
    """创建示例YAML配置."""
    yaml_path = os.path.join(DATA_DIR, "visdrone.yaml")

    # 如果已存在visdrone目录，使用实际路径
    train_path = "images/train" if os.path.exists(os.path.join(DATA_DIR, "images/train")) else "path/to/train/images"
    val_path = "images/val" if os.path.exists(os.path.join(DATA_DIR, "images/val")) else "path/to/val/images"

    yaml_content = f"""# 无人机目标检测数据集配置
# 数据集: VisDrone
# 类别 (11类):
#   0: pedestrian (行人)
#   1: person (人)
#   2: bicycle (自行车)
#   3: car (汽车)
#   4: van (货车)
#   5: truck (卡车)
#   6: tricycle (三轮车)
#   7: awning-tricycle (带棚三轮车)
#   8: bus (公交车)
#   9: motor (摩托车)
#   10: others (其他)

path: {DATA_DIR}
train: {train_path}
val: {val_path}

nc: 11
names:
  0: pedestrian
  1: person
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
  10: others
"""

    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"示例配置已创建: {yaml_path}")
    print()
    print("请下载数据集后，将数据放置在datasets目录下")
    print("目录结构:")
    print(f"""
{DATA_DIR}/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── visdrone.yaml
""")


if __name__ == "__main__":
    try_download()
