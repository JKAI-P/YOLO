#!/usr/bin/env python3
"""
测试不同置信度阈值下的Recall和Precision
使用predict模式而不是val模式，因为val模式的conf参数不控制NMS阈值
"""

import os
import yaml
import torch
from ultralytics import YOLO
from pathlib import Path
import numpy as np


def calculate_metrics(pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes, iou_threshold=0.5):
    """
    计算给定预测结果和真实标签的Recall和Precision
    """
    if len(pred_boxes) == 0:
        return 0.0, 0.0, 0
    
    # 按置信度降序排序
    sorted_indices = np.argsort(-pred_scores)
    pred_boxes = pred_boxes[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    pred_classes = pred_classes[sorted_indices]
    
    tp = 0  # True Positives
    fp = 0  # False Positives
    matched_gt = set()
    
    for i, (pred_box, pred_class) in enumerate(zip(pred_boxes, pred_classes)):
        best_iou = 0
        best_gt_idx = -1
        
        # 找到与当前预测框IoU最大的真实框
        for j, (gt_box, gt_class) in enumerate(zip(gt_boxes, gt_classes)):
            if j in matched_gt:
                continue
            if pred_class != gt_class:
                continue
                
            # 计算IoU
            x1 = max(pred_box[0], gt_box[0])
            y1 = max(pred_box[1], gt_box[1])
            x2 = min(pred_box[2], gt_box[2])
            y2 = min(pred_box[3], gt_box[3])
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
                gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
                union = pred_area + gt_area - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j
        
        # 判断是否为TP
        if best_iou >= iou_threshold and best_gt_idx != -1:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1
    
    recall = tp / len(gt_boxes) if len(gt_boxes) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    return recall, precision, tp


def load_ground_truth(label_path):
    """加载真实标签"""
    if not os.path.exists(label_path):
        return [], []
    
    boxes = []
    classes = []
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            
            class_id = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:5])
            
            # 转换为绝对坐标 (假设图像尺寸为640x640)
            x1 = (x_center - width/2) * 640
            y1 = (y_center - height/2) * 640  
            x2 = (x_center + width/2) * 640
            y2 = (y_center + height/2) * 640
            
            boxes.append([x1, y1, x2, y2])
            classes.append(class_id)
    
    return np.array(boxes), np.array(classes)


def test_conf_thresholds():
    """测试不同置信度阈值"""
    # 使用绝对路径
    model_path = '/mnt/workspace/YOLO/runs/detect/runs/train/iyuav_det_200ep5/weights/best.pt'
    data_yaml = '/mnt/workspace/YOLO/datasets/Det-Fly.v6i.yolov/data.yaml'
    valid_images_dir = '/mnt/workspace/YOLO/datasets/Det-Fly.v6i.yolov/valid/images/'
    valid_labels_dir = '/mnt/workspace/YOLO/datasets/Det-Fly.v6i.yolov/valid/labels/'
    
    # 加载模型
    model = YOLO(model_path)
    
    # 获取验证集图像列表
    image_files = [f for f in os.listdir(valid_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    print(f"找到 {len(image_files)} 张验证图像")
    
    # 测试的conf阈值
    conf_values = [0.25, 0.2, 0.15, 0.1, 0.05, 0.01, 0.001, 0.0001]
    
    print("\n🔍 测试不同置信度阈值...")
    print("=" * 80)
    
    results = {}
    
    for conf in conf_values:
        print(f"\n测试 conf={conf}...")
        
        total_tp = 0
        total_gt = 0
        total_pred = 0
        
        # 处理每张图像（先测试前50张加速）
        for img_file in image_files[:50]:
            img_path = os.path.join(valid_images_dir, img_file)
            label_path = os.path.join(valid_labels_dir, img_file.replace('.jpg', '.txt').replace('.png', '.txt'))
            
            # 获取预测结果
            results_pred = model.predict(
                img_path, 
                conf=conf, 
                iou=0.7,
                imgsz=640,
                device='0',
                verbose=False
            )
            
            pred_result = results_pred[0]
            if len(pred_result.boxes) > 0:
                pred_boxes = pred_result.boxes.xyxy.cpu().numpy()
                pred_scores = pred_result.boxes.conf.cpu().numpy()
                pred_classes = pred_result.boxes.cls.cpu().numpy().astype(int)
            else:
                pred_boxes = np.array([])
                pred_scores = np.array([])
                pred_classes = np.array([])
            
            # 获取真实标签
            gt_boxes, gt_classes = load_ground_truth(label_path)
            
            # 计算指标
            recall_img, precision_img, tp_img = calculate_metrics(
                pred_boxes, pred_scores, pred_classes, gt_boxes, gt_classes
            )
            
            total_tp += tp_img
            total_gt += len(gt_boxes)
            total_pred += len(pred_boxes)
        
        # 计算整体指标
        overall_recall = total_tp / total_gt if total_gt > 0 else 0
        overall_precision = total_tp / total_pred if total_pred > 0 else 0
        
        results[conf] = {
            'recall': overall_recall,
            'precision': overall_precision,
            'total_gt': total_gt,
            'total_pred': total_pred,
            'total_tp': total_tp
        }
        
        print(f"  ✓ Recall: {overall_recall:.4f} | Precision: {overall_precision:.4f} | TP: {total_tp}/{total_gt}")
    
    # 输出结果汇总
    print("\n" + "=" * 80)
    print("📊 最终结果汇总:")
    print("Conf      | Recall   | Precision | 目标达成")
    print("-" * 50)
    
    target_conf = None
    for conf in sorted(results.keys()):
        recall = results[conf]['recall']
        precision = results[conf]['precision']
        target_achieved = '✅' if recall >= 0.98 else '❌'
        print(f"{conf:<9} | {recall:<8.4f} | {precision:<9.4f} | {target_achieved}")
        
        if recall >= 0.98 and target_conf is None:
            target_conf = conf
    
    print("\n" + "=" * 80)
    if target_conf is not None:
        print(f"🎯 推荐配置 (Recall ≥ 98%):")
        print(f"   Confidence阈值: {target_conf}")
        print(f"   Recall: {results[target_conf]['recall']:.2%}")
        print(f"   Precision: {results[target_conf]['precision']:.2%}")
    else:
        print("❌ 未找到满足 Recall ≥ 98% 的配置")
        print("建议:")
        print("1. 尝试更低的conf阈值 (如 0.00001)")
        print("2. 检查数据标注质量")
        print("3. 考虑使用集成学习或模型融合")


if __name__ == "__main__":
    test_conf_thresholds()