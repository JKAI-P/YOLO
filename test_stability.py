#!/usr/bin/env python3
"""
数值稳定性测试脚本
测试CIoU和SIoU在各种极端情况下的表现
"""

import torch
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ultralytics.utils.metrics import bbox_iou

def test_extreme_cases():
    """测试极端情况下的数值稳定性"""
    print("🔍 开始数值稳定性测试...")
    
    # 测试用例1: 完全重叠的边界框 (IoU = 1)
    print("\n🧪 测试用例1: 完全重叠 (IoU = 1)")
    box1 = torch.tensor([[0, 0, 100, 100]], dtype=torch.float32)
    box2 = torch.tensor([[0, 0, 100, 100]], dtype=torch.float32)
    
    try:
        iou_ciou = bbox_iou(box1, box2, CIoU=True)
        iou_siou = bbox_iou(box1, box2, SIoU=True)
        print(f"✅ CIoU: {iou_ciou.item():.6f}, SIoU: {iou_siou.item():.6f}")
        assert not torch.isnan(iou_ciou) and not torch.isinf(iou_ciou)
        assert not torch.isnan(iou_siou) and not torch.isinf(iou_siou)
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    # 测试用例2: 完全不重叠的边界框 (IoU = 0)
    print("\n🧪 测试用例2: 完全不重叠 (IoU = 0)")
    box1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    box2 = torch.tensor([[100, 100, 110, 110]], dtype=torch.float32)
    
    try:
        iou_ciou = bbox_iou(box1, box2, CIoU=True)
        iou_siou = bbox_iou(box1, box2, SIoU=True)
        print(f"✅ CIoU: {iou_ciou.item():.6f}, SIoU: {iou_siou.item():.6f}")
        assert not torch.isnan(iou_ciou) and not torch.isinf(iou_ciou)
        assert not torch.isnan(iou_siou) and not torch.isinf(iou_siou)
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    # 测试用例3: 极端宽高比
    print("\n🧪 测试用例3: 极端宽高比")
    box1 = torch.tensor([[0, 0, 1000, 1]], dtype=torch.float32)  # 非常宽
    box2 = torch.tensor([[0, 0, 1, 1000]], dtype=torch.float32)  # 非常高
    
    try:
        iou_ciou = bbox_iou(box1, box2, CIoU=True)
        iou_siou = bbox_iou(box1, box2, SIoU=True)
        print(f"✅ CIoU: {iou_ciou.item():.6f}, SIoU: {iou_siou.item():.6f}")
        assert not torch.isnan(iou_ciou) and not torch.isinf(iou_ciou)
        assert not torch.isnan(iou_siou) and not torch.isinf(iou_siou)
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    # 测试用例4: 批量处理
    print("\n🧪 测试用例4: 批量处理 (batch_size=32)")
    batch_size = 32
    box1 = torch.rand(batch_size, 4) * 100
    box2 = torch.rand(batch_size, 4) * 100
    
    # 确保坐标格式正确 [x1, y1, x2, y2]
    box1[:, 2:] = box1[:, :2] + torch.abs(box1[:, 2:] - box1[:, :2]) + 1
    box2[:, 2:] = box2[:, :2] + torch.abs(box2[:, 2:] - box2[:, :2]) + 1
    
    try:
        iou_ciou = bbox_iou(box1, box2, CIoU=True)
        iou_siou = bbox_iou(box1, box2, SIoU=True)
        print(f"✅ 批量CIoU: min={iou_ciou.min().item():.6f}, max={iou_ciou.max().item():.6f}")
        print(f"✅ 批量SIoU: min={iou_siou.min().item():.6f}, max={iou_siou.max().item():.6f}")
        assert not torch.any(torch.isnan(iou_ciou)) and not torch.any(torch.isinf(iou_ciou))
        assert not torch.any(torch.isnan(iou_siou)) and not torch.any(torch.isinf(iou_siou))
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    # 测试用例5: 梯度计算
    print("\n🧪 测试用例5: 梯度计算")
    box1 = torch.tensor([[0, 0, 50, 50]], dtype=torch.float32, requires_grad=True)
    box2 = torch.tensor([[10, 10, 60, 60]], dtype=torch.float32, requires_grad=True)
    
    try:
        loss_ciou = 1 - bbox_iou(box1, box2, CIoU=True)
        loss_siou = 1 - bbox_iou(box1, box2, SIoU=True)
        
        loss_ciou.backward()
        grad_ciou = box1.grad.clone()
        box1.grad.zero_()
        
        loss_siou.backward()
        grad_siou = box1.grad.clone()
        
        print(f"✅ CIoU梯度: {grad_ciou}")
        print(f"✅ SIoU梯度: {grad_siou}")
        assert not torch.any(torch.isnan(grad_ciou)) and not torch.any(torch.isinf(grad_ciou))
        assert not torch.any(torch.isnan(grad_siou)) and not torch.any(torch.isinf(grad_siou))
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    # 测试用例6: 中心点y坐标相同 (s_ch = 0)
    print("\n🧪 测试用例6: 中心点y坐标相同 (s_ch = 0)")
    box1 = torch.tensor([[0, 50, 100, 150]], dtype=torch.float32)  # center_y = 100
    box2 = torch.tensor([[50, 50, 150, 150]], dtype=torch.float32)  # center_y = 100
    
    try:
        iou_ciou = bbox_iou(box1, box2, CIoU=True)
        iou_siou = bbox_iou(box1, box2, SIoU=True)
        print(f"✅ CIoU: {iou_ciou.item():.6f}, SIoU: {iou_siou.item():.6f}")
        assert not torch.isnan(iou_ciou) and not torch.isinf(iou_ciou)
        assert not torch.isnan(iou_siou) and not torch.isinf(iou_siou)
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    
    print("\n🎉 所有数值稳定性测试通过！")
    return True

if __name__ == "__main__":
    success = test_extreme_cases()
    if success:
        print("\n✅ 数值稳定性验证完成，可以安全开始训练！")
        sys.exit(0)
    else:
        print("\n❌ 数值稳定性测试失败，请修复代码后再训练！")
        sys.exit(1)