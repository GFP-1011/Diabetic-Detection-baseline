import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np

import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.models as models
from loaddata import load_datasets
from tqdm import tqdm
import matplotlib.pyplot as plt


def plot_auc_curve(all_labels, all_preds, group_labels=None, group_names=None):
    """
    绘制 AUC 曲线，包括 Overall 和分组 AUC。

    Args:
        all_labels: 所有样本的真实标签 (0 或 1)。
        all_preds: 模型预测的正类概率。
        group_labels: 分组标签 (可选，例如性别标签)。
        group_names: 分组名称 (可选，例如 ["Female", "Male"])。
    """
    # 计算整体 AUC 和 ROC 曲线
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    overall_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"Overall AUC = {overall_auc:.4f}")

    # 如果有分组，分别计算每组的 AUC 和绘制曲线
    if group_labels is not None and group_names is not None:
        group_labels = np.array(group_labels)
        for i, group_name in enumerate(group_names):
            group_indices = group_labels == i
            if group_indices.any():
                fpr_group, tpr_group, _ = roc_curve(all_labels[group_indices], all_preds[group_indices])
                group_auc = auc(fpr_group, tpr_group)
                plt.plot(fpr_group, tpr_group, label=f"{group_name} AUC = {group_auc:.4f}")

    # 图表设置
    plt.title("ROC Curve with AUC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

def test_model_with_roc_and_accuracy(model, test_loader, device):
    """
    测试模型，计算 Overall AUC、Accuracy 和按性别分组的 AUC、Accuracy，同时绘制 ROC 曲线。

    Args:
        model: 训练好的 PyTorch 模型。
        test_loader: 测试数据的 DataLoader。
        device: 使用的设备（'cpu' 或 'cuda'）。

    Returns:
        dict: 包含 overall AUC、Accuracy，male 和 female 的 AUC 和 Accuracy 的结果。
    """
    model.eval()  # 设置为评估模式
    all_preds = []
    all_labels = []
    all_genders = []
    all_pred_classes = []  # 用于计算准确率

    with torch.no_grad():  # 禁用梯度计算
        for slo_fundus, male, dr_class in tqdm(test_loader, desc="Testing"):
            model.to(device)
            slo_fundus = slo_fundus.to(device)
            dr_class = dr_class.to(device)

            # 模型预测
            outputs = model(slo_fundus)
            probabilities = torch.softmax(outputs, dim=1)[:, 1]  # 获取正类概率
            predicted_classes = outputs.argmax(dim=1)  # 获取预测类别

            # 收集结果
            all_preds.extend(probabilities.cpu().numpy())
            all_labels.extend(dr_class.cpu().numpy())
            all_genders.extend(male.cpu().numpy())
            all_pred_classes.extend(predicted_classes.cpu().numpy())

    # 转换为 NumPy 数组
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_genders = np.array(all_genders)
    all_pred_classes = np.array(all_pred_classes)

    # 计算 Overall ROC 曲线和 AUC
    fpr, tpr, _ = roc_curve(all_labels, all_preds)
    overall_auc = auc(fpr, tpr)
    overall_accuracy = np.mean(all_pred_classes == all_labels)

    # 分别计算 Male 和 Female 的 ROC、AUC 和 Accuracy
    male_indices = all_genders == 1
    female_indices = all_genders == 0

    male_fpr, male_tpr, male_auc, male_accuracy = None, None, None, None
    female_fpr, female_tpr, female_auc, female_accuracy = None, None, None, None

    if male_indices.any():
        male_fpr, male_tpr, _ = roc_curve(all_labels[male_indices], all_preds[male_indices])
        male_auc = auc(male_fpr, male_tpr)
        male_accuracy = np.mean(all_pred_classes[male_indices] == all_labels[male_indices])

    if female_indices.any():
        female_fpr, female_tpr, _ = roc_curve(all_labels[female_indices], all_preds[female_indices])
        female_auc = auc(female_fpr, female_tpr)
        female_accuracy = np.mean(all_pred_classes[female_indices] == all_labels[female_indices])

    # 绘制 ROC 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"Overall AUC = {overall_auc:.4f}")
    if male_fpr is not None and male_tpr is not None:
        plt.plot(male_fpr, male_tpr, label=f"Male AUC = {male_auc:.4f}")
    if female_fpr is not None and female_tpr is not None:
        plt.plot(female_fpr, female_tpr, label=f"Female AUC = {female_auc:.4f}")
    plt.title("ROC Curve with AUC")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.grid()
    plt.show()

    # 输出结果
    results = {
        "Overall AUC": overall_auc,
        "Overall Accuracy": overall_accuracy,
        "Male AUC": male_auc,
        "Male Accuracy": male_accuracy,
        "Female AUC": female_auc,
        "Female Accuracy": female_accuracy
    }

    print("Evaluation Metrics:")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}" if value is not None else f"{metric}: Not available")

    return results

if __name__ == '__main__':
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    model = models.densenet121(pretrained=True)  # 使用 DenseNet-121 作为示例

    # 修改最后的分类层
    model.classifier = nn.Linear(model.classifier.in_features, 2)  # 修改为二分类
    # 加载保存的模型参数
    model.load_state_dict(torch.load("best_model_dense.pth", map_location=device))

    # train_loader, val_loader, test_loader = load_datasets('./dataset', batch_size=32)
    train_loader, val_loader, test_loader = load_datasets('/data/fupeiguo/ORID', batch_size=32)

    results = test_model_with_roc_and_accuracy(model, test_loader, device)
