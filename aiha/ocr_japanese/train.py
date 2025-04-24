import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision import transforms
from torchvision.models import resnet18

from utils.dataset_utils import LABELS, ImageLabelDataset, set_seed
from utils.training_utils import evaluate_model, predict_test, train_model

# ===== シード固定 =====
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ===== 前処理（Data Augmentation付き）=====
transform = transforms.Compose([
    transforms.Grayscale(),                     # グレースケール化
    transforms.Resize((128, 128)),              # サイズ統一
    transforms.RandomRotation(5),               # 小さな回転
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # 平行移動
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) # 正規化
])

# ===== データ読み込み =====
full_dataset = ImageLabelDataset("data/train/low", transform=transform, with_label=True)
labels = [full_dataset[i][1] for i in range(len(full_dataset))]

# ===== Stratified データ分割 =====
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ===== モデル定義（ResNet18） =====
model = resnet18(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # グレースケール対応
model.fc = nn.Linear(model.fc.in_features, len(LABELS))  # 出力層
model = model.to(device)

summary(model, input_size=(1, 128, 128))

# ===== クラス不均衡対策（重み付き損失関数） =====
from collections import Counter
label_counts = Counter(labels)
weights = [1.0 / label_counts[i] for i in range(len(LABELS))]
weights_tensor = torch.FloatTensor(weights).to(device)
criterion = nn.CrossEntropyLoss(weight=weights_tensor)

# ===== オプティマイザ & スケジューラ =====
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ===== 学習ループ =====
for epoch in range(22):
    print(f"\nEpoch {epoch + 1}")
    train_model(model, train_loader, criterion, optimizer, device)
    print("Validation...")
    evaluate_model(model, val_loader, device, criterion)
    scheduler.step()

# ===== モデル保存 =====
model_save_dir = "checkpoints"
os.makedirs(model_save_dir, exist_ok=True)
model_save_path = os.path.join(model_save_dir, "model_final.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Saved model to {model_save_path}")

# ===== 推論（テストデータ）=====
print("\nPredicting on test data...")
test_dataset = ImageLabelDataset("data/test/low", transform=transform, with_label=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
results = predict_test(model, test_loader, device)

# ===== 結果保存（submission.csv）=====
submission_df = pd.DataFrame(results, columns=["id", "label"])
submission_df["sort_key"] = submission_df["id"].str.extract(r"(\d+)").astype(int)
submission_df = submission_df.sort_values("sort_key").drop(columns="sort_key")

os.makedirs("pred", exist_ok=True)
submission_path = os.path.join("pred", "submission.csv")
submission_df.to_csv(submission_path, index=False, encoding="shift_jis")
print(f"Saved predictions to {submission_path}")
