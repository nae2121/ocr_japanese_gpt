# === OCR + CNN + GPT-4o マルチモーダル完全統合コード ===
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Subset
from torchsummary import summary
from torchvision import transforms
from torchvision.models import resnet18

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pytesseract
import pandas as pd
from openai import OpenAI
from sklearn.model_selection import train_test_split

from utils.dataset_utils import LABELS, ImageLabelDataset, set_seed
from utils.training_utils import evaluate_model,predict_test,train_model


# === ラベル定義 ===
LABELS = ['玄関', 'ホール', '洋室', '廊下', 'クローゼット', '和室', '浴室', '収納', 'バルコニー', 'トイレ']
label_to_idx = {label: idx for idx, label in enumerate(LABELS)}

# === OpenAI クライアント初期化 ===
client = OpenAI(api_key="Your-OpenAI-API-Key", model="gpt-4o")
# === シード固定 ===
def set_seed(seed=42):  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# === データセットクラス ===
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# === 前処理定義 ===
transform = transforms.Compose([
    transforms.Grayscale(),                     # グレースケール化
    transforms.Resize((128, 128)),              # サイズ統一
    transforms.RandomRotation(5),               # 小さな回転
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),  # 平行移動
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]) # 正規化
])

# === 学習用データ準備 ===
def prepare_data():
    image_paths, labels = [], []
    for fname in os.listdir("data/train/low"):
        if fname.endswith(".jpg"):
            label_name = fname.split("_")[0]
            if label_name in label_to_idx:
                image_paths.append(os.path.join("data/train/low", fname))
                labels.append(label_to_idx[label_name])

    X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
    return ImageDataset(X_train, y_train, transform), ImageDataset(X_val, y_val, transform)

full_dataset = ImageLabelDataset("data/train/low", transform=transform, with_label=True)
labels = [full_dataset[i][1] for i in range(len(full_dataset))]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))
train_dataset = Subset(full_dataset, train_idx)
val_dataset = Subset(full_dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# === モデル構築 ===
def build_model():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    return model

# === モデル学習 ===
def train_model(model, train_loader, val_loader, device, epochs=12):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 検証
        model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f"Epoch {epoch+1}, Accuracy: {correct / total * 100:.2f}%")

    torch.save(model.state_dict(), "cnn_model.pth")
    return model

# === OCR + GPT-4oによるラベル補正 ===
def correct_with_gpt(image_path):
    print(f"OCR結果 [{image_path}]: '{ocr_result.strip()}'")
    if not ocr_result.strip():
        print(f"⚠️ OCR失敗: {image_path}")
        return "不明"
    ocr_result = pytesseract.image_to_string(Image.open(image_path), lang="jpn")
    prompt = f"""
    以下はOCRで読み取った日本語のテキストです：\n"{ocr_result.strip()}"
    次のカテゴリのいずれかと最も一致するものを1つだけ答えてください：{', '.join(LABELS)}。
    出力はカテゴリ名だけにしてください。
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("GPTエラー:", e)
        return "不明"

# === テスト画像に対して推論＋補正 ===
def predict_and_save(model, device):
    model.eval()
    test_dir = "data/test/test/low"
    results = []

    with torch.no_grad():
        for fname in sorted(os.listdir(test_dir)):
            if fname.lower().endswith(".jpg"):
                path = os.path.join(test_dir, fname)
                image_tensor = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
                output = model(image_tensor)
                pred_label = LABELS[output.argmax().item()]
                llm_label = correct_with_gpt(path)
                results.append({"id": fname, "label": llm_label})
                print(f"ファイル名: {fname}, CNN予測: {pred_label}, LLM補正: {llm_label}")
    # 結果をCSVに保存
    if len(results) == 0:
        print("⚠️ 推論結果が空です。画像がないか、すべての処理に失敗している可能性があります。")
    else:
        df = pd.DataFrame(results, columns=["id", "label"])
        df["sort_key"] = df["id"].str.extract(r"(\d+)")
        df = df.dropna(subset=["sort_key"])
        df["sort_key"] = df["sort_key"].astype(int)
        df = df.sort_values("sort_key").drop(columns="sort_key")
        os.makedirs("outputs", exist_ok=True)
        df.to_csv("outputs/results.csv", index=False, encoding="shift-jis")
        print("✅ 完了：outputs/results.csv に保存")

# === 実行 ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = prepare_data()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    model = build_model()
    model = train_model(model, train_loader, val_loader, device)
    predict_and_save(model, device)