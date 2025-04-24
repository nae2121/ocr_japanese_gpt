import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pytesseract
import openai
import pandas as pd
from sklearn.model_selection import train_test_split

# === OpenAI API設定 ===
openai.api_key = "Your-OpenAI-API-Key"

# === ラベル定義 ===
LABELS = ['玄関', 'ホール', '洋室', '廊下', 'クローゼット', '和室', '浴室', '収納', 'バルコニー', 'トイレ']
label_to_idx = {label: idx for idx, label in enumerate(LABELS)}

# === カスタムデータセットクラス ===
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        return self.transform(img), label

# === 前処理 ===
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# === データ準備 ===
def prepare_data():
    train_dir = "data/train/low"
    image_paths, labels = [], []

    for fname in os.listdir(train_dir):
        if fname.lower().endswith(".jpg"):
            label_name = fname.split("_")[0]
            if label_name in label_to_idx:
                image_paths.append(os.path.join(train_dir, fname))
                labels.append(label_to_idx[label_name])

    X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=0.2, random_state=42)
    train_ds = ImageDataset(X_train, y_train, transform)
    val_ds = ImageDataset(X_val, y_val, transform)
    return train_ds, val_ds

# === モデル定義 ===
def build_model():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, len(LABELS))
    return model

# === 学習 ===
def train_model(model, train_loader, val_loader, device, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 検証
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        print(f"Epoch {epoch+1}/{epochs}, Validation Accuracy: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), "cnn_model.pth")
    return model

# === OCR+LLM補正 ===
def ocr_llm_correct(image_path):
    ocr_text = pytesseract.image_to_string(Image.open(image_path), lang="jpn")
    prompt = f"""
    OCRの結果は「{ocr_text.strip()}」です。
    次のカテゴリのうちどれが最も近いですか？必ず1つだけ選んでください。
    カテゴリ：{', '.join(LABELS)}
    わからない場合はOCRの結果を選んでください
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    return response['choices'][0]['message']['content'].strip()

# === 推論と結果保存 ===

def predict_and_save(model, device):
    test_dir = "data/test"
    model.eval()
    results = []
    with torch.no_grad():
        for fname in sorted(os.listdir(test_dir)):
            if fname.lower().endswith(".jpg"):
                img_path = os.path.join(test_dir, fname)
                img_tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                output = model(img_tensor)
                pred_label = LABELS[output.argmax().item()]
                llm_label = ocr_llm_correct(img_path)

                # 提出ファイルにあわせ pred_label（CNN）またはllm_label（OCR+LLM）を使う
                final_label = llm_label  # LLM補正後のラベルを利用する場合はこちら
                # final_label = pred_label  # CNNの予測を利用する場合はこちら
                
                results.append({"filename": fname, "pred": final_label})

    df = pd.DataFrame(results)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/results.csv", index=False, encoding="utf-8-sig")
    print("結果がoutputs/results.csvに保存されました。")

# === 実行 ===
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds, val_ds = prepare_data()
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    model = build_model()
    trained_model = train_model(model, train_loader, val_loader, device)
    predict_and_save(trained_model, device)
