import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from PIL import Image
import pytesseract
import openai
import pandas as pd
from tqdm import tqdm

# ユーティリティ関数：再現性のため乱数シードを設定
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 再現性確保のための設定（必要に応じて）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # シード固定

# デバイスの設定（GPUが利用可能ならGPU、なければCPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ResNet18をベースとした分類モデルの構築
num_classes = None  # 後で訓練データからクラス数を決定
model = torchvision.models.resnet18(pretrained=False)  # 学習済みでないResNet18
# 最終全結合層の出力ユニット数を後でnum_classesに合わせて変更

# GitHubのTPGSRコードを組み込む（TPGSRディレクトリがプロジェクト内にある想定）
# 例えば、TPGSRのクラスやモデルをインポート
try:
    import sys
    sys.path.append('TPGSR')  # TPGSRレポジトリのパスを追加
    from interfaces.super_resolution import TextSR
    import yaml
    from easydict import EasyDict

    # TPGSRモデルの設定を読み込み（super_resolution.yamlを使用）
    config_path = os.path.join('TPGSR', 'config', 'super_resolution.yaml')
    config = EasyDict(yaml.safe_load(open(config_path, 'r')))
    # 必要な引数を設定（アーキテクチャや事前学習済み認識器の種類など）
    args = EasyDict({
        "arch": "tsrn_tl_cascade",       # 超解像モデル+テキスト誘導(cascade)
        "test_model": "CRNN",            # テキスト認識器にCRNNを使用
        "STN": False, "mask": False, "gradient": False, 
        "sr_share": False, "tpg_share": False, 
        "use_label": False, "use_distill": False, "ssim_loss": False
    })
    # テキスト認識器（CRNN）のオプション設定（文字セットなど）
    opt_TPG = EasyDict({
        "Transformation": "None",
        "FeatureExtraction": "ResNet",
        "SequenceModeling": "None",
        "Prediction": "CTC",
        "saved_model": "None-ResNet-None-CTC.pth",  # CRNNの学習済みモデルパス
        "character": "-0123456789abcdefghijklmnopqrstuvwxyz"
    })
    opt_TPG.num_class = len(opt_TPG.character)
    # TextSRクラスのインスタンス化（TPGSRモデルを構築）
    tpgsr_system = TextSR(config, args, opt_TPG)
    # 学習済みモデルの読み込み（別途学習済みパラメータが存在する場合）
    # 例えば: tpgsr_system.load_model('TPGSR/checkpoints/tsrn_tl_cascade_best.pth')
    tpgsr_system.model = tpgsr_system.model.to(device)
    tpgsr_system.model.eval()
except Exception as e:
    print("Warning: TPGSR model integration failed. Make sure the TPGSR repository and weights are available.", e)
    tpgsr_system = None

# 低解像度画像にTPGSRで超解像を適用する関数
def apply_tpgsr(lr_image: Image.Image) -> Image.Image:
    """低解像度のPIL画像をTPGSRモデルで超解像し、高解像度PIL画像を返す"""
    if tpgsr_system is None:
        # 万一TPGSRモデルが利用できない場合はバイキュービック補間で代用
        return lr_image.resize((lr_image.width * 2, lr_image.height * 2), Image.BICUBIC)
    # 前処理：グレースケール化（TPGSRは1チャネル想定）とテンソル変換
    lr_gray = lr_image.convert("L")
    lr_tensor = transforms.ToTensor()(lr_gray).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_tensor = tpgsr_system.model(lr_tensor)  # 超解像モデルの推論
    # 出力テンソルを画像に変換（1チャネル→Lモードに変換後、RGBに統合）
    sr_tensor = sr_tensor.squeeze(0).cpu()
    # TPGSR出力が1チャネルの場合
    try:
        sr_img = transforms.ToPILImage()(sr_tensor).convert("RGB")
    except:
        sr_img = transforms.ToPILImage()(sr_tensor)
    return sr_img

# データセットの準備（学習用）
train_low_dir = 'data/train/low'
train_high_dir = 'data/train/high'
classes = []  # クラス名リスト（文字列ラベル）
class_to_idx = {}  # クラス名→インデックス
# クラス名を低解像度・高解像度ディレクトリから取得し統合
if os.path.isdir(train_high_dir):
    classes.extend(name for name in os.listdir(train_high_dir) if os.path.isdir(os.path.join(train_high_dir, name)))
if os.path.isdir(train_low_dir):
    classes.extend(name for name in os.listdir(train_low_dir) if os.path.isdir(os.path.join(train_low_dir, name)))
classes = sorted(set(classes))
class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
num_classes = len(classes)
# ResNet18モデルの全結合層をクラス数に合わせて再定義
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# 画像変換の定義（学習時）。サイズを統一し正規化
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNetの入力サイズにリサイズ
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 学習データのカスタムデータセット構築（低解像度はTPGSR適用）
from torch.utils.data import Dataset, DataLoader, ConcatDataset

class TrainImageDataset(Dataset):
    def __init__(self, base_dir: str, quality: str, transform=None):
        """
        base_dir内の画像とクラスを読み込み。
        qualityが'low'の場合はTPGSRで超解像してから返す。
        """
        self.filepaths = []
        self.labels = []
        self.transform = transform
        self.quality = quality
        # ディレクトリ構造: base_dir/<class_name>/<image_files>
        for cls_name in os.listdir(base_dir):
            cls_path = os.path.join(base_dir, cls_name)
            if not os.path.isdir(cls_path) or cls_name not in class_to_idx:
                continue
            label_idx = class_to_idx[cls_name]
            for fname in os.listdir(cls_path):
                fpath = os.path.join(cls_path, fname)
                if os.path.isfile(fpath):
                    self.filepaths.append(fpath)
                    self.labels.append(label_idx)
    def __len__(self):
        return len(self.filepaths)
    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]
        # 画像読み込み
        img = Image.open(img_path)
        # モノクロやRGBAの場合はRGBに変換
        img = img.convert("RGB")
        # 品質が低の場合はTPGSRで高解像度化
        if self.quality == 'low':
            img = apply_tpgsr(img)
        # 変換適用
        if self.transform:
            img = self.transform(img)
        return img, label

# 低解像度データセットと高解像度データセットを作成し結合
train_dataset_low = TrainImageDataset(train_low_dir, quality='low', transform=transform_train)
train_dataset_high = TrainImageDataset(train_high_dir, quality='high', transform=transform_train)
train_dataset = ConcatDataset([train_dataset_high, train_dataset_low])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)

# 損失関数とオプティマイザの設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 学習ループ
num_epochs = 10
for epoch in range(1, num_epochs+1):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch} - Training loss: {avg_loss:.4f}")

# モデルの保存
os.makedirs('checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/model_final.pth')

# テストデータに対する推論
test_low_dir = 'data/test/low'
test_files = []
# data/test/low配下の画像ファイル一覧取得
for fname in os.listdir(test_low_dir):
    fpath = os.path.join(test_low_dir, fname)
    if os.path.isfile(fpath):
        test_files.append(fpath)
# ID順にソート（ファイル名から数字部分を抽出してソート）
test_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

# 推論結果を蓄積するリスト
results = []
model.eval()  # モデルを推論モードに
for fpath in tqdm(test_files, desc="Testing"):
    img_id = os.path.splitext(os.path.basename(fpath))[0]  # 拡張子を除いたID
    # 画像読み込みとTPGSR超解像
    img = Image.open(fpath).convert("RGB")
    img_sr = apply_tpgsr(img)
    # CNNモデルでラベル予測
    img_tensor = transform_train(img_sr).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, pred_idx = torch.max(output, 1)
    cnn_label = classes[pred_idx.item()] if num_classes is not None else ''
    # OCRでテキスト抽出
    ocr_text = pytesseract.image_to_string(img_sr, lang='eng', config='--psm 6').strip()
    # GPT-4 APIを使ってテキスト補正
    gpt_label = ""
    try:
        # OpenAI API呼び出し（GPT-4）
        openai.api_key = os.getenv('OPENAI_API_KEY')  # 環境変数からAPIキー取得（適宜設定）
        system_msg = {"role": "system", "content": "あなたはOCR結果を正確なテキストに修正するアシスタントです。"}
        user_msg = {"role": "user", "content": f"OCRの結果: '{ocr_text}'。これは誤認識や欠損があるかもしれません。正しいラベルを出力してください。わからない場合は不明とだけ答えてください。"}
        response = openai.ChatCompletion.create(model="gpt-4o", messages=[system_msg, user_msg])
        gpt_label = response["choices"][0]["message"]["content"].strip()
    except Exception as e:
        gpt_label = ""
        print(f"GPT補正エラー (ID={img_id}):", e)
    # 最終ラベルの決定
    final_label = gpt_label if gpt_label and gpt_label != "不明" else cnn_label
    results.append({
        "id": int(img_id) if img_id.isdigit() else img_id,
        "cnn_label": cnn_label,
        "ocr_text": ocr_text,
        "gpt_label": gpt_label,
        "final_label": final_label
    })

# 結果をデータフレームにしてCSV保存（idで昇順ソート）
df = pd.DataFrame(results)
# id列が数値の場合はソートのためint型に
try:
    df['id'] = df['id'].astype(int)
except:
    pass
df = df.sort_values('id').reset_index(drop=True)
os.makedirs('outputs', exist_ok=True)
df.to_csv('outputs/results.csv', index=False, columns=["id", "cnn_label", "ocr_text", "gpt_label", "final_label"])
print("Saved results to outputs/results.csv")
