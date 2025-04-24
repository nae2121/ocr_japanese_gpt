# === 統合システム main.py (ResNet152 + TPGSR超解像 + OCR + GPT補正) ===
import os
import random
import yaml
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
from types import SimpleNamespace

# 再現性のためのシード固定
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# デバイス設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# OpenAI APIキー設定 (環境変数 OPENAI_API_KEY が必要)
openai.api_key = os.getenv('OPENAI_API_KEY')

# TPGSR超解像モデルの統合
try:
    import sys
    sys.path.append('TPGSR')
    from interfaces.super_resolution import TextSR

    # 設定読み込み
    cfg_dict = yaml.safe_load(open('TPGSR/config/super_resolution.yaml', 'r'))
    cfg = SimpleNamespace(**cfg_dict)
    args = SimpleNamespace(
        arch='tsrn_tl_cascade', test_model='CRNN',
        STN=False, mask=False, gradient=False,
        sr_share=False, tpg_share=False,
        use_label=False, use_distill=False, ssim_loss=False
    )
    opt_dict = {
        'Transformation': 'None', 'FeatureExtraction': 'ResNet',
        'SequenceModeling': 'None', 'Prediction': 'CTC',
        'saved_model': 'None-ResNet-None-CTC.pth',
        'character': '-0123456789abcdefghijklmnopqrstuvwxyz'
    }
    opt = SimpleNamespace(**opt_dict)
    opt.num_class = len(opt.character)

    # TPGSRモデル構築
    tpgsr_system = TextSR(cfg, args, opt)
    tpgsr_system.model.to(device).eval()
except Exception as e:
    print('Warning: TPGSR integration failed.', e)
    tpgsr_system = None

# 超解像関数
def apply_tpgsr(img: Image.Image) -> Image.Image:
    if tpgsr_system is None:
        # バイキュービック補間
        return img.resize((img.width*2, img.height*2), Image.BICUBIC)
    lr = img.convert('L')
    lr_t = transforms.ToTensor()(lr).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_t = tpgsr_system.model(lr_t)
    sr = sr_t.squeeze(0).cpu()
    try:
        return transforms.ToPILImage()(sr).convert('RGB')
    except:
        return transforms.ToPILImage()(sr)

# クラス名一覧
from utils.dataset_utils import LABELS
classes = LABELS  # ['玄関','ホール','洋室',...]
num_classes = len(classes)

# モデル定義: ResNet152 + 単一チャネル入力対応
model = torchvision.models.resnet152(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

# 学習データ前処理 (白黒画像特化)
transform_train = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# データセットの構築
data_dir = 'data/train'
from torch.utils.data import Dataset, DataLoader, ConcatDataset
class SRTrainDataset(Dataset):
    def __init__(self, base_dir, quality, transform):
        self.files, self.labels = [], []
        self.transform = transform
        for fname in os.listdir(os.path.join(base_dir, quality)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            cls = fname.split('_')[0]
            if cls in classes:
                self.files.append(os.path.join(base_dir, quality, fname))
                self.labels.append(classes.index(cls))
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert('RGB')
        if 'low' in self.files[idx]:
            img = apply_tpgsr(img)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

train_ds = ConcatDataset([
    SRTrainDataset(data_dir, 'high', transform_train),
    SRTrainDataset(data_dir, 'low', transform_train)
])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)

# 損失関数とオプティマイザ
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 学習ループ
epochs = 10
for ep in range(1, epochs+1):
    model.train()
    total_loss = 0.0
    for imgs, lbls in tqdm(train_loader, desc=f'Training {ep}/{epochs}'):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {ep}: Loss={total_loss/len(train_loader):.4f}')

# モデル保存
os.makedirs('checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/model_final.pth')

# テストデータの準備
test_dir = 'data/test/low'
files = sorted(
    [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))],
    key=lambda x: int(os.path.splitext(x)[0])
)

# 推論 + OCR + GPT補正
def refine_with_gpt(ocr_txt: str) -> str:
    if not ocr_txt.strip():
        return ''
    messages = [
        {"role": "system", "content": "OCR結果を正確なカテゴリ名に補正するアシスタントです。"},
        {"role": "user", "content": f"OCR: '{ocr_txt}'. カテゴリ: {', '.join(classes)}。"}
    ]
    try:
        resp = openai.ChatCompletion.create(model='gpt-4o', messages=messages)
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print('GPT error:', e)
        return ''

results = []
model.eval()
for fname in tqdm(files, desc='Inference'):
    path = os.path.join(test_dir, fname)
    img = Image.open(path).convert('RGB')
    img_sr = apply_tpgsr(img)
    inp = transform_train(img_sr).unsqueeze(0).to(device)
    with torch.no_grad(): pred = model(inp).argmax(1).item()
    cnn_lbl = classes[pred]

    ocr_txt = pytesseract.image_to_string(img_sr, lang='jpn').strip()
    gpt_lbl = refine_with_gpt(ocr_txt)
    final_lbl = gpt_lbl if gpt_lbl else cnn_lbl
    results.append({
        'id': int(os.path.splitext(fname)[0]),
        'label': final_lbl
    })

# 提出用CSV出力
df = pd.DataFrame(results)
df = df.sort_values('id').reset_index(drop=True)
os.makedirs('outputs', exist_ok=True)
df.to_csv('outputs/results.csv', index=False, encoding='shift-jis')
print('Saved outputs/results.csv')