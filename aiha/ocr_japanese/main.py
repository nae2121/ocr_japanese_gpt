""""
- ResNet152 (ImageNet事前学習) を用いた画像分類
- TPGSRによる超解像 (低解像度画像前処理)
- DataAugment強化 (albumentations, RandAugment)
- FocalLoss/LabelSmoothing に対応
- Mixed Precision Training (torch.cuda.amp)
- CosineAnnealingLR スケジューラ
- Test-Time Augmentation (TTA)
- OCR: EasyOCR 併用 + pytesseract フォールバック
- GPT-4o によるテキスト補正 (cpt-4o)
"""
import os
import sys
import random
import yaml
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.models import resnet152, ResNet152_Weights
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import pytesseract
import easyocr
import openai
import pandas as pd
from tqdm import tqdm
from types import SimpleNamespace
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ----- シード固定 -----
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)

# ----- デバイス -----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# ----- OpenAI 設定 -----
openai.api_key = os.getenv('OPENAI_API_KEY')
GPT_MODEL = 'gpt-4o'

# ----- OCRエンジン -----
# EasyOCR reader (日本語対応)
reader = easyocr.Reader(['ja'], gpu=torch.cuda.is_available())

# ----- TPGSR 超解像統合 -----
try:
    sys.path.append('TPGSR')
    from interfaces.super_resolution import TextSR
    cfg = SimpleNamespace(**yaml.safe_load(open('TPGSR/config/super_resolution.yaml')))
    args = SimpleNamespace(arch='tsrn_tl_cascade', test_model='CRNN', STN=False,
                           mask=False, gradient=False, sr_share=False,
                           tpg_share=False, use_label=False,
                           use_distill=False, ssim_loss=False)
    opt = SimpleNamespace(Transformation='None', FeatureExtraction='ResNet',
                          SequenceModeling='None', Prediction='CTC',
                          saved_model='None-ResNet-None-CTC.pth',
                          character='-0123456789abcdefghijklmnopqrstuvwxyz')
    opt.num_class = len(opt.character)
    tpgsr_sys = TextSR(cfg, args, opt)
    tpgsr_sys.model.to(device).eval()
except Exception as e:
    print('Warning: TPGSR load failed.', e)
    tpgsr_sys = None

# 超解像関数
def apply_superres(img: Image.Image) -> Image.Image:
    if tpgsr_sys is None:
        return img.resize((img.width*2, img.height*2), Image.BICUBIC)
    lr = img.convert('L')
    lr_t = transforms.ToTensor()(lr).unsqueeze(0).to(device)
    with torch.no_grad():
        sr_t = tpgsr_sys.model(lr_t)
    sr = sr_t.squeeze(0).cpu()
    try:
        return transforms.ToPILImage()(sr).convert('RGB')
    except:
        return transforms.ToPILImage()(sr)

# ----- データとラベル -----
from utils.dataset_utils import LABELS
classes = LABELS  # 10カテゴリ
num_classes = len(classes)

# --- 画像前処理 ---
def preprocess_for_ocr(pil_img: Image.Image) -> np.ndarray:
    # PIL→OpenCV gray
    img = np.array(pil_img.convert('L'))
    # ノイズ除去
    img = cv2.medianBlur(img, 3)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    # 二値化（適応）
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,11,2)
    # 形態学開閉
    k = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    img = cv2.morphologyEx(img,cv2.MORPH_OPEN,k)
    img = cv2.morphologyEx(img,cv2.MORPH_CLOSE,k)
    # 傾き補正
    coords = np.column_stack(np.where(img>0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45: angle = -(90+angle)
    else: angle = -angle
    (h,w)=img.shape
    M=cv2.getRotationMatrix2D((w/2,h/2),angle,1)
    img=cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    # シャープ化
    gauss = cv2.GaussianBlur(img,(0,0),3)
    img = cv2.addWeighted(img,1.5,gauss,-0.5,0)
    return img
# ----- モデル構築 -----
model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# ----- データ拡張 (albumentations + RandAugment) -----
train_aug = A.Compose([
    A.Resize(224, 224),
    A.RandAugment(),
    A.GaussNoise(var_limit=(10,50), p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.PadIfNeeded(min_height=224, min_width=224),
    ToTensorV2()
])

# ----- カスタムデータセット -----
class OCRDataset(Dataset):
    def __init__(self, root, quality, transform=None):
        self.paths = []
        self.labels = []
        self.transform = transform
        for fname in os.listdir(os.path.join(root, quality)):
            if not fname.lower().endswith(('.jpg','.png','.jpeg')): continue
            cls = fname.split('_')[0]
            if cls in classes:
                self.paths.append(os.path.join(root, quality, fname))
                self.labels.append(classes.index(cls))
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if 'low' in self.paths[idx]:
            img = apply_superres(img)
        img = np.array(img.convert('L')) if self.transform is None else np.array(img)
        if self.transform:
            aug = self.transform(image=img)
            img = aug['image']
        else:
            img = transforms.ToTensor()(img)
        return img, self.labels[idx]

# データローダー作成
data_root = 'data/train'
train_ds = ConcatDataset([
    OCRDataset(data_root, 'high', transform=train_aug),
    OCRDataset(data_root, 'low',  transform=train_aug)
])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)

# ----- 損失関数とオプティマイザ -----
# LabelSmoothing
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
# Scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
# Mixed precision
scaler = GradScaler()

# ----- 学習ループ -----
epochs = 12
print('Training...')
for epoch in range(1, epochs+1):
    model.train()
    running_loss = 0.0
    for imgs, lbls in tqdm(train_loader, desc=f'Train {epoch}/{epochs}'):
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        with autocast(): out = model(imgs); loss = criterion(out, lbls)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        running_loss += loss.item()
    scheduler.step()
    print(f'Epoch {epoch} loss: {running_loss/len(train_loader):.4f}')
# モデル保存
os.makedirs('checkpoints', exist_ok=True)
torch.save(model.state_dict(), 'checkpoints/model_final.pth')

# ----- 推論 (TTA + OCR + GPT) -----
test_dir = 'data/test/low'
files = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg','.png'))],
               key=lambda x: int(os.path.splitext(x)[0]))
# TTA transforms (flips)
tta_transforms = [lambda x: x,
                  lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
                  lambda x: x.transpose(Image.FLIP_TOP_BOTTOM)]

results = []
model.eval()
for fname in tqdm(files, desc='Inference'):
    path = os.path.join(test_dir, fname)
    img0 = Image.open(path).convert('RGB')
    img_sr = apply_superres(img0)
    # CNN予測 with TTA
    preds = []
    for tfn in tta_transforms:
        img_t = tfn(img_sr).convert('L')
        tensor = transforms.ToTensor()(img_t).unsqueeze(0).to(device)
        with torch.no_grad(): preds.append(model(tensor).softmax(1))
    pred = torch.stack(preds).mean(0).argmax(1).item()
    cnn_lbl = classes[pred]
    # OCR前処理
    pre=cv2.cvtColor(preprocess_for_ocr(img_sr),cv2.COLOR_GRAY2RGB)
    # OCR: EasyOCR
    ocr_res = reader.readtext(np.array(img_sr), detail=0)
    ocr_txt = ' '.join(ocr_res) if ocr_res else pytesseract.image_to_string(img_sr, lang='jpn').strip()
    # GPT補正
    def refine(ocr_txt):
        if not ocr_txt: return ''
        msgs = [
            {'role':'system','content':'OCR結果の日本語テキストを正しいカテゴリ名に補正してください。'},
            {'role':'user','content':f"OCR: '{ocr_txt}'. カテゴリ: {', '.join(classes)}。"}
        ]
        try:
            rsp = openai.ChatCompletion.create(model=GPT_MODEL, messages=msgs, temperature=0)
            return rsp.choices[0].message.content.strip()
        except Exception as e:
            print('GPT error', e)
            return ''
    gpt_lbl = refine(ocr_txt)
    final_lbl = gpt_lbl if gpt_lbl else cnn_lbl
    results.append({'id':int(os.path.splitext(fname)[0]), 'label':final_lbl})
# 保存
df = pd.DataFrame(results).sort_values('id').reset_index(drop=True)
os.makedirs('outputs', exist_ok=True)
df.to_csv('outputs/results.csv', index=False, encoding='shift-jis')
print('Saved outputs/results.csv')
