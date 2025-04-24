# システム要件

Python	3.8 以上（ハッカソンで使用：3.10）

CUDA	NVIDIA GPU を使うなら CUDA 11.x

cuDNN	CUDA バージョンに対応した cuDNN

Git	git clone 用

Tesseract OCR	tesseract-ocr

Tesseract 日本語パック	tesseract-ocr-jpn

OpenCV（ヘッドレス）	libgl1-mesa-glx が必要な場合ある

# 環境構築

**Ubuntu 22.04 でのシステム依存ライブラリ**
```sudo apt update
sudo apt install -y \
    python3.10 python3.10-venv python3.10-distutils \
    build-essential git \
    tesseract-ocr tesseract-ocr-jpn \
    libgl1-mesa-glx \
    # GPU 環境なら
    cuda-toolkit-11-8 libcudnn8
```
**Python環境の構築**
プロジェクトルートで
```
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```
**必要 Python パッケージ**
```
pip install -r requirements.txt
```
ディレクトリ配置
```
your_project/
├─ main.py
├─ requirements.txt
├─ TPGSR/                 ← git clone した TPGSR コード一式
│   ├─ config/
│   └─ interfaces/
├─ utils/                 ← dataset_utils.py, training_utils.py
├─ data/
│   ├─ train/
│   │   ├─ high/
│   │   └─ low/
│   └─ test/ 
│       └─ low/
├─ 
```
**TPGSR clone**
```
git clone https://github.com/mjq11302010044/TPGSR.git
```
**TPGSR で必要な学習済みのエンジン**

Aster
```
git clone https://github.com/ayumiymk/aster.pytorch
```
MORAN
```
git clone https://github.com/Canjie-Luo/MORAN_v2
```
CRNN
```
git colne https://github.com/meijieru/crnn.pytorch
```
**環境変数**
・OpenAI API キー
```
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```
・Tesseract 実行パス（必要なら）
```
export TESSDATA_PREFIX=Tesseract_path
```
**GPU／AMP**

・GPU 環境で mixed-precision を使う場合、CUDA／cuDNN のインストールが必須。

・AMP（torch.cuda.amp）を利用するには PyTorch の GPU ビルドを利用してください。


# 使用技術
**1. ResNet152（ImageNet事前学習）**

役割：画像分類モデルの骨格。深い残差構造で多数の層を安定的に学習可能。

ポイント：ResNet152_Weights.IMAGENET1K_V2 を用いて ImageNet 上で事前学習された重みをロードし、最終出力層を本タスクのカテゴリ数に再定義。

メリット：少ないデータでも“転移学習”で高い精度が得やすい。

注意：1チャネル（白黒画像）入力に合わせて最初の Conv を置き換える必要あり。

**2. TPGSR（Text Prior Guided SR）**

役割：低解像度テキスト画像を、文字認識モデルの“事前知識（Text Prior）”でガイドしながら超解像。

ポイント：CRNN 等のテキスト認識器からの確率分布を用いて、文字形状を忠実に復元。

メリット：元画像の文字領域が鮮明になり、OCR精度が向上。

注意：リポジトリのコード・学習済みモデルをプロジェクト配下に正しく配置する必要がある。

**3. Albumentations + RandAugment（Data Augmentation）**

役割：学習用データに多彩な擬似的変換を加えることで、モデルの汎化性能を高める。

ポイント：OpenCVベースで高速に動作する Albumentations の RandAugment を中心に、ノイズ付加や明暗変化も併用。

メリット：回転・ノイズ・コントラスト変動などを学習し、未知の画像変化に強いモデルが得られる。

注意：過度すぎる強変換は逆効果。タスク特性に合わせて強度と頻度を調整。

**4. Label Smoothing（ラベル平滑化）**

役割：学習時の正解ラベルを完全な one-hot からわずかに分散させ、モデルの過信・過学習を抑制。

ポイント：PyTorch の CrossEntropyLoss(label_smoothing=ε) で実現（典型的には ε=0.1）。

メリット：正解との余裕を持たせることで、汎化性能が向上し、不確実性の扱いが改善。

注意：ε を大きくしすぎると学習が鈍くなるため注意。

**5. CosineAnnealingLR（学習率スケジューラ）**

役割：学習率を余弦カーブ状に減衰させ、最終的にゼロへ滑らかに近づける。

ポイント：T_max に総エポック数を指定し、各エポック終了後に scheduler.step()。

メリット：学習末期の微調整が安定し、「山なり」の学習率変化でより高精度に収束。

注意：周期を短くしすぎると再度上昇してしまうので、基本は一周期。

**6. Mixed Precision Training（AMP）**

役割：GPU 計算の一部を半精度（FP16）で行い、学習速度向上とメモリ使用量削減を両立。

ポイント：torch.cuda.amp.autocast() と torch.cuda.amp.GradScaler() を組み合わせる。

メリット：大きなバッチサイズや高解像度も扱いやすくなり、訓練時間短縮。

注意：極端な学習率や不安定なモデルでは勾配スケーリングが必要。

**7. Test-Time Augmentation（TTA）**

役割：推論時に画像を左右反転や上下反転など複数のバリエーションで評価し、予測を平均化して堅牢化。

ポイント：簡易的に Image.FLIP_LEFT_RIGHT / TOP_BOTTOM を掛けた３種 → 平均確率で最終予測。

メリット：ノイズや視点変化に対して頑健な予測が得られ、数％精度向上も狙える。

注意：推論コストは TTA 回数分増加する。

**8. OCR: EasyOCR + pytesseract フォールバック**

役割：EasyOCR（CNN+CTC+BeamSearch）でまずテキスト抽出し、結果が空なら Tesseract を使う。

ポイント：reader.readtext() で日本語対応、失敗時には pytesseract.image_to_string() で再試行。

メリット：NN ベースと伝統エンジンの両者を活用することで、抽出率と耐障害性を両立。

注意：EasyOCR は初回ロードが重い。Tesseract は日本語パックを別途インストール要。

**9. GPT-4o（cpt-4o）によるテキスト補正**

役割：OCR結果を「カテゴリ名のうちどれか」に補正するために LLM を活用。

ポイント：openai.ChatCompletion.create(model='cpt-4o', messages=…) を用い、システム＋ユーザープロンプトで補正指示。

メリット：OCRの誤認識を文脈的に修正でき、最終ラベルの精度向上に寄与。

注意：API 呼び出しコストとレイテンシ、誤補正リスクに注意。失敗時はCNN予測にフォールバック。

### 画像の前処理

**1. グレースケール化 (Grayscale)**

目的：色情報（RGB）は文字認識には不要。計算量削減とコントラスト強調の前提としてまず輝度情報のみに変換。

```　gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)```

**2. ノイズ除去 (Denoising)**

目的：JPEG圧縮ノイズや撮影時の点状ノイズを除去し、文字輪郭をクリアにする。

メディアンブラー：塩胡椒ノイズに効果的
```
denoised = cv2.medianBlur(gray, ksize=3)
```
ガウシアンブラー：滑らかなノイズ除去
```
denoised = cv2.GaussianBlur(gray, (5,5), sigmaX=0)
```
バイラテラルフィルタ：エッジを保護しながら平滑化
```
denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
```



**3. コントラスト強調 (Contrast Enhancement)**
目的：背景と文字の明暗差をはっきりさせ、文字領域を浮かび上がらせる。

ヒストグラム平坦化：全体的なコントラストを均一化
```
eq = cv2.equalizeHist(denoised)
```
CLAHE（局所的コントラスト強調）：局所ごとに微調整
```
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(denoised)
```

**4. 二値化 (Thresholding)**

目的：文字（前景）と背景を明確に分離し、OCRエンジンや輪郭検出の入力に最適化。
```
Otsu
_, th1 = cv2.threshold(clahe_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
```
大津の二値化（Otsu’s）：一括でしきい値を自動計算

適応的二値化（Adaptive）：照明ムラがある画像に強い
```
Adaptive
th2 = cv2.adaptiveThreshold(
    clahe_img, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    blockSize=11,
    C=2
)
```
**5. 形態学的処理 (Morphological Operations)**
目的：文字の連結を補助したり、小さな穴・ノイズを除去したり、形状を整える。

開処理 (Opening)：小さなノイズの除去
```
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
```
閉処理 (Closing)：小さな穴埋め
```
opened  = cv2.morphologyEx(th2, cv2.MORPH_OPEN,  kernel)
```

膨張 (Dilation)／収縮 (Erosion)：文字線の太さ調整
```
closed  = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
```

**6. 傾き補正 (Deskew)**

目的：撮影時に文字行が傾いているとOCR精度が落ちるため、文字行の角度を推定して水平回転。

輪郭座標から最小外接矩形を計算し、その角度で回転
```
coords = np.column_stack(np.where(closed > 0))
angle  = cv2.minAreaRect(coords)[-1]
if angle < -45: angle = -(90 + angle)
else:            angle = -angle
(h, w) = closed.shape
M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
deskew = cv2.warpAffine(closed, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
```
**7. シャープ化 (Sharpening)**
目的：文字のエッジを強調し、細部をくっきりさせてOCRの誤認識を減らす。

アンシャープマスク：元画像とぼかし画像の差分を加算
```
gauss    = cv2.GaussianBlur(deskew, (0,0), sigmaX=3)
sharpen  = cv2.addWeighted(deskew, 1.5, gauss, -0.5, 0)
```

**8. ROI（領域）切り出し**

目的：画面全体ではなく文字が含まれる領域だけをOCRに通す。

輪郭検出で大きさフィルタをかけ、文字ブロックを抽出

各領域ごとにOCR実行
```
contours, _ = cv2.findContours(sharpen, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rois = []
for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    if w*h > 1000:  # 最小面積閾値
        roi = sharpen[y:y+h, x:x+w]
        rois.append(roi)
```


低解像度画像 → TPGSR で超解像

学習時：Augmentation → ResNet152 (FP16+Cosine) → Label Smoothing

推論時：TTA → CNN予測 + OCR → EasyOCR⇔Tesseract → GPT補正 → フォールバック

出力：id,label の提出用CSV


# 参考文献

https://github.com/xinntao/Real-ESRGAN

https://github.com/mjq11302010044/TPGSR

https://github.com/ayumiymk/aster.pytorch

https://github.com/meijieru/crnn.pytorch

https://note.com/rosso_blog/n/nd5dc2eea3134#b361318e-fd9b-4f31-bedb-2c03a3cc22d0

https://qiita.com/ryuus/items/b6b8a34bf0ea11c3b378

https://pytorch.org/vision/main/models/generated/torchvision.models.resnet152.html

https://github.com/tesseract-ocr/tesseract

https://qiita.com/spc_ehara/items/e425b6dcc0398299c40d

https://github.com/JaidedAI/EasyOCR

https://qiita.com/cfiken/items/7cbf63357c7374f43372

https://qiita.com/bowdbeg/items/71c62cf8ef891d164ecd

https://qiita.com/kma-jp/items/81db6d5c549e50707e30

https://qiita.com/wing_man/items/a1d5ab1bba7d763d9369

https://qiita.com/kitfactory/items/d89457eeab5c185880be

https://qiita.com/takoroy/items/e2f1ee627311be5d879d

https://qiita.com/Takayoshi_Makabe/items/79c8a5ba692aa94043f7
