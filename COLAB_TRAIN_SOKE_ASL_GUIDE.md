# Huong Dan Train SOKE ASL Tren Google Colab Theo Huong Hybrid

Tai lieu nay dung cho luong train tren Colab:

```text
Code: clone tu GitHub vao /content/SOKE
Archive goc: luu tren Google Drive
Data/assets khi train: copy archive ve /content, giai nen vao /content/SOKE_COLAB_DATA
Checkpoint working: luu local /content de save nhanh
Checkpoint backup/results/token cache: sync/luu tren Google Drive de resume duoc
```

Ly do dung hybrid:

```text
Doc hang tram nghin file .pkl nho truc tiep tu Drive rat cham.
/content la local disk cua Colab, doc file nho nhanh hon nhieu.
Checkpoint nen ghi local truoc de nhanh, roi sync sang Drive moi epoch de resume duoc neu Colab ngat.
```

## 0. Cac Fix Colab Da Duoc Giu Lai Cho Dataset 21k_8s

Notebook/config 21k_8s van giu cac buoc xu ly loi da gap khi train 5k:

```text
notebooks/train_soke_colab_asl_21k_8s.ipynb
configs/soke_colab_asl_21k_8s.yaml
```

Cac fix quan trong:

```text
1. Khong cai requirements.txt goc vi co bpy/Blender va mot so package render de gay loi tren Colab.
2. Force reinstall numpy==1.26.4 de tranh loi numpy.dtype size changed.
3. Patch torch.load(..., weights_only=False) de tranh loi PyTorch 2.6 khi load tokenizer.ckpt/last.ckpt.
4. Dung PROGRESS_BAR: tqdm de tranh loi RichProgressBar pop from empty list trong Colab notebook.
5. Patch metric t2m cast vertices ve float32 de tranh loi bf16: float != c10::BFloat16 khi validation.
6. Tu sua layout t2m.tar.gz neu finest.tar bi giai nen long folder.
7. Tokenization co log /content/tokenize_debug.log va check so token theo CSV/config, khong hardcode 5000.
8. Train co log /content/train_debug.log de debug khi subprocess fail.
9. Checkpoint luu local /content truoc, sync sang Drive moi epoch, va co interrupted.ckpt khi exception/interruption mem.
10. Truoc khi sync checkpoint len Drive, code xoa file trung ten cu de tranh Google Drive tao nhieu last.ckpt cung ten.
```

## 1. Code Va File Chinh

Config train chinh cho dataset How2Sign 21k_8s tren A100 40GB:

```text
configs/soke_colab_asl_21k_8s.yaml
```

Notebook chinh cho dataset How2Sign 21k_8s:

```text
notebooks/train_soke_colab_asl_21k_8s.ipynb
```

Config/notebook 5k cu van duoc giu de doi chieu:

```text
configs/soke_colab_asl_5k.yaml
notebooks/train_soke_colab_asl_5k.ipynb
```

Config asset/output:

```text
configs/assets_colab.yaml
```

Requirements rieng cho Colab:

```text
requirements-colab.txt
```

Script tao dataset How2Sign 21k_8s:

```text
scripts/export_how2sign_colab_21k_8s_dataset.py
```

Script tao subset How2Sign 5K cu:

```text
scripts/export_how2sign_colab_subset.py
```

## 2. Nguyen Tac Luu Tru

Khong push data/assets/checkpoints len GitHub. `.gitignore` da ignore cac thu muc va artifact lon:

```text
data/
data_colab_5k/
data_colab_21k_8s/
deps/
experiments/
results/
upload_archives/
*.ckpt
*.pt
*.pkl
*.npy
*.npz
*.tar
*.tar.gz
```

Ban chi push code/config/notebook/guide len GitHub.

## 3. Tao Subset How2Sign 5K O Local

Chay tu root repo local:

```text
D:\MyProgress\6thSemes\python\SOKE\SOKE
```

Tao subset dung theo config Colab:

```powershell
python scripts\export_how2sign_colab_subset.py --output-folder data_colab_5k\How2Sign --no-zip
```

Lenh nay khong xoa data goc `data/How2Sign`. No tao folder moi:

```text
data_colab_5k/
  How2Sign/
    train/
      poses/       # 5000 sample, duration <= 8s
      re_aligned/
    val/
      poses/       # 256 sample, duration <= 8s
      re_aligned/
    test/
      poses/       # 256 sample, duration <= 8s
      re_aligned/
```

Kiem tra:

```powershell
(Get-ChildItem data_colab_5k\How2Sign\train\poses -Directory | Measure-Object).Count
(Get-ChildItem data_colab_5k\How2Sign\val\poses -Directory | Measure-Object).Count
(Get-ChildItem data_colab_5k\How2Sign\test\poses -Directory | Measure-Object).Count
```

Ket qua mong doi:

```text
5000
256
256
```

## 4. Tao Archive De Upload Len Drive

Tao folder tam:

```powershell
New-Item -ItemType Directory -Force upload_archives
```

Nen/copy cac file can upload:

```powershell
Compress-Archive -Path data_colab_5k\How2Sign -DestinationPath upload_archives\How2Sign.zip -Force
Compress-Archive -Path data\stats -DestinationPath upload_archives\stats.zip -Force
Compress-Archive -Path deps\mbart-h2s-csl-phoenix -DestinationPath upload_archives\mbart-h2s-csl-phoenix.zip -Force
Compress-Archive -Path deps\smpl_models -DestinationPath upload_archives\smpl_models.zip -Force
Copy-Item deps\t2m\t2m.tar.gz upload_archives\t2m.tar.gz -Force
Copy-Item experiments\mgpt\DETO_ASL\checkpoints\tokenizer.ckpt upload_archives\tokenizer.ckpt -Force
```

Neu `deps\t2m\t2m.tar.gz` khong ton tai nhung `deps\t2m` da giai nen san, dung:

```powershell
Compress-Archive -Path deps\t2m -DestinationPath upload_archives\t2m.zip -Force
```

Sau buoc nay, can co:

```text
upload_archives/
  How2Sign.zip
  stats.zip
  mbart-h2s-csl-phoenix.zip
  smpl_models.zip
  t2m.tar.gz
  tokenizer.ckpt
```

Luu y: archive phai chua folder top-level dung ten. Vi du `How2Sign.zip` khi giai nen phai ra:

```text
How2Sign/
  train/
  val/
  test/
```

khong phai:

```text
train/
val/
test/
```

## 5. Upload Archive Len Google Drive

Tao folder tren Drive:

```text
MyDrive/SOKE_COLAB/archives/
```

Upload:

```text
MyDrive/SOKE_COLAB/archives/How2Sign.zip
MyDrive/SOKE_COLAB/archives/stats.zip
MyDrive/SOKE_COLAB/archives/mbart-h2s-csl-phoenix.zip
MyDrive/SOKE_COLAB/archives/smpl_models.zip
MyDrive/SOKE_COLAB/archives/t2m.tar.gz
MyDrive/SOKE_COLAB/archives/tokenizer.ckpt
```

Google Drive se giu them cac output sau khi train:

```text
MyDrive/SOKE_COLAB/
  experiments/
  results/
  token_cache/
```

## 6. Colab Runtime Layout

Sau khi notebook chay cell setup, layout tren Colab se la:

```text
/content/SOKE/                         # repo clone tu GitHub
/content/SOKE_COLAB_ARCHIVES/          # archive copy tu Drive ve local
/content/SOKE_COLAB_DATA/
  data/
    How2Sign/
    stats/
  deps/
    mbart-h2s-csl-phoenix/
    smpl_models/
    t2m/
  pretrained/
    tokenizer.ckpt
/content/SOKE_COLAB_RUN/
  experiments/                         # checkpoint working local, save nhanh
```

Repo clone se tao symlink:

```text
/content/SOKE/deps/mbart-h2s-csl-phoenix -> /content/SOKE_COLAB_DATA/deps/mbart-h2s-csl-phoenix
/content/SOKE/deps/smpl_models -> /content/SOKE_COLAB_DATA/deps/smpl_models
/content/SOKE/deps/t2m -> /content/SOKE_COLAB_DATA/deps/t2m
```

Checkpoint backup va results tren Drive:

```text
/content/drive/MyDrive/SOKE_COLAB/experiments/
/content/drive/MyDrive/SOKE_COLAB/results/
```

Trong luc train, `Trainer` ghi checkpoint vao local:

```text
/content/SOKE_COLAB_RUN/experiments/mgpt/SOKE_COLAB_ASL_5K/checkpoints/
```

Callback se sync checkpoint sang Drive moi epoch:

```text
/content/drive/MyDrive/SOKE_COLAB/experiments/mgpt/SOKE_COLAB_ASL_5K/checkpoints/
```

## 7. Push Code Len GitHub

Sau khi sua code xong:

```bash
git status
git add .gitignore requirements-colab.txt configs/assets_colab.yaml configs/soke_colab_asl_5k.yaml notebooks/train_soke_colab_asl_5k.ipynb COLAB_TRAIN_SOKE_ASL_GUIDE.md scripts/export_how2sign_colab_subset.py mGPT train.py test.py
git commit -m "Add hybrid Colab SOKE ASL 5K training setup"
git push origin main
```

Khong add:

```text
data/
data_colab_5k/
deps/
experiments/
upload_archives/
```

## 8. Chay Notebook Tren Colab

Mo:

```text
notebooks/train_soke_colab_asl_5k.ipynb
```

Cell dau tien sua:

```python
GITHUB_REPO_URL = "https://github.com/YOUR_USERNAME/SOKE.git"
GITHUB_BRANCH = "main"
DRIVE_ROOT = "/content/drive/MyDrive/SOKE_COLAB"
LOCAL_ROOT = "/content/SOKE_COLAB_DATA"
LOCAL_ARCHIVE_ROOT = "/content/SOKE_COLAB_ARCHIVES"
LOCAL_RUN_ROOT = "/content/SOKE_COLAB_RUN"
```

Chay lan luot:

```text
1. Mount Drive va check GPU
2. Clone repo vao /content/SOKE
3. Cai requirements-colab.txt
4. Copy archive tu Drive ve /content/SOKE_COLAB_ARCHIVES
5. Giai nen vao /content/SOKE_COLAB_DATA
6. Kiem tra required paths va tao symlink deps
7. Restore token cache tu Drive neu co
8. Neu chua co token cache, sinh motion tokens tren local disk
9. Copy token cache ve Drive
10. Restore last.ckpt tu Drive ve local neu co
11. Train SOKE, checkpoint ghi local va sync Drive moi epoch
12. Liet ke checkpoint local va Drive
```

## 9. Config Train Dang Dung

File:

```text
configs/soke_colab_asl_5k.yaml
```

Thong so chinh cho A100 40GB:

```yaml
PRECISION: bf16-mixed
MATMUL_PRECISION: high
BENCHMARK: true

TRAIN:
  END_EPOCH: 8
  BATCH_SIZE: 8
  NUM_WORKERS: 6

EVAL:
  BATCH_SIZE: 4
  NUM_WORKERS: 4

DATASET:
  H2S:
    ROOT: /content/SOKE_COLAB_DATA/data/How2Sign
    MEAN_PATH: /content/SOKE_COLAB_DATA/data/stats/mean.pt
    STD_PATH: /content/SOKE_COLAB_DATA/data/stats/std.pt
    DATASET_NAME: how2sign
    FILTER:
      TRAIN_MAX_DURATION: 8.0
      TRAIN_MAX_SAMPLES: 5000
      VAL_MAX_DURATION: 8.0
      VAL_MAX_SAMPLES: 256
      TEST_MAX_DURATION: 8.0
      TEST_MAX_SAMPLES: 256
  CODE_PATH: TOKENS_how2sign_colab_5k
```

Tokenizer checkpoint:

```yaml
TRAIN:
  PRETRAINED_VAE: /content/SOKE_COLAB_DATA/pretrained/tokenizer.ckpt
```

Output checkpoint:

```yaml
# configs/assets_colab.yaml
FOLDER: /content/SOKE_COLAB_RUN/experiments
TEST:
  FOLDER: /content/drive/MyDrive/SOKE_COLAB/results

# configs/soke_colab_asl_5k.yaml
CHECKPOINT:
  EVERY_N_TRAIN_STEPS: 500      # save local last.ckpt nhanh
  EVERY_N_EPOCHS: 1             # save local last.ckpt va epoch-xxx.ckpt
  SYNC_DIRPATH: /content/drive/MyDrive/SOKE_COLAB/experiments/mgpt/SOKE_COLAB_ASL_5K/checkpoints
  SYNC_EVERY_N_TRAIN_STEPS: 0   # uu tien toc do, khong sync Drive theo step
  SYNC_EVERY_N_EPOCHS: 1        # sync Drive moi epoch
  SYNC_ON_EXCEPTION: true
```

## 10. Motion Token Cache

Vi config doc data tu local:

```text
/content/SOKE_COLAB_DATA/data/How2Sign
```

motion tokens khi sinh se nam o local:

```text
/content/SOKE_COLAB_DATA/data/How2Sign/TOKENS_how2sign_colab_5k/how2sign/*.npy
```

Notebook se copy token cache ve Drive:

```text
MyDrive/SOKE_COLAB/token_cache/TOKENS_how2sign_colab_5k/how2sign/*.npy
```

Lan sau, notebook restore token cache tu Drive ve local de bo qua buoc tokenize.

## 11. Checkpoint Va Resume

Checkpoint working duoc luu nhanh vao local:

```text
/content/SOKE_COLAB_RUN/experiments/mgpt/SOKE_COLAB_ASL_5K/checkpoints/
```

Cuoi moi epoch, callback sync checkpoint sang Drive:

```text
MyDrive/SOKE_COLAB/experiments/mgpt/SOKE_COLAB_ASL_5K/checkpoints/
```

Se co:

```text
last.ckpt
epoch-001.ckpt
epoch-002.ckpt
...
epoch-008.ckpt
interrupted.ckpt
```

Notebook tu dong resume neu thay:

```text
MyDrive/SOKE_COLAB/experiments/mgpt/SOKE_COLAB_ASL_5K/checkpoints/last.ckpt
```

Khi resume, notebook copy file nay ve local truoc:

```text
MyDrive/SOKE_COLAB/experiments/mgpt/SOKE_COLAB_ASL_5K/checkpoints/last.ckpt
-> /content/SOKE_COLAB_RUN/experiments/mgpt/SOKE_COLAB_ASL_5K/checkpoints/last.ckpt
```

Sau do `train.py` resume tu local checkpoint, tranh doc checkpoint truc tiep tu Drive trong qua trinh khoi dong trainer.

## 12. Luu Y Hieu Nang

Hybrid nhanh hon doc Drive truc tiep vi:

```text
archive duoc copy 1 lan ve local
file .pkl nho duoc doc tu /content
checkpoint van an toan tren Drive
checkpoint save nhanh tai local, Drive chi sync moi epoch
```

Neu OOM tren A100:

```yaml
TRAIN:
  BATCH_SIZE: 4
EVAL:
  BATCH_SIZE: 2
TEST:
  BATCH_SIZE: 2
```

Neu muon an toan hon va chap nhan cham hon, co the sync Drive theo step:

```yaml
CHECKPOINT:
  SYNC_EVERY_N_TRAIN_STEPS: 1000
```

Neu muon tiet kiem Drive:

```yaml
CHECKPOINT:
  KEEP_EPOCH_CHECKPOINTS: false
```
