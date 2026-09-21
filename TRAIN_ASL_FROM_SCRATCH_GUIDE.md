# Huong Dan Train ASL-Only Tu Dau Cho SOKE

Tai lieu nay danh cho truong hop **khong tim duoc SOKE generator checkpoint** va phai train lai pipeline de chay:

```text
English text -> ASL 3D sign motion/avatar
```

Huong dan nay uu tien **ASL-only / How2Sign-only**, khong train CSL-Daily hoac Phoenix-2014T.

## 0. Tong Quan Pipeline Can Train

Repo nay co 2 model chinh:

```text
1. DETO tokenizer
   SMPL-X pose sequence -> discrete motion tokens

2. SOKE generator
   English text + optional retrieved sign tokens -> motion tokens -> SMPL-X motion
```

Thu tu train bat buoc:

```text
Step 1: Chuan bi How2Sign SMPL-X poses + annotation csv + mean/std
Step 2: Train DETO tokenizer
Step 3: Encode How2Sign poses thanh motion-token .npy
Step 4: Train SOKE generator tren caption + motion tokens
Step 5: Test/inference va render
```

Khong the bo Step 2 neu ban khong co `tokenizer.ckpt`.

## 1. Yeu Cau May Va Moi Truong

Khuyen nghi:

```text
OS: Linux hoac WSL2 Ubuntu
Python: 3.10
GPU: NVIDIA CUDA
VRAM: toi thieu 24GB de thu nghiem, tot hon la multi-GPU
Disk: toi thieu 200GB neu dung full How2Sign poses/raw data
```

Tren Windows native co the gap loi voi `bpy`, `pyrender`, CUDA multiprocessing va shell scripts. Neu co the, nen train tren Linux/WSL2.

Tao moi truong:

```bash
conda create -n soke python=3.10
conda activate soke

# Cai torch phu hop CUDA cua may. Vi du CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Neu `bpy==3.4.0` loi khi install, co the tam thoi bo qua visualization packages de train truoc. Phan train/test model khong can Blender.

## 2. Folder Chuan Can Co

Dat du lieu va asset theo layout nay:

```text
SOKE/
  deps/
    smpl_models/
    mbart-h2s-csl-phoenix/
    t2m/
  data/
    How2Sign/
      train/
        poses/
        re_aligned/
      val/
        poses/
        re_aligned/
      test/
        poses/
        re_aligned/
      TOKENS_how2sign/
    stats/
      mean.pt
      std.pt
  experiments/
    mgpt/
      DETO_ASL/
      SOKE_ASL/
```

## 3. Download Asset Bat Buoc

### 3.1. SMPL/SMPL-X/MANO human models

Download tu README:

```text
https://drive.google.com/file/d/1YIXddvvBJPQVRuKON2Xc9EEDXikRTteo/view
```

Giai nen vao:

```text
SOKE/deps/smpl_models/
```

Cau truc dung:

```text
SOKE/deps/smpl_models/
  smpl/
  smplh/
  smplx/
  mano/
```

### 3.2. mBART model

Download tu README:

```text
https://drive.google.com/drive/folders/1GnaHrI0PC4ZRr-GK3FS2GXcQwlrpA5Gi
```

Dat vao:

```text
SOKE/deps/mbart-h2s-csl-phoenix/
```

Bat buoc can co:

```text
SOKE/deps/mbart-h2s-csl-phoenix/map_ids.pkl
```

### 3.3. Mean/std SMPL-X pose

Download tu README:

```text
mean: https://drive.google.com/file/d/1NH-eVtS0nNjMjCwae-A1ii5sxj44C3bo/view
std:  https://drive.google.com/file/d/1FHHWS0GPM2s6S2PB2JHv4ufdEbzezuKW/view
```

Doi ten va dat vao:

```text
SOKE/data/stats/mean.pt
SOKE/data/stats/std.pt
```

### 3.4. T2M evaluator assets

Dung cho metric/evaluation. Neu chi train nhanh de co checkpoint, co the disable val/test truoc, nhung nen chuan bi:

```bash
sh prepare/download_t2m_evalutors.sh
```

Luu y: ten file script trong repo la `download_t2m_evalutors.sh`, bi thieu chu `a` trong `evaluators`.

### 3.5. How2Sign data va SMPL-X poses

Can tai:

```text
1. How2Sign raw/split files tu https://how2sign.github.io/
2. Split files tu link README:
   https://drive.google.com/drive/folders/1sPhBwmiWCXLZSHtM3fpotbz3BDgoYmco
3. SMPL-X poses tu project homepage:
   https://2000zrl.github.io/soke/
```

Code dang ky vong How2Sign co dang:

```text
SOKE/data/How2Sign/
  train/
    poses/
      SAMPLE_NAME/
        SAMPLE_NAME_0_3D.pkl
        SAMPLE_NAME_1_3D.pkl
        ...
    re_aligned/
      how2sign_realigned_train_preprocessed_fps.csv
  val/
    poses/
      SAMPLE_NAME/
        SAMPLE_NAME_0_3D.pkl
        ...
    re_aligned/
      how2sign_realigned_val_preprocessed_fps.csv
  test/
    poses/
      SAMPLE_NAME/
        SAMPLE_NAME_0_3D.pkl
        ...
    re_aligned/
      how2sign_realigned_test_preprocessed_fps.csv
```

CSV can co cac cot ma code doc:

```text
SENTENCE_NAME
SENTENCE
fps
START_REALIGNED
END_REALIGNED
```

Moi `.pkl` frame can co cac key:

```text
smplx_root_pose
smplx_body_pose
smplx_lhand_pose
smplx_rhand_pose
smplx_jaw_pose
smplx_shape
smplx_expr
```

## 4. Tao Config ASL-Only

Nen tao 2 config rieng, de khong dung vao config multilingual goc:

```text
configs/deto_asl.yaml
configs/soke_asl.yaml
```

### 4.1. Config DETO ASL-only

Copy `configs/deto.yaml` thanh:

```text
configs/deto_asl.yaml
```

Sua cac truong chinh:

```yaml
NAME: DETO_ASL
ACCELERATOR: "gpu"
DEVICE: [0]

TRAIN:
  STAGE: vae
  NUM_WORKERS: 8
  BATCH_SIZE: 64
  END_EPOCH: 500
  PRETRAINED: ""

EVAL:
  BATCH_SIZE: 8
  SPLIT: val

TEST:
  SPLIT: test
  BATCH_SIZE: 8
  REPLICATION_TIMES: 1
  SAVE_PREDICTIONS: false

DATASET:
  target: mGPT.data.H2S.H2SDataModule
  H2S:
    DATASET_NAME: how2sign
    ROOT: data/How2Sign
    CSL_ROOT: data/CSL-Daily
    PHOENIX_ROOT: data/Phoenix_2014T
    MEAN_PATH: data/stats/mean.pt
    STD_PATH: data/stats/std.pt
    MAX_MOTION_LEN: 400
    MIN_MOTION_LEN: 40
    MAX_TEXT_LEN: 20
    PICK_ONE_TEXT: true
    FRAME_RATE: 20.0
    UNIT_LEN: 4
    STD_TEXT: false

METRIC:
  TYPE: ["MRMetrics"]

LOGGER:
  TYPE: []
  VAL_EVERY_STEPS: 10
  WANDB:
    params:
      project: null
```

Ghi chu:

```text
CSL_ROOT va PHOENIX_ROOT co the de placeholder, vi DATASET_NAME=how2sign nen code khong doc CSL/Phoenix.
```

### 4.2. Config SOKE ASL-only

Copy `configs/soke.yaml` thanh:

```text
configs/soke_asl.yaml
```

Sua cac truong chinh:

```yaml
NAME: SOKE_ASL
ACCELERATOR: "gpu"
DEVICE: [0]

TRAIN:
  STAGE: lm_pretrain
  NUM_WORKERS: 8
  BATCH_SIZE: 8
  END_EPOCH: 150
  RESUME: ""
  PRETRAINED: ""
  PRETRAINED_VAE: experiments/mgpt/DETO_ASL/checkpoints/last.ckpt

EVAL:
  BATCH_SIZE: 4
  SPLIT: val

TEST:
  CHECKPOINTS: null
  SPLIT: test
  BATCH_SIZE: 4
  REPLICATION_TIMES: 1
  SAVE_PREDICTIONS: true

DATASET:
  target: mGPT.data.H2S.H2SDataModule
  H2S:
    DATASET_NAME: how2sign
    ROOT: data/How2Sign
    CSL_ROOT: data/CSL-Daily
    PHOENIX_ROOT: data/Phoenix_2014T
    MEAN_PATH: data/stats/mean.pt
    STD_PATH: data/stats/std.pt
    MAX_MOTION_LEN: 400
    MIN_MOTION_LEN: 40
    MAX_TEXT_LEN: 40
    PICK_ONE_TEXT: true
    FRAME_RATE: 20.0
    UNIT_LEN: 4
    STD_TEXT: false
  CODE_PATH: TOKENS_how2sign

METRIC:
  TYPE: ["TM2TMetrics"]

model:
  target: mGPT.models.mgpt.MotionGPT
  params:
    condition: text
    task: t2m
    lm: ${lm.mbart_h2s_csl_phoenix}
    motion_vae: ${vq.re96}
    hand_vae_cfg: ${vq.hand192}
    rhand_vae_cfg: ${vq.hand192}

LOGGER:
  TYPE: []
  VAL_EVERY_STEPS: 5
  WANDB:
    params:
      project: null
```

Neu GPU yeu, giam:

```yaml
TRAIN:
  BATCH_SIZE: 1
  NUM_WORKERS: 2
EVAL:
  BATCH_SIZE: 1
TEST:
  BATCH_SIZE: 1
DEVICE: [0]
```

## 5. Patch Nho Can Luu Y Cho ASL-Only

Code callback hien tai monitor metric cua ca:

```text
how2sign
csl
phoenix
```

Neu train ASL-only, metric `csl_*` va `phoenix_*` co the khong ton tai, lam checkpoint callback loi khi validation.

Truoc khi train that, nen patch `mGPT/callback.py` de:

```text
1. Voi MRMetrics: chi monitor Metrics/how2sign_MPJPE_PA_hand
2. Voi TM2TMetrics: chi monitor Metrics/how2sign_DTW_MPJPE_PA_lhand
```

Hoac don gian hon:

```text
Tam thoi chi save last checkpoint, khong monitor csl/phoenix metric.
```

Khi ban san sang train, minh co the patch phan nay truc tiep de repo chay ASL-only sach hon.

## 6. Step 1 - Train DETO Tokenizer

Chay:

```bash
python -m train --cfg configs/deto_asl.yaml --nodebug --device 0 --use_gpus 0
```

Output mong doi:

```text
SOKE/experiments/mgpt/DETO_ASL/
  checkpoints/
    last.ckpt
    ...
  log_..._train.log
  config_..._train.yaml
```

File quan trong:

```text
SOKE/experiments/mgpt/DETO_ASL/checkpoints/last.ckpt
```

Neu train dung multi-GPU, vi du 4 GPU:

```bash
python -m train --cfg configs/deto_asl.yaml --nodebug --device 0 1 2 3 --use_gpus 0,1,2,3
```

## 7. Step 2 - Test DETO Tokenizer

Chay:

```bash
python -m test --cfg configs/deto_asl.yaml --nodebug --device 0 --use_gpus 0
```

Muc tieu:

```text
Kiem tra checkpoint DETO decode/reconstruct duoc SMPL-X pose.
```

Neu `TEST.CHECKPOINTS` null, code tu tim:

```text
experiments/mgpt/DETO_ASL/checkpoints/last.ckpt
```

## 8. Step 3 - Encode How2Sign Thanh Motion Tokens

README ghi sai module path. File that nam o:

```text
scripts/get_motion_code.py
```

Chay:

```bash
python -m scripts.get_motion_code --cfg configs/soke_asl.yaml --nodebug --device 0 --use_gpus 0
```

Output mong doi:

```text
SOKE/data/How2Sign/TOKENS_how2sign/how2sign/
  SAMPLE_NAME.npy
  ...
```

Moi `.npy` chua token cho:

```text
body
left hand
right hand
```

Luu y: script hien tai dung `datasets.train_dataloader()`, nen mac dinh chi encode split train. Dieu nay du de train LM vi `Text2MotionDatasetCB` cung dang force `split = train`. Neu muon encode val/test cho cac muc dich khac, can sua script rieng.

## 9. Step 4 - Train SOKE Generator ASL-Only

Chay:

```bash
python -m train --cfg configs/soke_asl.yaml --nodebug --device 0 --use_gpus 0
```

Output mong doi:

```text
SOKE/experiments/mgpt/SOKE_ASL/
  checkpoints/
    last.ckpt
    ...
  log_..._train.log
  config_..._train.yaml
```

File generator checkpoint can cho inference:

```text
SOKE/experiments/mgpt/SOKE_ASL/checkpoints/last.ckpt
```

## 10. Step 5 - Test SOKE Generator

Chay:

```bash
python -m test --cfg configs/soke_asl.yaml --task t2m --device 0 --use_gpus 0
```

Neu `TEST.SAVE_PREDICTIONS: true`, output se nam o:

```text
SOKE/results/mgpt/SOKE_ASL/test_rank_0/
  SAMPLE_NAME.pkl
  test_scores.json
```

Moi `.pkl` co:

```text
feats_rst  # generated SMPL-X motion features
feats_ref  # ground-truth motion features
text       # input English sentence
```

## 11. Step 6 - Build Inference Text Tu Do

Sau khi co:

```text
experiments/mgpt/SOKE_ASL/checkpoints/last.ckpt
experiments/mgpt/DETO_ASL/checkpoints/last.ckpt
```

co the build script:

```bash
python infer_asl.py --text "There is a customer service hotline." --out_dir outputs/asl_demo
```

Script inference nen:

```text
1. Load configs/soke_asl.yaml
2. Load DETO tokenizer checkpoint
3. Load SOKE_ASL generator checkpoint
4. Gan src="how2sign"
5. Generate body/lhand/rhand tokens
6. Decode thanh SMPL-X features
7. Xuat pkl/npy/video
```

## 12. Checklist Truoc Khi Train

Kiem tra cac path:

```text
SOKE/deps/smpl_models/smplx/
SOKE/deps/mbart-h2s-csl-phoenix/map_ids.pkl
SOKE/data/stats/mean.pt
SOKE/data/stats/std.pt
SOKE/data/How2Sign/train/re_aligned/how2sign_realigned_train_preprocessed_fps.csv
SOKE/data/How2Sign/val/re_aligned/how2sign_realigned_val_preprocessed_fps.csv
SOKE/data/How2Sign/test/re_aligned/how2sign_realigned_test_preprocessed_fps.csv
SOKE/data/How2Sign/train/poses/
SOKE/data/How2Sign/val/poses/
SOKE/data/How2Sign/test/poses/
SOKE/scripts/word2code.json
```

Sau Step 1:

```text
SOKE/experiments/mgpt/DETO_ASL/checkpoints/last.ckpt
```

Sau Step 3:

```text
SOKE/data/How2Sign/TOKENS_how2sign/how2sign/*.npy
```

Sau Step 4:

```text
SOKE/experiments/mgpt/SOKE_ASL/checkpoints/last.ckpt
```

## 13. Cac Loi De Gap

### Loi khong tim thay `map_ids.pkl`

Kiem tra:

```text
deps/mbart-h2s-csl-phoenix/map_ids.pkl
```

### Loi khong tim thay SMPL-X model

Kiem tra:

```text
deps/smpl_models/smplx/
```

Khong duoc de long folder.

### Loi khong tim thay CSV

Kiem tra:

```text
data/How2Sign/train/re_aligned/how2sign_realigned_train_preprocessed_fps.csv
```

### Loi khong tim thay pose frame `.pkl`

Voi sample `abc`, code se tim:

```text
data/How2Sign/train/poses/abc/abc_0_3D.pkl
data/How2Sign/train/poses/abc/abc_1_3D.pkl
...
```

### Loi checkpoint metric CSL/Phoenix khi train ASL-only

Can patch `mGPT/callback.py` de chi monitor `how2sign` metrics.

### Out of memory

Giam:

```yaml
TRAIN.BATCH_SIZE
EVAL.BATCH_SIZE
TEST.BATCH_SIZE
TRAIN.NUM_WORKERS
DEVICE
```

Bat dau voi batch size 1-4 neu GPU nho.

