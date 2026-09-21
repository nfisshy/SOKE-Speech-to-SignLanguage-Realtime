# Huong Dan Download Cho Luong Text -> ASL 3D Sign Language

Tai lieu nay danh cho truong hop ban chi can chay inference/demo **text tieng Anh -> ASL 3D avatar**, khong can train/evaluate tren CSL-Daily hoac Phoenix-2014T.

## Muc Tieu

Sau khi chuan bi xong, repo nen co cac asset toi thieu de build luong:

```text
SOKE/
  deps/
    smpl_models/
    mbart-h2s-csl-phoenix/
  experiments/
    mgpt/
      vae/
        checkpoints/
          tokenizer.ckpt
      SOKE/
        checkpoints/
          last.ckpt
  data/
    stats/
      mean.pt
      std.pt
  outputs/
```

Ghi chu: folder `mbart-h2s-csl-phoenix` van giu dung ten nay du chi dung ASL, vi config/model checkpoint cua repo dang ky vong duong dan va tokenizer theo ten do.

## 1. Human Models: SMPL/SMPL-X/MANO

Download human models tu link trong README:

```text
https://drive.google.com/file/d/1YIXddvvBJPQVRuKON2Xc9EEDXikRTteo/view
```

Giai nen vao:

```text
SOKE/deps/smpl_models/
```

Sau khi giai nen, cau truc dung nen giong:

```text
SOKE/deps/smpl_models/
  smpl/
  smplh/
  smplx/
  mano/
```

Can tranh cau truc bi long them mot cap, vi code dang hardcode:

```text
deps/smpl_models
```

Sai:

```text
SOKE/deps/smpl_models/smpl_models/smplx/
```

Dung:

```text
SOKE/deps/smpl_models/smplx/
```

## 2. mBART Model Cua SOKE

Download mBART model folder tu README:

```text
https://drive.google.com/drive/folders/1GnaHrI0PC4ZRr-GK3FS2GXcQwlrpA5Gi
```

Dat toan bo file ben trong folder Google Drive vao:

```text
SOKE/deps/mbart-h2s-csl-phoenix/
```

Folder nay bat buoc can co file:

```text
SOKE/deps/mbart-h2s-csl-phoenix/map_ids.pkl
```

Neu thieu `map_ids.pkl`, model se khong load duoc.

## 3. DETO Tokenizer Checkpoint

Download tokenizer checkpoint tu README:

```text
https://drive.google.com/file/d/18HdPeXwz4O6LY4FZMC5BZ9rja4pcUTFk/view
```

Doi ten file thanh:

```text
tokenizer.ckpt
```

Dat vao:

```text
SOKE/experiments/mgpt/vae/checkpoints/tokenizer.ckpt
```

Checkpoint nay dung de decode motion tokens thanh SMPL-X motion features.

## 4. Mean Va Std Cho SMPL-X Pose

Download 2 file mean/std tu README:

```text
mean: https://drive.google.com/file/d/1NH-eVtS0nNjMjCwae-A1ii5sxj44C3bo/view
std:  https://drive.google.com/file/d/1FHHWS0GPM2s6S2PB2JHv4ufdEbzezuKW/view
```

Doi ten dung:

```text
mean.pt
std.pt
```

Dat vao:

```text
SOKE/data/stats/mean.pt
SOKE/data/stats/std.pt
```

Ghi chu: README goi day la mean/std cua SMPL-X poses. Du chi chay ASL, van can 2 file nay de normalize/denormalize va convert output motion sang joints/vertices.

## 5. SOKE Generator Checkpoint

Day la file quan trong nhat de chay text -> ASL ma khong train lai.

Can co checkpoint generator da train xong. Dat file tai:

```text
SOKE/experiments/mgpt/SOKE/checkpoints/last.ckpt
```

Neu file tai ve co ten khac, hay doi ten thanh:

```text
last.ckpt
```

Luu y: README hien tai khong ghi ro link checkpoint generator cuoi cung. Neu khong co file nay, chi co the build code pipeline, nhung chua chay inference chat luong that duoc. Khi do can:

```text
1. Tim pretrained SOKE generator checkpoint tren project homepage hoac hoi tac gia.
2. Hoac train lai SOKE, nhung cach nay can full dataset va GPU manh.
```

## 6. Retrieval Files

Repo hien tai da co san cac file retrieval:

```text
SOKE/scripts/word2code.json
SOKE/scripts/name2kws_train.json
SOKE/scripts/name2kws_val.json
SOKE/scripts/name2kws_test.json
```

Voi inference text tu do, minh se build logic doc keyword truc tiep tu cau tieng Anh va tra trong `word2code.json`. Vi vay ban khong can download them retrieval dictionary neu chi muon demo ASL.

## 7. Dataset How2Sign Co Can Khong?

Neu chi can demo:

```text
Text tieng Anh -> ASL 3D motion/video
```

thi **khong bat buoc** download full How2Sign dataset.

Chi can download How2Sign neu ban muon:

```text
1. Chay lai test.py goc cua repo tren test split.
2. Tinh metrics voi ground truth.
3. Train/fine-tune model.
4. Render video so sanh voi raw video ground truth.
```

Neu can cac viec tren, dat How2Sign vao:

```text
SOKE/data/How2Sign/
```

Nhung voi luong inference ASL-only, minh se build script khong phu thuoc vao dataloader test split.

## 8. Lenh Tao Folder Tren Windows

Chay tu root repo `SOKE/`:

```powershell
New-Item -ItemType Directory -Force deps\smpl_models
New-Item -ItemType Directory -Force deps\mbart-h2s-csl-phoenix
New-Item -ItemType Directory -Force experiments\mgpt\vae\checkpoints
New-Item -ItemType Directory -Force experiments\mgpt\SOKE\checkpoints
New-Item -ItemType Directory -Force data\stats
New-Item -ItemType Directory -Force outputs
```

## 9. Checklist Cuoi

Truoc khi minh build pipeline, hay dam bao cac path nay ton tai:

```text
SOKE/deps/smpl_models/smplx/
SOKE/deps/mbart-h2s-csl-phoenix/map_ids.pkl
SOKE/experiments/mgpt/vae/checkpoints/tokenizer.ckpt
SOKE/experiments/mgpt/SOKE/checkpoints/last.ckpt
SOKE/data/stats/mean.pt
SOKE/data/stats/std.pt
SOKE/scripts/word2code.json
```

Khi cac file tren da san sang, luong ASL-only se duoc build theo huong:

```powershell
python infer_asl.py --text "There is a customer service hotline." --out_dir outputs\asl_demo
```

Output du kien:

```text
outputs/asl_demo/
  motion.pkl
  motion.npy
  joints.npy
  vertices.npy
  preview.mp4
```
