# SegFormer B5 Preprocessing Configuration

## Tóm tắt các thay đổi chính

### 1. **Sửa hệ màu: BGR → RGB** ✅
**Vấn đề cũ:**
- Dataset cũ dùng OpenCV đọc ảnh BGR, sau đó đảo ngược `image[:, :, ::-1]`
- SegFormer B5 được train với RGB format

**Giải pháp:**
- Tạo dataset mới trong `dataset/segformer_datasets.py`
- Sử dụng PIL để đọc ảnh (mặc định là RGB)
- **KHÔNG đảo ngược kênh màu** (bỏ `[:, :, ::-1]`)

### 2. **Cập nhật tham số chuẩn hóa** ✅

**Cấu hình cũ (ResNet/FIFO):**
```python
IMG_MEAN = [104.00698793, 116.66876762, 122.67891434]
IMG_STD = [1.0, 1.0, 1.0]  # Không chia cho std
```

**Cấu hình mới (SegFormer B5):**
```python
SEGFORMER_MEAN = [123.675, 116.28, 103.53]
SEGFORMER_STD = [58.395, 57.12, 57.375]
```

**Công thức chuẩn hóa:**
```python
# Cũ: (BGR - mean)
# Mới: (RGB - mean) / std
image = (image - SEGFORMER_MEAN) / SEGFORMER_STD
```

### 3. **Multi-scale Testing Strategy**

Code vẫn giữ chiến lược multi-scale testing (3 scales) nhưng với preprocessing đúng:

**Foggy Zurich:**
- Scale 1: 1152 × 648
- Scale 2: 1536 × 864  
- Scale 3: 1920 × 1080

**Foggy Driving:**
- Scale 1: 100% (original size)
- Scale 2: 80%
- Scale 3: 60%

**Cityscapes/Lindau:**
- Scale 1: 2048 × 1024
- Scale 2: 1638 × 819
- Scale 3: 1229 × 614

### 4. **Model Loading**

Code được cập nhật để hỗ trợ nhiều format checkpoint:

```python
# 1. Load pretrained từ HuggingFace (default)
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
    num_labels=19
)

# 2. Nếu có custom checkpoint (e.g., segformer_b5_cityscapes.pth)
checkpoint = torch.load(args.restore_from)
model.load_state_dict(checkpoint['state_dict'], strict=False)
```

## Files đã thay đổi

### 1. **dataset/segformer_datasets.py** (MỚI)
Chứa 3 dataset classes với preprocessing đúng cho SegFormer:
- `SegformerFoggyZurichDataSet`
- `SegformerFoggyDrivingDataSet`
- `SegformerCityscapesDataSet`

### 2. **evaluate.py** (CẬP NHẬT)
- Import dataset mới
- Cập nhật IMG_MEAN và thêm IMG_STD
- Thay thế tất cả old datasets bằng SegFormer datasets
- Cập nhật model loading để xử lý SegFormer outputs (`.logits`)

## Cách test

### Test với pretrained weights từ HuggingFace:
```bash
python evaluate.py --file-name "segformer_test" --restore-from "without_pretraining"
```

### Test với custom checkpoint:
```bash
python evaluate.py --file-name "segformer_custom" --restore-from "segformer_b5_cityscapes.pth"
```

## Kiểm tra kết quả

Sau khi chạy, kết quả sẽ được lưu tại:
- `./result_FZ/segformer_test/` - Foggy Zurich predictions
- `./result_FD/segformer_test/` - Foggy Driving predictions
- `./result_FDD/segformer_test/` - Foggy Driving Dense predictions
- `./result_Clindau/segformer_test/` - Clear Lindau predictions

## So sánh với model cũ

| Aspect | RefineNet (cũ) | SegFormer B5 (mới) |
|--------|----------------|-------------------|
| Color Format | BGR | **RGB** |
| Mean | [104, 116, 122] | **[123.675, 116.28, 103.53]** |
| Std | [1, 1, 1] | **[58.395, 57.12, 57.375]** |
| Output | 6 outputs | **1 output (.logits)** |
| Architecture | CNN (ResNet) | **Transformer** |

## Lưu ý quan trọng

1. **KHÔNG dùng dataset cũ** (`cityscapes_dataset.py`, `Foggy_Zurich_test.py`, `foggy_driving.py`) cho SegFormer
2. **BẮT BUỘC sử dụng RGB format** - không đảo ngược kênh màu
3. **PHẢI chia cho std** - không thể dùng std=[1,1,1]
4. **Model output là `.logits`** không phải tuple nhiều outputs như RefineNet

## Troubleshooting

### Nếu mIoU thấp (<20%):
- ✅ Kiểm tra xem có dùng đúng dataset mới không
- ✅ Kiểm tra preprocessing: RGB vs BGR
- ✅ Kiểm tra mean/std values

### Nếu model không load được checkpoint:
- ✅ Kiểm tra format của checkpoint (mmseg vs transformers)
- ✅ Thử `strict=False` khi load_state_dict
- ✅ In ra keys của checkpoint để debug

### Nếu CUDA out of memory:
- ✅ Giảm batch size (đã set =1)
- ✅ Giảm crop size trong evaluation
- ✅ Disable gradient: `with torch.no_grad()`
