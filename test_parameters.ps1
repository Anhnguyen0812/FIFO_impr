# Test Training Commands - Quick Parameter Testing

## 🧪 Test 1: Small Batch with Accumulation (8GB GPU)
Write-Host "Test 1: Small batch with gradient accumulation" -ForegroundColor Cyan
python main.py `
    --file-name 'test_b1a4' `
    --modeltrain 'train' `
    --use-segformer `
    --batch-size 1 `
    --accum-steps 4 `
    --num-steps 100 `
    --save-pred-every 50 `
    --gpu 0

## 🧪 Test 2: Medium Batch (16GB GPU)
Write-Host "Test 2: Medium batch" -ForegroundColor Cyan
python main.py `
    --file-name 'test_b2a2' `
    --modeltrain 'train' `
    --use-segformer `
    --batch-size 2 `
    --accum-steps 2 `
    --num-steps 100 `
    --save-pred-every 50 `
    --gpu 0

## 🧪 Test 3: Large Batch (24GB GPU)
Write-Host "Test 3: Large batch" -ForegroundColor Cyan
python main.py `
    --file-name 'test_b4a2' `
    --modeltrain 'train' `
    --use-segformer `
    --batch-size 4 `
    --accum-steps 2 `
    --num-steps 100 `
    --save-pred-every 50 `
    --gpu 0

## 🧪 Test 4: ResNet Baseline (small batch)
Write-Host "Test 4: ResNet baseline" -ForegroundColor Cyan
python main.py `
    --file-name 'test_resnet_b4' `
    --modeltrain 'train' `
    --batch-size 4 `
    --accum-steps 1 `
    --num-steps 100 `
    --save-pred-every 50 `
    --gpu 0

## 🧪 Test 5: Pretrain FogPassFilter only
Write-Host "Test 5: Pretrain FogPassFilter" -ForegroundColor Cyan
python main.py `
    --file-name 'test_fogpass_pretrain' `
    --modeltrain 'no' `
    --use-segformer `
    --freeze-segformer-encoder `
    --batch-size 4 `
    --num-steps 100 `
    --save-pred-every 50 `
    --gpu 0
