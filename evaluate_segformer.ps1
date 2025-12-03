# PowerShell script to evaluate SegFormer B5 checkpoint
# Usage: .\evaluate_segformer.ps1 -Checkpoint "segformer_b5_cityscapes.pth"

param(
    [string]$Checkpoint = "segformer_b5_cityscapes.pth",
    [int]$GPU = 0,
    [string]$FileName = "segformer_b5_eval"
)

if (-not (Test-Path $Checkpoint)) {
    Write-Host "❌ Error: Checkpoint not found: $Checkpoint" -ForegroundColor Red
    Write-Host ""
    Write-Host "Available checkpoints:" -ForegroundColor Cyan
    Get-ChildItem *.pth | Select-Object Name, @{Name="Size(MB)";Expression={[math]::Round($_.Length/1MB,2)}}, LastWriteTime | Format-Table
    exit 1
}

Write-Host "🔍 Evaluating SegFormer B5 Model" -ForegroundColor Green
Write-Host "  Checkpoint: $Checkpoint" -ForegroundColor Cyan
Write-Host "  GPU: $GPU" -ForegroundColor Cyan
Write-Host "  Output folder: result_FZ/$FileName" -ForegroundColor Cyan
Write-Host ""

# Auto-detect model type from checkpoint
Write-Host "📦 Loading checkpoint..." -ForegroundColor Yellow
python -c @"
import torch
ckpt = torch.load('$Checkpoint', map_location='cpu', weights_only=False)
print('Checkpoint keys:', list(ckpt.keys()))
if 'args' in ckpt:
    print('Training args found')
    if hasattr(ckpt['args'], 'use_segformer'):
        print(f'  use_segformer: {ckpt['args'].use_segformer}')
if 'state_dict' in ckpt:
    first_keys = list(ckpt['state_dict'].keys())[:5]
    print('First 5 model keys:')
    for k in first_keys:
        print(f'  - {k}')
    if any('segformer' in k for k in ckpt['state_dict'].keys()):
        print('✓ Detected: SegFormer model')
    else:
        print('✓ Detected: ResNet model')
if 'train_iter' in ckpt:
    print(f'Trained iterations: {ckpt['train_iter']}')
"@

Write-Host ""
Write-Host "🚀 Running evaluation..." -ForegroundColor Green
Write-Host ""

$cmd = @(
    "python", "evaluate.py",
    "--restore-from", $Checkpoint,
    "--gpu", $GPU,
    "--file-name", $FileName
)

Write-Host "Command:" -ForegroundColor Yellow
Write-Host ($cmd -join " ") -ForegroundColor Gray
Write-Host ""

& $cmd[0] $cmd[1..($cmd.Length-1)]

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Evaluation completed successfully!" -ForegroundColor Green
    Write-Host "  Results saved to: result_FZ/$FileName/" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To view results:" -ForegroundColor Yellow
    Write-Host "  cd result_FZ/$FileName" -ForegroundColor Gray
    Write-Host "  ls *.png" -ForegroundColor Gray
} else {
    Write-Host ""
    Write-Host "❌ Evaluation failed with exit code: $LASTEXITCODE" -ForegroundColor Red
}
