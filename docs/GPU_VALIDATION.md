# GPU Validation

Environment:
- Python: 3.11.9
- PyTorch: 2.12.0+cu126
- Torch CUDA build: 12.6
- GPU: NVIDIA GeForce RTX 4090 Laptop GPU
- CUDA available: True

Validation commands:
python -c "import torch; print('torch:', torch.__version__); print('torch cuda build:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python -c "import torch; x=torch.randn(4096,4096,device='cuda'); y=x@x; torch.cuda.synchronize(); print(y.shape, y.device)"

Result:
- CUDA-backed tensor operation completed on cuda:0.
