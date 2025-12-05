# train_model_first.py - 修改device配置

import torch
import math
from quixer.setup_training import get_train_evaluate

def main():
    print("="*60)
    print("TRAINING MINIQUIXER MODEL")
    print("="*60)
    
    # ===== 检查并使用GPU =====
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("⚠️  GPU not available, using CPU")
        # CPU优化
        torch.set_num_threads(8)  # 根据你的CPU核心数调整
        print(f"   CPU threads: {torch.get_num_threads()}")
    
    hyperparams = {
        "qubits": 4,
        "layers": 2,
        "ansatz_layers": 0,
        "window": 32,
        "epochs": 10,
        "restart_epochs": 30000,
        "dropout": 0.10,
        "lr": 0.002,
        "lr_sched": "cos",
        "wd": 0.0001,
        "eps": 1e-10,
        "batch_size": 32,
        "max_grad_norm": 5.0,
        "model": "MiniQuixerTQ",  # ← 还是用这个名字
        "print_iter": 50,
        "dimension": 96,
        "seed": 42,
    }
    
    print(f"\nConfiguration:")
    print(f"  Device: {device}")
    print(f"  Epochs: {hyperparams['epochs']}")
    print(f"  Batch size: {hyperparams['batch_size']}")
    
    if device.type == 'cuda':
        print(f"\nEstimated time: 3-10 minutes (GPU)")
    else:
        print(f"\nEstimated time: 25-35 minutes (CPU)")
    
    print(f"{'='*60}\n")
    
    # 训练
    import time
    start = time.time()
    
    train_evaluate = get_train_evaluate(device)
    test_loss = train_evaluate(hyperparams)
    
    elapsed = time.time() - start
    test_ppl = math.exp(test_loss)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test PPL:  {test_ppl:.2f}")
    
    if test_ppl > 300:
        print("\n⚠️  WARNING: PPL still high!")
    elif test_ppl > 200:
        print("\n✓ Model trained reasonably well")
    else:
        print("\n✓ Model trained successfully!")
    
    print("\nCheckpoint saved in: trained_models/")
    print("You can now run: python run_on_ionq.py")
    print("="*60)


if __name__ == "__main__":
    main()