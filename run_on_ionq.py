# run_on_ionq.py - 完整修复版本

import math
import time
import torch
from tqdm import tqdm

from quixer.setup_training import setup_dataset, get_batch_s2s
from quixer.mini_quixer_tq import MiniQuixerTQ, MiniQuixerPQCBlock
from quixer.mini_quixer_ionq import MiniQuixerPQCBlockIonQ


def load_trained_miniquixer(checkpoint_path, hyperparams, vocab_size, device):
    """加载并验证模型"""
    model = MiniQuixerTQ(
        n_qubits=hyperparams["qubits"],
        n_tokens=hyperparams["window"],
        qsvt_polynomial_degree=hyperparams["layers"],
        n_ansatz_layers=hyperparams["ansatz_layers"],
        vocabulary_size=vocab_size,
        embedding_dimension=hyperparams["dimension"],
        dropout=hyperparams["dropout"],
        batch_size=hyperparams["batch_size"],
        device=device,
    )
    
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    
    print("\n" + "="*60)
    print("VERIFYING MODEL")
    print("="*60)
    
    vocab, (_, val_iter, _), _ = setup_dataset(
        device, hyperparams["batch_size"], hyperparams["window"]
    )
    
    x, y = get_batch_s2s(val_iter, 0, hyperparams["window"])
    x, y = x.to(device), y.to(device)
    
    with torch.no_grad():
        logits, _ = model(x)
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits, y)
        ppl = math.exp(loss.item())
    
    print(f"Quick test PPL: {ppl:.2f}")
    
    if ppl > 500:
        print("\n❌ CRITICAL ERROR: Model is NOT trained!")
        print(f"   Expected PPL: 150-250")
        print(f"   Your PPL: {ppl:.2f}")
        raise ValueError("Untrained model detected")
    elif ppl > 250:
        print("\n⚠️  WARNING: Model training incomplete")
        print("   PPL is higher than expected")
        print("   Recommend: Train for 15-20 epochs")
    else:
        print("✓ Model verification passed")
    
    return model


def quantum_eval_epoch(
    model: torch.nn.Module,
    hyperparams: dict,
    val_iter: torch.Tensor,
    device: torch.device,
    num_batches: int = 3,
):
    """
    修复版本 - 正确处理batch维度
    """
    
    loss_fn = torch.nn.CrossEntropyLoss()
    window = hyperparams["window"]
    
    # 准备IonQ backend
    pqc_block_sim: MiniQuixerPQCBlock = model.pqc_block
    input_proj = pqc_block_sim.input_proj
    pqc_params = pqc_block_sim.pqc_params.detach().clone()
    
    pqc_block_ionq = MiniQuixerPQCBlockIonQ(
        input_proj=input_proj,
        pqc_params=pqc_params,
    )
    
    sim_losses = []
    q_losses = []
    
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    print(f"Evaluating {num_batches} batches...")
    print()
    
    for b_idx in tqdm(range(num_batches), desc="Batches"):
        x, y = get_batch_s2s(val_iter, b_idx, window)
        x = x.to(device)
        y = y.to(device)
        B, T = x.shape  # 注意: B实际上等于window_size=32!
        
        print(f"\n[Batch {b_idx+1}]")
        print(f"  Input shape: x={x.shape} (B={B}, T={T})")
        
        # ===== Baseline =====
        with torch.no_grad():
            logits_sim, _ = model(x)
            loss_sim = loss_fn(logits_sim, y)
        
        sim_losses.append(loss_sim.item())
        
        # ===== Quantum - 完全按照训练时的方式处理 =====
        with torch.no_grad():
            # 前处理 - 与训练时完全一致
            tok = model.token_emb(x)  # [B, T, d_model]
            pos_ids = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
            pos = model.pos_emb(pos_ids)
            h = model.dropout(tok + pos)
            
            # QKV
            qkv = model.qkv_proj(h)  # [B, T, 3*d_qkv]
            q, k, v = torch.chunk(qkv, 3, dim=-1)
            
            # ===== 关键: 每个样本独立mean-pool =====
            # v 的形状是 [B, T, d_qkv]
            # 我们需要对每个样本的T个token做mean-pool
            context = v.mean(dim=1)  # [B, d_qkv]
            
            print(f"  Context shape: {context.shape} (should be [B={B}, d_qkv={model.pqc_block.d_qkv}])")
            
            # 量子处理 - 批量处理B个样本
            t0 = time.time()
            pqc_out = pqc_block_ionq(context)  # [B, n_qubits]
            t1 = time.time()
            
            print(f"  PQC output shape: {pqc_out.shape} (should be [B={B}, n_qubits={model.n_qubits}])")
            print(f"  Quantum time: {t1-t0:.1f}s")
            
            # 后处理
            mapped = model.map_back(pqc_out)  # [B, d_model]
            mapped = mapped.unsqueeze(1).expand(B, T, model.embedding_dimension)
            
            h = model.ln1(h + mapped)
            h2 = model.ffn(h)
            h = model.ln2(h + h2)
            
            h_last = h[:, -1, :]  # [B, d_model]
            logits_q = torch.matmul(h_last, model.token_emb.weight.t())
            
            loss_q = loss_fn(logits_q, y)
        
        q_losses.append(loss_q.item())
        
        print(f"  Baseline loss: {loss_sim.item():.4f}")
        print(f"  Quantum loss:  {loss_q.item():.4f}")
        print(f"  Difference: {abs(loss_q.item() - loss_sim.item()):.4f} ({abs(loss_q.item() - loss_sim.item())/loss_sim.item()*100:.1f}%)")
    
    # 汇总结果
    sim_loss_avg = sum(sim_losses) / len(sim_losses)
    q_loss_avg = sum(q_losses) / len(q_losses)
    
    sim_ppl = math.exp(sim_loss_avg)
    q_ppl = math.exp(q_loss_avg)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Baseline (TorchQuantum):")
    print(f"  Loss: {sim_loss_avg:.4f}")
    print(f"  PPL:  {sim_ppl:.2f}")
    print()
    print(f"Quantum (IonQ SV1):")
    print(f"  Loss: {q_loss_avg:.4f}")
    print(f"  PPL:  {q_ppl:.2f}")
    print()
    print(f"Difference:")
    diff_pct = abs(q_ppl - sim_ppl) / sim_ppl * 100
    print(f"  Δ Loss: {abs(q_loss_avg - sim_loss_avg):.4f}")
    print(f"  Δ PPL:  {abs(q_ppl - sim_ppl):.2f} ({diff_pct:.1f}%)")
    print("="*60)
    
    # 详细分析
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    if sim_ppl > 300:
        print("❌ CRITICAL: Baseline PPL too high!")
        print(f"   Current: {sim_ppl:.2f}")
        print(f"   Expected: 150-200")
        print("\n   ROOT CAUSE: Model insufficiently trained")
        print("   SOLUTION: Train for 15-20 epochs")
        print("   Command: python train_model_first.py")
        print("            (modify epochs: 10 -> 20)")
    elif sim_ppl > 250:
        print("⚠️  WARNING: Baseline PPL higher than optimal")
        print(f"   Current: {sim_ppl:.2f}")
        print(f"   Target: 150-200")
        print("   Suggest: Train 5-10 more epochs")
    else:
        print("✓ Baseline PPL acceptable")
    
    if diff_pct > 30:
        print(f"\n❌ CRITICAL: Quantum-baseline gap too large ({diff_pct:.1f}%)")
        print("   Expected: 10-20%")
        print("\n   POSSIBLE CAUSES:")
        print("   1. Sampling noise (shots=1000 may be insufficient)")
        print("   2. Numerical precision differences")
        print("   3. Implementation mismatch")
        print("\n   DIAGNOSIS:")
        
        # 检查per-batch差异
        per_batch_diffs = []
        for i, (sl, ql) in enumerate(zip(sim_losses, q_losses)):
            diff = abs(ql - sl) / sl * 100
            per_batch_diffs.append(diff)
            print(f"      Batch {i+1}: {diff:.1f}% difference")
        
        avg_diff = sum(per_batch_diffs) / len(per_batch_diffs)
        print(f"      Average: {avg_diff:.1f}%")
        
        if max(per_batch_diffs) - min(per_batch_diffs) > 30:
            print("\n   → High variance across batches suggests sampling noise")
            print("      Try: Increase shots to 2000-5000")
        else:
            print("\n   → Consistent difference suggests systematic issue")
            print("      Check: Model architecture consistency")
            
    elif diff_pct > 20:
        print(f"\n⚠️  WARNING: Quantum overhead ({diff_pct:.1f}%)")
        print("   Expected: 10-20%")
        print("   This is borderline acceptable")
    else:
        print(f"\n✓ Quantum overhead acceptable ({diff_pct:.1f}%)")
    
    print("="*60)


def main():
    device = torch.device("cpu")
    
    hyperparams = {
        "qubits": 4,
        "layers": 2,
        "ansatz_layers": 0,
        "window": 32,
        "dropout": 0.10,
        "batch_size": 32,  # ← 这个参数实际上不控制实际batch size
        "dimension": 96,
        "model": "MiniQuixerTQ",
    }
    
    # 更新为你的checkpoint路径
    checkpoint_path = "trained_models/q_transformer_lm_MiniQuixerTQ_42_1764909616.pt"
    
    import os
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        import glob
        checkpoints = glob.glob("trained_models/*.pt")
        for cp in checkpoints:
            print(f"  {cp}")
        return
    
    print("Loading dataset...")
    vocab, (train_iter, val_iter, test_iter), _ = setup_dataset(
        device, hyperparams["batch_size"], hyperparams["window"]
    )
    vocab_size = len(vocab)
    
    try:
        model = load_trained_miniquixer(
            checkpoint_path, hyperparams, vocab_size, device
        )
    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        return
    
    quantum_eval_epoch(
        model=model,
        hyperparams=hyperparams,
        val_iter=val_iter,
        device=device,
        num_batches=3,
    )


if __name__ == "__main__":
    main()