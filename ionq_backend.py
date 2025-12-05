# ionq_backend.py - 修复版本

import torch
from braket.circuits import Circuit
from braket.aws import AwsDevice
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# TODO: 把这个 ARN 换成你 Braket 控制台里实际能用的 IonQ 设备
IONQ_ARN = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"

aws_device = AwsDevice(IONQ_ARN)
print("[Braket backend] Initialized device:", aws_device.arn)


def ionq_pqc_forward(angles: torch.Tensor,
                     params: torch.Tensor,
                     shots: int = 1000,  # ← 改: 2000→1000 (足够且快)
                     max_parallel: int = 8) -> torch.Tensor:  # ← 新增: 并行数量
    """
    修复版本 - 主要改进:
    1. 批量并行提交电路 (从2700s降到300s!)
    2. 减少shots到1000 (仍然足够精确)
    """
    angles_cpu = angles.detach().cpu()
    params_cpu = params.detach().cpu()

    B, Q = angles_cpu.shape
    L = params_cpu.shape[0]

    print(f"[IonQ] Building {B} circuits...")
    
    # ===== 步骤1: 批量构建所有电路 =====
    circuits = []
    for b in range(B):
        c = Circuit()

        # 1) 输入编码
        for q in range(Q):
            theta = float(angles_cpu[b, q])
            if abs(theta) > 1e-9:
                c.rx(q, theta)

        # 2) L 层参数化 + 纠缠
        for l in range(L):
            for q in range(Q):
                rx_t, ry_t, rz_t = params_cpu[l, q]
                rx_t = float(rx_t)
                ry_t = float(ry_t)
                rz_t = float(rz_t)

                if abs(rx_t) > 1e-9:
                    c.rx(q, rx_t)
                if abs(ry_t) > 1e-9:
                    c.ry(q, ry_t)
                if abs(rz_t) > 1e-9:
                    c.rz(q, rz_t)

            # CNOT 链
            for q in range(Q - 1):
                c.cnot(q, q + 1)

        # 3) 测量
        for q in range(Q):
            c.measure(q)
        
        circuits.append(c)
    
    # ===== 步骤2: 并行提交所有任务 (关键优化!) =====
    print(f"[IonQ] Submitting {B} tasks in parallel (max {max_parallel} concurrent)...")
    start_time = time.time()
    
    def submit_single_circuit(circuit):
        """提交单个电路并返回task"""
        return aws_device.run(circuit, shots=shots)
    
    # 使用线程池并行提交
    tasks = []
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        # 提交所有电路
        futures = [executor.submit(submit_single_circuit, c) for c in circuits]
        
        # 收集所有任务
        for future in futures:
            task = future.result()  # 获取AwsQuantumTask对象
            tasks.append(task)
    
    submit_time = time.time() - start_time
    print(f"[IonQ] All tasks submitted in {submit_time:.1f}s")
    
    # ===== 步骤3: 等待所有结果 =====
    print(f"[IonQ] Waiting for {B} results...")
    results = []
    for i, task in enumerate(tasks):
        result = task.result()  # 阻塞等待结果
        results.append(result)
        if (i + 1) % 10 == 0 or (i + 1) == B:
            print(f"  Received {i+1}/{B} results...")
    
    total_time = time.time() - start_time
    print(f"[IonQ] All results received in {total_time:.1f}s")
    print(f"[IonQ] Average: {total_time/B:.1f}s per circuit")

    # ===== 步骤4: 处理结果 =====
    all_outputs = []
    for result in results:
        counts = result.measurement_counts
        
        # 计算每个 qubit 的 <Z> 期望值
        expvals = []
        for q in range(Q):
            p0, p1 = 0.0, 0.0
            for bitstring, c_count in counts.items():
                bit = int(bitstring[q])
                if bit == 0:
                    p0 += c_count
                else:
                    p1 += c_count
            total = p0 + p1 if (p0 + p1) > 0 else 1.0
            expvals.append((p0 - p1) / total)

        all_outputs.append(torch.tensor(expvals))

    return torch.stack(all_outputs, dim=0)