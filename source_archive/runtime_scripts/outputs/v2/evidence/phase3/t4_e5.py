import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch, json
dev = torch.device("cuda:0")
torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True)
try:
    torch.backends.cuda.preferred_blas_library("cublas")
except Exception:
    pass
res = {}
# GEMM with tensors at address A1
a1 = torch.randn(3686, 2560, device=dev)
w1 = torch.randn(32, 2560, device=dev)
r1 = (a1 @ w1.t()).clone()
# churn allocator: allocate/free junk of varied sizes
for sz in [(1024, 4096), (7, 333333), (512, 2048)]:
    junk = torch.randn(*sz, device=dev); del junk
# SAME VALUES at (likely) different addresses
a2 = a1.clone(); w2 = w1.clone()
r2 = (a2 @ w2.t()).clone()
res["gemm_addr_dependent"] = bool(not torch.equal(r1, r2))
res["gemm_max_diff"] = float((r1 - r2).abs().max())
# same addresses back-to-back (reuse a1,w1)
r3 = (a1 @ w1.t()).clone()
res["gemm_same_addr_repro"] = bool(torch.equal(r1, r3))
# distance-matrix scale repro with fresh allocations mimicking per-call pattern
d1 = (a1 * a1).sum(1, keepdim=True) + (w1 * w1).sum(1).unsqueeze(0) - 2 * (a1 @ w1.t())
for sz in [(3, 999999)]:
    junk = torch.randn(*sz, device=dev); del junk
d2 = (a2 * a2).sum(1, keepdim=True) + (w2 * w2).sum(1).unsqueeze(0) - 2 * (a2 @ w2.t())
res["dist_addr_dependent"] = bool(not torch.equal(d1, d2))
res["dist_max_diff"] = float((d1 - d2).abs().max())
res["ptr_a1_a2_diff"] = a1.data_ptr() != a2.data_ptr()
print(json.dumps(res, indent=1))
print("E5_DONE")
