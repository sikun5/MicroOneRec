import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch, json, hashlib
dev = torch.device("cuda:0")
torch.manual_seed(123)
torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True)
def h(t): return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:12]
g0 = torch.Generator(device="cpu"); g0.manual_seed(7)
p0 = torch.randn(64, generator=g0).to(dev)
g_fixed = torch.randn(64, generator=g0).to(dev)
ea0 = (torch.randn(64, generator=g0) * 0.1).to(dev)
eas0 = (torch.rand(64, generator=g0) * 0.1).to(dev)
res = {}
# A: fresh optimizer each trial
outs = []
for t in range(5):
    p = torch.nn.Parameter(p0.clone())
    opt = torch.optim.AdamW([p], lr=1e-3, weight_decay=0.0)
    opt.state[p]["step"] = torch.tensor(40.0)
    opt.state[p]["exp_avg"] = ea0.clone()
    opt.state[p]["exp_avg_sq"] = eas0.clone()
    before = h(p)
    p.grad = g_fixed.clone()
    opt.step()
    outs.append((before, h(p), h(opt.state[p]["exp_avg"]), h(opt.state[p]["exp_avg_sq"])))
res["A_fresh_optimizer"] = {"trials": outs, "all_equal": len(set(outs)) == 1}
# B: reused optimizer with in-place state restore (harness-style)
p = torch.nn.Parameter(p0.clone())
opt = torch.optim.AdamW([p], lr=1e-3, weight_decay=0.0)
SNAP_W = p0.clone(); SNAP_EA = ea0.clone(); SNAP_EAS = eas0.clone()
outs = []
for t in range(5):
    with torch.no_grad(): p.copy_(SNAP_W)
    opt.state[p]["step"] = torch.tensor(40.0)
    opt.state[p]["exp_avg"] = SNAP_EA.clone()
    opt.state[p]["exp_avg_sq"] = SNAP_EAS.clone()
    before = h(p)
    p.grad = g_fixed.clone()
    opt.step()
    outs.append((before, h(p), h(opt.state[p]["exp_avg"])))
res["B_reused_inplace_restore"] = {"trials": outs, "all_equal": len(set(outs)) == 1}
# C: production-style load_state_dict restore
p = torch.nn.Parameter(p0.clone())
opt = torch.optim.AdamW([p], lr=1e-3, weight_decay=0.0)
# build a realistic state snapshot via one real step
opt.state[p]["step"] = torch.tensor(40.0)
opt.state[p]["exp_avg"] = ea0.clone()
opt.state[p]["exp_avg_sq"] = eas0.clone()
SNAP_OPT = json.loads(json.dumps(opt.state_dict())) if False else None
SNAP_OPT = {"state": {id(p) if False else 0: {"step": torch.tensor(40.0).clone(), "exp_avg": ea0.clone(), "exp_avg_sq": eas0.clone()}},
            "param_groups": [{"lr": 1e-3, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0,
                              "amsgrad": False, "foreach": None, "maximize": False, "capturable": False,
                              "differentiable": False, "fused": None, "params": [0]}]}
# remap param index to the actual param object key
p2 = torch.nn.Parameter(p0.clone())
opt2 = torch.optim.AdamW([p2], lr=1e-3, weight_decay=0.0)
outs = []
for t in range(5):
    with torch.no_grad(): p2.copy_(SNAP_W)
    opt2.load_state_dict({"state": {0: SNAP_OPT["state"][0]}, "param_groups": SNAP_OPT["param_groups"]})
    before = h(p2)
    p2.grad = g_fixed.clone()
    opt2.step()
    outs.append((before, h(p2), h(opt2.state[p2]["exp_avg"])))
res["C_load_state_dict_restore"] = {"trials": outs, "all_equal": len(set(outs)) == 1}
res["summary"] = {k: v["all_equal"] for k, v in res.items() if isinstance(v, dict)}
print(json.dumps(res, indent=1))
print("E7_DONE")
