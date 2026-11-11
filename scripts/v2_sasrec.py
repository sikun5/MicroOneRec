#!/usr/bin/env python
"""Phase 5 T5: B1 SASRec adaptation, smoke gates, seed42 training, validation selection.
SCHEMA: phase5-v2_sasrec-1.0

ADAPTATION from existing sasrec.py (SASRecModules_ori.py kept UNTOUCHED as the
traceability baseline; the attention module used here is the BOSS-AUTHORIZED
controlled patch MultiHeadAttentionV2 defined in this file):
- inherited: SASRec block (hidden 32 / 1 head / dropout 0.3 / state_size 10 /
  embedding item_num+1 with padding=item_num), causal tril attention,
  ff_out *= padding mask, BCEWithLogitsLoss with 1 negative per target
  (np.random.randint(item_num) retry != target; real-item domain [0,3686),
  padding excluded by range; numpy legacy global RNG seeded by setup_seed(42)
  - stream recorded), Adam(lr=1e-3, wd=1e-5, eps=1e-8).
- MINIMAL FIX (boss ruling 2026-11-11, technical-correctness blocker): original
  key/query masking guessed padding from sign(abs(sum(row))); LayerNorm rows can
  sum to EXACTLY 0.0 in fp32 and silently masked REAL items (327/2000 rows
  measured). MultiHeadAttentionV2 traces key and query validity to the EXPLICIT
  padding boolean mask (states == item_num). NOT behavior-equivalent to the
  original by design; structure/params/causality/BCE/data/budget unchanged.
- fixed diffs (boss rulings/protocol):
  1. selection metric = validation NDCG@10 (original took ndcg_list[-1]=NDCG@20);
  2. ZERO test evaluation anywhere (original evaluated test every eval round +
     final + saved logits);
  3. wandb fully disabled;
  4. DRO/propensity computation removed (original inactive at alpha=0);
  5. early stop: STOP at the 20th consecutive non-improving validation
     (first non-improvement counts 1; boss ruling fixes the original 21st-return
     off-by-one); tie = non-improvement, earliest best kept (strict >);
  6. data from outputs/v2/data/A/{train,valid}.csv (history_item_id/item_id,
     right-padded with item_num=3686); 36,259 samples/epoch, one target per row,
     no sequence compression;
  7. shuffle via np.random.permutation (original torch DataLoader shuffle) and
     num_workers=0 (original 8): infrastructure only, both seeded/recorded;
  8. forward uses .squeeze(1) (original .squeeze() would break on last batch of
     size 1); validation ranking: rank = 1 + #{logits strictly greater}, ties
     (measure-zero on continuous logits) get the best position - consistent with
     v2_evaluate first-hit semantics;
  9. final predictions: per-row full-catalog ordering via np.argsort(-logits,
     kind='stable') (ties by item id ascending, deterministic), evaluated by the
     shared v2_evaluate layer in STRICT reference mode (ground truth from frozen
     valid.csv, label coverage gate on).
No test contact anywhere. resume_used=false (single uninterrupted run)."""
import os, sys, json, time, math, hashlib, csv, ast, argparse
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np
import torch
from pathlib import Path
REPO = Path("/hpc2hdd/home/swu073/sikun/dsaa5020/MicroOneRec")
sys.path.insert(0, str(REPO))
from SASRecModules_ori import MultiHeadAttention as _OrigMHA


class MultiHeadAttentionV2(_OrigMHA):
    """BOSS-AUTHORIZED MINIMAL FIX (2026-11-11, T5 注意力缺陷裁定): the original
    module guesses padding from `sign(abs(sum(row)))` of raw embeddings/LN
    outputs; LayerNorm rows can sum to EXACTLY 0.0 in fp32, silently masking
    REAL items' attention readout (smoke-measured: 327/2000 rows triggered).
    This controlled patch replaces BOTH zero-sum key and query masks with the
    EXPLICIT padding boolean mask (states == item_num; real item 0 valid,
    padding 3686 invalid). Projections, head split, scaling, tril causality,
    softmax, dropout, weighted sum, restore, residual: line-identical to the
    original module. NOT behavior-equivalent to the original by design."""
    def forward(self, queries, keys, pad_mask=None):
        if pad_mask is None:  # fallback: no padding anywhere (all valid)
            pad_mask = torch.ones(keys.shape[0], keys.shape[1], device=keys.device)
        Q = self.linear_q(queries)
        K = self.linear_k(keys)
        V = self.linear_v(keys)
        split_size = self.hidden_size // self.num_heads
        Q_ = torch.cat(torch.split(Q, split_size, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, split_size, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, split_size, dim=2), dim=0)
        matmul_output = torch.bmm(Q_, K_.transpose(1, 2)) / self.hidden_size ** 0.5
        # Key masking: EXPLICIT padding mask (was zero-sum guess)
        key_mask = pad_mask.repeat(self.num_heads, 1)
        key_mask_reshaped = key_mask.unsqueeze(1).repeat(1, queries.shape[1], 1)
        key_paddings = torch.ones_like(matmul_output) * (-2 ** 32 + 1)
        matmul_output_m1 = torch.where(torch.eq(key_mask_reshaped, 0), key_paddings, matmul_output)
        # Causality - Future Blinding (unchanged)
        diag_vals = torch.ones_like(matmul_output[0, :, :])
        tril = torch.tril(diag_vals)
        causality_mask = tril.unsqueeze(0).repeat(matmul_output.shape[0], 1, 1)
        causality_paddings = torch.ones_like(causality_mask) * (-2 ** 32 + 1)
        matmul_output_m2 = torch.where(torch.eq(causality_mask, 0), causality_paddings, matmul_output_m1)
        matmul_output_sm = self.softmax(matmul_output_m2)
        # Query masking: EXPLICIT padding mask (was zero-sum guess)
        query_mask = pad_mask.repeat(self.num_heads, 1).unsqueeze(-1).repeat(1, 1, keys.shape[1])
        matmul_output_qm = matmul_output_sm * query_mask
        matmul_output_dropout = self.dropout(matmul_output_qm)
        output_ws = torch.bmm(matmul_output_dropout, V_)
        output = torch.cat(torch.split(output_ws, output_ws.shape[0] // self.num_heads, dim=0), dim=2)
        output_res = output + queries
        return output_res

BASE = REPO / "outputs/v2/baselines/B1"
EV = REPO / "outputs/v2/evidence/phase5"
ITEM_NUM, PAD_ID, SEQ_SIZE = 3686, 3686, 10
CFG = {"seed": 42, "hidden": 32, "heads": 1, "dropout": 0.3, "batch_size": 1024,
       "lr": 1e-3, "weight_decay": 1e-5, "adam_eps": 1e-8, "max_epochs": 500,
       "patience": 20, "ks": [3, 5, 10], "state_size": SEQ_SIZE}
HEADER = ["row_id", "user_id", "history_item_title", "item_title",
          "history_item_id", "item_id", "history_item_sid", "item_sid"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def setup_seed(seed):
    import numpy as np, random, torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_split(path):
    rows = []
    with open(path, newline="") as f:
        rd = csv.reader(f)
        assert next(rd) == HEADER
        for r in rd:
            rows.append({"row_id": r[0], "user_id": r[1],
                         "hist": [int(x) for x in ast.literal_eval(r[4])],
                         "target": int(r[5])})
    return rows


def pad_rows(rows):
    """Right-pad histories with PAD_ID to SEQ_SIZE; returns seq list + lens."""
    seqs, lens = [], []
    for r in rows:
        h = r["hist"][:SEQ_SIZE]
        seqs.append(h + [PAD_ID] * (SEQ_SIZE - len(h)))
        lens.append(len(h))
    return seqs, lens


def sample_negatives(targets):
    """1 negative per target; np.random.randint(3686) retry != target (original
    semantics; real items only, padding excluded by range). numpy legacy RNG."""
    negs = []
    for t in targets:
        n = int(np.random.randint(ITEM_NUM))
        while n == t:
            n = int(np.random.randint(ITEM_NUM))
        negs.append(n)
    return negs


def build_model(device):
    import torch
    from torch import nn
    from SASRecModules_ori import PositionwiseFeedForward

    class SASRec(nn.Module):  # structure identical to sasrec.py; attention module = patched V2
        def __init__(self, hidden_size, item_num, state_size, dropout, device, num_heads=1):
            super(SASRec, self).__init__()
            self.state_size = state_size
            self.hidden_size = hidden_size
            self.item_num = int(item_num)
            self.dropout = nn.Dropout(dropout)
            self.device = device
            self.item_embeddings = nn.Embedding(num_embeddings=item_num + 1, embedding_dim=hidden_size)
            nn.init.normal_(self.item_embeddings.weight, 0, 0.01)
            self.positional_embeddings = nn.Embedding(num_embeddings=state_size, embedding_dim=hidden_size)
            self.emb_dropout = nn.Dropout(dropout)
            self.ln_1 = nn.LayerNorm(hidden_size)
            self.ln_2 = nn.LayerNorm(hidden_size)
            self.ln_3 = nn.LayerNorm(hidden_size)
            self.mh_attn = MultiHeadAttentionV2(hidden_size, hidden_size, num_heads, dropout)
            self.feed_forward = PositionwiseFeedForward(hidden_size, hidden_size, dropout)
            self.s_fc = nn.Linear(hidden_size, item_num)

        def forward(self, states, len_states):
            inputs_emb = self.item_embeddings(states)
            inputs_emb += self.positional_embeddings(torch.arange(self.state_size).to(self.device))
            seq = self.emb_dropout(inputs_emb)
            mask = torch.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
            seq *= mask
            seq_normalized = self.ln_1(seq)
            mh_attn_out = self.mh_attn(seq_normalized, seq, mask.squeeze(-1))  # explicit pad mask
            ff_out = self.feed_forward(self.ln_2(mh_attn_out))
            ff_out *= mask
            ff_out = self.ln_3(ff_out)
            indices = (len_states - 1).view(-1, 1, 1).repeat(1, 1, self.hidden_size)
            state_hidden = torch.gather(ff_out, 1, indices)
            supervised_output = self.s_fc(state_hidden).squeeze(1)  # diff 8: explicit dim
            return supervised_output

        def forward_eval(self, states, len_states):
            import torch as _t
            inputs_emb = self.item_embeddings(states)
            inputs_emb += self.positional_embeddings(_t.arange(self.state_size).to(self.device))
            seq = self.emb_dropout(inputs_emb)
            mask = _t.ne(states, self.item_num).float().unsqueeze(-1).to(self.device)
            seq *= mask
            seq_normalized = self.ln_1(seq)
            mh_attn_out = self.mh_attn(seq_normalized, seq, mask.squeeze(-1))  # explicit pad mask
            ff_out = self.feed_forward(self.ln_2(mh_attn_out))
            ff_out *= mask
            ff_out = self.ln_3(ff_out)
            indices = (len_states - 1).view(-1, 1, 1).repeat(1, 1, self.hidden_size)
            state_hidden = _t.gather(ff_out, 1, indices)
            return self.s_fc(state_hidden).squeeze(1)

    return SASRec(CFG["hidden"], ITEM_NUM, SEQ_SIZE, CFG["dropout"], device)


def make_batches(rows, batch_size, shuffle_rng=None):
    idx = list(range(len(rows)))
    if shuffle_rng is not None:
        idx = shuffle_rng.permutation(len(rows)).tolist()
    return [idx[i:i + batch_size] for i in range(0, len(rows), batch_size)]


def eval_valid(model, valid_rows, device):
    """Training-time validation NDCG@10 (selection metric) + HR@10 log.
    rank = 1 + #{logits strictly greater} (ties -> best position, measure-zero)."""
    import torch
    import numpy as np
    model.eval()
    seqs, lens = pad_rows(valid_rows)
    ndcgs, hrs = [], []
    with torch.no_grad():
        for b in make_batches(valid_rows, CFG["batch_size"]):
            seq = torch.LongTensor([seqs[i] for i in b]).to(device)
            ln = torch.LongTensor([lens[i] for i in b]).to(device)
            logits = model.forward_eval(seq, ln)
            tgts = torch.LongTensor([valid_rows[i]["target"] for i in b]).to(device)
            tgt_scores = torch.gather(logits, 1, tgts.view(-1, 1))
            ranks = (logits > tgt_scores).sum(1).float() + 1.0
            for r in ranks.tolist():
                ndcgs.append(1.0 / math.log2(r + 1) if r <= 10 else 0.0)
                hrs.append(1.0 if r <= 10 else 0.0)
    model.train()
    return float(np.mean(ndcgs)), float(np.mean(hrs))


def smoke():
    import torch
    import numpy as np
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    res = {}
    setup_seed(CFG["seed"])
    # S1 causality on the PATCHED module: perturbing the LAST position must leave
    # earlier output positions bit-identical (tril future blinding preserved).
    mha = MultiHeadAttentionV2(CFG["hidden"], CFG["hidden"], 1, 0.0).to(device).eval()
    with torch.no_grad():
        x1 = torch.randn(1, SEQ_SIZE, CFG["hidden"], device=device)
        x2 = x1.clone()
        x2[0, -1] = torch.randn(CFG["hidden"], device=device)
        o1, o2 = mha(x1, x1), mha(x2, x2)
        res["causal_prefix_invariant"] = {
            "prefix_0_to_8_identical": torch.equal(o1[0, :-1], o2[0, :-1]),
            "last_position_differs": not torch.equal(o1[0, -1], o2[0, -1])}
    # S2 full model: padding isolation + DEFECT-GONE verification + BCE labels + grads
    model = build_model(device).to(device).eval()
    with torch.no_grad():
        seq_a = torch.LongTensor([[5, 0] + [PAD_ID] * 8]).to(device)
        ln_a = torch.LongTensor([2]).to(device)
        out_a1 = model.forward_eval(seq_a, ln_a)
        w_pad = model.item_embeddings.weight[PAD_ID].clone()
        model.item_embeddings.weight[PAD_ID] = w_pad + 99.0
        out_a2 = model.forward_eval(seq_a, ln_a)
        model.item_embeddings.weight[PAD_ID] = w_pad
        # DEFECT-GONE check: perturbing a NON-gather real position's embedding
        # (position 0, item 5) must now change the gathered output; for the seed42
        # init this was bitwise-invisible under the original zero-sum mask because
        # LN rows at valid positions summed to exactly 0.0.
        w5 = model.item_embeddings.weight[5].clone()
        shift5 = torch.zeros_like(w5); shift5[3] = 99.0
        model.item_embeddings.weight[5] = w5 + shift5
        out_a3 = model.forward_eval(seq_a, ln_a)
        model.item_embeddings.weight[5] = w5
        # LN row-sum evidence at valid positions (context of the defect)
        emb_ = model.item_embeddings(seq_a) + model.positional_embeddings(torch.arange(SEQ_SIZE).to(device))
        sn_ = model.ln_1(model.emb_dropout(emb_) * torch.ne(seq_a, PAD_ID).float().unsqueeze(-1))
        ln_sums = sn_.sum(dim=-1)[0]
        # key mask blocks padding: softmax probs over pad keys must be exactly 0
        cap = {}
        h = model.mh_attn.softmax.register_forward_hook(
            lambda m, i, o: cap.__setitem__("p", o.detach().clone()))
        _ = model.forward_eval(seq_a, ln_a)
        h.remove()
        pad_probs = cap["p"][0, 1, 2:]  # gather position 1, keys 2..9 are padding
        out_b1 = model.forward_eval(torch.LongTensor([[5, 0] + [PAD_ID] * 8]).to(device),
                                    torch.LongTensor([2]).to(device))
        out_b2 = model.forward_eval(torch.LongTensor([[5] + [PAD_ID] * 9]).to(device),
                                    torch.LongTensor([1]).to(device))
    res["padding_isolation"] = {
        "pad_embedding_perturbation_inert": torch.equal(out_a1, out_a2),
        "defect_gone_non_gather_real_perturbation_changes_output": not torch.equal(out_a1, out_a3),
        "attention_mixing_restored_maxdiff": float((out_a3 - out_a1).abs().max()),
        "key_mask_probs_on_padding_exactly_zero": bool((pad_probs == 0).all()),
        "valid_position_ln_rows_summing_exactly_0_in_this_draw":
            [float(x) for x in ln_sums[:2]],
        "item0_real_vs_truncated_differ": not torch.equal(out_b1, out_b2)}
    # S3 negative sampling: 50k draws vs fixed target
    np.random.seed(CFG["seed"])
    tgt = 7
    draws = []
    for _ in range(50000):
        n = int(np.random.randint(ITEM_NUM))
        while n == tgt:
            n = int(np.random.randint(ITEM_NUM))
        draws.append(n)
    res["negative_sampling"] = {"collisions_with_target": draws.count(tgt),
                                "out_of_domain": sum(1 for d in draws if not 0 <= d < ITEM_NUM),
                                "mean": float(np.mean(draws)),
                                "expected_mean_uniform": (ITEM_NUM - 1) / 2,
                                "rng": "numpy legacy global (np.random.seed(42))"}
    # S4 early-stop state machine (exact training bookkeeping)
    def bookkeeping(series):
        best, best_epoch, counter, stop_at = -1.0, -1, 0, None
        for ep, v in enumerate(series):
            if v > best:
                best, best_epoch, counter = v, ep, 0
            else:
                counter += 1
                if counter >= CFG["patience"]:
                    stop_at = ep
                    break
        return best_epoch, stop_at, counter
    be, sa, cf = bookkeeping([0.10, 0.30, 0.35] + [0.30] * 30)
    be2, sa2, cf2 = bookkeeping([0.35, 0.35, 0.20] + [0.20] * 25)
    res["early_stop_state_machine"] = {
        "best_epoch": be, "stopped_at_epoch": sa, "final_counter": cf,
        "expected": "best_epoch=2, stop at 20th consecutive non-improving eval (epoch 22, NOT 23)",
        "tie_best_epoch": be2, "tie_expected": "tie is non-improvement; earliest best (epoch 0) kept"}
    # S5 BCE labels construction + finite loss AND finite gradients on 3 real batches
    train_rows = load_split(REPO / "outputs/v2/data/A/train.csv")
    batches = make_batches(train_rows, CFG["batch_size"])[:3]
    model_t = build_model(device).to(device)
    opt = torch.optim.Adam(model_t.parameters(), lr=CFG["lr"], eps=CFG["adam_eps"],
                           weight_decay=CFG["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()
    finite, losses, grad_ok, labels_ok = True, [], True, True
    for b in batches:
        rows_b = [train_rows[i] for i in b]
        seqs, lens = pad_rows(rows_b)
        tgts = [r["target"] for r in rows_b]
        negs = sample_negatives(tgts)
        seq = torch.LongTensor(seqs).to(device)
        ln = torch.LongTensor(lens).to(device)
        out = model_t(seq, ln)
        t_idx = torch.LongTensor(tgts).view(-1, 1).to(device)
        n_idx = torch.LongTensor(negs).view(-1, 1).to(device)
        scores = torch.cat((torch.gather(out, 1, t_idx), torch.gather(out, 1, n_idx)), 0)
        labels_ = torch.cat((torch.ones(len(b), 1), torch.zeros(len(b), 1)), 0).to(device)
        labels_ok &= bool((labels_[:len(b)] == 1).all()) and bool((labels_[len(b):] == 0).all())
        loss = loss_fn(scores, labels_)
        opt.zero_grad()
        loss.backward()
        norms = {n: float(p.grad.norm()) for n, p in model_t.named_parameters() if p.grad is not None}
        grad_ok &= len(norms) == len(list(model_t.named_parameters())) and all(
            math.isfinite(v) and v > 0 for v in norms.values())
        opt.step()
        losses.append(float(loss))
        finite &= math.isfinite(loss.item()) and loss.item() >= 0
    res["bce_and_gradients"] = {"losses": losses, "all_finite_nonneg": finite,
                                "labels_pos1_neg0_exact": labels_ok,
                                "all_param_grads_finite_nonzero": grad_ok}
    res["device"] = str(device)
    ok = (res["causal_prefix_invariant"]["prefix_0_to_8_identical"]
          and res["causal_prefix_invariant"]["last_position_differs"]
          and res["padding_isolation"]["pad_embedding_perturbation_inert"]
          and res["padding_isolation"]["defect_gone_non_gather_real_perturbation_changes_output"]
          and res["padding_isolation"]["key_mask_probs_on_padding_exactly_zero"]
          and res["padding_isolation"]["item0_real_vs_truncated_differ"]
          and res["negative_sampling"]["collisions_with_target"] == 0
          and res["negative_sampling"]["out_of_domain"] == 0
          and res["early_stop_state_machine"]["best_epoch"] == 2
          and res["early_stop_state_machine"]["stopped_at_epoch"] == 22
          and res["early_stop_state_machine"]["tie_best_epoch"] == 0
          and res["bce_and_gradients"]["all_finite_nonneg"]
          and res["bce_and_gradients"]["labels_pos1_neg0_exact"]
          and res["bce_and_gradients"]["all_param_grads_finite_nonzero"])
    (EV / "t5_smoke_checks.json").write_text(json.dumps(
        {"schema": "phase5-t5-smoke-1.1", "date": "2026-11-11", "all_pass": ok,
         "fix": "MultiHeadAttentionV2 explicit padding mask (boss-authorized minimal fix 2026-11-11)",
         "code_sha256": sha256_file(__file__), "results": res}, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1)[:2600], flush=True)
    print("T5_SMOKE_SHA256:", sha256_file(EV / "t5_smoke_checks.json"), flush=True)
    print("T5_SMOKE_ALL_PASS:", ok, flush=True)
    print("T5_SMOKE_DONE", flush=True)


def train():
    import torch
    from torch import nn
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_seed(CFG["seed"])
    BASE.mkdir(parents=True, exist_ok=True)
    train_rows = load_split(REPO / "outputs/v2/data/A/train.csv")
    valid_rows = load_split(REPO / "outputs/v2/data/A/valid.csv")
    assert len(train_rows) == 36259 and len(valid_rows) == 4532
    model = build_model(device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["lr"], eps=CFG["adam_eps"],
                           weight_decay=CFG["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_epoch, counter = -1.0, -1, 0
    stop_reason = None
    log_path = BASE / "training_log.jsonl"
    log_f = open(log_path, "w")
    t0 = time.time()
    meta = {"resume_used": False, "run_kind": "formal", "seed": CFG["seed"]}
    for ep in range(CFG["max_epochs"]):
        ep_t0 = time.time()
        model.train()
        losses = []
        for b in make_batches(train_rows, CFG["batch_size"], shuffle_rng=np.random):
            rows_b = [train_rows[i] for i in b]
            seqs, lens = pad_rows(rows_b)
            tgts = [r["target"] for r in rows_b]
            negs = sample_negatives(tgts)
            seq = torch.LongTensor(seqs).to(device)
            ln = torch.LongTensor(lens).to(device)
            out = model(seq, ln)
            t_idx = torch.LongTensor(tgts).view(-1, 1).to(device)
            n_idx = torch.LongTensor(negs).view(-1, 1).to(device)
            scores = torch.cat((torch.gather(out, 1, t_idx), torch.gather(out, 1, n_idx)), 0)
            labels_ = torch.cat((torch.ones(len(b), 1), torch.zeros(len(b), 1)), 0).to(device)
            loss = loss_fn(scores, labels_)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss))
            if not math.isfinite(loss.item()):
                stop_reason = f"non-finite loss at epoch {ep}"
                break
        if stop_reason:
            break
        ndcg10, hr10 = eval_valid(model, valid_rows, device)
        improved = ndcg10 > best
        if improved:
            best, best_epoch, counter = ndcg10, ep, 0
            torch.save({"state_dict": model.state_dict(),
                        "meta": {"epoch": ep, "val_ndcg10": ndcg10, "val_hr10": hr10,
                                 "config": CFG, "resume_used": False}},
                       BASE / "best_checkpoint.pth")
        else:
            counter += 1
        rec = {"epoch": ep, "train_loss_mean": float(np.mean(losses)),
               "val_ndcg10": ndcg10, "val_hr10": hr10, "improved": improved,
               "best_so_far": best, "best_epoch": best_epoch,
               "non_improve_counter": counter, "epoch_wall_s": round(time.time() - ep_t0, 2)}
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()
        print("EPOCH", json.dumps(rec), flush=True)
        if counter >= CFG["patience"]:
            stop_reason = f"early stop: {CFG['patience']} consecutive non-improving validations (patience={CFG['patience']})"
            break
    wall = time.time() - t0
    log_f.close()
    summary = {"schema": "phase5-b1-train-summary-1.0", "date": "2026-11-11",
               "stop_reason": stop_reason or f"max epochs {CFG['max_epochs']} reached",
               "epochs_run": ep + 1, "best_epoch": best_epoch, "best_val_ndcg10": best,
               "resume_used": False, "wall_s": round(wall, 1),
               "peak_mem_allocated_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
               "device": torch.cuda.get_device_properties(0).name if device.type == "cuda" else "cpu",
               "config": CFG,
               "inputs_sha256": {p: sha256_file(REPO / p) for p in
                                 ("outputs/v2/data/A/train.csv", "outputs/v2/data/A/valid.csv")},
               "code_sha256": sha256_file(__file__),
               "neg_sampling": "np.random.randint(3686) retry != target; numpy legacy global RNG seeded by setup_seed(42); shuffle via np.random.permutation; num_workers=0",
               "history": "training ran uninterrupted (single process); resume_used=false"}
    (BASE / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print("TRAIN_SUMMARY:", json.dumps({k: summary[k] for k in
          ("stop_reason", "epochs_run", "best_epoch", "best_val_ndcg10", "wall_s", "peak_mem_allocated_bytes")}),
          flush=True)
    print("T5_TRAIN_SHA256:", sha256_file(BASE / "train_summary.json"), flush=True)
    print("T5_TRAIN_DONE", flush=True)


def finalize():
    """Fresh process: reload best checkpoint -> per-row full ordering -> shared
    evaluator (strict reference) -> metrics; double-eval reload consistency."""
    import torch
    import numpy as np
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    setup_seed(CFG["seed"])
    ckpt = torch.load(BASE / "best_checkpoint.pth", map_location=device, weights_only=False)
    model = build_model(device).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    valid_rows = load_split(REPO / "outputs/v2/data/A/valid.csv")
    catalog = [int(x) for x in json.load(open(REPO / "outputs/v2/data/A/Industrial_and_Scientific.index.json"))["item_ids"]]
    assert catalog == list(range(ITEM_NUM))
    seqs, lens = pad_rows(valid_rows)

    def predict_all():
        outs = []
        with torch.no_grad():
            for b in make_batches(valid_rows, CFG["batch_size"]):
                seq = torch.LongTensor([seqs[i] for i in b]).to(device)
                ln = torch.LongTensor([lens[i] for i in b]).to(device)
                outs.append(model.forward_eval(seq, ln).detach().cpu().numpy())
        return np.concatenate(outs, 0)

    logits1 = predict_all()
    orderings = np.argsort(-logits1, axis=1, kind="stable")
    # reload consistency: second evaluation pass must reproduce identical predictions
    logits2 = predict_all()
    orderings2 = np.argsort(-logits2, axis=1, kind="stable")
    consistency = bool(np.array_equal(orderings, orderings2))
    with open(BASE / "valid_predictions.jsonl", "w") as f:
        for i, r in enumerate(valid_rows):
            f.write(json.dumps({"row_id": r["row_id"], "user_id": r["user_id"],
                                "target_item_id": r["target"],
                                "ranked_item_ids": [int(x) for x in orderings[i]]}) + "\n")
    # shared evaluator, strict reference from frozen valid.csv
    sys.path.insert(0, str(REPO / "scripts"))
    from v2_evaluate import validate_and_align, evaluate_rows
    labels = {}
    for line in open(REPO / "outputs/v2/manifests/quality_features.jsonl"):
        d = json.loads(line)
        labels[int(d["item_id"])] = bool(d["original_description_missing"])
    reference = {r["row_id"]: {"user_id": r["user_id"], "target_item_id": r["target"]} for r in valid_rows}
    pred_rows = [{"row_id": r["row_id"], "user_id": r["user_id"], "target_item_id": r["target"],
                  "ranked_item_ids": [int(x) for x in orderings[i]]} for i, r in enumerate(valid_rows)]
    aligned = validate_and_align(pred_rows, [r["row_id"] for r in valid_rows], set(catalog), reference)
    metrics = evaluate_rows(aligned, set(catalog), tuple(CFG["ks"]), labels)
    summary = json.load(open(BASE / "train_summary.json"))
    out = {"schema": "phase5-b1-valid-metrics-1.0", "date": "2026-11-11",
           "config": {"checkpoint_sha256": sha256_file(BASE / "best_checkpoint.pth"),
                      "predictions_sha256": sha256_file(BASE / "valid_predictions.jsonl"),
                      "best_epoch": ckpt["meta"]["epoch"],
                      "best_val_ndcg10_train_log": ckpt["meta"]["val_ndcg10"],
                      "checkpoint_meta": ckpt["meta"],
                      "reload_consistency_predictions_identical": consistency,
                      "valid_csv_sha256": sha256_file(REPO / "outputs/v2/data/A/valid.csv"),
                      "catalog_sha256": sha256_file(REPO / "outputs/v2/data/A/Industrial_and_Scientific.index.json"),
                      "quality_features_sha256": sha256_file(REPO / "outputs/v2/manifests/quality_features.jsonl"),
                      "evaluator_code_sha256": sha256_file(REPO / "scripts/v2_evaluate.py"),
                      "finalizer_code_sha256": sha256_file(__file__),
                      "reference_mode": True, "n_rows": len(valid_rows),
                      "n_users": metrics["user_macro"]["n_users"]},
           "metrics": metrics}
    (BASE / "valid_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    cfg_out = {"schema": "phase5-b1-config-1.0", "date": "2026-11-11", "method": "B1-SASRec",
               "config": CFG, "train_summary": summary,
               "products_sha256": {p: sha256_file(BASE / p) for p in
                                   ("best_checkpoint.pth", "training_log.jsonl",
                                    "valid_predictions.jsonl", "valid_metrics.json",
                                    "train_summary.json")},
               "command_chain": ["--mode smoke", "--mode train", "--mode finalize"],
               "resume_used": False}
    (BASE / "config.json").write_text(json.dumps(cfg_out, ensure_ascii=False, indent=1))
    print("FINALIZE consistency:", consistency, flush=True)
    print("FINALIZE ndcg10:", metrics["ndcg@10"], "hr10:", metrics["hr@10"], flush=True)
    print("FINALIZE macro_ndcg10:", metrics["user_macro"]["ndcg@10"], flush=True)
    print("FINALIZE coverage:", metrics["coverage@10"]["value"], "illegal:", metrics["illegal_rate_global"], flush=True)
    print("FINALIZE slices:", json.dumps(metrics["slices"], ensure_ascii=False)[:260], flush=True)
    print("METRICS_SHA256:", sha256_file(BASE / "valid_metrics.json"), flush=True)
    print("CONFIG_SHA256:", sha256_file(BASE / "config.json"), flush=True)
    print("PREDICTIONS_SHA256:", sha256_file(BASE / "valid_predictions.jsonl"), flush=True)
    print("CHECKPOINT_SHA256:", sha256_file(BASE / "best_checkpoint.pth"), flush=True)
    print("T5_FINALIZE_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "train", "finalize"], required=True)
    args = ap.parse_args()
    {"smoke": smoke, "train": train, "finalize": finalize}[args.mode]()


if __name__ == "__main__":
    main()
