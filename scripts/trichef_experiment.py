r"""Tri-CHEF 문서 검색 파이프라인으로 매핑 후보를 찾는 실험 (단계별 누적).

실행 (retriever-eval 가상환경, GPU):
    .\.venv\Scripts\python.exe scripts\trichef_experiment.py "C:\공모전\충파모\data_raw\kaggle_product_demand\Historical Product Demand.csv"

DB_insight (커밋 6a50976) App/backend/services/trichef/unified_engine.py 의 search() 를 따른다.
    Re = SigLIP2 텍스트 인코더 (google/siglip2-so400m-patch16-naflex), Im = BGE-M3 dense (BAAI/bge-m3)
    Z = 0 (텍스트 질의에서는 Z 채널을 끈다: 원 코드 [Phase D]. DINOv2 는 이미지 전용)
    dense = sqrt(Re² + (α·Im)²), 문서 도메인 α = 0.8 (SEARCH_IM_ALPHA_DOC 기본값)
    질의 확장: [원문, expand_bilingual(원문), 그중 영문 토큰만] 임베딩 평균 후 정규화
    문자열 일치 가산(substring boost): 확장 질의 토큰이 문서 id 에 있으면 +0.05/개(+0.10 전부 일치), 본문에 원문 토큰 전부 +0.12 / 일부 +0.03
    lexical: BGE-M3 sparse 가중치 내적 (확장 질의)
    ASF: auto_vocab + asf_filter (DB_insight 기본값은 꺼짐)
    순위: 채널별 min-max 후 0.60·dense + 0.25·lexical + 0.15·asf (없는 채널 가중치는 dense 로)
    판정: dense ≥ τ 인 후보 중 융합 점수 1순위. 없으면 보류.
expand_bilingual, build_vocab, build_doc_token_sets, asf_scores 는 DB_insight 모듈을 그대로 import 해서 실행한다 (복사하지 않음).

과제
    A   열 이름 → 필드 (합성 평가셋, data/header_queries.csv)
    B   품목 표기 → 마스터 (합성 평가셋, data/value_queries.csv, master_items.csv)
    K2  제품 코드 → Kaggle 실제 제품 2,160개 (질의 표기는 합성 변형)
    K3  분류 코드 → Kaggle 실제 분류 33개 (질의 표기는 합성 변형)

단계 (앞 단계에 하나씩 더한다)
    s0_im  Im 만 / s1_tc2_a04  2축 α=0.4 / s2_tc2_a08  2축 α=0.8 / s3_expand  + 질의 확장
    s4_boost  + 문자열 일치 가산 / s5_sparse  + sparse 결합 (= DB_insight 문서 기본 구성) / s6_asf  + ASF
기준선 두 가지: dev 에서 정확도가 가장 높은 τ / Tri-CHEF 식 보정 τ = μ_null + 1.645σ_null (FAR 0.05).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import pathlib
import re
import statistics
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "results"
RES.mkdir(exist_ok=True)
DBI = pathlib.Path(r"C:\Honey\DB_insight\App\backend")
sys.path.insert(0, str(DBI))
from services.query_expand import expand_bilingual  # noqa: E402
from services.trichef import asf_filter, auto_vocab  # noqa: E402

STAGES = ["s0_im", "s1_tc2_a04", "s2_tc2_a08", "s3_expand", "s4_boost", "s5_sparse", "s6_asf"]
FULL = "s5_sparse"
DOC_ALPHA, FUSION_W, FAR_Z = 0.8, (0.60, 0.25, 0.15), 1.6449
KO_SUFFIXES = ("하는", "했던", "하고", "하기", "스러운", "되는", "된", "하던", "한", "인", "는", "을", "를", "의", "이")


# ───────────────────────── 모델 ─────────────────────────
class BgeM3:
    name = "BAAI/bge-m3"

    def __init__(self, device):
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModel.from_pretrained(self.name).eval().to(device)
        self.lin = torch.nn.Linear(self.model.config.hidden_size, 1)
        self.lin.load_state_dict(torch.load(hf_hub_download(self.name, "sparse_linear.pt"), map_location="cpu"))
        self.lin.to(device)
        self.special = {i for i in (self.tok.cls_token_id, self.tok.sep_token_id, self.tok.pad_token_id, self.tok.unk_token_id) if i is not None}
        self.device = device

    @torch.no_grad()
    def encode(self, texts, bs=64):
        """dense(L2 정규화) + sparse 원 가중치 (DB_insight lexical_scores 와 같은 내적용, 정규화하지 않음)."""
        dense, sparse = [], []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.device)
            h = self.model(**enc).last_hidden_state
            dense.append(F.normalize(h[:, 0], dim=-1).float().cpu().numpy())
            w = torch.relu(self.lin(h)).squeeze(-1).float().cpu().numpy()
            ids, mask = enc["input_ids"].cpu().numpy(), enc["attention_mask"].cpu().numpy()
            for r in range(ids.shape[0]):
                d: dict[int, float] = {}
                for t, x, m in zip(ids[r], w[r], mask[r]):
                    if m and int(t) not in self.special and x > d.get(int(t), 0.0):
                        d[int(t)] = float(x)
                sparse.append(d)
        return np.concatenate(dense), sparse


class Siglip2Text:
    """Tri-CHEF Re 축. DB_insight siglip2_re.embed_texts 와 같은 방식: padding=max_length, get_text_features, L2 정규화, CUDA 는 FP16."""
    name = "google/siglip2-so400m-patch16-naflex"

    def __init__(self, device):
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModel.from_pretrained(self.name, dtype=torch.float16 if device == "cuda" else torch.float32).to(device).eval()
        self.device = device

    @torch.no_grad()
    def encode(self, texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            enc = self.tok(texts[i:i + bs], padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(self.device)
            v = self.model.get_text_features(**enc)
            if not torch.is_tensor(v):
                v = v.pooler_output
            out.append(F.normalize(v.float(), dim=-1).cpu().numpy())
        return np.concatenate(out)


# ───────────────────────── 과제 ─────────────────────────
def read_csv(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def split_of(key: str) -> str:
    return "dev" if int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 10 < 4 else "test"


def synthetic_tasks():
    fields = json.loads((DATA / "fields.json").read_text(encoding="utf-8"))
    hq, vq, master = read_csv("header_queries.csv"), read_csv("value_queries.csv"), read_csv("master_items.csv")
    docs_a = {e: [(f, spec["strong"] + spec["weak"] + [spec["desc"], f]) for f, spec in fs.items()] for e, fs in fields.items()}
    qa = [{"text": r["header"], "scope": r["entity"], "gold": r["gold"], "accept": [], "category": r["category"], "split": r["split"],
           "rule_pred": r["rule_pred"]} for r in hq]
    docs_b = {"all": [(m["code"], [m["code"], m["name"], f'{m["code"]} {m["name"]}']) for m in master]}
    qb = [{"text": r["query"], "scope": "all", "gold": r["gold"], "accept": [c for c in r["accept"].split(",") if c], "category": r["category"],
           "split": r["split"], "rule_pred": r["rule_pred"]} for r in vq]
    return {"A": (qa, docs_a), "B": (qb, docs_b)}


def load_kaggle(path):
    sha = hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
    prod, cat, n = set(), set(), 0
    with open(path, encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        for row in r:
            n += 1
            prod.add(row[0])
            cat.add(row[2])
    return sorted(prod), sorted(cat), {"sha256": sha, "rows": n, "header": header, "n_products": len(prod), "n_categories": len(cat)}


def code_formats(en, ko, num, width):
    z = f"{num:0{width}d}"
    return {"case": f"{en.lower()}_{z}", "drop_zero": f"{en}_{num}", "hyphen_upper": f"{en.upper()}-{z}", "space_short": f"{en[:4]} {z}",
            "initial": f"{en[0]}{z}", "number_only": z, "number_nozero": str(num), "no_sep": f"{en}{z}", "korean": f"{ko}_{z}"}


def kaggle_task(name, master, en, ko, width, n_pick, trap_nums):
    """실제 코드에 다른 업체식 표기 9종을 적용한 질의 + 데이터에 없는 번호(함정). 난수 없음."""
    nums = {int(re.sub(r"\D", "", c)): c for c in master}
    ms = set(master)
    picks = sorted(nums, key=lambda k: hashlib.sha256(f"{name}|{k}".encode()).hexdigest())[:n_pick]
    raw = [(text, nums[k], fmt, k) for k in picks for fmt, text in code_formats(en, ko, k, width).items()]
    for k in trap_nums:
        assert k not in nums
        f = code_formats(en, ko, k, width)
        raw += [(f"{en}_{k:0{width}d}", "NONE", "trap_exact_form", k), (f["drop_zero"], "NONE", "trap_drop_zero", k), (f["initial"], "NONE", "trap_initial", k)]
    seen, out = set(), []
    for text, gold, cat, k in raw:
        if text in seen:
            continue
        seen.add(text)
        vk = [c for c in master if c.casefold().lstrip("0") == text.casefold().lstrip("0")]       # 백엔드 _variant_key 와 같은 식
        groups = re.findall(r"\d+", text)
        out.append({"text": text, "scope": "all", "gold": gold, "accept": [], "category": cat, "split": split_of(f"{name}|{k}"),
                    "rule_pred": text if text in ms else (vk[0] if len(vk) == 1 else ""),
                    "digits_pred": nums.get(int(groups[0]), "") if len(groups) == 1 else ""})
    return out, {"all": [(c, [c]) for c in master]}


# ───────────────────────── Tri-CHEF 검색 절차 ─────────────────────────
def variants_of(q):
    bil = expand_bilingual(q)
    v = [q]
    if bil != q:
        v.append(bil)
    en = " ".join(t for t in bil.split() if not re.search(r"[가-힣]", t)).strip()
    if en and en != q and en not in v:
        v.append(en)
    return v, bil


def avg_norm(M):
    v = M.astype(np.float32).mean(axis=0)
    return v / (float(np.linalg.norm(v)) + 1e-12)


def minmax_fusion(dense, lex, asf):
    def norm(x):
        mn, mx = float(x.min()), float(x.max())
        return np.full_like(x, 0.5) if mx - mn < 1e-6 else (x - mn) / (mx - mn + 1e-9)
    nd = norm(dense)
    out = FUSION_W[0] * nd
    out = out + FUSION_W[1] * (norm(lex) if lex is not None and float(lex.max()) > 0 else nd)
    out = out + FUSION_W[2] * (norm(asf) if asf is not None and float(asf.max()) > 0 else nd)
    return out


def boost_vec(query, bil, doc_ids, doc_caps):
    q_tokens = [t.lower() for t in bil.split() if len(t) >= 2]
    orig = [t.lower() for t in query.split() if len(t) >= 2]
    n_q = len(orig)
    b = np.zeros(len(doc_ids), dtype=np.float32)
    if not q_tokens:
        return b
    cands = [[ot] + [ot[:-len(s)] for s in KO_SUFFIXES if ot.endswith(s) and len(ot) - len(s) >= 2] for ot in orig]
    for i, (rid, cap) in enumerate(zip(doc_ids, doc_caps)):
        hay = rid.lower()
        m = sum(1 for tok in q_tokens if tok in hay)
        if m > 0:
            b[i] += 0.05 * m
            if m >= n_q:
                b[i] += 0.10
        if cap:
            hit = sum(1 for c in cands if any(x in cap for x in c))
            if n_q >= 2 and hit >= n_q:
                b[i] += 0.12
            elif hit >= 1:
                b[i] += 0.03
    return b


def sdot(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(x * b[k] for k, x in a.items() if k in b)


def score_task(queries, docs, E):
    scope = {}
    for sc, ds in docs.items():
        texts = [t for _, ts in ds for t in ts]
        joined = [" ".join(ts) for _, ts in ds]
        vocab = auto_vocab.build_vocab(joined, min_df=2, max_df_ratio=0.4)
        scope[sc] = {"ids": [d for d, _ in ds], "caps": [j.lower() for j in joined], "n": len(ds),
                     "owner": np.array([i for i, (_, ts) in enumerate(ds) for _ in ts]),
                     "Sg": np.stack([E["sig"][t] for t in texts]), "Dn": np.stack([E["dense"][t] for t in texts]),
                     "sp": [E["sparse"][t] for t in texts], "vocab": vocab, "sets": asf_filter.build_doc_token_sets(joined, vocab)}

    def dmax(vals, s):
        out = np.full(s["n"], -np.inf, dtype=np.float32)
        np.maximum.at(out, s["owner"], vals.astype(np.float32))
        return out

    rows = []
    for q in queries:
        s = scope[q["scope"]]
        v, bil = variants_of(q["text"])
        Re0, Im0 = dmax(s["Sg"] @ E["sig"][q["text"]], s), dmax(s["Dn"] @ E["dense"][q["text"]], s)
        ReX = dmax(s["Sg"] @ avg_norm(np.stack([E["sig"][t] for t in v])), s)
        ImX = dmax(s["Dn"] @ avg_norm(np.stack([E["dense"][t] for t in v])), s)
        boost = boost_vec(q["text"], bil, s["ids"], s["caps"])
        lex = dmax(np.array([sdot(E["sparse"][bil], x) for x in s["sp"]], dtype=np.float32), s)
        asf = asf_filter.asf_scores(q["text"], s["sets"], s["vocab"]) if s["vocab"] else np.zeros(s["n"], dtype=np.float32)
        d = {"s0_im": Im0, "s1_tc2_a04": np.sqrt(Re0 ** 2 + (0.4 * Im0) ** 2), "s2_tc2_a08": np.sqrt(Re0 ** 2 + (DOC_ALPHA * Im0) ** 2),
             "s3_expand": np.sqrt(ReX ** 2 + (DOC_ALPHA * ImX) ** 2)}
        d["s4_boost"] = d["s3_expand"] + boost
        f = dict(d)                                              # sparse 전 단계는 순위 = dense
        d["s5_sparse"], f["s5_sparse"] = d["s4_boost"], minmax_fusion(d["s4_boost"], lex, None)
        d["s6_asf"], f["s6_asf"] = d["s4_boost"], minmax_fusion(d["s4_boost"], lex, asf)
        rows.append({"dense": d, "fused": f, "ids": s["ids"], "Re": ReX, "Im": ImX, "lex": lex, "boost": boost,
                     "expanded": len(v) > 1, "asf_active": bool(asf.any())})
    return rows


# ───────────────────────── 평가 ─────────────────────────
def decide(row, stage, tau):
    d, f = row["dense"][stage], row["fused"][stage]
    ok = d >= tau
    return row["ids"][int(np.argmax(np.where(ok, f, -np.inf)))] if ok.any() else None


def outcome(pred, q):
    mappable = q["gold"] not in ("NONE", "AMBIG")
    if not pred:
        return "hold_miss" if mappable else "correct"
    return "correct" if mappable and pred == q["gold"] else "wrong"


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p, d = k / n, 1 + z * z / n
    c, e = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - e), 4), round(min(1.0, c + e), 4)]


def summarize(o):
    n, c, w, h = len(o), o.count("correct"), o.count("wrong"), o.count("hold_miss")
    return {"n": n, "accuracy": c / n, "wrong_rate": w / n, "hold_miss_rate": h / n, "acc_ci95": wilson(c, n), "wrong_ci95": wilson(w, n)}


def by_category(o, queries, idx):
    return {c: summarize([o[i] for i in idx if queries[i]["category"] == c])
            for c in sorted({q["category"] for q in queries}) if any(queries[i]["category"] == c for i in idx)}


def evaluate(rows, queries, stage, dev, test):
    devset = set(dev)
    grid = np.quantile(np.array([float(r["dense"][stage].max()) for r in rows])[dev], np.linspace(0, 1, 61))
    best = None
    for t in sorted(set([-1e9] + [float(x) for x in grid])):
        o = [outcome(decide(rows[i], stage, t), queries[i]) for i in dev]
        key = (o.count("correct"), -o.count("wrong"), t)
        if best is None or key > best[0]:
            best = (key, t)
    null, rel, hard = [], [], []
    for i, (q, r) in enumerate(zip(queries, rows)):
        ok = {q["gold"], *q["accept"]}
        nl = [float(x) for x, did in zip(r["dense"][stage], r["ids"]) if did not in ok]
        if i in devset:
            null += nl
        if nl:
            hard.append(max(nl))
        if q["gold"] not in ("NONE", "AMBIG"):
            rel.append(float(r["dense"][stage][r["ids"].index(q["gold"])]))
    mu, sd = statistics.fmean(null), statistics.pstdev(null)
    mr, sr = statistics.fmean(rel), statistics.pstdev(rel)
    res = {"tau_fit": best[1], "tau_cal": mu + FAR_Z * sd, "mu_null": mu, "sigma_null": sd, "mu_rel": mr, "sigma_rel": sr,
           "d_prime": (mr - mu) / math.sqrt((sd * sd + sr * sr) / 2 + 1e-12),
           "auc_rel_vs_hardnull": float(np.mean([(a > b) + 0.5 * (a == b) for a in rel for b in hard]))}
    t1 = r3 = n = 0
    for i in test:
        q = queries[i]
        if q["gold"] in ("NONE", "AMBIG"):
            continue
        ranked = [rows[i]["ids"][j] for j in np.argsort(-rows[i]["fused"][stage])[:3]]
        n += 1
        t1 += ranked[0] == q["gold"]
        r3 += q["gold"] in ranked
    res["ranking_test"] = {"n": n, "top1": t1 / n, "recall3": r3 / n}
    for name, tau in (("fit", res["tau_fit"]), ("cal", res["tau_cal"])):
        o = [outcome(decide(rows[i], stage, tau), queries[i]) for i in range(len(rows))]
        res[name] = {"test": summarize([o[i] for i in test]), "dev": summarize([o[i] for i in dev]), "by_category_test": by_category(o, queries, test)}
    # 기준선을 움직일 때 (오매핑률, 정답 매핑률) 곡선 — test 에서 직접 움직인 낙관적 상한
    n_map = sum(queries[i]["gold"] not in ("NONE", "AMBIG") for i in test)
    curve = []
    for t in sorted(set(float(x) for x in np.quantile(np.array([float(rows[i]["dense"][stage].max()) for i in test]), np.linspace(0, 1, 121))), reverse=True):
        o = [outcome(decide(rows[i], stage, t), queries[i]) for i in test]
        right = sum(1 for i, x in zip(test, o) if x == "correct" and queries[i]["gold"] not in ("NONE", "AMBIG"))
        curve.append((o.count("wrong") / len(test), right / n_map))
    dist = {"rel": rel, "hard": hard, "null_sample": null[::max(1, len(null) // 20000)]}
    return res, dist, curve


def plot_points(rows, queries):
    """산점도용: 질의마다 정답 + 가장 헷갈리는 무관 후보 + 무관 후보 표본. (kind, Re, Im, lexical, dense)"""
    pts = []
    for q, r in zip(queries, rows):
        ok = {q["gold"], *q["accept"]}
        order = np.argsort(-r["fused"][FULL])
        nulls = [j for j in order if r["ids"][j] not in ok]
        step = max(1, len(nulls) // 12)
        sel = [("hard", nulls[0])] + [("null", j) for j in nulls[1::step]]
        if q["gold"] not in ("NONE", "AMBIG"):
            sel.append(("gold", r["ids"].index(q["gold"])))
        for kind, j in sel:
            pts.append((kind, float(r["Re"][j]), float(r["Im"][j]), float(r["lex"][j]), float(r["dense"][FULL][j])))
    return pts


def main(csv_path):
    sys.stdout.reconfigure(encoding="utf-8")
    tasks = synthetic_tasks()
    products, categories, profile = load_kaggle(csv_path)
    assert profile["sha256"].startswith("114dfeb4"), "Kaggle 원본 해시가 다르다"
    prod_traps = [849, 850, 851, 852, 854, 855, 856, 857, 858, 859, 1849, 2040, 2173, 2180, 2199, 2250, 2400, 2999, 3000, 4000]
    tasks["K2"] = kaggle_task("K2", products, "Product", "제품", 4, 40, prod_traps)
    tasks["K3"] = kaggle_task("K3", categories, "Category", "분류", 3, 33, [34, 35, 36, 40, 45, 50, 60, 99])

    texts = set()
    for queries, docs in tasks.values():
        for ds in docs.values():
            for _, ts in ds:
                texts.update(ts)
        for q in queries:
            v, bil = variants_of(q["text"])
            texts.update(v)
            texts.add(bil)
    texts = sorted(texts)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    timing = {"device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None, "n_texts": len(texts)}
    print(f"texts={len(texts)} device={device}", flush=True)
    for key, cls in (("bge_m3", BgeM3), ("siglip2_text", Siglip2Text)):
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        m = cls(device)
        m.encode(texts[:8])
        t0 = time.perf_counter()
        out = m.encode(texts)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        lat = []
        for t in texts[:40]:
            s0 = time.perf_counter()
            m.encode([t])
            if device == "cuda":
                torch.cuda.synchronize()
            lat.append(time.perf_counter() - s0)
        timing[key] = {"encode_all_s": round(dt, 2), "per_text_ms": round(1000 * dt / len(texts), 2), "single_query_ms_median": round(1000 * statistics.median(lat), 1),
                       "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 2**20, 0) if device == "cuda" else None}
        if key == "bge_m3":
            dense, sparse = out
        else:
            sig = out
        del m
    E = {"dense": dict(zip(texts, dense)), "sparse": dict(zip(texts, sparse)), "sig": dict(zip(texts, sig))}
    print("timing", json.dumps(timing, ensure_ascii=False), flush=True)

    results = {"db_insight_commit": "6a50976", "models": {"Re": Siglip2Text.name, "Im": BgeM3.name}, "doc_im_alpha": DOC_ALPHA,
               "fusion_weights": list(FUSION_W), "far": 0.05, "stages": STAGES, "full_stage": FULL, "kaggle": profile, "timing": timing, "tasks": {}}
    plot = {}
    for name, (queries, docs) in tasks.items():
        rows = score_task(queries, docs, E)
        dev = [i for i, q in enumerate(queries) if q["split"] == "dev"]
        test = [i for i, q in enumerate(queries) if q["split"] == "test"]
        tr = {"n": len(queries), "n_dev": len(dev), "n_test": len(test), "n_docs": sum(len(ds) for ds in docs.values()), "stages": {}, "rules": {},
              "queries_expanded": sum(r["expanded"] for r in rows), "queries_asf_active": sum(r["asf_active"] for r in rows),
              "queries_boosted": sum(bool(r["boost"].any()) for r in rows)}
        for rname in ("rule_pred", "digits_pred"):
            if rname in queries[0]:
                o = [outcome(q[rname], q) for q in queries]
                tr["rules"][rname] = {"test": summarize([o[i] for i in test]), "by_category_test": by_category(o, queries, test)}
        plot[name] = {"dist": {}, "curve": {}, "points": plot_points(rows, queries)}
        for st in STAGES:
            tr["stages"][st], plot[name]["dist"][st], plot[name]["curve"][st] = evaluate(rows, queries, st, dev, test)
        results["tasks"][name] = tr
        print(f"\n== {name}  질의 {len(queries)} (dev {len(dev)} / test {len(test)}), 후보 {tr['n_docs']}  확장 {tr['queries_expanded']}, 가산 {tr['queries_boosted']}, ASF {tr['queries_asf_active']}")
        for rname, r in tr["rules"].items():
            print(f"{rname:12s}               | acc {r['test']['accuracy']:.3f} wrong {r['test']['wrong_rate']:.3f}")
        print(f"{'stage':12s} {'top1':>6s} {'R@3':>6s} | dev τ: acc wrong hold | 보정 τ: acc wrong hold | τfit  τcal   d′")
        for st in STAGES:
            r = tr["stages"][st]
            f, c = r["fit"]["test"], r["cal"]["test"]
            print(f"{st:12s} {r['ranking_test']['top1']:6.3f} {r['ranking_test']['recall3']:6.3f} | {f['accuracy']:.3f} {f['wrong_rate']:.3f} {f['hold_miss_rate']:.3f}  |"
                  f" {c['accuracy']:.3f} {c['wrong_rate']:.3f} {c['hold_miss_rate']:.3f}   | {r['tau_fit']:.3f} {r['tau_cal']:.3f} {r['d_prime']:.2f}", flush=True)
    (RES / "trichef_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    (RES / "trichef_plotdata.json").write_text(json.dumps(plot, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
