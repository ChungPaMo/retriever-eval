r"""Tri-CHEF 실험 그림 7장. 실행: .\.venv\Scripts\python.exe scripts\trichef_figures.py

tri1_ladder.png            단계별 판정 결과 (dev 기준선 / Tri-CHEF 식 보정 기준선)
tri2_distributions.png     문서 기본 구성의 dense 점수 분포와 정규분포 근사
tri3_three_axis.png        3축 공간 (Re SigLIP2 / Im BGE-M3 dense / Lexical BGE-M3 sparse)
tri4_re_im_plane.png       Re–Im 평면과 기준선(타원)
tri5_tradeoff.png          오매핑률 대비 정답 매핑률
tri6_category_heatmap.png  범주별 판정 정확도
tri7_stage_metrics.png     단계별 1순위 정답률과 분리도
"""
from __future__ import annotations

import json
import math
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
CRITICAL, NEUTRAL = "#d03b3b", "#c3c2b7"
plt.rcParams.update({
    "font.family": "Malgun Gothic", "axes.unicode_minus": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})
R = json.loads((RES / "trichef_results.json").read_text(encoding="utf-8"))
P = json.loads((RES / "trichef_plotdata.json").read_text(encoding="utf-8"))
TASKS = ("A", "B", "K2", "K3")
FULL, A = R["full_stage"], R["doc_im_alpha"]
TN = {"A": "A 열 이름 → 필드", "B": "B 품목 표기 → 마스터", "K2": "K2 제품 코드 (실제 2,160개)", "K3": "K3 분류 코드 (실제 33개)"}
SK = {"s0_im": "Im 만 (BGE-M3)", "s1_tc2_a04": "2축 α=0.4", "s2_tc2_a08": "2축 α=0.8", "s3_expand": "+ 질의 확장",
      "s4_boost": "+ 문자열 일치 가산", "s5_sparse": "+ sparse 결합 (문서 기본)", "s6_asf": "+ ASF"}
SK_SHORT = {"s0_im": "Im 만", "s1_tc2_a04": "2축\nα=0.4", "s2_tc2_a08": "2축\nα=0.8", "s3_expand": "+질의\n확장", "s4_boost": "+문자열\n가산",
            "s5_sparse": "+sparse\n(문서 기본)", "s6_asf": "+ASF"}
RK = {"rule_pred": "현재 규칙", "digits_pred": "숫자 추출 규칙"}
CAT = {"alias_exact": "별칭\n그대로", "alias_format": "별칭\n표기 변형", "synonym": "동의어\n·영문", "typo": "약어\n·오타", "compound": "복합\n표기",
       "none": "해당 없음\n열", "trap": "함정\n열", "code_variant": "코드\n표기 변형", "name_exact": "품명\n그대로", "name_spacing": "띄어\n쓰기",
       "name_reorder": "어순\n·괄호", "name_synonym": "동의어\n·영문", "spec_pick": "규격\n구분", "spec_trap": "없는\n규격", "ambiguous": "후보\n여럿",
       "conflict": "코드·품명\n충돌", "unregistered": "미등록\n품목", "case": "소문자", "drop_zero": "0 빠짐", "hyphen_upper": "대문자\n·하이픈",
       "space_short": "약어\n·띄어쓰기", "initial": "머리\n글자", "number_only": "숫자만", "number_nozero": "숫자만\n(0 없음)", "no_sep": "구분자\n없음",
       "korean": "한글\n접두어", "trap_exact_form": "함정\n결번", "trap_drop_zero": "함정\n결번\n(0 빠짐)", "trap_initial": "함정\n결번\n(머리글자)"}
CAT_ORDER = {"A": ["alias_exact", "alias_format", "synonym", "typo", "compound", "none", "trap"],
             "B": ["code_variant", "name_exact", "name_spacing", "name_reorder", "name_synonym", "spec_pick", "spec_trap", "ambiguous", "conflict", "unregistered"],
             "K2": ["case", "drop_zero", "hyphen_upper", "space_short", "initial", "number_only", "number_nozero", "no_sep", "korean", "trap_exact_form", "trap_drop_zero", "trap_initial"]}
CAT_ORDER["K3"] = CAT_ORDER["K2"]
NOTE = "A·B 는 합성 평가셋, K2·K3 는 Kaggle 실제 코드에 합성 표기 변형. Tri-CHEF 와 같은 모델(SigLIP2 텍스트 + BGE-M3), DB_insight 문서 검색 절차."


def normal_pdf(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


def stacked(ax, rows):
    y = np.arange(len(rows))[::-1]
    cc = np.array([v["accuracy"] for _, v in rows])
    hh = np.array([v["hold_miss_rate"] for _, v in rows])
    ww = np.array([v["wrong_rate"] for _, v in rows])
    ax.barh(y, cc, height=0.64, color=BLUE, label="맞음 (맞게 매핑했거나 맞게 보류)", edgecolor=SURF, linewidth=1.2)
    ax.barh(y, hh, left=cc, height=0.64, color=NEUTRAL, label="정답이 있는데 보류", edgecolor=SURF, linewidth=1.2)
    ax.barh(y, ww, left=cc + hh, height=0.64, color=CRITICAL, label="오매핑 (틀린 후보를 제안)", edgecolor=SURF, linewidth=1.2)
    for yi, (_, v), ci, hi_, wi in zip(y, rows, cc, hh, ww):
        if v.get("na"):
            ax.text(0.01, yi, "해당 없음 (숫자 코드가 아닌 과제)", ha="left", va="center", color=MUTED, fontsize=8.5)
            continue
        if ci > 0.12:
            ax.text(ci / 2, yi, f"{ci:.0%}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        else:
            ax.text(ci + 0.01, yi, f"{ci:.0%}", ha="left", va="center", color=INK, fontsize=8.5)
        if wi > 0.06:
            ax.text(ci + hi_ + wi / 2, yi, f"{wi:.0%}", ha="center", va="center", color="white", fontsize=8.5)
    ax.set_yticks(y, [n for n, _ in rows])
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "", "50%", "", ""])
    ax.grid(axis="y", visible=False)


def tri1():
    empty = {"accuracy": 0.0, "hold_miss_rate": 0.0, "wrong_rate": 0.0, "na": True}
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.6), sharey=True)
    for r, (mode, mname) in enumerate((("fit", "dev 에서 고른 기준선"), ("cal", "Tri-CHEF 식 보정 기준선 (μ+1.645σ, 무관 쌍 5% 허용)"))):
        for c, t in enumerate(TASKS):
            T = R["tasks"][t]
            rows = [(RK[k], T["rules"][k]["test"] if k in T["rules"] else empty) for k in RK] + [(SK[s], T["stages"][s][mode]["test"]) for s in R["stages"]]
            stacked(axes[r, c], rows)
            axes[r, c].axhline(len(R["stages"]) - 0.5, color=AXIS, linewidth=1)
            axes[r, c].set_title(f"{TN[t]}  (test {T['n_test']}건)" if r == 0 else TN[t])
        fig.text(0.012, 0.905 - r * 0.425, mname, fontsize=11.5, fontweight="bold", color=INK2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.012, 0.03), ncol=3, fontsize=10)
    fig.suptitle("단계별 판정 결과: Tri-CHEF 문서 검색 절차를 다 쌓아도 BGE-M3 단독(Im 만)과 차이가 작다", x=0.012, ha="left", fontsize=13.5, fontweight="bold")
    fig.text(0.012, 0.008, "단계는 위에서 아래로 하나씩 더한 것. 'sparse 결합'까지가 문서 기본 구성(ASF 는 기본 꺼짐). 판정: dense ≥ τ 인 후보 중 융합 점수 1순위. " + NOTE, color=MUTED, fontsize=8.6)
    fig.subplots_adjust(left=0.115, right=0.99, top=0.865, bottom=0.11, hspace=0.42, wspace=0.05)
    fig.savefig(FIG / "tri1_ladder.png", dpi=150)
    plt.close(fig)


def tri2():
    fig, axes = plt.subplots(1, 4, figsize=(17, 5))
    for ax, t in zip(axes, TASKS):
        d, s = P[t]["dist"][FULL], R["tasks"][t]["stages"][FULL]
        allv = d["rel"] + d["hard"] + d["null_sample"]
        lo, hi = min(allv) - 0.03, max(allv) + 0.05
        bins, x = np.linspace(lo, hi, 44), np.linspace(lo, hi, 400)
        ax.hist(d["null_sample"], bins=bins, density=True, color=NEUTRAL, alpha=0.75, label="무관 쌍")
        ax.hist(d["hard"], bins=bins, density=True, histtype="step", color=ORANGE, linewidth=2, label="질의마다 가장 헷갈리는 무관 후보")
        ax.hist(d["rel"], bins=bins, density=True, color=BLUE, alpha=0.55, label="정답 쌍")
        ax.plot(x, normal_pdf(x, s["mu_null"], s["sigma_null"]), color=INK2, linewidth=2, linestyle="--", label="무관 쌍 정규분포 근사")
        ax.plot(x, normal_pdf(x, s["mu_rel"], s["sigma_rel"]), color="#1c5cab", linewidth=2, linestyle="--", label="정답 쌍 정규분포 근사")
        ax.axvline(s["tau_cal"], color=INK, linewidth=1.5)
        ax.axvline(s["tau_fit"], color=INK, linewidth=1.5, linestyle=":")
        below = sum(v < s["tau_cal"] for v in d["rel"]) / len(d["rel"])
        above = sum(v >= s["tau_cal"] for v in d["hard"]) / len(d["hard"])
        ax.text(0.02, 0.97, f"무관 μ={s['mu_null']:.3f}, σ={s['sigma_null']:.3f}\n정답 μ={s['mu_rel']:.3f}, σ={s['sigma_rel']:.3f}\n분리도 d′={s['d_prime']:.2f}\n"
                            f"보정 τ={s['tau_cal']:.3f} (실선)\ndev τ={s['tau_fit']:.3f} (점선)\n보정 τ 아래 정답 {below:.0%}\n보정 τ 넘는 헷갈리는 후보 {above:.0%}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.6, color=INK2, linespacing=1.45, bbox={"facecolor": SURF, "edgecolor": "none", "alpha": 0.85, "pad": 2})
        ax.set_xlim(lo, hi)
        ax.set_xlabel(f"dense 점수 √(Re² + ({A}·Im)²) + 문자열 일치 가산")
        ax.set_title(TN[t])
    axes[0].set_ylabel("밀도")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.03, 0.045), ncol=5, fontsize=9.5)
    fig.suptitle("문서 기본 구성의 점수 분포: 보정 기준선(실선)을 넘는 헷갈리는 후보가 많고(B 71%), 실제 코드(K2·K3)는 정답 절반이 기준선 아래다", x=0.012, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.012, 0.008, "정규분포 근사는 참고용(정답 쌍 분포는 한쪽으로 치우친다). " + NOTE, color=MUTED, fontsize=8.6)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.87, bottom=0.22, wspace=0.16)
    fig.savefig(FIG / "tri2_distributions.png", dpi=150)
    plt.close(fig)


def groups(t):
    pts = P[t]["points"]
    return [(kind, np.array([p[1:] for p in pts if p[0] == kind])) for kind in ("null", "hard", "gold")]


STYLE = {"null": (NEUTRAL, 6, 0.35, "무관 후보 (표본)"), "hard": (ORANGE, 20, 0.85, "질의마다 가장 헷갈리는 무관 후보"), "gold": (BLUE, 20, 0.9, "정답")}


def tri3():
    fig = plt.figure(figsize=(15, 12.4))
    for k, t in enumerate(TASKS):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        zmax = 0.0
        for kind, p in groups(t):
            color, size, alpha, label = STYLE[kind]
            ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=size, c=color, alpha=alpha, label=f"{label} ({len(p):,})", depthshade=False, linewidths=0)
            zmax = max(zmax, float(p[:, 2].max()))
        ax.set_xlabel("Re: SigLIP2 텍스트", labelpad=6)
        ax.set_ylabel("Im: BGE-M3 dense", labelpad=6)
        ax.set_zlabel("Lexical: BGE-M3 sparse 내적", labelpad=6)
        ax.set_xlim(0.3, 1), ax.set_ylim(0.2, 1), ax.set_zlim(0, zmax * 1.02)
        ax.view_init(elev=20, azim=-60)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor(SURF)
            axis.pane.set_edgecolor(GRID)
            axis._axinfo["grid"]["color"] = GRID
        ax.set_title(TN[t], loc="left", pad=0)
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.97), fontsize=8.5, handletextpad=0.3)
    fig.suptitle("Tri-CHEF 3축 공간: 정답은 Im·Lexical 이 높은 쪽에 모이고, Re(SigLIP2)는 정답과 무관 후보를 거의 가르지 못한다", x=0.012, ha="left", fontsize=13.5, fontweight="bold")
    fig.text(0.012, 0.012, "축: Tri-CHEF 문서 검색이 쓰는 세 신호 (Re, Im 은 질의 확장 후 코사인 유사도, Lexical 은 BGE-M3 sparse 가중치 내적). Z(DINOv2)는 텍스트 질의에서 0.\n" + NOTE, color=MUTED, fontsize=8.6)
    fig.subplots_adjust(left=0.0, right=0.98, top=0.94, bottom=0.055, wspace=0.02, hspace=0.06)
    fig.savefig(FIG / "tri3_three_axis.png", dpi=150)
    plt.close(fig)


def tri4():
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.4))
    for ax, t in zip(axes, TASKS):
        for kind, p in groups(t):
            color, size, alpha, label = STYLE[kind]
            ax.scatter(p[:, 0], p[:, 1], s=size + 2, c=color, alpha=alpha, label=label, linewidths=0.5 if kind != "null" else 0, edgecolors=SURF, zorder={"null": 1, "hard": 2, "gold": 3}[kind])
        s = R["tasks"][t]["stages"][FULL]
        for tau, style, lab in ((s["tau_cal"], "-", "보정 기준선"), (s["tau_fit"], "--", "dev 기준선")):
            x = np.linspace(0, min(tau, 1.02), 400)
            ax.plot(x, np.sqrt(np.clip(tau * tau - x * x, 0, None)) / A, color=INK, linewidth=1.5, linestyle=style, label=f"{lab} τ={tau:.3f}", zorder=4)
        ax.set_xlim(0.3, 1.02), ax.set_ylim(0.2, 1.02)
        ax.set_xlabel("Re: SigLIP2 텍스트 코사인")
        ax.set_title(TN[t])
        ax.legend(loc="lower left", fontsize=7.8, handletextpad=0.3)
    axes[0].set_ylabel("Im: BGE-M3 dense 코사인")
    fig.suptitle(f"Re–Im 평면: α={A} 에서는 기준선이 대각선에 가까워 두 축을 함께 본다. 그래도 헷갈리는 후보가 정답과 같은 영역에 있다", x=0.012, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.012, 0.012, f"기준선은 √(Re² + ({A}·Im)²) = τ (문자열 일치 가산 전 기준이라 실제 판정과 조금 다르다). " + NOTE, color=MUTED, fontsize=8.6)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.87, bottom=0.14, wspace=0.14)
    fig.savefig(FIG / "tri4_re_im_plane.png", dpi=150)
    plt.close(fig)


def tri5():
    series = [("s0_im", BLUE, 2.0), ("s2_tc2_a08", AQUA, 2.0), (FULL, VIOLET, 2.8)]
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.2), sharey=True)
    table = {}
    for ax, t in zip(axes, TASKS):
        table[t] = {}
        for st, color, lw in series:
            c = np.array(P[t]["curve"][st])
            x, y = np.concatenate([[0], c[:, 0]]), np.concatenate([[0], c[:, 1]])
            ax.step(x, y, where="post", color=color, linewidth=lw, label=SK[st])
            table[t][st] = {f"le_{int(cap * 100)}pct": float(y[x <= cap + 1e-9].max()) for cap in (0.0, 0.02, 0.05)}
        T = R["tasks"][t]
        n_map = T["stages"][FULL]["ranking_test"]["n"]
        for k, (rname, marker) in enumerate((("rule_pred", "o"), ("digits_pred", "D"))):
            if rname in T["rules"]:
                rr = T["rules"][rname]["test"]
                rec = sum(v["accuracy"] * v["n"] for cn, v in T["rules"][rname]["by_category_test"].items() if not cn.startswith("trap") and cn not in ("none", "trap", "spec_trap", "ambiguous", "conflict", "unregistered")) / n_map
                ax.scatter([rr["wrong_rate"]], [rec], s=80, marker=marker, color=INK, zorder=5, edgecolors=SURF, linewidths=1.5, label=f"{RK[rname]} (정답 매핑 {rec:.0%})")
                table[t][rname] = {"wrong": rr["wrong_rate"], "recall": rec}
        ax.axvline(0.02, color=MUTED, linewidth=1, linestyle=":")
        ax.set_xlim(-0.006, 0.3), ax.set_ylim(0, 1.03)
        ax.set_xticks([0, 0.1, 0.2, 0.3], ["0%", "10%", "20%", "30%"])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("오매핑률")
        ax.set_title(TN[t])
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("정답 매핑률 (정답이 있는 질의 중)")
    fig.suptitle("오매핑을 낮게 묶을수록 맞게 매핑하는 비율이 줄어든다: 왼쪽 위에 가까울수록 좋고, 규칙(검은 점)은 오매핑 0 에 있다", x=0.012, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.012, 0.012, "기준선 τ 를 test 에서 직접 움직인 곡선이라 실제 운영보다 낙관적이다. 세로 점선 = 오매핑 2%. " + NOTE, color=MUTED, fontsize=8.6)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.87, bottom=0.14, wspace=0.06)
    fig.savefig(FIG / "tri5_tradeoff.png", dpi=150)
    plt.close(fig)
    (RES / "trichef_tradeoff.json").write_text(json.dumps(table, ensure_ascii=False, indent=1), encoding="utf-8")


def tri6():
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"])
    fig, axes = plt.subplots(2, 2, figsize=(17, 8.2), gridspec_kw={"width_ratios": [7, 10]})
    axes = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]
    layout = {"A": 0, "B": 1, "K3": 2, "K2": 3}
    for t in TASKS:
        ax = axes[layout[t]]
        T = R["tasks"][t]
        rows = [(RK[k], T["rules"][k]["by_category_test"]) for k in RK if k in T["rules"]]
        rows += [("Im 만 (dev τ)", T["stages"]["s0_im"]["fit"]["by_category_test"]), ("문서 기본 (dev τ)", T["stages"][FULL]["fit"]["by_category_test"]),
                 ("문서 기본 (보정 τ)", T["stages"][FULL]["cal"]["by_category_test"])]
        cats = CAT_ORDER[t] if t not in ("K2", "K3") or True else []
        cats = [c for c in CAT_ORDER[t]]
        if t == "K3":
            cats = cats
        M = np.array([[bc.get(c, {}).get("accuracy", np.nan) for c in cats] for _, bc in rows])
        N = [rows[-1][1].get(c, {}).get("n", 0) for c in cats]
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.0%}", ha="center", va="center", fontsize=9, color="white" if M[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(cats)), [f"{CAT[c]}\n(n={n})" for c, n in zip(cats, N)], fontsize=7.8, color=INK2)
        ax.set_yticks(range(len(rows)), [n for n, _ in rows], color=INK)
        ax.tick_params(length=0)
        ax.grid(False)
        ax.set_xticks(np.arange(-0.5, len(cats)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
        ax.grid(which="minor", color=SURF, linewidth=2)
        ax.tick_params(which="minor", length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(TN[t])
    fig.suptitle("범주별 판정 정확도: Tri-CHEF 는 새 표기에 강하고, 매핑하면 안 되는 것(함정·없는 규격·결번)에 약하다", x=0.012, ha="left", fontsize=13.5, fontweight="bold")
    fig.text(0.012, 0.008, "test 분할. 칸 = 맞게 매핑했거나 맞게 보류한 비율. 범주별 건수가 적어 칸 하나의 값은 크게 흔들릴 수 있다. " + NOTE, color=MUTED, fontsize=8.6)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.91, bottom=0.13, hspace=0.5, wspace=0.27)
    fig.savefig(FIG / "tri6_category_heatmap.png", dpi=150)
    plt.close(fig)


def tri7():
    colors = {"A": BLUE, "B": ORANGE, "K2": AQUA, "K3": YELLOW}
    x = np.arange(len(R["stages"]))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    specs = [("1순위 정답률 (기준선 없이 순위만)", lambda s: s["ranking_test"]["top1"], True),
             ("분리도 d′ (정답 쌍과 무관 쌍의 평균 차이 ÷ 표준편차)", lambda s: s["d_prime"], False),
             ("정답이 가장 헷갈리는 무관 후보보다 높을 확률 (AUC)", lambda s: s["auc_rel_vs_hardnull"], True)]
    for ax, (title, fn, pct) in zip(axes, specs):
        for t in TASKS:
            y = [fn(R["tasks"][t]["stages"][s]) for s in R["stages"]]
            ax.plot(x, y, color=colors[t], linewidth=2, marker="o", markersize=6, markeredgecolor=SURF, markeredgewidth=1.2, label=TN[t])
            ax.text(x[-1] + 0.12, y[-1], t, color=INK2, fontsize=9, va="center")
        ax.axvspan(4.5, 5.5, color=GRID, alpha=0.5, zorder=0)
        ax.set_xticks(x, [SK_SHORT[s] for s in R["stages"]], fontsize=8.3, color=INK2)
        ax.set_xlim(-0.3, len(x) - 0.4)
        if pct:
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.set_title(title, fontsize=10.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower left", bbox_to_anchor=(0.03, 0.045), ncol=4, fontsize=9.5)
    fig.suptitle("단계별 지표: α=0.4 에서 떨어졌다가 0.8 에서 회복하고, 문자열 일치 가산이 코드 과제(K2·K3)를 끌어올린다. 질의 확장·sparse·ASF 는 변화가 거의 없다", x=0.012, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.012, 0.012, "회색 띠 = DB_insight 문서 기본 구성. 분리도·AUC 는 dense 점수 기준. " + NOTE, color=MUTED, fontsize=8.6)
    fig.subplots_adjust(left=0.04, right=0.975, top=0.86, bottom=0.24, wspace=0.16)
    fig.savefig(FIG / "tri7_stage_metrics.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import matplotlib.ticker  # noqa: F401
    tri1(), tri2(), tri3(), tri4(), tri5(), tri6(), tri7()
    print("saved", sorted(p.name for p in FIG.glob("tri*.png")))
    print(json.dumps(json.loads((RES / "trichef_tradeoff.json").read_text(encoding="utf-8")), ensure_ascii=False))
    for t in TASKS:
        print(t, {s: round(R["tasks"][t]["stages"][s]["auc_rel_vs_hardnull"], 3) for s in ("s0_im", "s1_tc2_a04", "s5_sparse")})
