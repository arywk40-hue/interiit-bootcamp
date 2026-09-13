from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUT = Path(__file__).resolve().parents[1] / "architecture.png"


BG = "#F7F9FC"
INK = "#17233C"
MUTED = "#56627A"
LINE = "#6B7890"
BLUE = "#DCEBFF"
BLUE_EDGE = "#3274C8"
TEAL = "#D9F3F0"
TEAL_EDGE = "#198A82"
PURPLE = "#ECE3FF"
PURPLE_EDGE = "#7651B8"
ORANGE = "#FFF0D9"
ORANGE_EDGE = "#C77B17"
GREEN = "#E2F4E3"
GREEN_EDGE = "#3A9145"
GREY = "#EEF1F5"


def add_box(ax, x, y, w, h, title, body, face, edge, title_size=11, body_size=8.2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.55,rounding_size=1.8",
        linewidth=1.5, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + 1.1, y + h - 2.1, title, va="top", ha="left",
            fontsize=title_size, fontweight="bold", color=INK)
    ax.text(x + 1.1, y + h - 5.2, body, va="top", ha="left",
            fontsize=body_size, linespacing=1.28, color=INK)


def arrow(ax, start, end, color=LINE, dashed=False, lw=1.5):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="-|>", mutation_scale=13, color=color,
                        linewidth=lw, linestyle="--" if dashed else "-",
                        shrinkA=5, shrinkB=5),
    )


def main():
    fig, ax = plt.subplots(figsize=(19, 13), dpi=180)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 97.3, "Roman Indic NLU — Hinglish Multi-Task Architecture",
            ha="center", va="top", fontsize=22, fontweight="bold", color=INK)
    ax.text(50, 94.7, "Research + deployment design for spelling drift, code mixing, emojis and tight latency",
            ha="center", va="top", fontsize=10.5, color=MUTED)

    # Ingress and governance.
    add_box(ax, 3, 84, 27, 8.2, "1  Request contract",
            "task + raw Hinglish text\nconversation_id / optional QA context",
            BLUE, BLUE_EDGE)
    add_box(ax, 35, 84, 27, 8.2, "2  Data governance",
            "provenance • hashes • writer/source split\ndeduplication • locked test sets",
            GREY, LINE)
    add_box(ax, 67, 84, 30, 8.2, "3  Explicit task routing",
            "no generative router\nrun exactly one selected endpoint",
            TEAL, TEAL_EDGE)
    arrow(ax, (30, 88), (35, 88))
    arrow(ax, (62, 88), (67, 88))

    # Shared representation layer.
    ax.add_patch(FancyBboxPatch((5, 57), 90, 23,
                                boxstyle="round,pad=0.8,rounding_size=2.2",
                                linewidth=1.7, edgecolor=BLUE_EDGE,
                                facecolor="#F0F6FF"))
    ax.text(7, 77.6, "SHARED ROMANIZATION-AWARE LANGUAGE LAYER",
            fontsize=12, fontweight="bold", color=BLUE_EDGE, va="top")

    add_box(ax, 8, 62.5, 22, 10.4, "Safe text views",
            "RAW: emojis, punctuation, casing\nORTHO: conservative cleanup\nPHONETIC: spelling-variant view",
            BLUE, BLUE_EDGE, body_size=7.9)
    add_box(ax, 34, 62.5, 25, 10.4, "Robust representation",
            "byte/character embeddings\nshallow char-CNN for local drift\noptional task-safe subword tokenizer",
            TEAL, TEAL_EDGE, body_size=7.9)
    add_box(ax, 63, 62.5, 24, 10.4, "Compact shared encoder",
            "small Transformer encoder\ngated raw + phonetic fusion\npooled representation z",
            PURPLE, PURPLE_EDGE, body_size=7.9)
    arrow(ax, (30, 67.7), (34, 67.7))
    arrow(ax, (59, 67.7), (63, 67.7))

    # Task heads.
    ax.text(50, 54.4, "TASK-SPECIFIC HEADS (same z, separate data + metrics)",
            ha="center", va="top", fontsize=11, fontweight="bold", color=MUTED)
    xs = [2, 26.5, 51, 75.5]
    titles = ["SENTIMENT", "INTENT", "SUMMARIZATION", "QUESTION ANSWERING"]
    bodies = [
        "SentiMix-style 3-way\npositive / neutral / negative\nmacro-F1 + calibration",
        "Human Hinglish taxonomy\n10–15 support intents\nmacro-F1 + confusion matrix",
        "GupShup Hinglish → English\ncompact encoder-decoder\nROUGE + factuality + latency",
        "Retriever → extractive reader\nspan or explicit no_answer\nEM / token F1 / recall@k",
    ]
    colors = [(ORANGE, ORANGE_EDGE), (GREEN, GREEN_EDGE), (PURPLE, PURPLE_EDGE), (TEAL, TEAL_EDGE)]
    for x, title, body, (face, edge) in zip(xs, titles, bodies, colors):
        add_box(ax, x, 37, 22.5, 14.3, title, body, face, edge,
                title_size=10.2, body_size=8.0)
        arrow(ax, (50, 57), (x + 11.25, 51.3), color=BLUE_EDGE)

    # Data notes associated with branches.
    add_box(ax, 2, 27.1, 22.5, 6.7, "Locked benchmark",
            "uploaded 14k / 3k / 3k split\nremove exact train/test duplicate",
            GREY, LINE, title_size=9.2, body_size=7.6)
    add_box(ax, 26.5, 27.1, 22.5, 6.7, "Human-written data",
            "taxonomy + examples\nquestion-type data is auxiliary only",
            GREY, LINE, title_size=9.2, body_size=7.6)
    add_box(ax, 51, 27.1, 22.5, 6.7, "Licensed / requested",
            "GupShup access + terms\nreleased checkpoints = baselines",
            GREY, LINE, title_size=9.2, body_size=7.6)
    add_box(ax, 75.5, 27.1, 22.5, 6.7, "Controlled QA corpus",
            "Code-Mixed QA Hinglish subset\nhuman-reviewed support context",
            GREY, LINE, title_size=9.2, body_size=7.6)
    for x in xs:
        arrow(ax, (x + 11.25, 37), (x + 11.25, 33.8), color=LINE, dashed=True, lw=1.2)

    # Cross-cutting evaluation and deployment gate.
    ax.add_patch(FancyBboxPatch((5, 4), 90, 19.3,
                                boxstyle="round,pad=0.8,rounding_size=2.2",
                                linewidth=1.7, edgecolor=ORANGE_EDGE,
                                facecolor="#FFF9EE"))
    ax.text(7, 21.2, "EVALUATION + DEPLOYMENT GATE",
            fontsize=12, fontweight="bold", color=ORANGE_EDGE, va="top")
    add_box(ax, 8, 8.2, 24, 9.2, "Robustness challenge sets",
            "spelling drift • typos • code-mix\nemoji pragmatics • punctuation\nraw-only vs raw+phonetic ablations",
            ORANGE, ORANGE_EDGE, body_size=7.7)
    add_box(ax, 38, 8.2, 24, 9.2, "Reproducible reporting",
            "macro-F1 / EM / ROUGE\nper-condition + calibration\nerror slices + data cards",
            BLUE, BLUE_EDGE, body_size=7.7)
    add_box(ax, 68, 8.2, 24, 9.2, "CPU-first serving",
            "distill → int8 quantize\nbatch size 1: p50 / p95\nparams • memory • model hash",
            TEAL, TEAL_EDGE, body_size=7.7)
    arrow(ax, (20, 27), (20, 17.6), color=ORANGE_EDGE, dashed=True, lw=1.3)
    arrow(ax, (50, 27), (50, 17.6), color=ORANGE_EDGE, dashed=True, lw=1.3)
    arrow(ax, (80, 27), (80, 17.6), color=ORANGE_EDGE, dashed=True, lw=1.3)

    ax.text(50, 1.6,
            "Solid arrows = inference flow   |   Dashed arrows = data / evaluation controls   |   Hinglish scope is explicit",
            ha="center", va="bottom", fontsize=8.5, color=MUTED)
    fig.savefig(OUT, dpi=180, facecolor=BG, bbox_inches="tight", pad_inches=0.18)
    print(OUT)


if __name__ == "__main__":
    main()
