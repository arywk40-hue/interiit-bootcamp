from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parents[1] / "reports" / "sentiment" / "sentiment_architecture.png"


def box(ax, x, y, w, h, title, lines, face, edge="#263650"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.35,rounding_size=0.8",
            facecolor=face, edgecolor=edge, linewidth=1.4,
        )
    )
    ax.text(x + 0.7, y + h - 1.0, title, ha="left", va="top",
            fontsize=11.5, fontweight="bold", color="#15223A")
    ax.text(x + 0.7, y + h - 2.5, lines, ha="left", va="top",
            fontsize=8.9, linespacing=1.35, color="#263650")


def arrow(ax, a, b, color="#61708A", dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            a, b, arrowstyle="-|>", mutation_scale=14,
            linewidth=1.5, color=color,
            linestyle="--" if dashed else "-",
            shrinkA=5, shrinkB=5,
        )
    )


def main():
    bg = "#F8FAFD"
    fig, ax = plt.subplots(figsize=(17, 9.5), dpi=180)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")

    ax.text(50, 57.4, "Hinglish Sentiment — Selected Architecture",
            ha="center", va="top", fontsize=21, fontweight="bold", color="#15223A")
    ax.text(50, 54.8,
            "Character-first baseline chosen by measured development macro-F1 and batch-one CPU latency",
            ha="center", va="top", fontsize=10.5, color="#5A667C")

    box(ax, 3, 37, 19, 12, "Raw Hinglish", "Romanized Hindi + English\nemoji and punctuation retained\nno transliteration", "#E2F0FF", "#3478C5")
    box(ax, 28, 37, 20, 12, "Character TF-IDF", "3–5 character n-grams\nmin document frequency: 2\n180,000-feature cap", "#DDF3EF", "#258A7E")
    box(ax, 54, 37, 20, 12, "Linear classifier", "class-balanced one-vs-rest\nlogistic regression\n540,003 learned values", "#ECE4FF", "#7854B7")
    box(ax, 80, 37, 17, 12, "Prediction", "negative / neutral / positive\nclass probabilities\nconfidence + latency", "#FFF0D8", "#C57A17")
    arrow(ax, (22, 43), (28, 43))
    arrow(ax, (48, 43), (54, 43))
    arrow(ax, (74, 43), (80, 43))

    ax.text(50, 31.9, "TRAINING AND SELECTION EVIDENCE",
            ha="center", va="top", fontsize=10.5, fontweight="bold", color="#5A667C")
    box(ax, 3, 17, 27, 11, "Leakage-safe corpus", "13,997 train after cleaning\n3,000 dev • 3,000 locked test\n2 train/test leaks removed\n2,777 corrupt token rows repaired", "#EEF1F5", "#6B7890")
    box(ax, 36.5, 17, 27, 11, "Frozen selection rule", "models within 0.002 macro-F1\nof best dev result are equivalent\nchoose the lowest p95 latency", "#EEF1F5", "#6B7890")
    box(ax, 70, 17, 27, 11, "Final measured result", "test macro-F1: 0.6857\nmodel size: 5.59 MB\nCPU p50 / p95: 0.59 / 0.83 ms\n1,635 sequential predictions/s", "#E4F4E4", "#438E49")
    arrow(ax, (16.5, 28), (34, 37), dashed=True)
    arrow(ax, (50, 28), (64, 37), dashed=True)
    arrow(ax, (83.5, 28), (88.5, 37), dashed=True)

    ax.text(3, 11.8, "Ablations retained as results", fontsize=10.3,
            fontweight="bold", color="#15223A")
    ax.text(3, 9.5,
            "Word + character: +0.00005 dev macro-F1 but ~2.1× p95 latency   •   Raw + canonical view: −0.0115 macro-F1 and ~2.7× p95 latency",
            fontsize=9.3, color="#263650")
    ax.text(3, 6.5,
            "Emoji-symbol channel: −0.0138 dev F1   •   vowel deletion: −0.0032   •   repeated-letter noise: −0.0080   •   emoji removal: −0.0064",
            fontsize=9.3, color="#263650")
    ax.text(3, 3.3,
            "Open limitation: synthetic perturbations do not replace a human-written spelling + emoji contrast set.",
            fontsize=9.2, fontweight="bold", color="#9A5B0B")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.2, facecolor=bg)
    print(OUT)


if __name__ == "__main__":
    main()
