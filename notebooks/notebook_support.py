"""Configure the English-only experiment and present SQL-derived results."""

import os

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Markdown, display


def configure(method):
    if method not in ("strip_stopwords", "normalize"):
        raise ValueError("CLEANING_METHOD must be 'strip_stopwords' or 'normalize'")
    previous = os.environ.get("VGI_NLP_NO_MODEL_DOWNLOAD")
    os.environ["VGI_NLP_NO_MODEL_DOWNLOAD"] = "1"
    return previous


def cleanup(previous):
    if previous is None:
        os.environ.pop("VGI_NLP_NO_MODEL_DOWNLOAD", None)
    else:
        os.environ["VGI_NLP_NO_MODEL_DOWNLOAD"] = previous


def prose(text: str) -> None:
    display(Markdown(text))


def theme() -> None:
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.dpi": 145,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": "#E8EDE9",
            "text.color": "#243B2F",
            "axes.labelcolor": "#405047",
            "text.parse_math": False,
            "legend.frameon": False,
        },
    )


def chart(fig, title: str, subtitle: str, source: str) -> None:
    fig.text(0.08, 0.98, title, ha="left", va="top", fontsize=17, fontweight="semibold")
    fig.text(0.08, 0.90, subtitle, ha="left", va="top", fontsize=10, color="#58685F")
    fig.tight_layout(rect=(0.02, 0.11, 1, 0.82))
    fig.text(0.08, 0.03, source, fontsize=8, color="#6C7D73")
    plt.show()
