"""Present SQL-returned English text transformation and extraction audits."""

import matplotlib.pyplot as plt
import numpy as np
from vgi_notebooks import finish

SOURCE = (
    "Authored English inputs · VGI NLP v0.4.0 · spaCy en_core_web_sm 3.8.0\n"
    "Local inference · labels and spans are model outputs, not verified facts"
)


def scores(comparison_frame, CLEANING_METHOD):
    """Display returned values with their declared comparison population."""
    available = comparison_frame[comparison_frame.comparable].copy()
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    for i, item in enumerate(available.itertuples()):
        ax.plot(
            [item.original_score, item.cleaned_score],
            [i, i],
            color="#B16D44" if item.label_differs else "#ABB8B0",
            lw=2.4,
        )
    ax.scatter(
        available.original_score,
        range(len(available)),
        color="#889A90",
        s=42,
        marker="o",
        label="Original",
        zorder=3,
    )
    ax.scatter(
        available.cleaned_score,
        range(len(available)),
        color="#327659",
        s=48,
        marker="D",
        label="Cleaned",
        zorder=4,
    )
    ax.axvline(-0.05, color="#B6C1BA", ls=":")
    ax.axvline(0.05, color="#B6C1BA", ls=":")
    ax.set_yticks(range(len(available)), available.case_name, fontsize=9)
    ax.invert_yaxis()
    ax.set(xlabel="VADER compound score (dimensionless; not a probability)", xlim=(-1.05, 1.05))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=2, fontsize=9)
    finish(
        fig,
        "Cleaning can change a lexicon model’s polarity label",
        f"Cleaning: {CLEANING_METHOD} • orange connectors: label changed • dotted lines: ±0.05 thresholds",
        "Authored English probes • actual worker VADER 3.3.2 outputs • no human sentiment ground truth.\n"
        "Only pairs with both scores are plotted; coincident symbols indicate equal scores.",
        name="sentiment-pairs",
    )


def tokens(tokens_frame, CLEANING_METHOD):
    """Display returned values with their declared comparison population."""
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    categories = list(tokens_frame.token_class.unique())
    for offset, variant, color in [(-0.18, "Original", "#A7B6AD"), (0.18, "Cleaned", "#327659")]:
        values = [
            int(
                tokens_frame[
                    (tokens_frame.token_class == name) & (tokens_frame.variant == variant)
                ].tokens.iloc[0]
            )
            for name in categories
        ]
        bars = ax.bar(
            [i + offset for i in range(len(categories))], values, width=0.34, color=color, label=variant
        )
        ax.bar_label(bars, padding=4)
    ax.set_xticks(
        range(len(categories)), ["Listed negation\ntokens", "Other\nstopwords", "Punctuation", "Other tokens"]
    )
    ax.set_ylabel("Tokens across all authored inputs (count)")
    ax.set_ylim(0, max(1, tokens_frame.tokens.max()) * 1.28)
    ax.legend(loc="upper right", fontsize=9)
    finish(
        fig,
        "Check which language features survive the selected transform",
        f"Cleaning: {CLEANING_METHOD} • same spaCy en_core_web_sm 3.8.0 model for both forms",
        "Authored English probes • worker tokenization + SQL classification "
        "• listed negators are not exhaustive.\n"
        "Token counts describe the transform; they do not establish sentiment preservation in general.",
        name="token-categories",
    )


def extraction_counts(audit, form):
    """Keep every known input ID on independent sentence/entity count axes."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.7), sharey=True)
    y = np.arange(len(audit))
    for ax, column, label in zip(
        axes, ["sentence_count", "entity_count"], ["Sentence rows", "Entity rows"], strict=True
    ):
        ax.barh(y, audit[column], color="#397d62" if column == "entity_count" else "#397f91")
        for i, v in enumerate(audit[column]):
            ax.text(v + 0.06, i, str(v), va="center", fontsize=10)
        ax.set_xticks(range(0, int(audit[column].max()) + 1))
        ax.set_xlim(0, max(1, audit[column].max()) + 0.7)
        ax.set_xlabel(label + " (count)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [f"{r.id} · {r.case_name}" for r in audit.itertuples()])
    axes[0].invert_yaxis()
    finish(
        fig,
        "Extraction emits different row populations",
        f"Text form: {form} · all six input IDs retained · separate count axes",
        SOURCE,
        name="extraction-counts",
    )


def characters(audit, form):
    """Reconcile entity-span lengths with the exact processed character population."""
    fig, ax = plt.subplots(figsize=(11.5, 5.7))
    y = np.arange(len(audit))
    ax.barh(y, audit.entity_characters, color="#397d62", label="Returned entity spans")
    ax.barh(
        y,
        audit.other_characters,
        left=audit.entity_characters,
        color="#d8e1dc",
        label="Other processed characters",
    )
    for i, r in enumerate(audit.itertuples()):
        label = (
            "missing input"
            if r.missing_input
            else f"{int(r.entity_characters)} / {int(r.characters_for_accounting)}"
        )
        ax.text(r.characters_for_accounting + 1, i, label, va="center", fontsize=10)
    ax.set_yticks(y, [f"{r.id} · {r.case_name}" for r in audit.itertuples()])
    ax.invert_yaxis()
    ax.set_xlim(0, max(audit.characters_for_accounting) + 18)
    ax.set_xlabel("Processed Unicode characters (count)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize=9)
    finish(
        fig,
        "Span lengths account for part of processed text",
        f"Text form: {form} · nonoverlapping half-open offsets · annotation coverage, not recall",
        SOURCE,
        name="entity-character-coverage",
    )
