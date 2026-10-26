#!/usr/bin/env python3
"""Generate model vector-size and parameter-count report for presentation slides.

This script avoids notebook execution so it can be run safely from a terminal
while notebooks are busy.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.model import HybridStockfishModel


@dataclass(frozen=True)
class Model1Preset:
    name: str
    conv_channels: int
    residual_blocks: int
    move_dim: int
    transformer_dim: int
    n_layers: int
    n_heads: int
    board_embedding_dim: int
    feature_hidden_dim: int
    fusion_hidden_dim: int


MODEL1_PRESETS = [
    Model1Preset(
        name="tiny",
        conv_channels=16,
        residual_blocks=1,
        move_dim=64,
        transformer_dim=128,
        n_layers=1,
        n_heads=2,
        board_embedding_dim=128,
        feature_hidden_dim=32,
        fusion_hidden_dim=64,
    ),
    Model1Preset(
        name="small",
        conv_channels=32,
        residual_blocks=2,
        move_dim=128,
        transformer_dim=256,
        n_layers=2,
        n_heads=4,
        board_embedding_dim=256,
        feature_hidden_dim=64,
        fusion_hidden_dim=128,
    ),
    Model1Preset(
        name="paper",
        conv_channels=64,
        residual_blocks=6,
        move_dim=320,
        transformer_dim=1024,
        n_layers=12,
        n_heads=8,
        board_embedding_dim=512,
        feature_hidden_dim=64,
        fusion_hidden_dim=256,
    ),
]


def build_model1(preset: Model1Preset, num_classes: int = 2, feature_dim: int = 33) -> HybridStockfishModel:
    board_encoder_kwargs = {}
    if preset.name != "paper":
        board_encoder_kwargs = {
            "n_layers": preset.n_layers,
            "n_heads": preset.n_heads,
            "conv_channels": preset.conv_channels,
            "residual_blocks": preset.residual_blocks,
            "move_dim": preset.move_dim,
            "transformer_dim": preset.transformer_dim,
            "feedforward_dim": preset.transformer_dim * 2,
        }

    return HybridStockfishModel(
        num_classes=num_classes,
        stockfish_feature_dim=feature_dim,
        stockfish_hidden_dim=preset.feature_hidden_dim,
        board_embedding_dim=preset.board_embedding_dim,
        fusion_hidden_dim=preset.fusion_hidden_dim,
        board_encoder_kwargs=board_encoder_kwargs,
    )


def count_trainable_params(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def model1_table(num_classes: int, feature_dim: int, timesteps: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for preset in MODEL1_PRESETS:
        model = build_model1(preset, num_classes=num_classes, feature_dim=feature_dim)
        rows.append(
            {
                "model": "Model 1",
                "preset": preset.name,
                "input_tensor": f"(B, {timesteps}, 34, 8, 8)",
                "move_encoder_channels": preset.conv_channels,
                "residual_blocks": preset.residual_blocks,
                "move_vector_dim": preset.move_dim,
                "transformer_layers": preset.n_layers,
                "transformer_heads": preset.n_heads,
                "transformer_width": preset.transformer_dim,
                "board_embedding_dim": preset.board_embedding_dim,
                "feature_vector_dim": feature_dim,
                "feature_embedding_dim": preset.feature_hidden_dim,
                "fusion_vector_dim": preset.board_embedding_dim + preset.feature_hidden_dim,
                "logit_dim": num_classes,
                "trainable_params": count_trainable_params(model),
            }
        )
    return pd.DataFrame(rows)


def model2_table(feature_dim: int, binary_classes: int) -> pd.DataFrame:
    linear_svm_binary_params = feature_dim + 1  # weights + bias
    rows = [
        {
            "model": "Model 2",
            "classifier": "Linear SVM",
            "input_vector_dim": feature_dim,
            "output_dim": binary_classes,
            "parameter_count": linear_svm_binary_params,
            "count_note": "Binary pairwise case: one linear separator has d weights + 1 bias.",
        },
        {
            "model": "Model 2",
            "classifier": "RBF SVM",
            "input_vector_dim": feature_dim,
            "output_dim": binary_classes,
            "parameter_count": "data-dependent",
            "count_note": "Depends on number of support vectors (not fixed before fitting).",
        },
        {
            "model": "Model 2",
            "classifier": "Random Forest",
            "input_vector_dim": feature_dim,
            "output_dim": binary_classes,
            "parameter_count": "data-dependent",
            "count_note": "Uses 300 trees, max_depth=4 in this project; exact node counts depend on fitted splits.",
        },
    ]
    return pd.DataFrame(rows)


def markdown_report(df1: pd.DataFrame, df2: pd.DataFrame) -> str:
    def frame_to_markdown(df: pd.DataFrame) -> str:
        headers = [str(column) for column in df.columns]
        rows = [[str(value) for value in row] for row in df.to_numpy()]
        all_rows = [headers] + rows
        widths = [max(len(row[i]) for row in all_rows) for i in range(len(headers))]

        def fmt_row(values: list[str]) -> str:
            return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

        sep = "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"
        lines = [fmt_row(headers), sep]
        lines.extend(fmt_row(row) for row in rows)
        return "\n".join(lines)

    lines = [
        "# Model Vector Sizes and Parameter Counts",
        "",
        "## Model 1 (Neural Hybrid)",
        "",
        frame_to_markdown(df1),
        "",
        "## Model 2 (Classical)",
        "",
        frame_to_markdown(df2),
        "",
        "### Presentation shorthand",
        "",
        "- Model 1 (small preset): 34 -> 128 -> 256 + 64 -> 320 -> 2 logits, 1,254,350 trainable params.",
        "- Model 2 (pairwise): 33 -> 2 scores/probabilities; Linear SVM has 34 learned parameters in binary case.",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate architecture size and parameter reports.")
    parser.add_argument("--timesteps", type=int, default=80, help="Padded game length used by Model 1.")
    parser.add_argument("--feature-dim", type=int, default=33, help="Model 2 feature vector size.")
    parser.add_argument("--num-classes", type=int, default=2, help="Output classes for Model 1 report.")
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Output directory where CSV and Markdown reports will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_model1 = model1_table(num_classes=args.num_classes, feature_dim=args.feature_dim, timesteps=args.timesteps)
    df_model2 = model2_table(feature_dim=args.feature_dim, binary_classes=2)

    model1_csv = out_dir / "model1_size_report.csv"
    model2_csv = out_dir / "model2_size_report.csv"
    report_md = out_dir / "model_size_report.md"

    df_model1.to_csv(model1_csv, index=False)
    df_model2.to_csv(model2_csv, index=False)
    report_md.write_text(markdown_report(df_model1, df_model2), encoding="utf-8")

    print(f"Wrote {model1_csv}")
    print(f"Wrote {model2_csv}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
