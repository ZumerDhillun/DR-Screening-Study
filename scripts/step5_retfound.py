"""Run the RETFound arm through the existing Step 5 training pipeline.

This launcher deliberately delegates all training, validation, temperature
scaling, threshold selection, checkpointing, and output writing to
``step5_train.py`` so the RETFound arm follows the same code path as the
completed ConvNeXt-V2-Large arm.

Example (Colab):
    python scripts/step5_retfound.py \
        --checkpoint /content/RETFound_cfp_weights.pth \
        --out_dir "/content/drive/My Drive/retinavision_models"
"""
import argparse
from pathlib import Path

from step5_train import train_model


def main():
    parser = argparse.ArgumentParser(
        description="Continue Step 5 by training RETFound with the shared pipeline."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the official RETFound CFP .pth checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--out_dir",
        default="/content/drive/My Drive/retinavision_models",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        parser.error(f"RETFound checkpoint not found: {args.checkpoint}")

    train_model(
        model_key="retfound",
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=args.out_dir,
        retfound_checkpoint=str(args.checkpoint),
    )


if __name__ == "__main__":
    main()
