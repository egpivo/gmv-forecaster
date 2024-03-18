import argparse

import torch

from forecaster.logger.logging import setup_logger
from forecaster.training.rolling_window_trainer import RollingWindowTrainer

LOGGER = setup_logger()


def fetch_args() -> argparse.Namespace:
    """Fetch command-line arguments."""
    arg_parser = argparse.ArgumentParser(description="Arguments for training model.")
    arg_parser.add_argument("--user_data_path", type=str, help="Path to user data.")
    arg_parser.add_argument(
        "--transaction_data_path", type=str, help="Path to transaction data."
    )
    arg_parser.add_argument("--store_data_path", type=str, help="Path to store data.")
    arg_parser.add_argument(
        "--start_month",
        type=str,
        default="202101",
        help="Start date for training in 'yyyymm' format.",
    )
    arg_parser.add_argument(
        "--end_month",
        type=str,
        default="202112",
        help="End date for training in 'yyyymm' format.",
    )
    arg_parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate."
    )
    arg_parser.add_argument(
        "--embed_dim", type=int, default=64, help="Embedding dimension."
    )
    arg_parser.add_argument("--batch_size", type=int, default=1024, help="Batch size.")
    arg_parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    arg_parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight decay."
    )
    arg_parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoint/",
        help="Directory for saving models.",
    )
    arg_parser.add_argument(
        "--model_name", type=str, default="xdfm", help="Model name."
    )
    arg_parser.add_argument("--epoch", type=int, default=3, help="Number of epochs.")
    arg_parser.add_argument(
        "--num_workers",
        type=int,
        default=32,
        help="Number of CPU cores for parallel computing.",
    )
    return arg_parser.parse_args()


def run_job(args: argparse.Namespace, device: torch.device) -> None:
    """Run the training job."""
    trainer = RollingWindowTrainer(
        start_month=args.start_month,
        end_month=args.end_month,
        user_data_path=args.user_data_path,
        transaction_data_path=args.transaction_data_path,
        store_data_path=args.store_data_path,
        embed_dim=args.embed_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        epoch=args.epoch,
        dropout=args.dropout,
        num_workers=args.num_workers,
        model_name=args.model_name,
        device=device,
        logger=LOGGER,
    )
    trainer.train()


if __name__ == "__main__":
    args = fetch_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")
    run_job(args, device)
