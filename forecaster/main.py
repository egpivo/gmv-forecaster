from argparse import ArgumentParser

import torch

from forecaster.logger.logging import setup_logger
from forecaster.training.RollingWindowTrainer import RollingWindowTrainer

LOGGER = setup_logger()


def fetch_args() -> "argparse.Namespace":
    arg_parser = ArgumentParser()

    arg_parser.add_argument(
        "--user_data_path",
        type=str,
        dest="user_data_path",
        help="User data path",
    )
    arg_parser.add_argument(
        "--transaction_data_path",
        type=str,
        dest="transaction_data_path",
        help="Transaction data path",
    )
    arg_parser.add_argument(
        "--store_data_path",
        type=str,
        dest="store_data_path",
        help="Store data path",
    )
    arg_parser.add_argument(
        "--start_month",
        type=str,
        default="202101",
        dest="start_month",
        help="Start date for training with format `yyyymm`",
    )
    arg_parser.add_argument(
        "--end_month",
        type=str,
        default="202112",
        dest="end_month",
        help="Start date for training with format `yyyymm`",
    )
    arg_parser.add_argument(
        "--learning_rate",
        default=1e-3,
        type=float,
        dest="learning_rate",
        help="Learning rate",
    )
    arg_parser.add_argument(
        "--embed_dim",
        default=128,
        type=int,
        dest="embed_dim",
        help="Embedding dim",
    )
    arg_parser.add_argument(
        "--batch_size",
        default=1024,
        type=int,
        dest="batch_size",
        help="Batch size",
    )
    arg_parser.add_argument(
        "--dropout",
        default=0.2,
        type=float,
        dest="dropout",
        help=f"Dropout",
    )
    arg_parser.add_argument(
        "--weight_decay",
        default=0.1,
        type=float,
        dest="weight_decay",
        help=f"Weight decay",
    )
    arg_parser.add_argument(
        "--save_dir",
        type=str,
        dest="save_dir",
        default="checkpoint/",
        help="Directory for model saving",
    )
    arg_parser.add_argument(
        "--model_name",
        type=str,
        dest="model_name",
        default="xdfm",
        help="Model name",
    )
    arg_parser.add_argument(
        "--epoch",
        type=int,
        dest="epoch",
        default=5,
        help="Number of epochs",
    )
    arg_parser.add_argument(
        "--num_workers",
        type=int,
        dest="num_workers",
        default=32,
        help="Number of CPU cores for parallel computing",
    )
    return arg_parser.parse_args()


def run_job(args: "argparse.Namespace", device: torch.device) -> None:
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
    if torch.cuda.is_available():
        device = torch.device("cuda")
        LOGGER.info(
            "Using GPU:", torch.cuda.get_device_name(0)
        )  # Print the GPU device name
    else:
        device = torch.device("cpu")
        LOGGER.info("CUDA is not available. Using CPU instead.")

    run_job(args, device)
