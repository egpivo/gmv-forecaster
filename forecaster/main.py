from argparse import ArgumentParser

from forecaster.training.trainer import Trainer


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
        "--learning_rate",
        default=1e-3,
        type=float,
        dest="learning_rate",
        help="Learning rate",
    )
    arg_parser.add_argument(
        "--batch_size",
        default=128,
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
        default=10,
        help="Number of ephocs",
    )
    arg_parser.add_argument(
        "--num_workers",
        type=int,
        dest="num_workers",
        default=32,
        help="Number of CPU cores for parallel computing",
    )
    return arg_parser.parse_args()


def run_job(args: "argparse.Namespace") -> None:
    trainer = Trainer(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        save_dir=args.save_dir,
        user_data_path=args.user_data_path,
        transaction_data_path=args.transaction_data_path,
        store_data_path=args.store_data_path,
        epoch=args.epoch,
        dropout=args.dropout,
        num_workers=args.num_workers,
        model_name=args.model_name,
    )
    trainer.train()


if __name__ == "__main__":
    args = fetch_args()
    run_job(args)
