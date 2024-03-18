import argparse
import os
import warnings
from collections import defaultdict
from datetime import datetime, timedelta

import pandas as pd

from forecaster.data import UNSEEN_USER_ID
from forecaster.data.data_preprocessor import UserDataPreprocessor
from forecaster.inference.forecaster import UserGmvForecaster
from forecaster.logger.logging import setup_logger

# Filter out specific warning messages or categories
warnings.filterwarnings("ignore")

LOGGER = setup_logger()


def fetch_args() -> argparse.Namespace:
    arg_parser = argparse.ArgumentParser()

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
        "--start_date",
        type=str,
        default="20220101",
        dest="start_date",
        help="Start date for forecasting with format `yyyymmdd`",
    )
    arg_parser.add_argument(
        "--end_date",
        type=str,
        default="20220131",
        dest="end_date",
        help="End date for forecasting with format `yyyymmdd`",
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
        "--user_result_name",
        type=str,
        dest="user_result_name",
        default="user_gmv",
        help="User GMV forecasting result",
    )
    arg_parser.add_argument(
        "--daily_result_name",
        type=str,
        dest="daily_result_name",
        default="daily_gmv",
        help="PayPay daily GMV forecasting result",
    )
    return arg_parser.parse_args()


def run_job(args: argparse.Namespace) -> None:
    forecaster = UserGmvForecaster(
        model_path=os.path.join(args.save_dir, f"{args.model_name}.pt"),
        users_csv_path=args.user_data_path,
        stores_csv_path=args.store_data_path,
        transactions_csv_path=args.transaction_data_path,
    )
    start_date = datetime.strptime(args.start_date, "%Y%m%d")
    end_date = datetime.strptime(args.end_date, "%Y%m%d")
    date_range = [
        start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)
    ]

    result = defaultdict(dict)
    for predicted_date in [date.strftime("%Y%m%d") for date in date_range]:
        LOGGER.info(f"Forecast user GMV on {predicted_date}")
        result[predicted_date] = forecaster.forecast(predicted_date)

    user_mapping = UserDataPreprocessor(args.user_data_path).process()[
        ["user_id", "user_id_label"]
    ]

    data = [
        {"date": date, "user_id_label": user_id, "gmv": gmv}
        for date, gmv_data in result.items()
        for user_id, gmv in gmv_data.items()
    ]
    df = pd.DataFrame(data)
    df_merged = pd.merge(df, user_mapping, on="user_id_label", how="left")
    df_merged.drop(columns=["user_id_label"], inplace=True)
    total_user_gmv_pdf = (
        df_merged[df_merged["user_id"] != UNSEEN_USER_ID]
        .groupby("user_id")["gmv"]
        .sum()
        .reset_index()
    )
    user_gmv_path = os.path.join(
        "results", f"{args.user_result_name}_{args.start_date}_{args.end_date}.csv"
    )
    total_user_gmv_pdf.to_csv(user_gmv_path, index=False)

    daily_gmv_pdf = df_merged.groupby("date")["gmv"].sum().reset_index()
    daily_gmv_path = os.path.join(
        "results", f"{args.daily_result_name}_{args.start_date}_{args.end_date}.csv"
    )
    daily_gmv_pdf.to_csv(daily_gmv_path, index=False)


if __name__ == "__main__":
    args = fetch_args()
    run_job(args)
