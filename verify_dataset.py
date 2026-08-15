import pandas as pd


EXPECTED_COLUMNS = [
    "order_id",
    "product_category",
    "price_inr",
    "discount_pct",
    "payment_method",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
    "returned",
]


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def report_shape(df: pd.DataFrame) -> None:
    print("A. Dataset shape")
    print("- rows:", len(df))
    print("- columns:", len(df.columns))
    print("- column names:", list(df.columns))
    print()


def report_overall_statistics(df: pd.DataFrame) -> None:
    return_rate = df["returned"].mean()
    missing_rating_pct = df["rating_given"].isna().mean()

    print("B. Overall statistics")
    print("- overall return rate:", format_percent(return_rate))
    print("- missing rating_given percentage:", format_percent(missing_rating_pct))
    print()


def report_group_statistics(df: pd.DataFrame) -> None:
    print("C. Return rate by product_category")
    print(df.groupby("product_category")["returned"].mean().apply(format_percent).to_string())
    print()

    print("D. Return rate by payment_method")
    print(df.groupby("payment_method")["returned"].mean().apply(format_percent).to_string())
    print()

    print("E. Missing rating_given percentage by payment_method")
    missing_by_payment = (
        df.groupby("payment_method")["rating_given"].apply(lambda s: s.isna().mean())
    )
    print(missing_by_payment.apply(format_percent).to_string())
    print()


def report_missingness_analysis(df: pd.DataFrame) -> None:
    cod_missing_rate = df.loc[df["payment_method"] == "COD", "rating_given"].isna().mean()
    non_cod_missing_rate = df.loc[df["payment_method"] != "COD", "rating_given"].isna().mean()
    gap = cod_missing_rate - non_cod_missing_rate

    print("G. Missingness analysis")
    print("- rating_given missing rate for COD orders:", format_percent(cod_missing_rate))
    print("- rating_given missing rate for non-COD orders:", format_percent(non_cod_missing_rate))
    print("- gap (COD minus non-COD):", format_percent(gap))
    print("- explanation: rating_given missingness depends on the observed payment_method, so this is MAR.")
    print()


def verify_acceptance_criteria(df: pd.DataFrame) -> bool:
    row_count_ok = len(df) == 6000
    col_count_ok = len(df.columns) == 13
    columns_ok = list(df.columns) == EXPECTED_COLUMNS
    return_rate = df["returned"].mean()
    missing_rating_pct = df["rating_given"].isna().mean()
    return_rate_ok = 0.18 <= return_rate <= 0.27
    missing_rating_ok = 0.08 <= missing_rating_pct <= 0.18

    print("F. Acceptance criteria")
    print(f"- exact rows == 6000: {'PASS' if row_count_ok else 'FAIL'}")
    print(f"- exact columns == 13: {'PASS' if col_count_ok else 'FAIL'}")
    print(f"- expected column names: {'PASS' if columns_ok else 'FAIL'}")
    print(
        f"- return rate between 18% and 27%: {'PASS' if return_rate_ok else 'FAIL'}"
    )
    print(
        f"- missing rating_given between 8% and 18%: {'PASS' if missing_rating_ok else 'FAIL'}"
    )
    print()

    if not row_count_ok:
        print("  actual rows:", len(df))
    if not col_count_ok:
        print("  actual columns:", len(df.columns))
    if not columns_ok:
        print("  actual column names:", list(df.columns))
    if not return_rate_ok:
        print("  actual return rate:", format_percent(return_rate))
    if not missing_rating_ok:
        print("  actual missing rating_given:", format_percent(missing_rating_pct))

    return all(
        [
            row_count_ok,
            col_count_ok,
            columns_ok,
            return_rate_ok,
            missing_rating_ok,
        ]
    )


def assert_data_quality(df: pd.DataFrame) -> None:
    assert len(df) == 6000, "Dataset must contain exactly 6000 rows."
    assert len(df.columns) == 13, "Dataset must contain exactly 13 columns."
    assert list(df.columns) == EXPECTED_COLUMNS, (
        "Dataset columns do not match the expected schema. "
        f"Found: {list(df.columns)}"
    )


def main() -> None:
    df = load_dataset("orders_dataset.csv")

    report_shape(df)
    report_overall_statistics(df)
    report_group_statistics(df)
    verify_acceptance_criteria(df)
    report_missingness_analysis(df)
    assert_data_quality(df)

    print("Verification complete.")


if __name__ == "__main__":
    main()
