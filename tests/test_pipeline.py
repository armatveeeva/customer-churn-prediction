from pathlib import Path

import pandas as pd
import pytest

from src.constants import SERVICE_COLS
from src.data import create_advanced_features, load_raw_data
from src.model_training import TARGET, build_dataset, prepare_features

NEW_FEATURES = [
    "Группа срока",
    "Соотношение платежей",
    "Количество услуг",
    "Есть оптоволокно",
    "Есть стриминг",
    "Способ оплаты_TE",
]


def test_create_advanced_features_adds_columns(prepared_df: pd.DataFrame) -> None:
    result = create_advanced_features(prepared_df.copy())

    for col in NEW_FEATURES:
        assert col in result.columns
    assert result["Количество услуг"].between(0, len(SERVICE_COLS)).all()
    assert result["Группа срока"].notna().all()


def test_target_encoding_uses_given_means(prepared_df: pd.DataFrame) -> None:
    means = pd.Series({"Электронный чек": 0.5, "Чек почтой": 0.1, "Банковский перевод": 0.2})

    result = create_advanced_features(prepared_df.copy(), payment_means=means)

    known = result["Способ оплаты"] == "Электронный чек"
    assert (result.loc[known, "Способ оплаты_TE"] == 0.5).all()

    # Категория, которой нет в переданных средних, получает среднее по ним, а не NaN
    unseen = result["Способ оплаты"] == "Кредитная карта"
    assert unseen.any()
    assert result["Способ оплаты_TE"].notna().all()
    assert result.loc[unseen, "Способ оплаты_TE"].eq(means.mean()).all()


def test_prepare_features_shape_and_columns(prepared_df: pd.DataFrame) -> None:
    df = create_advanced_features(prepared_df.copy())

    X, y = prepare_features(df)

    assert len(X) == len(y) == len(df)
    assert set(y.unique()) <= {0, 1}
    for col in ["Отток", TARGET] + [c + "_num" for c in SERVICE_COLS]:
        assert col not in X.columns
    assert X.notna().all().all()
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes)


def test_build_dataset_split_sizes(prepared_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.model_training.load_and_prepare_data", lambda: prepared_df.copy())

    X_train, X_test, y_train, y_test = build_dataset()

    assert set(X_train.index).isdisjoint(X_test.index)
    assert len(X_train) + len(X_test) == len(prepared_df)
    assert len(X_test) == pytest.approx(0.2 * len(prepared_df), abs=1)
    assert list(X_train.columns) == list(X_test.columns)
    assert (X_train.index == y_train.index).all()
    assert (X_test.index == y_test.index).all()


def test_target_encoding_is_computed_on_train_only(
    prepared_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Тест на утечку: у всех строк (и train, и test) target encoding равен средней доле оттока по train."""
    monkeypatch.setattr("src.model_training.load_and_prepare_data", lambda: prepared_df.copy())

    X_train, X_test, y_train, _ = build_dataset()

    payment_train = prepared_df.loc[X_train.index, "Способ оплаты"]
    train_means = y_train.groupby(payment_train).mean()

    for X in (X_train, X_test):
        payment = prepared_df.loc[X.index, "Способ оплаты"]
        expected = payment.map(train_means).to_numpy(dtype=float)
        actual = X["Способ оплаты_TE"].to_numpy(dtype=float)
        assert actual == pytest.approx(expected)

    # Средние по всему датасету отличаются от средних по train, так что тест реально различает два варианта
    full_means = prepared_df.groupby("Способ оплаты")[TARGET].mean()
    assert not full_means.equals(train_means)


def test_load_raw_data_downloads_once_and_then_uses_cache(
    prepared_df: pd.DataFrame, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.csv"
    prepared_df.to_csv(source, index=False)
    cache = tmp_path / "data" / "telco_churn.csv"

    # Вместо сети «скачиваем» локальный файл
    monkeypatch.setattr("src.data.DATA_URL", str(source))
    first = load_raw_data(cache)
    assert cache.exists()

    # После появления кэша источник больше не нужен
    source.unlink()
    second = load_raw_data(cache)

    assert first.shape == second.shape == prepared_df.shape
