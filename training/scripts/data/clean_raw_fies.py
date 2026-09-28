"""Create reproducible, HFCE-aligned annual FIES anchors from RAW/uncleaned."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import logging
import os
import secrets
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SPEC_VERSION = "2026-09-24.2"
NCR_REGION_CODE = "13"
LOGGER = logging.getLogger(__name__)
PHP_TOLERANCE = 0.01
CATEGORIES = (
    "food",
    "alcohol_tobacco",
    "clothing_footwear",
    "housing_water_utilities",
    "furnishings_household",
    "health",
    "transport",
    "communication",
    "recreation_culture",
    "education",
    "restaurants_hotels",
    "miscellaneous_goods_services",
)
READ_CHUNK_SIZE = 1_000
PROGRESS_INTERVAL = 25_000


@dataclass(frozen=True)
class YearSpec:
    source: str
    dictionary: str
    keys: tuple[str, ...]
    region: str
    total: str
    income: str
    base_fields: dict[str, tuple[str, ...]]
    transport_weight: float | None = None
    include_trcom_cards: bool = False
    unresolved_food_restaurants: bool = False
    restaurant_proxy: bool = False
    restaurant_caveat: str | None = None


SPECS = {
    2000: YearSpec(
        "Volume 2/FIES PUF 2002 Vol.2.csv",
        "Volume 2/fies_2000_v2_metadata(dictionary) .xlsx",
        ("REGION", "PROVINCE", "DOMAIN", "HCN"),
        "REGION",
        "TOTEX",
        "TOINC",
        {
            "food": ("FOOD", "FDOUT"), "alcohol_tobacco": ("ALBEV", "TBCCO"),
            "clothing_footwear": ("CLOTH",), "housing_water_utilities": ("FUEL", "HOUSE"),
            "furnishings_household": ("HOPER", "NDFUR", "DUFUR"), "health": ("MEDIC",),
            "recreation_culture": ("RCRTN",), "education": ("EDUC",),
            "restaurants_hotels": ("FDOUT",), "miscellaneous_goods_services": ("PRCRE",),
        },
        transport_weight=242119 / 311227,
        include_trcom_cards=True,
        restaurant_proxy=True,
        restaurant_caveat="food_service_proxy",
    ),
    2003: YearSpec(
        "Final Version/FIES PUF 2003 final version.CSV",
        "Final Version/fies_2003_fv_metadata(dictionary).xlsx",
        ("REGN", "EA", "SHSN", "HCN"), "REGN", "TOTEX", "TOINC",
        {
            "food": ("FOOD", "FDOUT"), "alcohol_tobacco": ("ALBEV", "TBCCO"),
            "clothing_footwear": ("CLOTH",), "housing_water_utilities": ("FUEL", "HOUSE"),
            "furnishings_household": ("HOPER", "NDFUR", "DUFUR"), "health": ("MEDIC",),
            "recreation_culture": ("RCRTN",), "education": ("EDUC",),
            "restaurants_hotels": ("FDOUT",), "miscellaneous_goods_services": ("PRCRE",),
        },
        transport_weight=322604 / 447099,
        include_trcom_cards=True,
        restaurant_proxy=True,
        restaurant_caveat="food_service_proxy",
    ),
    2006: YearSpec(
        "Volume 2/FIES PUF 2006 Vol.2.CSV",
        "Volume 2/fies_2006_v2_metadata(dictionary).xlsx",
        ("PUFHHSEQNO",), "W_REGN", "TOTEX", "TOINC",
        {
            "food": ("FOOD", "FDOUT"), "alcohol_tobacco": ("ALBEV", "TBCCO"),
            "clothing_footwear": ("CLOTH",), "housing_water_utilities": ("FUEL1", "HOUSE", "RPAIR"),
            "furnishings_household": ("HOPER", "NDFUR", "DUFUR"), "health": ("MEDIC",),
            "recreation_culture": ("RCRTN",), "education": ("EDUC",),
            "restaurants_hotels": ("FDOUT",), "miscellaneous_goods_services": ("PRCRE",),
        },
        transport_weight=535617 / 720830,
        restaurant_proxy=True,
        restaurant_caveat="food_service_proxy",
    ),
    2009: YearSpec(
        "Volume 2/FIES PUF 2009 Vol.2.CSV",
        "Volume 2/fies_2009_v2_metadata(dictionary).xlsx",
        ("W_REGN", "W_ID", "W_SHSN", "W_HCN"), "W_REGN", "TOTEX", "TOINC",
        {
            "food": ("FOOD", "FDOUT"), "alcohol_tobacco": ("ALBEV", "TBCCO"),
            "clothing_footwear": ("CLOTH",), "housing_water_utilities": ("FUEL", "HOUSE", "RPAIR"),
            "furnishings_household": ("HOPER", "NDFUR", "DUFUR"), "health": ("MEDIC",),
            "recreation_culture": ("RCRTN",), "education": ("EDUC",),
            "restaurants_hotels": ("FDOUT",), "miscellaneous_goods_services": ("PRCRE",),
        },
        transport_weight=665474 / 889373,
        restaurant_proxy=True,
        restaurant_caveat="food_service_proxy",
    ),
    2012: YearSpec(
        "Volume 2/FIES PUF 2012 Vol.2.CSV", "Volume 2/fies_2012_v2_metadata(dictionary).xlsx",
        ("W_REGN", "W_OID", "W_SHSN", "W_HCN"), "W_REGN", "TTOTEX", "TOINC",
        {"food": ("TFOOD", "TFOODOUTSIDE"), "alcohol_tobacco": ("TALCOHOL", "TTOBACCO"),
         "clothing_footwear": ("TCLOTHFOOTWEAR",), "housing_water_utilities": ("THOUSINGWATER",),
         "furnishings_household": ("TFURNISHING",), "health": ("THEALTH",),
         "transport": ("TTRANSPORT",), "communication": ("TCOMMUNICATION",),
         "recreation_culture": ("TRECREATION",), "education": ("TEDUCATION",),
         "restaurants_hotels": ("TRESTAURANT",), "miscellaneous_goods_services": ("TMISCELLANEOUS",)},
    ),
    2015: YearSpec(
        "Volume 1/FIES PUF 2015 Vol.1.CSV", "Volume 1/fies_2015_v1_metadata(dictionary).xlsx",
        ("W_ID", "W_SHSN", "W_HCN"), "W_REGN", "TOTEX", "TOINC",
        {"food": ("FOOD", "FOOD_OUTSIDE"), "alcohol_tobacco": ("ALCOHOL", "TOBACCO"),
         "clothing_footwear": ("CLOTH",), "housing_water_utilities": ("HOUSING_WATER",),
         "furnishings_household": ("FURNISHING",), "health": ("HEALTH",),
         "transport": ("TRANSPORT",), "communication": ("COMMUNICATION",),
         "recreation_culture": ("RECREATION",), "education": ("EDUCATION",),
         "restaurants_hotels": ("FOOD_OUTSIDE", "FOOD_ACCOM_SRVC"), "miscellaneous_goods_services": ("MISCELLANEOUS",)},
    ),
    2018: YearSpec(
        "Volume 2/2018_FIES_vol2.csv", "Volume 2/fies_2018_v2_metadata(dictionary).xlsx",
        ("W_REGN", "W_PROV", "SEQUENCE_NO"), "W_REGN", "TTOTEX", "TOINC",
        {"food": ("TFOOD",), "alcohol_tobacco": ("TALCOHOL", "TTOBACCO"),
         "clothing_footwear": ("TCLOTHFOOTWEAR",), "housing_water_utilities": ("THOUSINGWATER",),
         "furnishings_household": ("TFURNISHING",), "health": ("THEALTH",),
         "transport": ("TTRANSPORT",), "communication": ("TCOMMUNICATION",),
         "recreation_culture": ("TRECREATION",), "education": ("TEDUCATION",),
         "restaurants_hotels": ("TRESTAURANT",), "miscellaneous_goods_services": ("TMISCELLANEOUS",)},
        unresolved_food_restaurants=True,
    ),
    2021: YearSpec(
        "Volume 1/FIES PUF 2021 Volume 1 - HHLD_SUMMARY.CSV", "Volume 1/fies_2021_v1_metadata(dictionary).xlsx",
        ("W_REGN", "W_PROV", "SEQ_NO"), "W_REGN", "TOTEX", "TOINC",
        {"food": ("FOOD", "FOOD_OUTSIDE"), "alcohol_tobacco": ("ALCOHOL", "TOBACCO"),
         "clothing_footwear": ("CLOTH",), "housing_water_utilities": ("HOUSING_WATER",),
         "furnishings_household": ("FURNISHING",), "health": ("HEALTH",), "transport": ("TRANSPORT",),
         "communication": ("COMMUNICATION",), "recreation_culture": ("RECREATION",), "education": ("EDUCATION",),
         "restaurants_hotels": ("FOOD_OUTSIDE", "FOOD_ACCOM_SRVC"), "miscellaneous_goods_services": ("MISCELLANEOUS",)},
    ),
    2023: YearSpec(
        "Volume 1/FIES PUF 2023 Volume1.CSV", "Volume 1/fies_2023_vol1_metadata(dictionary).xlsx",
        ("W_REGN", "W_PROV", "SEQ_NO"), "W_REGN", "TOTEX", "TOINC",
        {"food": ("FOOD", "FOOD_OUTSIDE"), "alcohol_tobacco": ("ALCOHOL", "TOBACCO"),
         "clothing_footwear": ("CLOTH",), "housing_water_utilities": ("HOUSING_WATER",),
         "furnishings_household": ("FURNISHING",), "health": ("HEALTH",), "transport": ("TRANSPORT",),
         "communication": ("COMMUNICATION",), "recreation_culture": ("RECREATION",), "education": ("EDUCATION",),
         "restaurants_hotels": ("FOOD_OUTSIDE", "FOOD_ACCOM_SRVC"), "miscellaneous_goods_services": ("MISCELLANEOUS",)},
    ),
    2025: YearSpec(
        "Volume 1/FIES_PUF_2025_Volume1.CSV", "Volume 1/fies_2025_vol1_metadata(dictionary).xlsx",
        ("W_REGN", "W_PROV", "SEQNO"), "W_REGN", "TOTEX", "TOINC",
        {"food": ("FOOD", "FOOD_OUTSIDE"), "alcohol_tobacco": ("ALCOHOL", "TOBACCO"),
         "clothing_footwear": ("CLOTH",), "housing_water_utilities": ("HOUSING_WATER",),
         "furnishings_household": ("FURNISHING",), "health": ("HEALTH",), "transport": ("TRANSPORT",),
         "communication": ("COMMUNICATION",), "recreation_culture": ("RECREATION",), "education": ("EDUCATION",),
         "restaurants_hotels": ("FOOD_OUTSIDE",), "miscellaneous_goods_services": ("MISCELLANEOUS",)},
        restaurant_proxy=True,
        restaurant_caveat="broad_outside_food_proxy",
    ),
}

# Supplementary files are validation inputs, never additional household samples.
SUPPLEMENTARY_SOURCES: dict[int, tuple[tuple[str, str, tuple[str, ...]], ...]] = {
    2000: (("Volume 3/FIES PUF 2000 Vol.3.CSV", "Volume 3/fies_2000_v3_metadata(dictionary).xlsx", ("REGION", "PROVINCE", "DOMAIN", "HCN")),),
    2003: (("Volume 3/FIES PUF 2003 Vol.3.CSV", "Volume 3/fies_2003_v3_metadata(dictionary).xlsx", ("REGN", "EA", "SHSN", "HCN")),),
    2015: (("Volume 2/FIES PUF 2015 Vol.2.csv", "Volume 2/fies_2015_v2_metadata(dictionary).xlsx", ("W_ID", "W_SHSN", "W_HCN")),),
    2018: (("Volume 1/FIES PUF 2018 Vol.1.CSV", "Volume 1/fies_2018_v1_metadata(dictionary).xlsx", ("W_REGN", "W_PROV", "SEQUENCE_NO")),),
    2021: (
        ("Volume 2/FIES PUF 2021 Volume 2 - HHLD_SUMMARY.CSV", "Volume 2/fies_2021_v2_metadata(dictionary).xlsx", ("W_REGN", "W_PROV", "SEQ_NO")),
        ("Volume 2/FIES PUF 2021 Volume 2 - FOOD_EX.CSV", "Volume 2/fies_2021_v2_metadata(dictionary).xlsx", ("W_REGN", "W_PROV", "SEQ_NO")),
        ("Volume 2/FIES PUF 2021 Volume 2 - NONFOOD.CSV", "Volume 2/fies_2021_v2_metadata(dictionary).xlsx", ("W_REGN", "W_PROV", "SEQ_NO")),
    ),
    2023: (
        ("Volume 2/FIES PUF 2023 Volume2 Household Summary.CSV", "Volume 2/fies_2023_vol2_metadata(dictionary).xlsx", ("W_REGN", "W_PROV", "SEQ_NO")),
        ("Volume 2/FIES PUF 2023 Volume2 FIES Record.CSV", "Volume 2/fies_2023_vol2_metadata(dictionary).xlsx", ("W_REGN", "W_PROV", "SEQ_NO")),
    ),
}


def append_csv(frame: pd.DataFrame, path: Path) -> None:
    """Append a table, writing the header only when the file is new."""
    frame.to_csv(path, index=False, mode="a", header=not path.exists())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_row_count(path: Path) -> int:
    """Count data rows without loading the CSV into memory."""
    newline_count = 0
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            newline_count += block.count(b"\n")
    return max(newline_count - 1, 0)


def required_columns(spec: YearSpec) -> set[str]:
    columns = set(spec.keys) | {spec.region, spec.total, spec.income, "FSIZE"}
    columns.update(column for fields in spec.base_fields.values() for column in fields)
    if spec.transport_weight is not None:
        columns.add("TRCOM")
        if spec is SPECS[2009]:
            return columns
        columns.update(
            {
                "A2022", "A2032", "A2042", "A2052", "A2062", "A2072", "A2082",
                "A2092", "A2102", "A2112", "A2122", "A3022", "A3032", "A3042", "A3052",
                "A3062", "A3072", "A3082", "A3092", "A3102", "A3112", "A3122",
            }
        )
        if spec.include_trcom_cards:
            columns.update({"PCARD", "GPCARD"})
        else:
            columns.update({"A2132", "A3132"})
    return columns


def dictionary_header_ids(path: Path, header: list[str]) -> list[str]:
    """Return dictionary IDs in their published order, restricted to CSV headers."""
    values = pd.read_excel(path, header=None, dtype=str).stack().dropna().str.strip().tolist()
    header_ids = set(header)
    # Some dictionaries repeat an ID in a group label; the first occurrence is
    # the published field position and subsequent occurrences are not columns.
    return list(dict.fromkeys(value for value in values if value in header_ids))


def validation_columns(year: int) -> set[str]:
    columns = {
        2000: {"FHOME"},
        2003: {"FHOME", "MLSCH", "MLWRK", "MLRES", "SNACK"},
        2006: {"FHOME", "MLSCH", "MLWRK", "MLRES", "SNACK"},
        2009: {"FHOME", "MLSCH", "MLWRK", "MLRES", "SNACK"},
        2012: {"TFOODHOME", "TFOODOUTSIDE", "TRESTAURANT"},
        2015: {"FOOD_HOME"},
        2018: {"TFOODHOME", "TFOODOUTSIDE", "TNFOOD"},
        2021: {"FOOD_HOME", "NFOOD"},
        2023: {"FOOD_HOME", "NFOOD"},
        2025: {"FOOD_HOME", "NFOOD"},
    }
    return columns[year]


def relationship_failures(year: int, frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Re-run documented within-source identities without altering source amounts."""
    failures: dict[str, pd.Series] = {}
    if year in {2000, 2003, 2006, 2009}:
        failures["food_total_identity"] = (frame["FOOD"] - frame["FHOME"] - frame["FDOUT"]).abs() > PHP_TOLERANCE
    if year in {2003, 2006, 2009}:
        failures["outside_food_identity"] = (
            frame["FDOUT"] - frame[["MLSCH", "MLWRK", "MLRES", "SNACK"]].sum(axis=1)
        ).abs() > PHP_TOLERANCE
    elif year == 2012:
        failures["food_total_identity"] = (frame["TFOOD"] - frame["TFOODHOME"] - frame["TFOODOUTSIDE"]).abs() > PHP_TOLERANCE
        failures["outside_food_within_restaurants"] = frame["TFOODOUTSIDE"] > frame["TRESTAURANT"] + PHP_TOLERANCE
    elif year == 2018:
        failures["food_total_identity"] = (frame["TFOOD"] - frame["TFOODHOME"] - frame["TFOODOUTSIDE"]).abs() > 1
        failures["total_expense_identity"] = (frame["TTOTEX"] - frame["TFOOD"] - frame["TNFOOD"]).abs() > 9_000
    elif year in {2021, 2023, 2025}:
        tolerance = {2021: 0, 2023: 1e-8, 2025: 0.061}[year]
        total_tolerance = {2021: 0, 2023: 1e-8, 2025: 0.50}[year]
        failures["food_total_identity"] = (frame["FOOD"] - frame["FOOD_HOME"] - frame["FOOD_OUTSIDE"]).abs() > tolerance
        failures["total_expense_identity"] = (frame["TOTEX"] - frame["FOOD"] - frame["NFOOD"]).abs() > total_tolerance
    elif year == 2015:
        failures["food_total_identity"] = (frame["FOOD"] - frame["FOOD_HOME"] - frame["FOOD_OUTSIDE"]).abs() > PHP_TOLERANCE
    return failures


def validate_supplementary_sources(
    year: int, raw_root: Path, primary_keys: pd.DataFrame
) -> list[dict[str, object]]:
    """Validate documented one-to-one joins without exporting raw household keys."""
    results = []
    for source_name, dictionary_name, keys in SUPPLEMENTARY_SOURCES.get(year, ()):
        source = raw_root / str(year) / source_name
        dictionary = raw_root / str(year) / dictionary_name
        if not source.is_file() or not dictionary.is_file():
            raise ValueError(f"{year}: required supplementary validation source is missing: {source_name}")
        header = list(pd.read_csv(source, nrows=0, dtype=str).columns)
        if header != dictionary_header_ids(dictionary, header):
            raise ValueError(f"{year}: supplementary header and dictionary differ: {source_name}")
        if set(keys) - set(header):
            raise ValueError(f"{year}: supplementary source lacks documented key: {source_name}")
        secondary_keys = pd.read_csv(source, usecols=list(keys), dtype=str, na_values=[".", ""])
        invalid_key = secondary_keys.isna().any(axis=1) | secondary_keys.eq("").any(axis=1)
        duplicate_key = secondary_keys.duplicated(keep=False)
        if invalid_key.any() or duplicate_key.any():
            raise ValueError(f"{year}: supplementary source has invalid or duplicate keys: {source_name}")
        joined = primary_keys.merge(secondary_keys, how="outer", indicator=True)
        if len(primary_keys) != len(secondary_keys) or len(joined) != len(primary_keys) or not (joined["_merge"] == "both").all():
            raise ValueError(f"{year}: documented one-to-one join failed: {source_name}")
        results.append(
            {
                "source": source_name,
                "dictionary": dictionary_name,
                "source_size_bytes": source.stat().st_size,
                "source_sha256": sha256(source),
                "dictionary_sha256": sha256(dictionary),
                "rows": len(secondary_keys),
                "join_cardinality": "one_to_one",
                "unmatched_primary": int((joined["_merge"] == "left_only").sum()),
                "unmatched_supplementary": int((joined["_merge"] == "right_only").sum()),
                "duplicate_summary_hash_matches_primary": sha256(source) == sha256(raw_root / str(year) / SPECS[year].source),
            }
        )
    return results


def canonical_region_code(values: pd.Series) -> pd.Series:
    """Normalize the documented region field to a two-digit code."""
    text = values.astype("string").str.strip()
    numeric = pd.to_numeric(text, errors="coerce")
    integer_like = numeric.notna() & numeric.eq(numeric.round())
    from_number = numeric.round().astype("Int64").astype("string").str.zfill(2)
    return text.str.zfill(2).mask(integer_like, from_number)


def pseudonymize(year: int, frame: pd.DataFrame, keys: tuple[str, ...], secret: bytes) -> pd.Series:
    values = frame.loc[:, keys].astype(str).apply("\x1f".join, axis=1)
    return values.map(
        lambda value: "fies_" + hmac.new(secret, f"{year}\x1f{value}".encode(), hashlib.sha256).hexdigest()
    )


def local_hmac_key(raw_root: Path) -> bytes:
    """Load the local stable key, creating it with owner-only permissions if needed."""
    environment_key = os.environ.get("FIES_HMAC_KEY")
    if environment_key:
        return environment_key.encode()
    key_path = raw_root.parent / ".fies_hmac_key"
    if key_path.exists():
        LOGGER.info("Using existing local HMAC key at %s", key_path)
        return key_path.read_bytes()
    key_path.write_text(secrets.token_urlsafe(48) + "\n")
    key_path.chmod(0o600)
    LOGGER.info("Created local HMAC key at %s", key_path)
    return key_path.read_bytes()


def category_values(frame: pd.DataFrame, spec: YearSpec) -> tuple[pd.DataFrame, dict[str, str | None]]:
    values: dict[str, pd.Series] = {}
    methods: dict[str, str | None] = {}
    for category, fields in spec.base_fields.items():
        if category == "food":
            values[category] = frame[fields[0]]
        else:
            values[category] = frame.loc[:, fields].sum(axis=1, min_count=len(fields))
        methods[category] = None
    if spec.transport_weight is not None:
        if spec is SPECS[2009]:
            values["transport"] = frame["TRCOM"] * spec.transport_weight
            values["communication"] = frame["TRCOM"] * (1 - spec.transport_weight)
            method = "full TRCOM split by 2009 HFCE current-price annual share"
            methods["transport"] = method
            methods["communication"] = method
            return pd.DataFrame(values), methods
        transport_fields = [f"A{prefix}2" for prefix in (*range(202, 209), *range(302, 309))]
        communication_fields = [f"A{prefix}2" for prefix in (*range(209, 212), *range(309, 312))]
        if "A2132" in frame:
            transport_fields.extend(["A2132", "A3132"])
        if "PCARD" in frame:
            communication_fields.extend(["PCARD", "GPCARD"])
        ambiguous = frame["A2122"] + frame["A3122"]
        values["transport"] = frame[transport_fields].sum(axis=1) + ambiguous * spec.transport_weight
        values["communication"] = frame[communication_fields].sum(axis=1) + ambiguous * (1 - spec.transport_weight)
        method = "observed item detail plus HFCE current-price annual share of ambiguous TRCOM"
        methods["transport"] = method
        methods["communication"] = method
    return pd.DataFrame(values), methods


def category_source_variables(spec: YearSpec) -> dict[str, list[str]]:
    sources = {category: list(fields) for category, fields in spec.base_fields.items()}
    if spec.transport_weight is not None:
        if spec is SPECS[2009]:
            sources.update({"transport": ["TRCOM"], "communication": ["TRCOM"]})
        else:
            sources.update(
                {
                    "transport": ["TRCOM", "A2022-A2082", "A2132", "A3022-A3082", "A3132", "A2122", "A3122"],
                    "communication": ["TRCOM", "A2092-A2112", "PCARD", "GPCARD", "A3092-A3112", "A2122", "A3122"],
                }
            )
    return sources


def clean_year(
    year: int, raw_root: Path, secret: bytes
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    spec = SPECS[year]
    source = raw_root / str(year) / spec.source
    dictionary = raw_root / str(year) / spec.dictionary
    LOGGER.info("%s: validating %s", year, source)
    if not source.is_file() or not dictionary.is_file():
        raise ValueError(f"{year}: selected source or dictionary is missing")
    header = list(pd.read_csv(source, nrows=0, dtype=str).columns)
    dictionary_header = dictionary_header_ids(dictionary, header)
    if header != dictionary_header:
        missing = sorted(set(header) - set(dictionary_header))
        extra = sorted(set(dictionary_header) - set(header))
        raise ValueError(
            f"{year}: CSV header and dictionary IDs differ or are ordered differently; "
            f"missing={missing}, extra={extra}"
        )
    required = required_columns(spec) | validation_columns(year)
    missing = required - set(header)
    if missing:
        raise ValueError(f"{year}: selected source is missing required columns: {sorted(missing)}")
    total_rows = csv_row_count(source)
    LOGGER.info("%s: reading %s required fields", year, len(required))
    chunks = []
    rows_read = 0
    for chunk in pd.read_csv(
        source,
        dtype={key: str for key in spec.keys + (spec.region,)},
        na_values=[".", ""],
        low_memory=False,
        usecols=lambda column: column in required,
        chunksize=READ_CHUNK_SIZE,
    ):
        chunks.append(chunk)
        rows_read += len(chunk)
        if rows_read % PROGRESS_INTERVAL == 0 or rows_read == total_rows:
            LOGGER.info(
                "checking %s rows of %s total - FIES %s",
                f"{rows_read:,}",
                f"{total_rows:,}",
                year,
            )
    frame = pd.concat(chunks, ignore_index=True)
    supplementary_validation = validate_supplementary_sources(year, raw_root, frame.loc[:, spec.keys])
    numeric = sorted(required - set(spec.keys) - {spec.region})
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    invalid_key = frame.loc[:, spec.keys].isna().any(axis=1) | frame.loc[:, spec.keys].eq("").any(axis=1)
    duplicate_key = frame.duplicated(list(spec.keys), keep=False)
    values, methods = category_values(frame, spec)
    relationship_rules = relationship_failures(year, frame)
    invalid_amount = values.isna().any(axis=1) | ~values.apply(
        lambda column: pd.Series(pd.notna(column), index=column.index)
    ).all(axis=1)
    invalid_amount |= values.lt(0).any(axis=1) | frame[[spec.total, spec.income]].isna().any(axis=1)
    invalid_amount |= frame[[spec.total, spec.income]].lt(0).any(axis=1)
    mapped_total = values.sum(axis=1)
    residual = frame[spec.total] - mapped_total
    # Food includes meals purchased from restaurants, so category anchors are not additive.
    # A negative residual reflects that intentional overlap rather than an invalid source row.
    relationship_invalid = pd.Series(False, index=frame.index)
    for failed in relationship_rules.values():
        relationship_invalid |= failed.fillna(True)
    row_invalid = invalid_key | duplicate_key | invalid_amount | relationship_invalid
    ncr = canonical_region_code(frame[spec.region]).eq(NCR_REGION_CODE).fillna(False)
    invalid = ncr & row_invalid
    reasons = pd.Series("", index=frame.index)
    reasons.loc[ncr & invalid_key] += "missing_household_key;"
    reasons.loc[ncr & duplicate_key] += "duplicate_household_key;"
    reasons.loc[ncr & invalid_amount] += "missing_negative_or_unreconciled_amount;"
    for rule, failed in relationship_rules.items():
        reasons.loc[ncr & failed.fillna(True)] += f"{rule};"
    quarantined = pd.DataFrame(
        {"source_row": frame.index[invalid] + 2, "reason_code": reasons.loc[invalid]}
    )
    retain = ncr & ~row_invalid
    valid = frame.loc[retain].reset_index(drop=True)
    valid_values = values.loc[retain].reset_index(drop=True)
    valid_residual = residual.loc[retain].clip(lower=0).reset_index(drop=True)
    households = pd.DataFrame(
        {
            "household_id": pseudonymize(year, valid, spec.keys, secret),
            "survey_year": year,
            "source_release": f"FIES {year}",
            "source_file": str(source.relative_to(raw_root)),
            "period": "annual",
            "annual_income_php": valid[spec.income],
            "annual_total_expense_php": valid[spec.total],
            "household_size": valid["FSIZE"],
            "source_region_code": valid[spec.region],
            "region_code": NCR_REGION_CODE,
            "region": "NCR",
            "unmapped_expense_residual_php": valid_residual,
            "cleaning_spec_version": SPEC_VERSION,
        }
    )
    long = valid_values.assign(household_id=households["household_id"]).melt(
        id_vars=["household_id"], var_name="category", value_name="amount_php"
    )
    long = long.assign(
        survey_year=year,
        source_release=f"FIES {year}",
        source_file=str(source.relative_to(raw_root)),
        period="annual",
        observation_status="observed",
        coverage_status="direct",
        allocation_method=long["category"].map(methods),
        source_variables=long["category"].map(
            lambda category: json.dumps(category_source_variables(spec)[category])
        ),
        coverage_caveat=pd.NA,
        cleaning_spec_version=SPEC_VERSION,
    )
    long["amount_php"] = long["amount_php"].astype("float64")
    long["allocation_method"] = long["allocation_method"].astype("string")
    long["coverage_caveat"] = long["coverage_caveat"].astype("string")
    households[["annual_income_php", "annual_total_expense_php", "household_size", "unmapped_expense_residual_php"]] = households[
        ["annual_income_php", "annual_total_expense_php", "household_size", "unmapped_expense_residual_php"]
    ].astype("float64")
    long.loc[long.category.eq("recreation_culture"), "coverage_status"] = "partial"
    if spec.restaurant_proxy:
        long.loc[long.category.eq("restaurants_hotels"), "coverage_status"] = "proxy"
        long.loc[long.category.eq("restaurants_hotels"), "coverage_caveat"] = spec.restaurant_caveat
    if spec.unresolved_food_restaurants:
        long.loc[long.category.isin(["food", "restaurants_hotels"]), "coverage_status"] = "unresolved"
    long["cleaning_spec_version"] = SPEC_VERSION
    manifest = {
        "year": year,
        "source": str(source.relative_to(raw_root)),
        "dictionary": str(dictionary.relative_to(raw_root)),
        "source_size_bytes": source.stat().st_size,
        "source_sha256": sha256(source),
        "dictionary_sha256": sha256(dictionary),
        "input_rows": len(frame),
        "excluded_non_ncr": int((~ncr).sum()),
        "retained_households": len(valid),
        "quarantined_households": int(invalid.sum()),
        "output_rows": len(long),
        "category_provenance": {
            category: {"source_variables": fields, "allocation_method": methods[category]}
            for category, fields in category_source_variables(spec).items()
        },
        "validation_rules": {
            "header_dictionary_identity": {"failures": 0, "tolerance_php": 0},
            "missing_household_key": {"failures": int((ncr & invalid_key).sum()), "tolerance_php": 0},
            "duplicate_household_key": {"failures": int((ncr & duplicate_key).sum()), "tolerance_php": 0},
            "missing_negative_or_unreconciled_amount": {
                "failures": int((ncr & invalid_amount).sum()),
                "tolerance_php": PHP_TOLERANCE,
            },
            **{
                rule: {
                    "failures": int((ncr & failed.fillna(True)).sum()),
                    "tolerance_php": {2023: 1e-8, 2025: 0.061}.get(year, PHP_TOLERANCE),
                }
                for rule, failed in relationship_rules.items()
            },
        },
        "supplementary_validation": supplementary_validation,
    }
    LOGGER.info(
        "%s: retained %s NCR households of %s; excluded %s non-NCR; quarantined %s; produced %s anchors",
        year,
        len(valid),
        len(frame),
        int((~ncr).sum()),
        int(invalid.sum()),
        len(long),
    )
    return long, households, quarantined, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("RAW/uncleaned"))
    parser.add_argument("--output-dir", type=Path, default=Path("RAW/cleaned"))
    parser.add_argument("--years", type=int, nargs="+", default=sorted(SPECS))
    parser.add_argument("--accept-input-changes", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOGGER.info("Starting FIES cleaning for survey years: %s", ", ".join(map(str, args.years)))
    secret = local_hmac_key(args.raw_root)
    if set(args.years) - set(SPECS):
        raise ValueError(f"Unsupported years: {sorted(set(args.years) - set(SPECS))}")
    previous = args.output_dir / "run_manifest.json"
    if previous.exists() and not args.accept_input_changes:
        prior = json.loads(previous.read_text())
        prior_hashes = {item["year"]: item["source_sha256"] for item in prior["inputs"]}
        current_hashes = {year: sha256(args.raw_root / str(year) / SPECS[year].source) for year in args.years}
        if any(prior_hashes.get(year) not in (None, digest) for year, digest in current_hashes.items()):
            raise ValueError("Input fingerprint changed; inspect the source and rerun with --accept-input-changes")
    staging = args.output_dir.with_name(f"{args.output_dir.name}.staging")
    if staging.exists():
        LOGGER.info("Removing incomplete staging output at %s", staging)
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    all_quarantined, manifests = [], []
    anchor_writer: pq.ParquetWriter | None = None
    household_writer: pq.ParquetWriter | None = None
    for year in args.years:
        anchors, households, quarantined, manifest = clean_year(year, args.raw_root, secret)
        anchor_path = staging / "annual_anchors.parquet"
        LOGGER.info("Writing %s anchors for FIES %s", f"{len(anchors):,}", year)
        anchor_table = pa.Table.from_pandas(anchors, preserve_index=False)
        if anchor_writer is None:
            anchor_writer = pq.ParquetWriter(anchor_path, anchor_table.schema, compression="snappy")
        anchor_writer.write_table(anchor_table)
        append_csv(anchors, staging / "annual_anchors.csv")
        household_path = staging / "households.parquet"
        LOGGER.info("Writing %s household controls for FIES %s", f"{len(households):,}", year)
        household_table = pa.Table.from_pandas(households, preserve_index=False)
        if household_writer is None:
            household_writer = pq.ParquetWriter(
                household_path, household_table.schema, compression="snappy"
            )
        household_writer.write_table(household_table)
        append_csv(households, staging / "households.csv")
        quarantined.insert(0, "survey_year", year)
        all_quarantined.append(quarantined)
        manifests.append(manifest)
        del anchors, anchor_table, households, household_table
    if anchor_writer is not None:
        anchor_writer.close()
    if household_writer is not None:
        household_writer.close()
    LOGGER.info("Writing quarantine report to %s", staging / "quarantined_households.csv")
    pd.concat(all_quarantined, ignore_index=True).to_csv(staging / "quarantined_households.csv", index=False)
    (staging / "run_manifest.json").write_text(json.dumps({"cleaning_spec_version": SPEC_VERSION, "inputs": manifests}, indent=2) + "\n")
    if args.output_dir.exists():
        LOGGER.info("Replacing previous cleaned output at %s", args.output_dir)
        shutil.rmtree(args.output_dir)
    staging.rename(args.output_dir)
    LOGGER.info("Completed FIES cleaning; outputs are available at %s", args.output_dir)


if __name__ == "__main__":
    main()
