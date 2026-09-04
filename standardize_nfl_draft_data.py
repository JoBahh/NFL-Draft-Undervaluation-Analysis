
"""
standardize_nfl_draft_data_v2.py

Purpose
-------
Clean, organize, and standardize the 3 source files before merging.

Design choices
--------------
1. Keep the cleaning conservative. We standardize names, schools, positions, and years,
   but we do not heavily alter the actual football data.
2. Use one main style for identity columns in each output file:
      - player_name
      - player_name_clean
      - draft_year
      - position
      - position_group
      - school
3. Keep position_group because it is very useful for analysis and merging.
4. Avoid redundant columns in the standardized output files.
5. Create only a few helper fields that are genuinely useful, such as combine
   missingness flags.

Inputs
------
- cleaned_combine_data.csv
- cleaned_post_draft_player_data.csv
- cleaned_pre_draft_player_data.csv

Outputs
-------
- standardized_combine_data_v2.csv
- standardized_post_draft_player_data_v2.csv
- standardized_pre_draft_player_data_v2.csv
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

COMBINE_INPUT = BASE_DIR / "cleaned/cleaned_combine_data.csv"
POST_DRAFT_INPUT = BASE_DIR / "cleaned/cleaned_post_draft_player_data.csv"
PRE_DRAFT_INPUT = BASE_DIR / "cleaned/cleaned_pre_draft_player_data.csv"

COMBINE_OUTPUT = BASE_DIR / "standardized/standardized_combine_data.csv"
POST_DRAFT_OUTPUT = BASE_DIR / "standardized/standardized_post_draft_player_data.csv"
PRE_DRAFT_OUTPUT = BASE_DIR / "standardized/standardized_pre_draft_player_data.csv"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def clean_text(value):
    """
    Basic text cleaner used for names and schools.
    This keeps the cleaning simple and conservative.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_player_name(value):
    """
    Create a standardized player name for joining.

    Important:
    - We remove punctuation and extra spaces.
    - We keep suffixes such as Jr and Sr because removing them can create
      false collisions between different players.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()
    text = re.sub(r"[\'\.\,\-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_school_name(value):
    """
    Standardize school names lightly.
    """
    if pd.isna(value):
        return np.nan

    text = clean_text(value)
    text = text.replace("&", "and")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def standardize_position(value):
    """
    Standardize the raw position label without making it too broad.
    """
    if pd.isna(value):
        return np.nan

    text = str(value).strip().upper()

    position_map = {
        "HB": "RB",
        "TB": "RB",
        "FB": "FB",
        "QB": "QB",
        "RB": "RB",
        "WR": "WR",
        "TE": "TE",
        "G": "OG",
        "T": "OT",
        "OL": "OL",
        "C": "C",
        "OG": "OG",
        "OT": "OT",
        "DE": "DE",
        "DT": "DT",
        "DL": "DL",
        "EDGE": "EDGE",
        "NT": "NT",
        "LB": "LB",
        "ILB": "ILB",
        "MLB": "MLB",
        "OLB": "OLB",
        "CB": "CB",
        "DB": "DB",
        "S": "S",
        "SAF": "S",
        "FS": "S",
        "SS": "S",
        "K": "K",
        "P": "P",
        "LS": "LS",
    }

    return position_map.get(text, text)


def create_position_group(value):
    """
    Create a broader position group for analysis and merging.
    """
    if pd.isna(value):
        return np.nan

    position = standardize_position(value)

    group_map = {
        "QB": "QB",
        "RB": "RB",
        "FB": "RB",
        "WR": "WR",
        "TE": "TE",
        "OT": "OL",
        "OG": "OL",
        "C": "OL",
        "OL": "OL",
        "DE": "DL",
        "DT": "DL",
        "DL": "DL",
        "EDGE": "DL",
        "NT": "DL",
        "LB": "LB",
        "ILB": "LB",
        "MLB": "LB",
        "OLB": "LB",
        "CB": "DB",
        "DB": "DB",
        "S": "DB",
        "K": "ST",
        "P": "ST",
        "LS": "ST",
    }

    return group_map.get(position, position)


def to_int_series(series):
    """
    Convert a column to pandas nullable integer type.
    """
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def add_metric_availability_flags(df, metric_columns):
    """
    Add simple yes/no flags for missing combine metrics.
    """
    for column in metric_columns:
        flag_name = f"has_{column}"
        df[flag_name] = df[column].notna().astype(int)

    return df


def remove_ambiguous_duplicate_keys(df, key_columns, source_name):
    """
    If a source has duplicate rows on the exact merge key, remove those rows
    entirely so they do not create one-to-many merge explosions later.

    We use this only for clearly ambiguous cases.
    """
    duplicate_mask = df.duplicated(subset=key_columns, keep=False)

    if duplicate_mask.any():
        duplicate_rows = df.loc[duplicate_mask].copy()
        print(f"[{source_name}] Removed ambiguous rows on merge key: {len(duplicate_rows)}")
        df = df.loc[~duplicate_mask].copy()

    return df


# ---------------------------------------------------------------------
# Standardization functions for each file
# ---------------------------------------------------------------------

def standardize_combine_data():
    df = pd.read_csv(COMBINE_INPUT)

    df = df.rename(
        columns={
            "player": "player_name",
            "year": "draft_year",
            "position": "position",
            "school": "school",
            "team": "draft_team",
            "round": "draft_round",
            "pick": "draft_pick",
        }
    )

    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["player_name_clean"] = df["player_name"].apply(clean_player_name)
    df["draft_year"] = to_int_series(df["draft_year"])

    df["position"] = df["position"].apply(standardize_position)
    df["position_group"] = df["position"].apply(create_position_group)

    df["school"] = df["school"].apply(clean_school_name)
    df["draft_team"] = df["draft_team"].apply(clean_text)
    df["draft_round"] = to_int_series(df["draft_round"])
    df["draft_pick"] = to_int_series(df["draft_pick"])

    metric_columns = [
        "height_in",
        "weight_lb",
        "forty_yd_dash",
        "vertical_jump",
        "bench_press",
        "broad_jump",
        "three_cone_drill",
        "twenty_yd_shuttle",
    ]

    for column in metric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = add_metric_availability_flags(
        df,
        [
            "forty_yd_dash",
            "vertical_jump",
            "bench_press",
            "broad_jump",
            "three_cone_drill",
            "twenty_yd_shuttle",
        ],
    )

    # Prevent ambiguous combine rows from blowing up joins later.
    merge_key = ["player_name_clean", "draft_year", "position_group"]
    df = remove_ambiguous_duplicate_keys(df, merge_key, "combine")

    output_columns = [
        "player_name",
        "player_name_clean",
        "draft_year",
        "position",
        "position_group",
        "school",
        "height_in",
        "weight_lb",
        "forty_yd_dash",
        "vertical_jump",
        "bench_press",
        "broad_jump",
        "three_cone_drill",
        "twenty_yd_shuttle",
        "has_forty_yd_dash",
        "has_vertical_jump",
        "has_bench_press",
        "has_broad_jump",
        "has_three_cone_drill",
        "has_twenty_yd_shuttle",
        "draft_team",
        "draft_round",
        "draft_pick",
    ]

    df = df[output_columns].sort_values(
        by=["draft_year", "player_name", "position"],
        na_position="last"
    )

    df.to_csv(COMBINE_OUTPUT, index=False)
    return df


def standardize_post_draft_data():
    df = pd.read_csv(POST_DRAFT_INPUT)

    df = df.rename(
        columns={
            "pfr_player_name": "player_name",
            "season": "draft_year",
            "college": "school",
            "team": "draft_team",
            "round": "draft_round",
            "pick": "draft_pick",
        }
    )

    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["player_name_clean"] = df["player_name"].apply(clean_player_name)
    df["draft_year"] = to_int_series(df["draft_year"])

    df["position"] = df["position"].apply(standardize_position)
    df["position_group"] = df["position"].apply(create_position_group)

    df["school"] = df["school"].apply(clean_school_name)
    df["draft_team"] = df["draft_team"].apply(clean_text)
    df["draft_round"] = to_int_series(df["draft_round"])
    df["draft_pick"] = to_int_series(df["draft_pick"])

    numeric_columns = [
        "age",
        "to",
        "allpro",
        "probowls",
        "seasons_started",
        "w_av",
        "dr_av",
        "games",
        "pass_completions",
        "pass_attempts",
        "pass_yards",
        "pass_tds",
        "pass_ints",
        "rush_atts",
        "rush_yards",
        "rush_tds",
        "receptions",
        "rec_yards",
        "rec_tds",
        "def_solo_tackles",
        "def_ints",
        "def_sacks",
        "AV_year1",
        "AV_year2",
        "AV_year3",
        "AV_year4",
        "AV_4yr_sum",
        "AV_4yr_avg",
        "AV_4yr_avg_percentile",
        "car_av_filled",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    output_columns = [
        "player_name",
        "player_name_clean",
        "draft_year",
        "position",
        "position_group",
        "school",
        "draft_team",
        "draft_round",
        "draft_pick",
        "hof",
        "category",
        "side",
        "age",
        "to",
        "allpro",
        "probowls",
        "seasons_started",
        "w_av",
        "dr_av",
        "games",
        "pass_completions",
        "pass_attempts",
        "pass_yards",
        "pass_tds",
        "pass_ints",
        "rush_atts",
        "rush_yards",
        "rush_tds",
        "receptions",
        "rec_yards",
        "rec_tds",
        "def_solo_tackles",
        "def_ints",
        "def_sacks",
        "AV_year1",
        "AV_year2",
        "AV_year3",
        "AV_year4",
        "AV_4yr_sum",
        "AV_4yr_avg",
        "AV_4yr_avg_percentile",
        "car_av_filled",
    ]

    df = df[output_columns].sort_values(
        by=["draft_year", "player_name", "position"],
        na_position="last"
    )

    df.to_csv(POST_DRAFT_OUTPUT, index=False)
    return df


def standardize_pre_draft_data():
    df = pd.read_csv(PRE_DRAFT_INPUT)

    df = df.rename(
        columns={
            "name": "player_name",
            "season": "college_season",
        }
    )

    df["player_name"] = df["player_name"].astype(str).str.strip()
    df["player_name_clean"] = df["player_name"].apply(clean_player_name)
    df["draft_year"] = to_int_series(df["draft_year"])
    df["college_season"] = to_int_series(df["college_season"])

    df["position"] = df["position"].apply(standardize_position)
    df["position_group"] = df["position"].apply(create_position_group)
    df["school"] = df["school"].apply(clean_school_name)

    numeric_columns = [
        "countable_plays",
        "avg_ppa_all",
        "total_ppa_all",
        "total_ppa_pass",
        "total_ppa_rush",
        "career_avg_ppa",
        "career_total_ppa",
        "ppa_per_play",
        "pass_ppa_share",
        "rush_ppa_share",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    output_columns = [
        "player_name",
        "player_name_clean",
        "draft_year",
        "college_season",
        "position",
        "position_group",
        "school",
        "countable_plays",
        "avg_ppa_all",
        "total_ppa_all",
        "total_ppa_pass",
        "total_ppa_rush",
        "career_avg_ppa",
        "career_total_ppa",
        "ppa_per_play",
        "pass_ppa_share",
        "rush_ppa_share",
    ]

    df = df[output_columns].sort_values(
        by=["draft_year", "player_name", "position"],
        na_position="last"
    )

    df.to_csv(PRE_DRAFT_OUTPUT, index=False)
    return df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    combine_df = standardize_combine_data()
    post_draft_df = standardize_post_draft_data()
    pre_draft_df = standardize_pre_draft_data()

    print("\nStandardization complete.")
    print(f"Combine rows: {len(combine_df)}")
    print(f"Post-draft rows: {len(post_draft_df)}")
    print(f"Pre-draft rows: {len(pre_draft_df)}")


if __name__ == "__main__":
    main()
