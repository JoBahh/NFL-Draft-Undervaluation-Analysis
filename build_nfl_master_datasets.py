
"""
build_nfl_master_datasets_v2.py

Purpose
-------
Build the 2 merged datasets discussed in the project plan.

Datasets
--------
Dataset A:
- Rich 3-way merge
- Uses combine + post-draft + pre-draft
- Draft years 2014 to 2018

Dataset B:
- Wider 2-way merge
- Uses combine + post-draft
- Draft years 2000 to 2018

Design choices
--------------
1. Use the post-draft file as the base because it contains drafted players and the
   outcome variables that matter most for the project.
2. Keep one canonical set of identity columns in the merged outputs:
      - player_name
      - player_name_clean
      - draft_year
      - position
      - position_group
      - school
3. Keep combine metrics and outcome metrics, but avoid redundant identity columns.
4. Add source flags so it is clear which rows matched to which source.

Inputs
------
- standardized_combine_data.csv
- standardized_post_draft_player_data.csv
- standardized_pre_draft_player_data.csv

Outputs
-------
- nfl_master_dataset_a_2014_2018.csv
- nfl_master_dataset_b_2000_2018.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

COMBINE_INPUT = BASE_DIR / "standardized/standardized_combine_data.csv"
POST_DRAFT_INPUT = BASE_DIR / "standardized/standardized_post_draft_player_data.csv"
PRE_DRAFT_INPUT = BASE_DIR / "standardized/standardized_pre_draft_player_data.csv"

DATASET_A_OUTPUT = BASE_DIR / "master/nfl_master_dataset_a_2014_2018.csv"
DATASET_B_OUTPUT = BASE_DIR / "master/nfl_master_dataset_b_2000_2018.csv"


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def coalesce_columns(df, output_column, candidate_columns):
    """
    Create one canonical column by taking the first non-missing value from a list
    of source columns.
    """
    result = pd.Series(np.nan, index=df.index, dtype=object)

    for column in candidate_columns:
        if column in df.columns:
            result = result.fillna(df[column])

    df[output_column] = result
    return df


def merge_on_project_keys(left_df, right_df, how="left"):
    """
    Merge using the agreed project keys.
    """
    merge_keys = ["player_name_clean", "draft_year", "position_group"]
    return left_df.merge(right_df, on=merge_keys, how=how)


def prepare_combine_for_merge(df):
    """
    Keep only the columns we want from the combine file before merging.
    """
    keep_columns = [
        "player_name_clean",
        "draft_year",
        "position_group",
        "player_name",
        "position",
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
    ]
    df = df[keep_columns].copy()

    rename_map = {
        "player_name": "player_name_combine",
        "position": "position_combine",
        "school": "school_combine",
    }
    df = df.rename(columns=rename_map)
    return df


def prepare_post_draft_for_base(df):
    """
    Keep the post-draft file as the base dataset.
    """
    return df.copy()


def prepare_pre_draft_for_merge(df):
    """
    Keep only the columns we want from the pre-draft file before merging.
    """
    keep_columns = [
        "player_name_clean",
        "draft_year",
        "position_group",
        "player_name",
        "position",
        "school",
        "college_season",
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
    df = df[keep_columns].copy()

    rename_map = {
        "player_name": "player_name_pre_draft",
        "position": "position_pre_draft",
        "school": "school_pre_draft",
    }
    df = df.rename(columns=rename_map)
    return df


def finalize_identity_columns(df):
    """
    Build one clean identity block for the merged dataset.
    """
    df = coalesce_columns(
        df,
        "player_name",
        ["player_name", "player_name_pre_draft", "player_name_combine"],
    )

    df = coalesce_columns(
        df,
        "position",
        ["position", "position_pre_draft", "position_combine"],
    )

    df = coalesce_columns(
        df,
        "school",
        ["school", "school_pre_draft", "school_combine"],
    )

    return df


def add_source_flags(df):
    """
    Add simple indicators for whether each source matched.
    """
    df["has_post_draft_data"] = 1
    df["has_combine_data"] = df["height_in"].notna().astype(int)

    if "career_total_ppa" in df.columns:
        df["has_pre_draft_data"] = df["career_total_ppa"].notna().astype(int)

    return df


def drop_redundant_identity_columns(df, include_pre_draft_columns):
    """
    Remove extra identity columns after the canonical versions have been built.
    """
    columns_to_drop = [
        "player_name_combine",
        "position_combine",
        "school_combine",
    ]

    if include_pre_draft_columns:
        columns_to_drop.extend([
            "player_name_pre_draft",
            "position_pre_draft",
            "school_pre_draft",
        ])

    columns_to_drop = [column for column in columns_to_drop if column in df.columns]
    df = df.drop(columns=columns_to_drop)

    return df


def reorder_columns(df, include_pre_draft_columns):
    """
    Put the columns in a clean, readable order.
    """
    identity_columns = [
        "player_name",
        "player_name_clean",
        "draft_year",
        "position",
        "position_group",
        "school",
        "draft_team",
        "draft_round",
        "draft_pick",
        "has_post_draft_data",
        "has_combine_data",
    ]

    if include_pre_draft_columns:
        identity_columns.append("has_pre_draft_data")

    combine_columns = [
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
    ]

    pre_draft_columns = [
        "college_season",
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

    outcome_columns = [
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

    ordered_columns = identity_columns + combine_columns

    if include_pre_draft_columns:
        ordered_columns += pre_draft_columns

    ordered_columns += outcome_columns

    ordered_columns = [column for column in ordered_columns if column in df.columns]
    return df[ordered_columns].copy()


# ---------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------

def build_dataset_b(post_draft_df, combine_df):
    """
    Build Dataset B: 2000 to 2018, using post-draft + combine.
    """
    base_df = prepare_post_draft_for_base(post_draft_df)
    base_df = base_df[(base_df["draft_year"] >= 2000) & (base_df["draft_year"] <= 2018)].copy()

    combine_subset = prepare_combine_for_merge(combine_df)
    combine_subset = combine_subset[(combine_subset["draft_year"] >= 2000) & (combine_subset["draft_year"] <= 2018)].copy()

    merged_df = merge_on_project_keys(base_df, combine_subset, how="left")
    merged_df = finalize_identity_columns(merged_df)
    merged_df = add_source_flags(merged_df)
    merged_df = drop_redundant_identity_columns(merged_df, include_pre_draft_columns=False)
    merged_df = reorder_columns(merged_df, include_pre_draft_columns=False)

    merged_df = merged_df.sort_values(
        by=["draft_year", "draft_round", "draft_pick", "player_name"],
        na_position="last"
    )

    merged_df.to_csv(DATASET_B_OUTPUT, index=False)
    return merged_df


def build_dataset_a(post_draft_df, combine_df, pre_draft_df):
    """
    Build Dataset A: 2014 to 2018, using post-draft + combine + pre-draft.
    """
    base_df = prepare_post_draft_for_base(post_draft_df)
    base_df = base_df[(base_df["draft_year"] >= 2014) & (base_df["draft_year"] <= 2018)].copy()

    combine_subset = prepare_combine_for_merge(combine_df)
    combine_subset = combine_subset[(combine_subset["draft_year"] >= 2014) & (combine_subset["draft_year"] <= 2018)].copy()

    pre_draft_subset = prepare_pre_draft_for_merge(pre_draft_df)
    pre_draft_subset = pre_draft_subset[(pre_draft_subset["draft_year"] >= 2014) & (pre_draft_subset["draft_year"] <= 2018)].copy()

    merged_df = merge_on_project_keys(base_df, combine_subset, how="left")
    merged_df = merge_on_project_keys(merged_df, pre_draft_subset, how="left")

    merged_df = finalize_identity_columns(merged_df)
    merged_df = add_source_flags(merged_df)
    merged_df = drop_redundant_identity_columns(merged_df, include_pre_draft_columns=True)
    merged_df = reorder_columns(merged_df, include_pre_draft_columns=True)

    merged_df = merged_df.sort_values(
        by=["draft_year", "draft_round", "draft_pick", "player_name"],
        na_position="last"
    )

    merged_df.to_csv(DATASET_A_OUTPUT, index=False)
    return merged_df


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    combine_df = pd.read_csv(COMBINE_INPUT)
    post_draft_df = pd.read_csv(POST_DRAFT_INPUT)
    pre_draft_df = pd.read_csv(PRE_DRAFT_INPUT)

    dataset_a = build_dataset_a(post_draft_df, combine_df, pre_draft_df)
    dataset_b = build_dataset_b(post_draft_df, combine_df)

    print("\nMerged dataset build complete.")
    print(f"Dataset A rows: {len(dataset_a)}")
    print(f"Dataset B rows: {len(dataset_b)}")


if __name__ == "__main__":
    main()
