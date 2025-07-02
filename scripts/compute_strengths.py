import pandas as pd
import os

"""
Compute team attacking/defensive xG strengths from historical match data.
Produces "data/team_strengths.csv" with columns:
  atk_home, def_home, atk_away, def_away
Supports versioning and customizable null handling strategies.
"""

def compute_strengths(
    input_csv: str,
    output_csv: str,
    fill_method: str = 'league_mean',  # options: 'league_mean', 'team_median', 'zero'
    version: str = 'v1'
) -> None:
    """
    Compute and save team strengths with null-handling.

    Args:
        input_csv: Path to historical xG CSV with columns home_team, away_team, home_xg, away_xg.
        output_csv: Desired output path for strengths CSV.
        fill_method: How to fill missing values:
            - 'league_mean': fill with overall league mean xG.
            - 'team_median': fill with per-team median xG.
            - 'zero': fill missing with 0.
        version: Version tag to include in output filename (e.g., 'v2').
    """
    # Load data
    df = pd.read_csv(input_csv)

    # Compute home/away strengths
    home_atk = df.groupby('home_team').home_xg.mean().rename('atk_home')
    df['home_concede'] = df.away_xg
    home_def = df.groupby('home_team').home_concede.mean().rename('def_home')

    away_atk = df.groupby('away_team').away_xg.mean().rename('atk_away')
    df['away_concede'] = df.home_xg
    away_def = df.groupby('away_team').away_concede.mean().rename('def_away')

    strengths = pd.concat([home_atk, home_def, away_atk, away_def], axis=1)

    # Choose fill strategy
    if fill_method == 'league_mean':
        fill_value = df[['home_xg','away_xg']].values.mean()
        strengths = strengths.fillna(fill_value)
    elif fill_method == 'team_median':
        medians = df.groupby('home_team').home_xg.median().combine_first(
            df.groupby('away_team').away_xg.median()
        )
        strengths = strengths.fillna(medians.to_dict())
    elif fill_method == 'zero':
        strengths = strengths.fillna(0)
    else:
        raise ValueError(f"Unknown fill_method: {fill_method}")

    # Versioning: append version to filename if provided
    base, ext = os.path.splitext(output_csv)
    versioned_path = f"{base}_{version}{ext}"

    os.makedirs(os.path.dirname(versioned_path), exist_ok=True)
    strengths.to_csv(versioned_path)
    print(f"Saved team strengths ({version}) to {versioned_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Compute team xG strengths with null handling')
    parser.add_argument('--input', '-i', required=True, help='CSV path with xG history')
    parser.add_argument('--output', '-o', default='data/team_strengths.csv', help='Output strengths CSV')
    parser.add_argument('--fill', '-f', default='league_mean', choices=['league_mean','team_median','zero'], help='Null-fill strategy')
    parser.add_argument('--version', '-v', default='v1', help='Version tag for output file')
    args = parser.parse_args()
    compute_strengths(args.input, args.output, fill_method=args.fill, version=args.version)
