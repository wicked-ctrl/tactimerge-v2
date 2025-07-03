from dotenv import load_dotenv
import os

# 1) Load environment variables
load_dotenv()
SECRET_KEY      = os.getenv("SECRET_KEY")
API_URL         = os.getenv("API_URL")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")

# 2) Configure new OpenAI client
from openai import OpenAI
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=API_URL
)

# 3) Core dependencies
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from scipy.stats import poisson

# 4) Instantiate FastAPI
app = FastAPI(
    title="TactiMerge API",
    version="0.1.0",
    description="Endpoints for team analysis, match prediction, and comparison"
)

# 5) Health check
@app.get("/health")
def health_check():
    if not SECRET_KEY or not API_URL or not OPENAI_API_KEY:
        raise HTTPException(500, detail="Missing SECRET_KEY, API_URL or OPENAI_API_KEY")
    return {"status": "ok", "api_url": API_URL}

# 6) Root welcome
@app.get("/")
def read_root():
    return {"message": "Welcome to TactiMerge API"}

# 7) Lazy-load team strengths CSV
_strengths = None
def load_strengths():
    global _strengths
    if _strengths is None:
        path = os.path.join("data", "team_strengths.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        _strengths = pd.read_csv(path, index_col=0)
    return _strengths

# 8) Pydantic models
class AnalysisRequest(BaseModel):
    team: str
    era: str
    injured: Optional[List[str]] = []
    new_signings: Optional[List[str]] = []

class AnalysisResponse(BaseModel):
    team: str
    era: str
    playstyle_summary: str

class PredictRequest(BaseModel):
    home_team: str
    away_team: str
    era: str
    injuries: Optional[List[str]] = []
    new_signings: Optional[List[str]] = []

class PredictResponse(BaseModel):
    home_team: str
    away_team: str
    predicted_score: str
    expected_xg: float

class CompareRequest(BaseModel):
    team_a: str
    team_b: str

class TeamCompareStats(BaseModel):
    team: str
    atk_home: float
    def_home: float
    atk_away: float
    def_away: float
    atk_home_ratio: float
    def_home_ratio: float
    atk_away_ratio: float
    def_away_ratio: float

class CompareResponse(BaseModel):
    league_avg: float
    stats: List[TeamCompareStats]
    expected_xg_a_vs_b: float
    expected_xg_b_vs_a: float

# 9) Helper: Poisson distribution for score probabilities
def poisson_score(atk: float, def_: float, league_avg: float):
    lam = atk * def_ / league_avg
    return {k: poisson.pmf(k, lam) for k in range(6)}

# 10) Analysis endpoint
@app.post("/analyze", response_model=AnalysisResponse)
def analyze(request: AnalysisRequest):
    prompt = (
        f"Summarize the playing style of {request.team} during the {request.era} era. "
        "Include formation, attack/defense balance, and key tactical traits."
    )

    try:
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a football tactics analyst."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        summary = resp.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(500, detail=f"OpenAI error: {e}")

    return AnalysisResponse(
        team=request.team,
        era=request.era,
        playstyle_summary=summary
    )

# 11) Prediction endpoint
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        strengths = load_strengths()
    except FileNotFoundError as e:
        raise HTTPException(500, detail=str(e))

    atk_h      = strengths.at[request.home_team, "atk_home"]
    def_a      = strengths.at[request.away_team, "def_away"]
    atk_a      = strengths.at[request.away_team, "atk_away"]
    def_h      = strengths.at[request.home_team, "def_home"]
    league_avg = strengths[["atk_home", "atk_away"]].values.mean()

    dist_h     = poisson_score(atk_h, def_a, league_avg)
    dist_a     = poisson_score(atk_a, def_h, league_avg)

    home_goals = max(dist_h, key=dist_h.get)
    away_goals = max(dist_a, key=dist_a.get)
    exp_xg     = (atk_h * def_a + atk_a * def_h) / (2 * league_avg)

    return PredictResponse(
        home_team=request.home_team,
        away_team=request.away_team,
        predicted_score=f"{home_goals}-{away_goals}",
        expected_xg=round(exp_xg, 2)
    )

# 12) Compare endpoint with extended metrics
@app.post("/compare", response_model=CompareResponse)
def compare(request: CompareRequest):
    try:
        strengths = load_strengths()
    except FileNotFoundError as e:
        raise HTTPException(500, detail=str(e))

    # compute league average across all four metrics
    league_avg = strengths[["atk_home", "def_home", "atk_away", "def_away"]].values.mean()

    def get_team_stats(team: str):
        if team not in strengths.index:
            raise HTTPException(404, detail=f"Team '{team}' not found")
        row = strengths.loc[team]
        return TeamCompareStats(
            team=team,
            atk_home=row["atk_home"],
            def_home=row["def_home"],
            atk_away=row["atk_away"],
            def_away=row["def_away"],
            atk_home_ratio=row["atk_home"] / league_avg,
            def_home_ratio=row["def_home"] / league_avg,
            atk_away_ratio=row["atk_away"] / league_avg,
            def_away_ratio=row["def_away"] / league_avg,
        )

    stats_a = get_team_stats(request.team_a)
    stats_b = get_team_stats(request.team_b)

    # head-to-head expected xG
    expected_xg_a_vs_b = stats_a.atk_home * stats_b.def_away / league_avg
    expected_xg_b_vs_a = stats_b.atk_home * stats_a.def_away / league_avg

    return CompareResponse(
        league_avg=round(league_avg, 3),
        stats=[stats_a, stats_b],
        expected_xg_a_vs_b=round(expected_xg_a_vs_b, 2),
        expected_xg_b_vs_a=round(expected_xg_b_vs_a, 2),
    )
