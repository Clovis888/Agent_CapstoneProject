import json
import pandas as pd
from typing import Dict, List, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dataclasses import dataclass
import re

@dataclass
class EvaluationResult:
    scenario: str
    relevance_score: float
    personalization_score: float
    trend_alignment: float
    event_coverage: float
    post_quality_avg: float
    overall_score: float
    passed: bool

def load_evaluation_dataset(path: str = "data/evaluation_dataset.json") -> List[Dict]:
    """Load ground-truth scenarios for evaluation."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {path} not found. Using fallback dataset.")
        return [
            {
                "id": 1,
                "user_input": "I want to create sales and promotion event from November to December, target customer group is 20-40 years old women in Singapore",
                "expected": {
                    "category": "beauty|perfume|skincare|fragrance",
                    "season": "Nov.*Dec|Christmas|Black Friday|Holiday",
                    "age_group": "20-40",
                    "gender": "women",
                    "market": "Singapore",
                    "trends": ["floral", "gift set", "sustainable", "long-lasting"],
                    "events": ["Black Friday", "Christmas", "12.12", "New Year"],
                    "required_keywords": ["discount", "bundle", "gift", "limited", "holiday", "Singapore"],
                    "social_post_quality": 0.8
                }
            },
            {
                "id": 2,
                "user_input": "Perfume promotion for Christmas and New Year, women 25-35 in US",
                "expected": {
                    "category": "perfume|fragrance",
                    "season": "Christmas|New Year|Winter",
                    "events": ["Christmas", "After Christmas Sale", "New Year"],
                    "required_keywords": ["perfume", "gift", "holiday", "scent", "bundle"]
                }
            }
        ]

def cosine_text_similarity(text1: str, text2: str) -> float:
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    try:
        tfidf = vectorizer.fit_transform([text1, text2])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0])
    except:
        return 0.0

def regex_match_score(text: str, patterns: List[str]) -> float:
    score = 0
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += 1
    return score / max(len(patterns), 1)

def evaluate_social_post(post: str, expected_keywords: List[str]) -> Dict[str, float]:
    length_ok = 50 <= len(post.split()) <= 280
    has_hashtag = bool(re.search(r"#\w+", post))
    has_emoji = bool(re.search(r"[\U0001F300-\U0001F9FF]|[\U0001F600-\U0001F64F]", post))
    keyword_score = sum(1 for kw in expected_keywords if kw.lower() in post.lower()) / len(expected_keywords)
    
    quality = (
        0.3 * keyword_score +
        0.3 * (1 if length_ok else 0) +
        0.2 * (1 if has_hashtag else 0) +
        0.2 * (1 if has_emoji else 0)
    )
    return {"quality": quality, "keyword_coverage": keyword_score}

def compute_overall_score(results: List[EvaluationResult]) -> Dict[str, float]:
    df = pd.DataFrame([{
        "scenario": r.scenario,
        "overall": r.overall_score,
        "passed": r.passed
    } for r in results])
    
    return {
        "total_scenarios": len(df),
        "pass_rate": df["passed"].mean(),
        "avg_overall_score": df["overall"].mean(),
        "best_scenario": df.loc[df["overall"].idxmax(), "scenario"] if not df.empty else "N/A",
        "worst_scenario": df.loc[df["overall"].idxmin(), "scenario"] if not df.empty else "N/A"
    }