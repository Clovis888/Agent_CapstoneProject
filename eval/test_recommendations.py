import pytest
import asyncio
import json
from typing import Dict, Any
from concierge_agent.agent import OrchestrationAgent
from eval.evaluation_utils import (
    load_evaluation_dataset, cosine_text_similarity,
    regex_match_score, evaluate_social_post, EvaluationResult,
    compute_overall_score
)

# Global agent instance (reuse to avoid reinitialization)
@pytest.fixture(scope="module")
def agent():
    return OrchestrationAgent()
 
@pytest.mark.asyncio
@pytest.mark.evaluation
async def test_full_evaluation_suite(agent):
    """Run full evaluation against golden dataset."""
    dataset = load_evaluation_dataset()
    results = []
    
    print("\nConcierge AI Agent — Evaluation Report")
    print("=" * 60)
    
    for item in dataset:
        user_input = item["user_input"]
        expected = item["expected"]
        
        print(f"\nScenario {item['id']}: {user_input[:70]}{'...' if len(user_input)>70 else ''}")
        
        try:
            # Run the actual agent
            output = await asyncio.wait_for(agent.run(user_input), timeout=60)
        except Exception as e:
            print(f"   Failed: {e}")
            results.append(EvaluationResult(
                scenario=user_input[:50],
                relevance_score=0, personalization_score=0,
                trend_alignment=0, event_coverage=0,
                post_quality_avg=0, overall_score=0, passed=False
            ))
            continue
        
        # Extract outputs
        recommendations = output.get("recommendations", {})
        promotions = output.get("promotions", [])
        social_posts = output.get("social_posts", {})
        
        rec_text = recommendations.get("suggestions", "") if isinstance(recommendations, dict) else str(recommendations)
        promo_text = " | ".join([f"{p.get('offer','')} {p.get('discount','')}" for p in promotions])
        all_posts = " ".join(social_posts.values())
        
        # Scoring
        relevance = cosine_text_similarity(user_input + " " + rec_text + promo_text, user_input)
        personalization = regex_match_score(all_posts, [
            expected.get("age_group", ""), 
            expected.get("gender", ""), 
            expected.get("market", "Singapore|US")
        ])
        trend_align = regex_match_score(all_posts, expected.get("trends", []))
        event_cov = regex_match_score(all_posts, expected.get("events", []))
        
        post_qualities = [
            evaluate_social_post(post, expected.get("required_keywords", []))["quality"]
            for post in social_posts.values()
        ]
        post_quality_avg = sum(post_qualities) / len(post_qualities) if post_qualities else 0
        
        overall = (
            0.25 * relevance +
            0.20 * personalization +
            0.15 * trend_align +
            0.15 * event_cov +
            0.25 * post_quality_avg
        )
        
        passed = overall >= 0.75
        
        result = EvaluationResult(
            scenario=user_input[:50],
            relevance_score=round(relevance, 3),
            personalization_score=round(personalization, 3),
            trend_alignment=round(trend_align, 3),
            event_coverage=round(event_cov, 3),
            post_quality_avg=round(post_quality_avg, 3),
            overall_score=round(overall, 3),
            passed=passed
        )
        results.append(result)
        
        status = "PASSED" if passed else "FAILED"
        print(f"   → Overall: {overall:.3f} | Post Quality: {post_quality_avg:.3f} | [{status}]")
    
    # Final Summary
    summary = compute_overall_score(results)
    print("\n" + "=" * 60)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total Scenarios: {summary['total_scenarios']}")
    print(f"Pass Rate:       {summary['pass_rate']:.1%}")
    print(f"Average Score:   {summary['avg_overall_score']:.3f}/1.0")
    print(f"Best:            {summary['best_scenario']}")
    print(f"Worst:           {summary['worst_scenario']}")
    print("=" * 60)
    
    # Save detailed results
    with open("eval_results.json", "w") as f:
        json.dump([r.__dict__ for r in results], f, indent=2)
    
    # Final assertion for CI/CD
    assert summary['pass_rate'] >= 0.7, f"Evaluation failed: Only {summary['pass_rate']:.1%} passed (need ≥70%)"

# Run with:
# uv run pytest eval/test_recommendations.py -v
# Or for full suite:
# uv run pytest eval/ -v --tb=short