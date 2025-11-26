import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from concierge_agent.agent import OrchestrationAgent
from concierge_agent.tools import search_trends, get_sales_events
from concierge_agent.config import ConciergeConfig

# Mock data for scenario
MOCK_CUSTOMER_PROFILE = {
    'category': 'perfume', 'season': 'Dec-Jan', 'age_group': '25-35',
    'gender': 'women', 'market': 'US', 'purchase_history': ['fragrances']
}
MOCK_TRENDS = {'top_products': ['Jo Malone Peony', 'Holiday scents'], 'stats': 'Perfume sales up 25% for holidays'}
MOCK_EVENTS = [{'event': 'Christmas', 'promo': 'Buy1Get1 on gifts'}, {'event': 'New Year', 'promo': '15% off bundles'}]
EXPECTED_KEYWORDS = ['perfume', 'Christmas', 'New Year', 'discount', 'bundle', 'women']

@pytest.mark.asyncio
async def test_perfume_holiday_scenario():
    """Test full scenario: Perfume promotion for Christmas/New Year, women 25-35, US."""
    agent = OrchestrationAgent()
    
    user_input = "Create perfume promotion for Christmas and New Year, target 25-35 women in US"
    
    # Patch tools and sub-agents for scenario
    with patch('concierge_agent.sub_agents.user_preference_agent.UserPreferenceAgent.generate', return_value=MOCK_CUSTOMER_PROFILE) as mock_pref, \
         patch('concierge_agent.tools.search_trends', return_value=MOCK_TRENDS) as mock_trends, \
         patch('concierge_agent.tools.get_sales_events', return_value=MOCK_EVENTS) as mock_events, \
         patch('concierge_agent.sub_agents.promotion_agent.PromotionAgent.craft', return_value=[{'offer': 'Holiday Perfume Set', 'discount': '20%'}]) as mock_promo, \
         patch('concierge_agent.tools.generate_social_post', new_callable=AsyncMock) as mock_social:
        
        mock_social.side_effect = [
            "X: Unwrap festive scents this Christmas! 20% off perfume bundles for her. #HolidayGifts",
            "FB: New Year glow with our perfume deals – perfect for 25-35 women! 🎉",
            "IG: Sparkle into 2026 with discounted holiday perfumes. Tag a friend! ✨"
        ]
        
        result = await agent.run(user_input)
    
    # Assertions on recommendations and promotions
    assert 'suggestions' in result['recommendations']
    assert any('perfume' in result['recommendations']['suggestions'].lower() for _ in range(1))
    promotions = result['promotions']
    assert len(promotions) >= 1
    assert 'discount' in promotions[0]
    
    # Check social posts
    social_posts = result['social_posts']
    assert len(social_posts) == 3
    for platform, post in social_posts.items():
        assert any(keyword in post.lower() for keyword in EXPECTED_KEYWORDS)
    
    # Relevance metric: Cosine similarity between expected and generated (simple TF-IDF)
    vectorizer = TfidfVectorizer().fit_transform([user_input, post])
    similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:])[0][0]
    assert similarity > 0.3  # Threshold for relevance
    
    # Verify mocks were called
    mock_pref.assert_called_once()
    mock_trends.assert_called_once()
    mock_events.assert_called_once()

@pytest.mark.asyncio
async def test_scenario_with_validation_failure():
    """Test scenario with refinement loop (e.g., invalid promo triggers retry)."""
    agent = OrchestrationAgent()
    user_input = "Invalid promo test for holidays"
    
    # Mock failure in promotion, then success after retry
    with patch('concierge_agent.sub_agents.promotion_agent.PromotionAgent.craft', side_effect=[[], [{'offer': 'Valid Bundle', 'discount': '25%'}]]), \
         patch('concierge_agent.sub_agents.refiner_agent.RefinerAgent.validate', side_effect=[False, True]):
        
        result = await agent.run(user_input)
        assert len(result['promotions']) == 1  # Refined to valid