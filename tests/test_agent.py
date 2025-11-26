import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from concierge_agent.agent import OrchestrationAgent, ContentWriterAgent
from concierge_agent.sub_agents.user_preference_agent import UserPreferenceAgent
from concierge_agent.sub_agents.trend_analysis_agent import TrendAnalysisAgent
from concierge_agent.sub_agents.sales_event_agent import SalesEventAgent
from concierge_agent.sub_agents.promotion_agent import PromotionAgent
from concierge_agent.sub_agents.refiner_agent import RefinerAgent
from concierge_agent.tools import search_trends, get_sales_events, generate_social_post
from concierge_agent.config import ConciergeConfig
import google.generativeai as genai

# Mock the Gemini model to avoid real API calls during tests
@pytest.fixture
def mock_model():
    mock = AsyncMock()
    mock.generate_content.return_value.text = '{"category": "beauty", "season": "winter", "age_group": "20-40", "gender": "women", "market": "Singapore"}'
    genai.GenerativeModel.return_value = mock
    return mock

@pytest.mark.asyncio
async def test_orchestration_agent_integration(mock_model):
    """Test end-to-end orchestration: parse input, delegate to sub-agents, generate outputs."""
    agent = OrchestrationAgent()
    
    # Mock user input
    user_input = "Perfume promo for Dec, women 20-40, Singapore"
    
    # Patch tools for determinism
    with patch('concierge_agent.tools.search_trends', return_value={'top_products': ['floral perfumes'], 'stats': 'Up 30% YoY'}) as mock_search, \
         patch('concierge_agent.tools.get_sales_events', return_value=[{'event': 'Christmas', 'promo': '20% off'}]) as mock_events, \
         patch('concierge_agent.tools.generate_social_post', new_callable=AsyncMock) as mock_social:
        
        mock_social.return_value = "Mock X post"
        
        # Run the agent
        result = await agent.run(user_input)
    
    # Assertions
    assert 'recommendations' in result
    assert 'social_posts' in result
    assert len(result['social_posts']) == 3  # X, Facebook, Instagram
    mock_search.assert_called_once()
    mock_events.assert_called_once()

@pytest.mark.asyncio
async def test_sub_agent_delegation(mock_model):
    """Test delegation from OrchestrationAgent to sub-agents."""
    orch_agent = OrchestrationAgent()
    user_pref = UserPreferenceAgent()
    trend = TrendAnalysisAgent()
    
    user_input = "Holiday beauty event for young women"
    
    # Test UserPreferenceAgent
    profile = await user_pref.generate(user_input)
    assert 'category' in profile
    assert profile['market'] == 'Singapore'  # From mock
    
    # Test TrendAnalysisAgent (with mocked tool)
    with patch('concierge_agent.tools.search_trends', return_value={'trends': 'Mock trends'}):
        trends = await trend.analyze(profile)
        assert 'trends' in trends

@pytest.mark.asyncio
async def test_loop_refinement(mock_model):
    """Test RefinerAgent with LoopController (retry logic)."""
    from concierge_agent.sub_agents.refiner_agent import LoopController
    
    # Mock invalid then valid promotions
    initial_promos = []  # Invalid (empty)
    valid_promos = [{'offer': 'Bundle', 'discount': '25%'}]
    
    loop_controller = LoopController()
    # Simulate validation failure first, then success (in real: via checker)
    with patch.object(loop_controller.sub_agents[0], 'validate', side_effect=[False, True]):
        refined = await loop_controller.run(initial_promos)
        assert refined == valid_promos  # Assume fallback/success

@pytest.mark.asyncio
async def test_content_writer_output():
    """Test final output formatting."""
    writer = ContentWriterAgent()
    social_posts = {
        'X': 'Mock X post',
        'Facebook': 'Mock FB post',
        'Instagram': 'Mock IG post'
    }
    output = await writer.run(social_posts)
    assert 'Final Social Media Posts:' in output
    for platform in social_posts:
        assert platform in output