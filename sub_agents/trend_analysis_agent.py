from google_adk import Agent
import google.generativeai as genai
from ...tools import search_trends
from ...config import ConciergeConfig
model = genai.GenerativeModel(ConciergeConfig.model_name)
class TrendAnalysisAgent(Agent):
    """Surfaces trending products and market insights."""
    def __init__(self):
        super().__init__(
            name="trend_analysis_agent",
            model=model,
            description="Analyze trends using search.",
            instruction="Use search_trends tool for latest stats on products/competitors.",
            tools=[search_trends]
        )
async def analyze(self, profile: dict) -> dict:
        query = f"Trends for {profile['category']} in {profile['market']} {profile['season']}"
        trends_data = await search_trends(query)
        prompt = f"Summarize trends: {trends_data}"
        response = self.model.generate_content(prompt)
        return {"trends": response.text, "stats": trends_data}
