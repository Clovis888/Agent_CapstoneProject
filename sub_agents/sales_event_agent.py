from google_adk import Agent
import google.generativeai as genai
from ...tools import get_sales_events
from ...config import ConciergeConfig
model = genai.GenerativeModel(ConciergeConfig.model_name)
class SalesEventAgent(Agent):
    """Identifies sales periods and promotions."""
    def __init__(self):
        super().__init__(
            name="sales_event_agent",
            model=model,
            description="Find events based on period.",
            instruction="Use get_sales_events for holidays/promos.",
            tools=[get_sales_events]
        )
async def find_events(self, profile: dict) -> list:
        events_data = await get_sales_events(profile['season'], profile['market'])
        prompt = f"List relevant events: {events_data}"
        response = self.model.generate_content(prompt)
        return [{"event": e, "promo": p} for e, p in eval(response.text).items()]
