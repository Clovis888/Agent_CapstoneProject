from google_adk import Agent
import google.generativeai as genai
from ...config import ConciergeConfig
model = genai.GenerativeModel(ConciergeConfig.model_name)
class PromotionAgent(Agent):
    """Crafts personalized offers and bundles."""
    def __init__(self):
        super().__init__(
            name="promotion_agent",
            model=model,
            description="Generate offers.",
            instruction="Create discounts/bundles based on profile, trends, events."
        )
async def craft(self, state) -> list:
        prompt = f"Craft promotions for {state.customer_profile} with {state.trends} and {state.events}."
        response = self.model.generate_content(prompt)
        return [{"offer": o, "discount": d} for o, d in eval(response.text).items()]
