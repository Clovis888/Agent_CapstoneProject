from google_adk import Agent
import google.generativeai as genai
from ...config import ConciergeConfig
model = genai.GenerativeModel(ConciergeConfig.model_name)
class UserPreferenceAgent(Agent):
    """Extracts user preferences: category, season, demographics."""
    def __init__(self):
        super().__init__(
            name="user_preference_agent",
            model=model,
            description="Parse user input for preferences.",
            instruction="Extract product category, season/sales period, age/gender/market from input."
        )
async def generate(self, user_input: str) -> dict:
        prompt = f"Parse: {user_input} into JSON: {{'category': str, 'season': str, 'age_group': str, 'gender': str, 'market': str}}"
        response = self.model.generate_content(prompt)
        return eval(response.text)  # Parse JSON (use json.loads in prod)
