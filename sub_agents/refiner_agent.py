from google_adk import Agent, LoopAgent, BaseAgent, EventActions
import google.generativeai as genai
from ...config import ConciergeConfig
model = genai.GenerativeModel(ConciergeConfig.model_name)
class RefinerAgent(BaseAgent):
    """Validates and refines promotions/posts."""
    async def validate(self, promotions: list) -> bool:
        # Simple check (expand with metrics)
        return len(promotions) > 0 and all("discount" in p for p in promotions)
class LoopController(LoopAgent):
    """Orchestrates refinement loop (up to 3 retries)."""
    def __init__(self):
        refiner = RefinerAgent(model=model)
        super().__init__(
            sub_agents=[refiner],
            max_iterations=3,
            instruction="Retry if validation fails."
        )
async def run(self, input_data):
        for _ in range(self.max_iterations):
            result = await self.sub_agents[0].run(input_data)
            if await self.sub_agents[0].validate(result):
                return EventActions(escalate=True)  # Proceed
        return input_data  # Fallback
