import os
from typing import Dict, Any, List
from dataclasses import dataclass
import google.generativeai as genai
from google_adk import Agent, LoopAgent, BaseAgent, EventActions
from google_adk.sessions import InMemorySessionService
from google_adk.tools import Tool
from .config import ConciergeConfig
from .tools import search_trends, get_sales_events, generate_social_post
from .memory import MemoryBank
from .sub_agents import (
    UserPreferenceAgent, TrendAnalysisAgent, SalesEventAgent,
    PromotionAgent, RefinerAgent
)
# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(ConciergeConfig.model_name)
@dataclass
class ConciergeState:
    user_input: str
    customer_profile: Dict[str, Any]
    trends: Dict[str, Any]
    events: List[Dict[str, Any]]
    promotions: List[Dict[str, Any]]
    recommendations: Dict[str, Any]
    social_posts: Dict[str, str]  # {platform: post}
class OrchestrationAgent(Agent):
    """Main Orchestration Agent: Coordinates sub-agents for concierge experience."""
    def __init__(self):
        super().__init__(
            name="orchestration_agent",
            model=model,
            description="Orchestrates personalized recommendations and promotions.",
            instruction="Coordinate sub-agents sequentially and in parallel where possible. Use memory for state. Output final social media posts.",
            sub_agents=[
                UserPreferenceAgent(),
                TrendAnalysisAgent(),
                SalesEventAgent(),
                PromotionAgent(),
                RefinerAgent()
            ],
            tools=[search_trends, get_sales_events, generate_social_post],
            session_service=InMemorySessionService(),
            memory_bank=MemoryBank()
        )
async def run(self, user_input: str) -> Dict[str, Any]:
        state = ConciergeState(user_input=user_input, customer_profile={}, trends={}, events=[], promotions=[], recommendations={}, social_posts={})
        
        # Sequential: User prefs
        state.customer_profile = await self.sub_agents[0].generate(state.user_input)
        
        # Parallel: Trends + Events
        trends_task = self.sub_agents[1].analyze(state.customer_profile)
        events_task = self.sub_agents[2].find_events(state.customer_profile)
        state.trends, state.events = await asyncio.gather(trends_task, events_task)
        
        # Sequential: Promotions
        state.promotions = await self.sub_agents[3].craft(state)
        
        # Loop: Refine (up to 3 retries)
        loop_agent = LoopAgent(sub_agents=[self.sub_agents[4]], max_iterations=3)
        refined = await loop_agent.run(state.promotions)
        state.promotions = refined if refined else state.promotions
        
        # Generate recommendations
        state.recommendations = self._generate_recommendations(state)
        
        # Social posts
        state.social_posts = await self._generate_social_posts(state.recommendations)
        
        # Save to memory
        self.memory_bank.save(state.user_input, state.social_posts)
        
        return {
            "recommendations": state.recommendations,
            "promotions": state.promotions,
            "social_posts": state.social_posts
        }
def _generate_recommendations(self, state: ConciergeState) -> Dict[str, Any]:
        # Personalization Engine logic (simple LLM call)
        prompt = f"Generate tailored recommendations for {state.customer_profile} based on {state.trends}, {state.events}, {state.promotions}."
        response = model.generate_content(prompt)
        return {"suggestions": response.text, "bundles": "Discounted perfume sets"}
async def _generate_social_posts(self, recommendations: Dict[str, Any]) -> Dict[str, str]:
        posts = {}
        for platform in ["X", "Facebook", "Instagram"]:
            posts[platform] = await generate_social_post(platform, recommendations)
        return posts
# Content Writer Agent (simple wrapper for final output)
class ContentWriterAgent(BaseAgent):
    """Outputs final social media posts."""
    async def run(self, social_posts: Dict[str, str]) -> str:
        output = "Final Social Media Posts:\n"
        for platform, post in social_posts.items():
            output += f"\n{platform}:\n{post}\n"
        return output
