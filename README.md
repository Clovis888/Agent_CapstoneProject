# Agent_CapstoneProject
Agent_CapstoneProject

It is a multi-agent system that automates personalized product recommendations and promotions for retailers. It integrates customer profiles, seasonal events (e.g., Black Friday, Christmas), and trend analysis to deliver timely, tailored offers across channels, boosting sales.

## Project Overview
Retailers struggle with generic recommendations, missing personalization for preferences, seasons, and trends. This leads to lost sales.
**Solution**: A concierge agent that proactively crafts and promotes tailored offers, using multi-agent orchestration for research, generation, and refinement. Outputs include social media posts for X, Facebook, Instagram.
Built with Google Agent Development Kit (ADK). Supports async parallel/sequential/loop agents, sessions/memory, observability.

## Agent Structure
- **Multi-Agent System**: Orchestration Agent coordinates sub-agents (sequential for prefs/promos, parallel for trends/events, loop for refinement).
- **LLM**: Google Gemini 2.5 Flash for all agents.
- **Sessions & Memory**: InMemorySessionService for state; MemoryBank for long-term (e.g., past recommendations).
- **Observability**: ADK logging/tracing/metrics; custom logging in agents.
- **Tools**: Custom (search, events); built-in (Google Search); long-running ops via async.
- **Enhanced**: Async pipelines for real-time; eval framework.

## Main Agent Description
**OrchestrationAgent** (`agent.py`): Central coordinator. Parses input, delegates to sub-agents, generates recommendations, and outputs social posts. Uses personalization engine for suggestions and promotion engine for offers.

## Sub-Agent Descriptions
- **UserPreferenceAgent** (`sub_agents/user_preference_agent.py`): Parses input for category, season, demographics (age/gender/market).
- **TrendAnalysisAgent** (`sub_agents/trend_analysis_agent.py`): Searches/analyzes trends, competitor stats (parallel-executable).
- **SalesEventAgent** (`sub_agents/sales_event_agent.py`): Identifies events/promos by period/market (parallel).
- **PromotionAgent** (`sub_agents/promotion_agent.py`): Crafts offers/bundles/discounts.
- **RefinerAgent** (`sub_agents/refiner_agent.py`): Validates/refines outputs; LoopController retries up to 3x if invalid.
**ContentWriterAgent**: Final output formatter for social posts.

## Tools Used in the Project
- **search_trends**: LLM-simulated search for market insights (custom).
- **get_sales_events**: Loads seasonal data (custom, from JSON).
- **generate_social_post**: Platform-specific post generation (custom).
- Built-in: Google Search (via ADK), Code Execution (for analytics).
See `tools.py`.

## Installation
Built for Python 3.14.0 (latest stable as of Nov 2025).
1. Install Python 3.14.0 from https://www.python.org/downloads/release/python-3140/.
2. Install uv: `winget install astral-sh.uv`.
3. `uv venv && .\.venv\Scripts\activate.bat`.
4. `uv pip install -r requirements.txt`.
5. Set `.env`: `GOOGLE_API_KEY=your_key`.

## Running the Agent
`adk web` for interactive UI.

## Evaluation
`uv run pytest` (from root). Requires deps in requirements.txt.
Example: Tests recommendation relevance using sklearn metrics on mock data.

## Testing
`uv run python -m tests.test_agent`.
**Test Scenario**: Promotion for perfume in Christmas/New Year.
- Input: "Perfume promo for Dec, women 25-35, US."
- Expected: Tailored bundles, social posts with #HolidayGifts.

## Workflow
1. User input → UserPreferenceAgent.
2. Parallel: TrendAnalysis + SalesEvent.
3. PromotionAgent crafts offers.
4. Loop: Refiner validates/refines.
5. Orchestration generates recs/posts.
6. ContentWriter outputs.

## Example of Conversation
**User Input**: "I want to create sales and promotion event from November to December, target customer group is 20-40 years old women in Singapore".
**Agent Response** (Step-by-Step in UI):
- **Step 1 (UserPreference)**: Parsed: {'category': 'fashion/beauty', 'season': 'Nov-Dec', 'age_group': '20-40', 'gender': 'women', 'market': 'Singapore'}.
- **Step 2 (Parallel)**: Trends: "Trending: Sustainable perfumes, floral scents (up 30% YoY per Google Trends). Competitors: Sephora bundles." Events: [{"event": "Christmas", "promo": "20% off gifts"}, {"event": "Black Friday", "promo": "Buy1Get1"}].
- **Step 3 (Promotion)**: Offers: [{"offer": "Festive Perfume Bundle", "discount": "25% off"}, {"offer": "Personalized Scent Quiz", "discount": "Free sample"}].
- **Step 4 (Refine)**: Validated (no retries needed).
- **Step 5 (Recommendations)**: Suggestions: "Recommend Jo Malone Peony & Blush Suede for young professionals; bundle with holiday candles for gifting."
**Final Output (Social Media Posts)**:

  
X:
Singapore ladies 20-40! Unwrap joy this Nov-Dec with 25% off festive perfume bundles – perfect for Christmas gifting! Smell the savings on floral trends. #HolidayPerfume #BlackFridayDeals [Link to shop]”
Facebook: “Dear Singaporean women aged 20-40, make this holiday season scentsational! From Black Friday to Christmas, enjoy personalized perfume promotions: 25% off bundles + free samples. What’s your signature scent? Comment below!
#SingaporeShopping #FestiveFragrances [Image: Perfume set] [Shop Now]”

Instagram: 
Holiday Glow-Up Alert!
For our 20-40yo queens in SG: Nov-Dec exclusives – 25% off trending floral perfumes & gift bundles. Black Friday steals to Christmas cheers! Tag a friend who’d love this.

#PerfumeLovers #SingaporeHolidays #GiftIdeas







