from google.adk.agents import LlmAgent, LoopAgent, SequentialAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from typing import Optional

GROQ_MODEL = LiteLlm(model="groq/llama-3.3-70b-versatile")


# ── Callback: runs before setup agent, seeds user_request into state ──
def seed_user_request(callback_context: CallbackContext, llm_request: LlmRequest) -> Optional[LlmRequest]:
    """Capture the user's message into session state so {user_request} works."""
    if "user_request" not in callback_context.state:
        # Pull the last user message from the LLM request contents
        for content in reversed(llm_request.contents):
            if content.role == "user":
                for part in content.parts:
                    if hasattr(part, "text") and part.text:
                        callback_context.state["user_request"] = part.text
                        callback_context.state["feedback"] = "No feedback yet. Write a first draft."
                        callback_context.state["email_draft"] = ""
                        break
                break
    return None  # None = let the agent run normally


# ── Setup agent: now just a pass-through to seed state via callback ───
setup = LlmAgent(
    name="setup",
    model=GROQ_MODEL,
    instruction="""You are a setup agent. Say only: 'Setup complete. Starting email pipeline.'""",
    output_key="setup_status",
    before_model_callback=seed_user_request,   # ← seeds all 3 state vars
)

# ── Writer ────────────────────────────────────────────────────────────
writer = LlmAgent(
    name="writer",
    model=GROQ_MODEL,
    instruction="""Write or improve an email.

User request: {user_request}

Latest feedback:
{feedback}

If feedback says 'No feedback yet', write a professional first draft.
Otherwise, revise the draft incorporating all feedback points.

Output ONLY the email text, nothing else.""",
    output_key="email_draft",
)

# ── Critic ────────────────────────────────────────────────────────────
critic = LlmAgent(
    name="critic",
    model=GROQ_MODEL,
    instruction="""Review this email draft:

EMAIL DRAFT:
{email_draft}

Score on: Clarity (1-10), Tone (1-10), Conciseness (1-10), Professionalism (1-10)

If ANY score is below 7:
- List SPECIFIC improvements needed
- End your response with: VERDICT: REVISE

If ALL scores are 7 or above:
- Write: APPROVED: This email meets quality standards.
- End your response with: VERDICT: APPROVED""",
    output_key="feedback",
)

# ── Escalation check (stops the loop when critic approves) ────────────
exit_checker = LlmAgent(
    name="exit_checker",
    model=GROQ_MODEL,
    instruction="""Check this feedback: {feedback}

If it contains 'VERDICT: APPROVED', respond with exactly: escalate
If it contains 'VERDICT: REVISE', respond with exactly: continue""",
    output_key="loop_control",
)

# ── Loop ──────────────────────────────────────────────────────────────
quality_loop = LoopAgent(
    name="quality_loop",
    sub_agents=[writer, critic],
    max_iterations=3,
)

# ── Root pipeline ─────────────────────────────────────────────────────
root_agent = SequentialAgent(
    name="email_pipeline",
    sub_agents=[setup, quality_loop],
)