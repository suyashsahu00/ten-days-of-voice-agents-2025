import logging
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext,
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# ------------ Wellness Data Management ------------

WELLNESS_LOG_FILE = "wellness_log.json"

class WellnessLogger:
    def __init__(self):
        self.log_file = Path(WELLNESS_LOG_FILE)
        self.entries = self._load_entries()

    def _load_entries(self):
        """Load existing wellness log entries from JSON file"""
        if self.log_file.exists():
            try:
                with open(self.log_file, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse {WELLNESS_LOG_FILE}, starting fresh")
                return []
        return []

    def get_last_entry(self):
        """Get the most recent wellness check-in"""
        if self.entries:
            return self.entries[-1]
        return None

    def add_entry(self, mood: str, energy: str, objectives: list, stress: str = ""):
        """Add a new wellness check-in entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "date": datetime.now().strftime("%Y-%m-%d"),
            "time": datetime.now().strftime("%H:%M"),
            "mood": mood,
            "energy": energy,
            "stress": stress,
            "objectives": objectives,
            "summary": f"User reported feeling {mood} with {energy} energy. Goals: {', '.join(objectives)}"
        }
        
        self.entries.append(entry)
        
        # Save to file
        with open(self.log_file, "w") as f:
            json.dump(self.entries, f, indent=2)
        
        logger.info(f"Saved wellness entry: {entry}")
        return entry

    def get_context_summary(self):
        """Get a summary of recent entries for agent context"""
        if not self.entries:
            return "This is the user's first check-in."
        
        last_entry = self.entries[-1]
        return f"Last check-in on {last_entry['date']}: User felt {last_entry['mood']} with {last_entry['energy']} energy. Their goals were: {', '.join(last_entry['objectives'])}."

# ------------ Wellness Check-in State ------------

class CheckInState:
    def __init__(self):
        self.state = {
            "mood": "",
            "energy": "",
            "stress": "",
            "objectives": []
        }

    def is_complete(self):
        """Check if all required fields are filled"""
        return all([
            self.state["mood"],
            self.state["energy"],
            len(self.state["objectives"]) > 0
        ])

    def missing_fields(self):
        """Return list of fields that still need to be collected"""
        missing = []
        if not self.state["mood"]:
            missing.append("mood")
        if not self.state["energy"]:
            missing.append("energy level")
        if len(self.state["objectives"]) == 0:
            missing.append("today's objectives")
        return missing

# ------------ Health & Wellness Agent ------------

class HealthWellnessAgent(Agent):
    def __init__(self):
        # Load previous check-ins
        self.wellness_logger = WellnessLogger()
        context_summary = self.wellness_logger.get_context_summary()
        
        super().__init__(
            instructions=f"""
You are a warm, supportive Health & Wellness Voice Companion. Your role is to conduct brief daily check-ins with users about their mental and physical well-being.

**IMPORTANT GUIDELINES:**
- You are NOT a medical professional or therapist
- NEVER provide medical diagnoses or clinical advice
- Keep conversations supportive, practical, and grounded
- Focus on simple, actionable wellness tips
- Be empathetic but realistic

**Your Check-in Process:**
1. Greet the user warmly
2. Ask about their MOOD (how they're feeling emotionally)
3. Ask about their ENERGY level (physical energy today)
4. Ask if anything is STRESSING them out (optional)
5. Ask about their OBJECTIVES for today (1-3 things they want to accomplish)
6. Offer simple, practical advice or reflections based on what they share
7. Recap the conversation: mood + energy + objectives
8. Ask "Does this sound right?" for confirmation

**Context from previous check-ins:**
{context_summary}

**Advice Style (keep it simple):**
- Break large goals into smaller steps
- Suggest short breaks or walks
- Encourage self-care activities
- Validate their feelings
- Offer grounding techniques (deep breaths, stretching)

Keep responses conversational, warm, and concise.
""",
        )
        self.checkin_state = CheckInState()

    @function_tool
    async def update_checkin(
        self,
        context: RunContext,
        mood: str = "",
        energy: str = "",
        stress: str = "",
        objectives: str = ""
    ):
        """
        Update the daily wellness check-in state.
        LLM calls this as user shares their mood, energy, stress, and objectives.
        """
        logger.info(f"Updating check-in: mood={mood}, energy={energy}, stress={stress}, objectives={objectives}")

        if mood:
            self.checkin_state.state["mood"] = mood
        if energy:
            self.checkin_state.state["energy"] = energy
        if stress:
            self.checkin_state.state["stress"] = stress
        if objectives:
            # Parse comma-separated objectives
            obj_list = [obj.strip() for obj in objectives.split(",") if obj.strip()]
            if obj_list:
                self.checkin_state.state["objectives"] = obj_list

        # Check if check-in is complete
        if self.checkin_state.is_complete():
            # Save to wellness log
            entry = self.wellness_logger.add_entry(
                mood=self.checkin_state.state["mood"],
                energy=self.checkin_state.state["energy"],
                objectives=self.checkin_state.state["objectives"],
                stress=self.checkin_state.state["stress"]
            )
            
            # Generate summary for user
            summary = f"""
Great! Let me recap today's check-in:

- Mood: {entry['mood']}
- Energy: {entry['energy']}
{f"- Stress: {entry['stress']}" if entry['stress'] else ""}
- Your goals for today: {', '.join(entry['objectives'])}

Does this sound right?
"""
            logger.info(f"Check-in complete: {entry}")
            return summary

        # If not complete, ask for missing fields
        missing = self.checkin_state.missing_fields()
        if missing:
            return f"Thank you for sharing! I'd also like to know about your {missing[0]}. Could you tell me more?"
        
        return "Thank you for sharing."

    @function_tool
    async def get_previous_checkin(self, context: RunContext):
        """
        Retrieve the last wellness check-in for reference.
        LLM can call this to personalize the conversation.
        """
        last_entry = self.wellness_logger.get_last_entry()
        if last_entry:
            return f"Last check-in was on {last_entry['date']} at {last_entry['time']}. {last_entry['summary']}"
        return "This is the user's first check-in."

# ------------ Prewarm and Entrypoint ------------

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()

async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Voice agent session pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="Iris",  # Warm, friendly voice
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection (optional)
    usage_collector = metrics.UsageCollector()
    
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
    
    ctx.add_shutdown_callback(log_usage)

    # Start session using HealthWellnessAgent
    await session.start(
        agent=HealthWellnessAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
