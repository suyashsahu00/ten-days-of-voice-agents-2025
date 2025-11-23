import logging
import json

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

# ------------ Coffee Order State Definition ------------

ORDER_FIELDS = ["drinkType", "size", "milk", "extras", "name"]
DRINK_TYPES = ["coffee", "latte", "espresso", "cappuccino", "americano"]
SIZES = ["small", "medium", "large"]
MILK_TYPES = ["whole", "skim", "soy", "almond", "oat"]
EXTRAS = ["whipped cream", "caramel", "vanilla", "chocolate", "none"]

class OrderState:
    def __init__(self):
        self.state = {
            "drinkType": "",
            "size": "",
            "milk": "",
            "extras": [],
            "name": "",
        }

    def is_complete(self):
        return all(
            self.state[field] if field != "extras" else True
            for field in ORDER_FIELDS
        )

    def missing_fields(self):
        return [
            field for field in ORDER_FIELDS
            if field != "extras" and not self.state[field]
        ]

    def to_dict(self):
        return self.state

# ------------ Coffee Barista Agent Implementation ------------

class CoffeeBaristaAgent(Agent):
    def __init__(self):
        # Persona prompt for LLM
        super().__init__(
            instructions=f"""
You are a friendly and efficient coffee shop barista for Falcon Cafe. 
Greet the customer and gather all order information step by step. Ask clear questions until you have:
- drinkType (one of {DRINK_TYPES})
- size (one of {SIZES})
- milk (one of {MILK_TYPES})
- extras (like {EXTRAS}, can be empty/none)
- name (customer's first name)
After all fields are filled, thank the customer and summarize the order.
""",
        )
        self.order_state = OrderState()  # One order per session

    @function_tool  # makes this function callable by the LLM
    async def update_order(
        self,
        context: RunContext,
        drinkType: str = "",
        size: str = "",
        milk: str = "",
        extras: str = "",
        name: str = "",
    ):
        """Update the order state. LLM fills in fields as user answers."""
        logger.info("Updating order with: "
                    f"drinkType={drinkType}, size={size}, milk={milk}, extras={extras}, name={name}")

        if drinkType:
            self.order_state.state["drinkType"] = drinkType
        if size:
            self.order_state.state["size"] = size
        if milk:
            self.order_state.state["milk"] = milk
        if extras:
            # Accept comma-separated or single string
            normalized = [e.strip() for e in extras.split(",") if e.strip() and e.lower() != "none"]
            self.order_state.state["extras"] = normalized if normalized else []
        if name:
            self.order_state.state["name"] = name

        # If complete, write to JSON and summarize.
        if self.order_state.is_complete():
            with open("latest_order.json", "w") as f:
                json.dump(self.order_state.to_dict(), f, indent=2)
            summary = self.order_state.to_dict()
            logger.info(f"Order complete: {summary}")
            return f"Thank you, {summary['name']}! Your order: {summary['size']} {summary['drinkType']} with {summary['milk']} milk" + \
                   (f", extras: {', '.join(summary['extras'])}" if summary["extras"] else "") + \
                   " has been placed."

        # If not, ask for missing fields
        missing = self.order_state.missing_fields()
        if missing:
            missing_pretty = ', '.join(missing)
            return f"Could you please tell me your {missing_pretty}?"
        else:
            return "Please provide any remaining details for your order."

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
            voice="Iris",
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # For metrics (optional)
    usage_collector = metrics.UsageCollector()
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
    ctx.add_shutdown_callback(log_usage)

    # Start session using CoffeeBaristaAgent
    await session.start(
        agent=CoffeeBaristaAgent(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))

