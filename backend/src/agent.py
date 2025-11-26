import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

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

# ------------ Company FAQ & Content Management ------------

FAQ_FILE = Path(__file__).parent.parent / "shared-data" / "day5_company_faq.json"
LEADS_FILE = Path(__file__).parent.parent / "leads_captured.json"

class CompanyKnowledgeBase:
    def __init__(self):
        self.data = self._load_faq()
        self.company_name = self.data.get("company", {}).get("name", "our company")
    
    def _load_faq(self):
        """Load company FAQ and content from JSON file"""
        if not FAQ_FILE.exists():
            logger.error(f"FAQ file not found: {FAQ_FILE}")
            FAQ_FILE.parent.mkdir(parents=True, exist_ok=True)
            # Create minimal default
            default_data = {
                "company": {
                    "name": "Razorpay",
                    "tagline": "India's Leading Payment Gateway",
                    "description": "Payment solutions for businesses"
                },
                "faqs": [
                    {
                        "question": "What does Razorpay do?",
                        "answer": "Razorpay helps businesses accept online payments easily."
                    }
                ]
            }
            with open(FAQ_FILE, "w") as f:
                json.dump(default_data, f, indent=2)
            return default_data
        
        try:
            with open(FAQ_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing FAQ file: {e}")
            return {"company": {}, "faqs": []}
    
    def search_faq(self, query: str) -> Optional[dict]:
        """Simple keyword-based FAQ search"""
        query_lower = query.lower()
        
        # Search through FAQs
        for faq in self.data.get("faqs", []):
            question = faq.get("question", "").lower()
            answer = faq.get("answer", "").lower()
            
            # Check if query keywords match question or answer
            if any(word in question for word in query_lower.split()) or \
               any(word in query_lower for word in question.split()):
                return faq
        
        return None
    
    def get_company_intro(self) -> str:
        """Get company introduction"""
        company = self.data.get("company", {})
        return f"{company.get('name', 'Our company')} - {company.get('tagline', '')}. {company.get('description', '')}"
    
    def get_products_summary(self) -> str:
        """Get summary of products"""
        products = self.data.get("products", [])
        if not products:
            return "We offer various solutions for businesses."
        
        summary = "Our main products include: "
        summary += ", ".join([p.get("name", "") for p in products[:3]])
        return summary
    
    def get_pricing_info(self) -> str:
        """Get pricing information"""
        pricing = self.data.get("pricing", {})
        if not pricing:
            return "Please contact us for pricing details."
        
        pg_pricing = pricing.get("payment_gateway", {})
        return f"Our payment gateway charges {pg_pricing.get('transaction_fee', 'competitive rates')} with {pg_pricing.get('setup_fee', 'no setup fee')}."

# ------------ Lead Capture State ------------

class LeadData:
    def __init__(self):
        self.data = {
            "name": "",
            "company": "",
            "email": "",
            "role": "",
            "use_case": "",
            "team_size": "",
            "timeline": "",
            "notes": "",
            "captured_at": ""
        }
    
    def is_complete(self) -> bool:
        """Check if minimum required fields are filled"""
        required = ["name", "company", "email", "use_case"]
        return all(self.data.get(field) for field in required)
    
    def missing_fields(self) -> list:
        """Get list of missing required fields"""
        required = ["name", "company", "email", "use_case"]
        return [field for field in required if not self.data.get(field)]
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {**self.data, "captured_at": datetime.now().isoformat()}

# ------------ Lead Storage ------------

class LeadStorage:
    @staticmethod
    def save_lead(lead_data: dict):
        """Save lead to JSON file"""
        # Load existing leads
        if LEADS_FILE.exists():
            try:
                with open(LEADS_FILE, "r") as f:
                    leads = json.load(f)
            except json.JSONDecodeError:
                leads = []
        else:
            LEADS_FILE.parent.mkdir(parents=True, exist_ok=True)
            leads = []
        
        # Add new lead
        leads.append(lead_data)
        
        # Save back to file
        with open(LEADS_FILE, "w") as f:
            json.dump(leads, f, indent=2)
        
        logger.info(f"Lead saved: {lead_data.get('name')} from {lead_data.get('company')}")

# ------------ SDR Agent ------------

class SDRAgent(Agent):
    """Sales Development Representative voice agent"""
    def __init__(self, knowledge_base: CompanyKnowledgeBase):
        self.kb = knowledge_base
        self.lead = LeadData()
        self.conversation_stage = "greeting"
        
        company_intro = knowledge_base.get_company_intro()
        products_summary = knowledge_base.get_products_summary()
        
        super().__init__(
            instructions=f"""
You are a friendly and professional Sales Development Representative (SDR) for {knowledge_base.company_name}.

**Company Information:**
{company_intro}

{products_summary}

**Your Role:**
You're here to understand the visitor's needs, answer their questions, and capture their information as a potential lead.

**Conversation Flow:**

1. **GREETING STAGE**
   - Greet warmly: "Hi! Welcome to {knowledge_base.company_name}!"
   - Ask: "What brings you here today?"
   - Ask: "Tell me a bit about what you're working on"

2. **DISCOVERY STAGE**
   - Listen carefully to understand their needs
   - Ask clarifying questions about their business
   - Identify their pain points
   - Use 'search_faq' when they ask questions

3. **FAQ STAGE**
   - Answer questions using the 'search_faq' function
   - Stay accurate - only use FAQ content
   - If you don't know, say so and offer to connect them with the team

4. **LEAD CAPTURE STAGE**
   - Naturally collect information during conversation
   - Required: name, company, email, use_case
   - Optional: role, team_size, timeline
   - Use 'update_lead' to store info as you learn it
   - Make it conversational, not like filling a form!
   
   Example natural flow:
   - "By the way, what's your name?"
   - "Which company are you with?"
   - "What's the best email to reach you?"
   - "What would you primarily use this for?"

5. **CLOSING STAGE**
   - Detect when they're done: "that's all", "I'm done", "thanks bye"
   - Use 'finalize_lead' to save and summarize
   - Thank them warmly
   - Mention next steps: "Our team will reach out within 24 hours"

**Important Guidelines:**
- Be warm, helpful, and professional
- Listen more than you talk
- Ask open-ended questions
- Don't be pushy or salesy
- Focus on understanding their needs
- Collect lead info naturally through conversation
- Use FAQ for accurate answers only

**Conversation Tips:**
- Start friendly: "Hey! How can I help you today?"
- Be curious: "Tell me more about that..."
- Validate: "That makes sense..."
- Transition smoothly: "By the way, I'd love to follow up with you..."

Your goal: Understand if {knowledge_base.company_name} can help them and capture their info for follow-up.
""",
        )
    
    @function_tool
    async def search_faq(self, context: RunContext, question: str):
        """
        Search the company FAQ for answers to user questions.
        
        Args:
            question: The user's question about product, pricing, or company
        """
        logger.info(f"Searching FAQ for: {question}")
        
        result = self.kb.search_faq(question)
        
        if result:
            return f"{result['answer']}"
        else:
            # Fallback to general company info
            if "price" in question.lower() or "cost" in question.lower() or "fee" in question.lower():
                return self.kb.get_pricing_info()
            elif "product" in question.lower() or "do" in question.lower() or "what" in question.lower():
                return self.kb.get_company_intro()
            else:
                return "I don't have specific information about that in my knowledge base. Let me connect you with our team who can provide detailed answers. Can I get your contact information so they can reach out?"
    
    @function_tool
    async def update_lead(
        self,
        context: RunContext,
        name: str = "",
        company: str = "",
        email: str = "",
        role: str = "",
        use_case: str = "",
        team_size: str = "",
        timeline: str = "",
        notes: str = ""
    ):
        """
        Update the lead information as you learn about the visitor.
        Call this function each time you learn a new piece of information.
        
        Args:
            name: Visitor's full name
            company: Company name
            email: Email address
            role: Their role/position (e.g., Founder, CTO, Product Manager)
            use_case: What they want to use the product for
            team_size: Size of their team (e.g., "1-10", "11-50", "50+")
            timeline: When they want to start (e.g., "immediately", "this month", "exploring")
            notes: Any additional context about their needs
        """
        logger.info(f"Updating lead: name={name}, company={company}, email={email}")
        
        # Update fields that are provided
        if name:
            self.lead.data["name"] = name
        if company:
            self.lead.data["company"] = company
        if email:
            self.lead.data["email"] = email
        if role:
            self.lead.data["role"] = role
        if use_case:
            self.lead.data["use_case"] = use_case
        if team_size:
            self.lead.data["team_size"] = team_size
        if timeline:
            self.lead.data["timeline"] = timeline
        if notes:
            # Append notes instead of replacing
            existing_notes = self.lead.data.get("notes", "")
            self.lead.data["notes"] = f"{existing_notes} {notes}".strip()
        
        # Check if we have minimum info
        if self.lead.is_complete():
            return "Great! I have all the key information. Is there anything else you'd like to know?"
        else:
            missing = self.lead.missing_fields()
            if len(missing) == 1:
                field_name = missing[0].replace("_", " ")
                return f"Just need one more thing - could you share your {field_name}?"
            else:
                return "Thanks for that information!"
    
    @function_tool
    async def finalize_lead(self, context: RunContext):
        """
        Finalize and save the lead when the conversation is ending.
        Call this when the user signals they're done (e.g., "that's all", "thanks bye").
        """
        logger.info("Finalizing lead capture")
        
        if not self.lead.is_complete():
            missing = self.lead.missing_fields()
            missing_text = ", ".join([f.replace("_", " ") for f in missing])
            return f"Before you go, could you quickly share your {missing_text}? This will help our team reach out to you properly."
        
        # Save the lead
        lead_dict = self.lead.to_dict()
        LeadStorage.save_lead(lead_dict)
        
        # Generate summary
        summary = f"""Perfect! Let me summarize what I learned:

📋 **Lead Summary:**
- Name: {lead_dict['name']}
- Company: {lead_dict['company']}
- Email: {lead_dict['email']}
{f"- Role: {lead_dict['role']}" if lead_dict.get('role') else ""}
- Use Case: {lead_dict['use_case']}
{f"- Team Size: {lead_dict['team_size']}" if lead_dict.get('team_size') else ""}
{f"- Timeline: {lead_dict['timeline']}" if lead_dict.get('timeline') else ""}

Your information has been saved, and our team will reach out to you within 24 hours to discuss how {self.kb.company_name} can help with {lead_dict['use_case']}.

Thank you for your time, {lead_dict['name'].split()[0]}! Looking forward to working with {lead_dict['company']}! 🚀"""
        
        return summary
    
    @function_tool
    async def get_product_info(self, context: RunContext, product_name: str = ""):
        """
        Get detailed information about a specific product.
        
        Args:
            product_name: Name of the product to get info about
        """
        products = self.kb.data.get("products", [])
        
        if not product_name:
            # Return all products
            product_list = "\n".join([f"- {p.get('name')}: {p.get('description')}" for p in products])
            return f"Our products:\n{product_list}"
        
        # Search for specific product
        for product in products:
            if product_name.lower() in product.get("name", "").lower():
                return f"{product.get('name')}: {product.get('description')} Perfect for: {product.get('use_case')}"
        
        return "I don't have information about that specific product. Let me share our main offerings..."

# ------------ Prewarm and Entrypoint ------------

def prewarm(proc: JobProcess):
    """Prewarm function to load models and data before agent starts"""
    proc.userdata["vad"] = silero.VAD.load()
    
    # Preload FAQ data
    knowledge_base = CompanyKnowledgeBase()
    proc.userdata["knowledge_base"] = knowledge_base
    logger.info(f"Prewarmed with FAQ for {knowledge_base.company_name}")

async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    
    logger.info("🚀 Starting SDR Agent...")
    
    # Load knowledge base from prewarm or create new
    if "knowledge_base" in ctx.proc.userdata:
        knowledge_base = ctx.proc.userdata["knowledge_base"]
        logger.info(f"Using prewarmed knowledge base for {knowledge_base.company_name}")
    else:
        knowledge_base = CompanyKnowledgeBase()
        logger.info(f"Created new knowledge base for {knowledge_base.company_name}")

    # Voice agent session pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="Iris",  # Professional, friendly voice for SDR
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True,
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()
    
    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)
    
    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")
    
    ctx.add_shutdown_callback(log_usage)

    # Start session with SDR Agent
    logger.info("🎙️ Starting SDR agent session...")
    await session.start(
        agent=SDRAgent(knowledge_base),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    logger.info("🔗 Connecting to room...")
    await ctx.connect()
    logger.info(f"✅ SDR Agent for {knowledge_base.company_name} connected successfully!")

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))
