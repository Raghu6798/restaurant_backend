from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from typing import List, Optional
from langgraph.graph import StateGraph, START, END
from langchain.prompts import ChatPromptTemplate
from langchain_sambanova import ChatSambaNovaCloud
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger
import warnings
import motor.motor_asyncio
from uuid import uuid4
import os
from bson import ObjectId
from dotenv import load_dotenv
import base64
import httpx
from typing import Any, Dict, Iterator, List, Optional, Union
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, SecretStr
from langchain_core.utils.utils import secret_from_env
from openai import OpenAI
warnings.filterwarnings("ignore")

load_dotenv()

# Configure logging
logger.add("logs/app.log", rotation="1 MB", retention="14 days", level="INFO")

# Constants
TOTAL_TABLES = 10
MONGO_URI = os.getenv("MONGO_URI")
SESSION_ID = str(uuid4())

# MongoDB connection
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client["restaurant"]
bookings_collection = db["bookings"]

class ChatMoonshotAI(BaseChatModel):
    model_name: str = Field(default="kimi-k1.5-preview")
    temperature: Optional[float] = 0.5
    max_tokens: Optional[int] = 1024
    timeout: Optional[int] = None
    stop: Optional[List[str]] = None
    max_retries: int = 2
    kimi_api_key: Optional[SecretStr] = Field(
        alias="api_key", default_factory=secret_from_env("KIMI_API_KEY", default=None)
    )

    @staticmethod
    def encode_image(image_path: str) -> str:
      if not image_path.startswith("http"):
        with open(image_path, "rb") as image_file:
          return base64.b64encode(image_file.read()).decode('utf-8')
      else:
        response = httpx.get(image_path)
        response.raise_for_status()
        return base64.b64encode(response.content).decode('utf-8')

    def _generate(
        self,
        messages: List[BaseMessage],
        image: Optional[Union[str, bytes]] = None,
        image_is_base64: bool = False,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a response from Kimi API, with optional image input."""

        client = OpenAI(api_key=self.kimi_api_key.get_secret_value(), base_url="https://api.moonshot.ai/v1")


        formatted_messages = [
            {"role": "system", "content": "You are an  Intelligent Assistant who assists with user queries. You are developed by Moonshot AI , and your name is "},
            {
                "role": "user",
                "name": "user",
                "content": [{"type": "text", "text": msg.content} for msg in messages],
            }
        ]

        if image:
            image_payload = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}" if image_is_base64 else image
                }
            }
            formatted_messages[1]["content"].append(image_payload)

        # API call
        response = client.chat.completions.create(
            model=self.model_name,
            messages=formatted_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        aimessage = AIMessage(
            content=response.choices[0].message.content,
            usage_metadata=UsageMetadata(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )

        return ChatResult(generations=[ChatGeneration(message=aimessage)])

    def _stream(
        self,
        messages: List[BaseMessage],
        image: Optional[Union[str, bytes]] = None,
        image_is_base64: bool = False,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream responses from Kimi API with optional image input."""

        client = OpenAI(api_key=self.kimi_api_key.get_secret_value(), base_url="https://api.minimaxi.chat/v1")

        formatted_messages = [
            {"role": "system", "content": "You are kimi-k1.5-model, which is a multimodal model that supports uploading images.You are developed to help user queries with Image understanding"},
            {
                "role": "user",
                "name": "user",
                "content": [{"type": "text", "text": msg.content} for msg in messages],
            }
        ]

        if image:
            image_payload = {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}" if image_is_base64 else image
                }
            }
            formatted_messages[1]["content"].append(image_payload)

        stream = client.chat.completions.create(
            model=self.model_name,
            messages=formatted_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=chunk.choices[0].delta.content)
                )

    @property
    def _llm_type(self) -> str:
        return "Minimax-Text-01"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"model_name": self.model_name}



def serialize_booking(booking):
    booking["_id"] = str(booking["_id"])
    return booking

class Booking(BaseModel):
    id: Optional[str] = Field(None, alias="_id")
    date: str
    time: str
    guests: int
    specialRequests: Optional[str] = None
    timestamp: datetime
    model_config = ConfigDict(arbitrary_types_allowed=True)

class DashboardState(BaseModel):
    messages: List[AnyMessage] = []
    total_bookings: int = 0
    todays_guests: int = 0
    tables_needed: int = 0
    available_tables: int = 0
    bookings: List[Booking] = []
    user_input: Optional[str] = None

# FastAPI app
app = FastAPI()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Llama31 = ChatSambaNovaCloud(
    model="Llama-3.1-Tulu-3-405B",
    temperature=0.5,
    max_tokens=2048,
)

Kimi_k_15 = ChatMoonshotAI(
    model_name="kimi-k1.5-preview",
    temperature=0.5,
    max_tokens=1024,
)
# SYSTEM_PROMPT template
SYSTEM_PROMPT = """
You are a restaurant assistant. Here is the current dashboard state:
- Total Bookings: {total_bookings}
- Today's Guests: {todays_guests}
- Available Tables: {available_tables}
- Total Tables in the restaurant: {total_tables}
- Booking Details:
{booking_details}

You are managing table reservations for a restaurant with the following seating rules:
- The restaurant has {total_tables} tables in total.
- Each reservation gets a dedicated table.
- If a reservation has more than 2 guests, they will need multiple tables.

Now, based on these rules, here are today's bookings:
{booking_details}

Given these bookings:
1. **Total tables occupied**: {tables_needed}
2. **Total tables still available**: {available_tables}
3. **Can a stranger join an already booked table?** No.

If a user asks about a date, check the number of bookings for that date.
If `bookings_per_date[date] < available_tables`, say "Yes, tables are available."
Otherwise, say "No, tables are fully booked."

User: "{user_input}"
Today's date: "{today}"
"""

# Fetch dashboard statistics
async def fetch_dashboard_statistics(state: DashboardState) -> DashboardState:
    """Fetch the latest restaurant dashboard statistics from MongoDB."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Fetching dashboard stats for date = {today}")

        # Await the count_documents coroutine
        state.total_bookings = await bookings_collection.count_documents({})

        # Await the find coroutine and convert the cursor to a list
        today_bookings = await bookings_collection.find({"date": today}).to_list(None)
        state.todays_guests = sum(booking["guests"] for booking in today_bookings)

        # Calculate tables needed
        state.tables_needed = sum(
            b["guests"] // 2 + (b["guests"] % 2) for b in today_bookings
        )
        state.available_tables = TOTAL_TABLES - state.tables_needed

        # Fetch the latest bookings
        latest_bookings = await bookings_collection.find().sort("timestamp", -1).limit(10).to_list(None)
        state.bookings = [
            Booking(**{**doc, "_id": str(doc["_id"])}) for doc in latest_bookings
        ]

        logger.info("Dashboard statistics fetched successfully!")

    except Exception as e:
        logger.error(f"Error fetching dashboard statistics: {e}")

    return state

# Restaurant chatbot function
async def restaurant_chatbot(state: DashboardState) -> DashboardState:
    """Use Llama-3.1 to generate responses based on restaurant dashboard state."""
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
    chain = prompt | Kimi_k_15| StrOutputParser()
    
    today = datetime.now().strftime("%Y-%m-%d")
    booking_details = "\n".join(
        [f"- {b.date} | {b.time} | {b.guests} guests | Ordered: {b.specialRequests if b.specialRequests else 'No food order'}"
         for b in state.bookings]
    ) or "No bookings yet."

    response = await chain.ainvoke({
        "total_bookings": state.total_bookings,
        "todays_guests": state.todays_guests,
        "available_tables": state.available_tables,
        "total_tables": TOTAL_TABLES,
        "tables_needed": state.tables_needed,
        "booking_details": booking_details,
        "user_input": state.user_input,
        "today": today
    })
    
    state.messages.append(AIMessage(content=response))
    return state

# StateGraph setup
rest_graph = StateGraph(state_schema=DashboardState)
rest_graph.add_node("Fetch_Dashboard", fetch_dashboard_statistics)
rest_graph.add_node("Restaurant_ChatBot", restaurant_chatbot)

rest_graph.add_edge(START, "Fetch_Dashboard")
rest_graph.add_edge("Fetch_Dashboard", "Restaurant_ChatBot")
rest_graph.add_edge("Restaurant_ChatBot", END)

# Compile the graph
memory = MemorySaver()
compiled_graph = rest_graph.compile(checkpointer=memory)

# /chat endpoint
@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        user_input = data.get("user_input")

        if not user_input:
            raise HTTPException(status_code=400, detail="User input is required")

        # Initialize the state with the user input
        input_state = DashboardState(messages=[HumanMessage(content=user_input)], user_input=user_input)

        # Fetch dashboard statistics (await the coroutine)
        input_state = await fetch_dashboard_statistics(input_state)

        # Stream the response from the compiled graph
        response_content = ""
        async for chunk in compiled_graph.astream(input_state, {"configurable": {"thread_id": SESSION_ID}}, stream_mode="values"):
            if "messages" in chunk:
                response_content = chunk["messages"][-1].content

        return {"response": response_content}

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Other endpoints (dashboard, bookings, etc.)
@app.get("/api/dashboard")
async def get_dashboard_stats():
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Fetching dashboard stats for date = {today}")

        # Fetch total bookings
        total_bookings = await bookings_collection.count_documents({})

        # Fetch today's bookings
        today_bookings = await bookings_collection.find({"date": today}).to_list(None)
        today_guests = sum(booking["guests"] for booking in today_bookings)

        # Calculate tables needed based on guests per booking
        tables_needed = sum(
            booking["guests"] // 2 + (booking["guests"] % 2) for booking in today_bookings
        )
        available_tables = TOTAL_TABLES - tables_needed

        # Fetch the latest bookings
        latest_bookings = (
            await bookings_collection.find()
            .sort("timestamp", -1)
            .limit(10)
            .to_list(None)
        )

        logger.success("Dashboard statistics fetched successfully!")

        return {
            "total_bookings": total_bookings,
            "todays_guests": today_guests,
            "tables_needed": tables_needed,
            "available_tables": available_tables,
            "bookings": [serialize_booking(booking) for booking in latest_bookings],
        }

    except Exception as e:
        logger.error(f"Dashboard Fetch Error: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch dashboard data")

@app.post("/api/bookings/")
async def create_booking(booking: Booking):
    try:
        new_booking = booking.dict()
        new_booking["timestamp"] = datetime.now()
        result = await bookings_collection.insert_one(new_booking)
        new_booking["_id"] = str(result.inserted_id)  # Set id after insertion

        # Fetch updated dashboard statistics
        today = datetime.now().strftime("%Y-%m-%d")
        total_bookings = await bookings_collection.count_documents({})
        today_bookings = await bookings_collection.find({"date": today}).to_list(None)
        today_guests = sum(booking["guests"] for booking in today_bookings)
        booked_tables_today = len(today_bookings)
        available_tables = max(0, TOTAL_TABLES - booked_tables_today)

        logger.success("Booking created successfully!")
        return {
            "message": "Booking confirmed successfully!",
            "booking": new_booking,
            "dashboard_stats": {
                "total_bookings": total_bookings,
                "todays_guests": today_guests,
                "available_tables": available_tables,
            },
        }
    except Exception as e:
        logger.error(f"Booking Error: {e}")
        raise HTTPException(status_code=500, detail="Could not create booking")

@app.get("/api/bookings/")
async def get_bookings():
    try:
        bookings = await bookings_collection.find().to_list(100)
        return [serialize_booking(booking) for booking in bookings]
    except Exception as e:
        print("Fetch Error:", e)
        raise HTTPException(status_code=500, detail="Could not fetch bookings")
