from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import os

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")  # set in env or Azure config

llm = ChatGroq(
    temperature=0.8,
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    streaming=True
)

# --------------------------------------------------
# APP INIT
# --------------------------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# 1️⃣ PROFILE ANALYSIS
# --------------------------------------------------
@app.post("/analyze")
def analyze_profile(profile: dict):

    prompt = f"""
You are an expert LinkedIn profile analyst.

PROFILE:
Name: {profile.get("name")}
Headline: {profile.get("headline")}
About: {profile.get("about")}
Experience: {profile.get("experience")}

Return:
1. Short professional summary
2. Core strengths
3. Key gaps
4. Career suggestions
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "analysis": response.content
    }

# --------------------------------------------------
# 2️⃣ JOB MATCHING (PROFILE ↔ JOB)
# --------------------------------------------------
@app.post("/job-match")
def job_match(payload: dict):

    profile = payload.get("profile")
    job_desc = payload.get("job")

    prompt = f"""
You are an AI recruiter.

PROFILE:
{profile}

JOB DESCRIPTION:
{job_desc}

Return clearly:
- Match score (0–100)
- Strengths vs job
- Missing skills
- Final recommendation
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "result": response.content
    }

# --------------------------------------------------
# 3️⃣ CHATBOT (ASK ABOUT PROFILE)
# --------------------------------------------------
@app.post("/chat")
def chat(payload: dict):

    profile = payload.get("profile")
    question = payload.get("question")

    prompt = f"""
You are a helpful AI assistant analyzing a LinkedIn profile.

PROFILE:
{profile}

USER QUESTION:
{question}

Answer concisely and professionally.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        "answer": response.content
    }

# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------
@app.get("/")
def health():
    return {"status": "ok"}

