import logging
import json
import re
from typing import List, Dict
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Question Evaluation API",
    description="API for evaluating user answers with scoring and educational links using Gemini models.",
    version="1.0.0"
)

# Pydantic models for input validation
class QuestionInput(BaseModel):
    question: str
    gemini_answer: str
    user_answer: str
    topic: str
    classification: str

class TopicInput(BaseModel):
    topic: str
    questions: List[QuestionInput]

class InputData(BaseModel):
    topics: List[TopicInput]

class QuestionOutput(BaseModel):
    question: str
    gemini_answer: str
    user_answer: str
    topic: str
    classification: str
    links: List[str]
    score_explanation: str
    score: float

class TopicOutput(BaseModel):
    topic: str
    questions: List[QuestionOutput]

class OutputData(BaseModel):
    topics: List[TopicOutput]

# Scoring and link generation prompts
SCORING_PROMPT = """
You are an expert evaluator AI scoring user answers for an educational interview system. Assess if the user understands the core concept, focusing on ideas, not grammar or style.

Question: {question}
Reference Answer: {reference_answer}
User Answer: {user_answer}

Evaluation Guide:
- Score 0.9–1.0: Excellent answer — captures core idea with clear details.
- Score 0.5–0.8: Partial understanding — correct but lacks clarity or details.
- Score below 0.5: Misunderstood, unrelated, or incorrect answer.

Instructions:
- Award a score between 0 and 1 (rounded to 4 decimals).
- Provide a concise explanation (1-2 sentences) in a conversational tone, addressing the user directly (e.g., 'Your answer is great because...').
- If the answer is empty or very short, note this and suggest adding details.
- End with a brief 'Strengths' and 'Weaknesses' section, each with 1 short point.
- Return as plain text in the format:
Score: <float>
Explanation: <string>
Strengths: <string>
Weaknesses: <string>
- Do not include code blocks or extra whitespace.
"""

LINK_GENERATION_PROMPT = """
You are an educational assistant tasked with finding relevant, high-quality educational resources for a given question and its correct answer. Based on the question and reference answer provided, return 1–3 URLs to authoritative, educational websites or documentation that directly address the topic of the question. Ensure the links are specific to the concepts discussed (e.g., strings in programming for questions about strings) and avoid unrelated or generic sources. Prioritize official documentation, educational platforms, or reputable tutorials.

Question: {question}
Reference Answer: {reference_answer}

Provide the response as plain text with one URL per line, up to 3 URLs. Do not include code blocks, extra whitespace, or any other formatting.
Example:
https://example.com/resource1
https://example.com/resource2
https://example.com/resource3
"""

class Evaluator:
    """A system to evaluate answers and generate educational links using Google Gemini API."""
    
    def __init__(self):
        """Initialize the evaluator."""
        self.google_api_key = None
        self.model = None
        self.rate_limit_delay = 2

    def setup_environment(self) -> None:
        """Load environment variables and configure Google API."""
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            logger.error("GOOGLE_API_KEY environment variable is not set")
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        genai.configure(api_key=self.google_api_key)
        logger.info("Environment setup completed.")

    def initialize_models(self) -> None:
        """Initialize the Gemini model."""
        try:
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            logger.info("Model initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def generate_content(self, prompt: str, response_type: str = "text/plain") -> str:
        """Generate content using the Gemini API with retry on rate limits."""
        try:
            async with asyncio.timeout(10):
                response = self.model.generate_content(
                    prompt,
                    generation_config={
                        "response_mime_type": response_type,
                        "temperature": 0.8
                    }
                )
                logger.info(f"Raw Gemini API response: {response.text}")
                return response.text
        except asyncio.TimeoutError:
            logger.error("Gemini API call timed out")
            raise HTTPException(status_code=504, detail="Gemini API request timed out")
        except Exception as e:
            logger.error(f"Content generation failed: {e}")
            raise HTTPException(status_code=500, detail=f"Content generation failed: {str(e)}")

    async def get_links(self, question: str, reference_answer: str) -> List[str]:
        """Generate educational links using Gemini API."""
        try:
            prompt = LINK_GENERATION_PROMPT.format(question=question, reference_answer=reference_answer)
            response_text = await self.generate_content(prompt, response_type="text/plain")
            # Clean response
            response_text = response_text.strip()
            logger.info(f"Cleaned links response: {response_text}")
            # Split into lines and filter valid URLs
            links = [line.strip() for line in response_text.split("\n") if line.strip().startswith("http")]
            return links[:3]  # Limit to 3 links
        except Exception as e:
            logger.error(f"Failed to generate links: {str(e)}, response: {response_text}")
            return []

    async def evaluate_answer(self, question: str, reference_answer: str, user_answer: str) -> Dict:
        """Evaluate user answer and generate score, explanation, and links."""
        try:
            prompt = SCORING_PROMPT.format(
                question=question,
                reference_answer=reference_answer,
                user_answer=user_answer
            )
            response_text = await self.generate_content(prompt, response_type="text/plain")
            # Clean response
            response_text = response_text.strip()
            logger.info(f"Raw scoring response: {response_text}")

            # Parse text response with flexible regex
            try:
                score_match = re.search(r"Score:\s*(\d*\.?\d+)", response_text, re.IGNORECASE)
                explanation_match = re.search(r"Explanation:\s*(.*?)(?:\nStrengths:|\Z)", response_text, re.DOTALL | re.IGNORECASE)
                strengths_match = re.search(r"Strengths:\s*(.*?)(?:\nWeaknesses:|\Z)", response_text, re.DOTALL | re.IGNORECASE)
                weaknesses_match = re.search(r"Weaknesses:\s*(.*?)(?:\n|$|\Z)", response_text, re.DOTALL | re.IGNORECASE)

                score = float(score_match.group(1)) if score_match else 0.0
                explanation = explanation_match.group(1).strip() if explanation_match else "No explanation could be parsed."
                strengths = strengths_match.group(1).strip() if strengths_match else "None identified."
                weaknesses = weaknesses_match.group(1).strip() if weaknesses_match else "None identified."
                if not explanation:
                    explanation = "No explanation could be parsed."
            except Exception as e:
                logger.error(f"Failed to parse text response: {str(e)}, response: {response_text}")
                explanation = "We couldn’t generate an explanation due to an issue."
                strengths = "None identified."
                weaknesses = "None identified."
                score = 0.0

            # Handle empty or short user answers
            if not user_answer.strip():
                score = 0.0
                explanation = "Your answer is empty."
                strengths = "None."
                weaknesses = "No response provided."
            elif len(user_answer.split()) < 3:
                score = min(score, 0.3)
                explanation = f"Your answer is too brief for '{question}'."
                strengths = "Attempted to answer."
                weaknesses = f"Needs more details like: '{reference_answer}'."

            # Fallback to keyword-based scoring if parsing failed
            if explanation in ["No explanation could be parsed.", "We couldn’t generate an explanation due to an issue."]:
                core_keywords = reference_answer.lower().split()
                user_words = user_answer.lower().split()
                matching_keywords = len(set(core_keywords) & set(user_words))
                keyword_ratio = matching_keywords / len(core_keywords) if core_keywords else 0

                if keyword_ratio >= 0.8:
                    score = round(0.9 + (keyword_ratio * 0.1), 4)
                    explanation = f"Your answer captures the main idea of '{question}'."
                    strengths = "Good grasp of core concept."
                    weaknesses = f"Could add details like: '{reference_answer}'."
                elif keyword_ratio >= 0.4:
                    score = round(0.5 + (keyword_ratio * 0.3), 4)
                    explanation = f"Your answer partly addresses '{question}'."
                    strengths = "Some relevant points included."
                    weaknesses = f"Misses key details in: '{reference_answer}'."
                else:
                    score = round(keyword_ratio * 0.5, 4)
                    explanation = f"Your answer misses the core of '{question}'."
                    strengths = "Attempted to answer."
                    weaknesses = f"Needs focus on: '{reference_answer}'."

            # Combine explanation with strengths and weaknesses
            full_explanation = f"{explanation}\nStrengths: {strengths}\nWeaknesses: {weaknesses}"

            links = await self.get_links(question, reference_answer)
            return {
                "score": score,
                "score_explanation": full_explanation,
                "links": links
            }
        except Exception as e:
            logger.error(f"Error evaluating answer: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error evaluating answer: {str(e)}")

@app.post("/evaluate", response_model=OutputData)
async def evaluate_answers(data: InputData):
    """Evaluate user answers, assign scores, and provide educational links."""
    evaluator = Evaluator()
    try:
        evaluator.setup_environment()
        evaluator.initialize_models()
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize evaluator: {str(e)}")

    try:
        output_topics = []
        for topic in data.topics:
            output_questions = []
            for question in topic.questions:
                evaluation = await evaluator.evaluate_answer(
                    question.question,
                    question.gemini_answer,
                    question.user_answer
                )
                output_question = QuestionOutput(
                    question=question.question,
                    gemini_answer=question.gemini_answer,
                    user_answer=question.user_answer,
                    topic=question.topic,
                    classification=question.classification,
                    links=evaluation["links"],
                    score_explanation=evaluation["score_explanation"],
                    score=evaluation["score"]
                )
                output_questions.append(output_question)
            output_topic = TopicOutput(
                topic=topic.topic,
                questions=output_questions
            )
            output_topics.append(output_topic)
        logger.info("Successfully evaluated answers")
        return OutputData(topics=output_topics)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    """Return a welcome message for the root path."""
    return {
        "message": "FastAPI evaluation API with Gemini AI link generation is running",
        "documentation": "/docs",
        "endpoints": {
            "POST /evaluate": "Evaluate user answers with scoring and educational links"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
