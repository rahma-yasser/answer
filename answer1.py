# import logging
# import json
# import re
# from typing import List, Optional
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel, Field
# import google.generativeai as genai
# import os
# import asyncio
# from tenacity import retry, stop_after_attempt, wait_exponential

# # Logging setup
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
# )
# logger = logging.getLogger(__name__)

# # FastAPI app setup
# app = FastAPI(
#     title="Question Scoring API",
#     description="API for scoring interview questions using Gemini models.",
#     version="1.0.3"
# )

# # Track definitions
# TRACKS = {
#     "1": {
#         "name": "flutter developer",
#         "default_topic": "flutter developer",
#         "tuned_model": "tunedModels/fluttermodel-2cx3qf2cm72f"
#     },
#     "2": {
#         "name": "machine learning",
#         "default_topic": "machine learning",
#         "tuned_model": "tunedModels/chk1-607sqy6pv5wt"
#     }
# }

# # Scoring prompt
# SCORING_PROMPT = """
# You are an expert evaluator AI helping score user answers in an educational interview system. Your goal is to assess whether the user demonstrates a basic understanding of the core concept, even if their phrasing, examples, or terminology differ from the ideal answer. Focus on the idea and understanding, not grammar, style, or advanced explanations.

# Question: {question}
# Reference Answer: {reference_answer}
# User Answer: {user_answer}

# Evaluation Guide:
# - Score 0.9–1.0: Excellent answer — captures the core idea well, with clear and accurate details.
# - Score 0.5–0.8: Partial understanding — correct direction but lacks clarity or key details.
# - Score below 0.5: Misunderstood, unrelated, or significantly incorrect answer.

# Instructions:
# - Award a score between 0 and 1 (rounded to 1 decimal) based on how well the user understood the core concept.
# - Provide a short, clear, and constructive explanation of the score, mentioning what was done well and what could be improved.
# - Include 1–3 relevant, high-quality educational URLs to help the user learn more about the topic. Return an empty list [] if no links are applicable.
# - Encourage the user if they show partial understanding, offering tips for improvement.

# Provide in JSON format:
# {
#   "score": [float between 0 and 1, rounded to 1 decimal],
#   "score_explanation": "Short and constructive explanation of the reasoning behind the score.",
#   "links": [
#     "https://example-link1.com",
#     "https://example-link2.com"
#   ]
# }
# """

# # Pydantic models for input
# class QuestionInput(BaseModel):
#     question: str
#     gemini_answer: str
#     user_answer: str
#     topic: str
#     classification: str

# class TopicQuestionsInput(BaseModel):
#     topic: str
#     questions: List[QuestionInput]

# class ScoreQuestionsRequest(BaseModel):
#     topics: List[TopicQuestionsInput]

# # Pydantic models for output
# class QuestionOutput(BaseModel):
#     question: str
#     gemini_answer: str
#     user_answer: str
#     topic: str
#     classification: str
#     score: Optional[float] = None
#     score_explanation: Optional[str] = None
#     links: Optional[List[str]] = None

# class TopicQuestionsOutput(BaseModel):
#     topic: str
#     questions: List[QuestionOutput]

# class ScoreQuestionsResponse(BaseModel):
#     topics: List[TopicQuestionsOutput]

# # Pydantic model for single answer scoring
# class ScoreResponse(BaseModel):
#     score: float = Field(..., ge=0.0, le=1.0, description="Score between 0 and 1")
#     score_explanation: str = Field(..., description="Explanation of the score")
#     links: List[str] = Field(default_factory=list, description="Educational links for improvement")

# # QuestionGenerator class
# class QuestionGenerator:
#     def __init__(self):
#         """Initialize the question generator."""
#         self.google_api_key = None
#         self.question_model = None
#         self.rate_limit_delay = 2

#     def setup_environment(self) -> None:
#         """Load environment variables and configure Google API."""
#         self.google_api_key = os.getenv("GOOGLE_API_KEY")
#         if not self.google_api_key:
#             logger.error("GOOGLE_API_KEY environment variable is not set")
#             raise ValueError("GOOGLE_API_KEY environment variable is not set")
#         genai.configure(api_key=self.google_api_key)
#         logger.info("Environment setup completed.")

#     def initialize_models(self) -> None:
#         """Initialize the question model."""
#         try:
#             self.question_model = genai.GenerativeModel("gemini-1.5-flash")
#             logger.info("Model initialized successfully: gemini-1.5-flash")
#         except Exception as e:
#             logger.error(f"Failed to initialize model: {e}")
#             raise

#     @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
#     async def generate_content(self, prompt: str, response_type: str = "text/plain") -> str:
#         """Generate content using the Gemini API with retry on rate limits."""
#         try:
#             async with asyncio.timeout(10):
#                 response = self.question_model.generate_content(
#                     prompt,
#                     generation_config={
#                         "response_mime_type": response_type,
#                         "temperature": 0.8
#                     }
#                 )
#                 return response.text
#         except asyncio.TimeoutError:
#             logger.error("Gemini API call timed out")
#             raise HTTPException(status_code=504, detail="Gemini API request timed out")
#         except Exception as e:
#             logger.error(f"Content generation failed: {e}")
#             raise HTTPException(status_code=500, detail=f"Content generation failed: {str(e)}")

#     async def score_answer(self, question: str, reference_answer: str, user_answer: str) -> ScoreResponse:
#         """Score a user's answer against the reference answer using Gemini API."""
#         for attempt in range(2):  # Try twice to handle parsing issues
#             try:
#                 prompt = SCORING_PROMPT.format(
#                     question=question,
#                     reference_answer=reference_answer,
#                     user_answer=user_answer
#                 )
#                 response_text = await self.generate_content(
#                     prompt,
#                     response_type="application/json"
#                 )

#                 # Log raw response for debugging
#                 logger.debug(f"Raw Gemini API response: {response_text}")

#                 try:
#                     # Sanitize response: remove whitespace, newlines, control characters, and JSON prefixes
#                     cleaned_response = re.sub(r'```json\n?', '', response_text)  # Remove ```json prefix
#                     cleaned_response = re.sub(r'```', '', cleaned_response)  # Remove closing ```
#                     cleaned_response = re.sub(r'[\n\r\t]+', ' ', cleaned_response.strip())  # Replace newlines/tabs
#                     cleaned_response = re.sub(r'\s+', ' ', cleaned_response)  # Collapse spaces
#                     cleaned_response = re.sub(r'[^\x20-\x7E]', '', cleaned_response)  # Remove non-printable characters

#                     # Log sanitized response
#                     logger.debug(f"Sanitized Gemini API response: {cleaned_response}")

#                     # Try to extract JSON if response is malformed
#                     json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
#                     if json_match:
#                         cleaned_response = json_match.group(0)
#                     else:
#                         raise ValueError("No valid JSON object found in response")

#                     score_data = json.loads(cleaned_response)
                    
#                     # Validate required fields
#                     if not all(key in score_data for key in ["score", "score_explanation", "links"]):
#                         logger.error(f"Invalid scoring response format: {score_data}")
#                         raise ValueError("Invalid scoring response format: missing required fields")
                    
#                     # Validate score
#                     score = float(score_data["score"])
#                     if not 0.0 <= score <= 1.0:
#                         logger.warning(f"Score {score} out of range, clamping to 0.0-1.0")
#                         score = max(0.0, min(1.0, score))
                    
#                     # Validate links
#                     if not isinstance(score_data["links"], list) or not all(isinstance(link, str) for link in score_data["links"]):
#                         logger.warning("Invalid links format, setting to empty list")
#                         score_data["links"] = []
                    
#                     return ScoreResponse(
#                         score=round(score, 1),
#                         score_explanation=score_data["score_explanation"],
#                         links=score_data["links"]
#                     )
#                 except json.JSONDecodeError as e:
#                     logger.error(f"Failed to parse scoring response as JSON: {str(e)}, response: {response_text}, sanitized: {cleaned_response}")
#                     if attempt == 1:  # Last attempt
#                         return ScoreResponse(
#                             score=0.0,
#                             score_explanation=f"Unable to score answer due to persistent JSON parsing error: {str(e)}",
#                             links=[]
#                         )
#                     continue  # Retry if not the last attempt
#                 except ValueError as e:
#                     logger.error(f"Validation error in scoring response: {str(e)}")
#                     if attempt == 1:
#                         return ScoreResponse(
#                             score=0.0,
#                             score_explanation=f"Unable to score answer due to response validation error: {str(e)}",
#                             links=[]
#                         )
#                     continue
#             except Exception as e:
#                 logger.error(f"Error scoring answer: {str(e)}")
#                 return ScoreResponse(
#                     score=0.0,
#                     score_explanation=f"Unable to score answer due to an error: {str(e)}",
#                     links=[]
#                 )

# # Scoring endpoint for multiple questions
# @app.post("/score-questions", response_model=ScoreQuestionsResponse)
# async def score_questions(request: ScoreQuestionsRequest):
#     """Score user answers for multiple questions across topics."""
#     generator = QuestionGenerator()
#     try:
#         generator.setup_environment()
#         generator.initialize_models()
#     except Exception as e:
#         logger.error(f"Failed to initialize generator: {e}")
#         raise HTTPException(status_code=500, detail=f"Failed to initialize generator: {str(e)}")

#     topics_output = []
#     for topic_input in request.topics:
#         questions_output = []
#         for question_input in topic_input.questions:
#             question_output = QuestionOutput(
#                 question=question_input.question,
#                 gemini_answer=question_input.gemini_answer,
#                 user_answer=question_input.user_answer,
#                 topic=question_input.topic,
#                 classification=question_input.classification
#             )

#             # Score only if user_answer is non-empty and gemini_answer is valid
#             if question_input.user_answer.strip() and question_input.gemini_answer.strip() and "Failed to generate" not in question_input.gemini_answer:
#                 try:
#                     score_result = await generator.score_answer(
#                         question=question_input.question,
#                         reference_answer=question_input.gemini_answer,
#                         user_answer=question_input.user_answer
#                     )
#                     question_output.score = score_result.score
#                     question_output.score_explanation = score_result.score_explanation
#                     question_output.links = score_result.links
#                 except Exception as e:
#                     logger.warning(f"Failed to score question '{question_input.question}': {str(e)}")
#                     question_output.score_explanation = f"Unable to score answer due to an error: {str(e)}"
            
#             questions_output.append(question_output)
        
#         topics_output.append(TopicQuestionsOutput(topic=topic_input.topic, questions=questions_output))
    
#     return ScoreQuestionsResponse(topics=topics_output)

# # Root endpoint
# @app.get("/")
# async def root():
#     """Return a welcome message for the root path."""
#     return {
#         "message": "Welcome to the Question Scoring API",
#         "documentation": "/docs",
#         "endpoints": {
#             "GET /tracks": "List available tracks",
#             "POST /score-questions": "Score multiple user answers across topics"
#         }
#     }

# # Tracks endpoint
# @app.get("/tracks")
# async def get_tracks():
#     """Return available tracks."""
#     return TRACKS

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uuid
import requests  # For potential HTTP-based Gemini API calls

app = FastAPI()

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

# Updated scoring prompt
SCORING_PROMPT = """
You are an expert evaluator AI helping score user answers in an educational interview system. Your goal is to assess whether the user demonstrates a basic understanding of the core concept, even if their phrasing, examples, or terminology differ from the ideal answer. Focus on the idea and understanding, not grammar, style, or advanced explanations.

Question: {question}
Reference Answer: {reference_answer}
User Answer: {user_answer}

Evaluation Guide:
- Score 0.9–1.0: Excellent answer — captures the core idea well, with clear and accurate details.
- Score 0.5–0.8: Partial understanding — correct direction but lacks clarity or key details.
- Score below 0.5: Misunderstood, unrelated, or significantly incorrect answer.

Instructions:
- Award a score between 0 and 1 (rounded to 4 decimals) based on how well the user understood the core concept.
- Provide a short, clear, and constructive explanation of the score, mentioning what was done well and what could be improved.
- Include 1–3 relevant, high-quality educational URLs to help the user learn more about the topic. Return an empty list [] if no links are applicable.
- Encourage the user if they show partial understanding, offering tips for improvement.
"""

# Refined prompt for Gemini AI to generate relevant links
LINK_GENERATION_PROMPT = """
You are an educational assistant tasked with finding relevant, high-quality educational resources for a given question and its correct answer. Based on the question and reference answer provided, return 1–3 URLs to authoritative, educational websites or documentation that directly address the topic of the question. Ensure the links are specific to the concepts discussed (e.g., strings in programming for questions about strings) and avoid unrelated or generic sources. Prioritize official documentation, educational platforms, or reputable tutorials.

Question: {question}
Reference Answer: {reference_answer}

Provide the response in JSON format:
{
  "links": [
    "https://example-link1.com",
    "https://example-link2.com",
    "https://example-link3.com"
  ]
}
"""

# Simulated Gemini AI function to generate links (replace with actual API call)
def get_links_from_gemini(question: str, reference_answer: str) -> List[str]:
    """
    Simulates calling Gemini AI to generate relevant educational links.
    Replace this with actual Gemini AI API integration.
    """
    # Mock response tailored to the question's topic
    mock_responses = {
        "What is a string in programming, and how is its length determined?": [
            "https://docs.python.org/3/tutorial/introduction.html#strings",
            "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String",
            "https://www.w3schools.com/java/java_strings.asp"
        ],
        "What is Dart?": [
            "https://dart.dev/",
            "https://flutter.dev/docs/get-started/install",
            "https://dartpad.dev/"
        ],
        "Why was Dart created?": [
            "https://dart.dev/guides",
            "https://flutter.dev/docs",
            "https://www.tutorialspoint.com/dart_programming/"
        ],
        "What is Flutter?": [
            "https://flutter.dev/docs",
            "https://api.flutter.dev/",
            "https://dart.dev/guides/libraries"
        ]
    }
    
    # Simulate Gemini AI response based on question
    for key in mock_responses:
        if key.lower() in question.lower():
            return mock_responses[key]
    
    # Default mock response for generic programming topics
    if "string" in question.lower() or "string" in reference_answer.lower():
        return [
            "https://docs.python.org/3/tutorial/introduction.html#strings",
            "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String",
            "https://www.w3schools.com/java/java_strings.asp"
        ]
    
    # Fallback for unrecognized questions
    return [
        "https://www.programiz.com/",
        "https://www.w3schools.com/"
    ]

    # Example of actual Gemini AI API call (uncomment and configure):
    """
    try:
        # Replace with actual Gemini API endpoint and authentication
        api_key = "YOUR_GEMINI_API_KEY"
        url = "https://api.gemini.google.com/v1/generate"  # Hypothetical endpoint
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        prompt = LINK_GENERATION_PROMPT.format(question=question, reference_answer=reference_answer)
        payload = {"prompt": prompt, "model": "gemini-2.5-pro"}  # Adjust based on API docs
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        return result.get("links", [])
    except Exception as e:
        print(f"Error calling Gemini AI: {str(e)}")
        return []
    """

# Simulated scoring logic
def evaluate_answer(question: str, reference_answer: str, user_answer: str) -> Dict:
    # Placeholder logic to simulate scoring based on the prompt
    core_keywords = reference_answer.lower().split()
    user_words = user_answer.lower().split()
    matching_keywords = len(set(core_keywords) & set(user_words))
    keyword_ratio = matching_keywords / len(core_keywords) if core_keywords else 0

    # Scoring logic based on keyword overlap and prompt guidelines
    if keyword_ratio >= 0.8:
        score = round(0.9 + (keyword_ratio * 0.1), 4)  # Excellent answer
        explanation = (
            f"The user answer captures the core idea of '{question}' well, aligning closely with the reference answer. "
            f"To improve, consider adding more specific details like those in the reference answer."
        )
    elif keyword_ratio >= 0.4:
        score = round(0.5 + (keyword_ratio * 0.3), 4)  # Partial understanding
        explanation = (
            f"The user answer shows partial understanding of '{question}' but misses some key details present in the reference answer. "
            f"Try incorporating more specific terms or examples to enhance clarity."
        )
    else:
        score = round(keyword_ratio * 0.5, 4)  # Misunderstood or incorrect
        explanation = (
            f"The user answer does not fully address the core concept of '{question}'. "
            f"Review the reference answer and focus on including key concepts for a stronger response."
        )

    # Fetch links from Gemini AI
    links = get_links_from_gemini(question, reference_answer)

    return {
        "score": score,
        "score_explanation": explanation,
        "links": links
    }

@app.post("/evaluate", response_model=OutputData)
async def evaluate_answers(data: InputData):
    try:
        output_topics = []
        
        for topic in data.topics:
            output_questions = []
            for question in topic.questions:
                # Evaluate the user answer and fetch links
                evaluation = evaluate_answer(
                    question.question,
                    question.gemini_answer,
                    question.user_answer
                )
                
                # Create output question
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
            
            # Create output topic
            output_topic = TopicOutput(
                topic=topic.topic,
                questions=output_questions
            )
            output_topics.append(output_topic)
        
        return OutputData(topics=output_topics)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.get("/")
async def root():
    return {"message": "FastAPI evaluation API with Gemini AI link generation is running"}