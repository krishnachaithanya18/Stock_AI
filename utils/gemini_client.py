"""
utils/gemini_client.py — Gemini API Client (Phase 7)
======================================================
WHAT THIS MODULE DOES:
  Provides a simple, reusable function to call the Google Gemini
  API using Python's 'requests' library — no extra SDK needed.

WHY DIRECT REST API INSTEAD OF google-generativeai SDK?
  The google-generativeai package is not installed in this environment.
  The Gemini REST API is straightforward and 100% equivalent.
  Using requests directly also makes the code more transparent —
  you can see exactly what is sent to and received from the API.

HOW GEMINI API WORKS:
  1. We send an HTTP POST to Google's endpoint
  2. The request body contains our prompt as JSON
  3. Google's servers process it and return a JSON response
  4. We extract the generated text from the response

API ENDPOINT:
  POST https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
  Header: x-goog-api-key: YOUR_KEY
  Body:   {"contents": [{"parts": [{"text": "your prompt"}]}]}
"""

import requests
import json
from config import GOOGLE_API_KEY, GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS


# Base URL for Gemini REST API (v1beta supports gemini-1.5-flash)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"


def call_gemini(prompt: str, temperature: float = None,
                max_tokens: int = None) -> dict:
    """
    Call the Gemini API with a text prompt.

    Parameters:
        prompt      : The full text prompt to send to Gemini
        temperature : Creativity level (0.0–1.0). Lower = more factual.
        max_tokens  : Maximum response length in tokens.

    Returns:
        dict with:
            "status"  : "success" or "error"
            "text"    : The generated text (empty string on error)
            "error"   : Error message (None on success)
            "model"   : Which model was used
            "tokens_used": Approximate token count (if available)
    """

    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_gemini_api_key_here":
        return {
            "status": "error",
            "text"  : "",
            "error" : "GOOGLE_API_KEY not set. Please add your key to the .env file.",
            "model" : GEMINI_MODEL,
        }

    temp       = temperature if temperature is not None else GEMINI_TEMPERATURE
    max_tok    = max_tokens  if max_tokens  is not None else GEMINI_MAX_TOKENS
    model_name = GEMINI_MODEL

    # --- Build the API endpoint URL ---
    url = f"{GEMINI_BASE_URL}/{model_name}:generateContent"

    # --- Build the request body ---
    # This is the standard Gemini API request format
    request_body = {
        "contents": [
            {
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature"    : temp,
            "maxOutputTokens": max_tok,
            "topP"           : 0.95,
        },
        "safetySettings": [
            # Relaxed safety for financial analysis (factual content)
            {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
        ]
    }

    headers = {
        "Content-Type"  : "application/json",
        "x-goog-api-key": GOOGLE_API_KEY,
    }

    try:
        # --- Make the HTTP POST request ---
        response = requests.post(
            url,
            headers=headers,
            json=request_body,
            timeout=60   # 60-second timeout for large responses
        )

        # --- Handle HTTP errors (4xx, 5xx) ---
        if response.status_code != 200:
            error_body = response.json() if response.content else {}
            error_msg  = (
                error_body.get("error", {}).get("message", "Unknown API error")
                or f"HTTP {response.status_code}"
            )
            return {
                "status": "error",
                "text"  : "",
                "error" : f"Gemini API error: {error_msg}",
                "model" : model_name,
            }

        # --- Parse successful response ---
        resp_json = response.json()

        # Navigate the response structure to get the text
        # Structure: candidates[0] → content → parts[0] → text
        candidates = resp_json.get("candidates", [])
        if not candidates:
            return {
                "status": "error",
                "text"  : "",
                "error" : "Gemini returned no candidates (content may have been blocked)",
                "model" : model_name,
            }

        generated_text = (
            candidates[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
        )

        if not generated_text:
            return {
                "status": "error",
                "text"  : "",
                "error" : "Gemini returned empty text in response",
                "model" : model_name,
            }

        # Token usage (if available in response)
        usage      = resp_json.get("usageMetadata", {})
        total_tok  = usage.get("totalTokenCount", "N/A")

        return {
            "status"      : "success",
            "text"        : generated_text,
            "error"       : None,
            "model"       : model_name,
            "tokens_used" : total_tok,
        }

    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "text"  : "",
            "error" : "Gemini API request timed out after 60 seconds.",
            "model" : model_name,
        }
    except requests.exceptions.ConnectionError:
        return {
            "status": "error",
            "text"  : "",
            "error" : "Could not connect to Gemini API. Check your internet connection.",
            "model" : model_name,
        }
    except Exception as e:
        return {
            "status": "error",
            "text"  : "",
            "error" : f"Unexpected error calling Gemini: {str(e)}",
            "model" : model_name,
        }


def test_gemini_connection() -> bool:
    """
    Quick sanity-check: send a minimal prompt to verify the API key works.
    Returns True if connection is successful, False otherwise.
    """
    result = call_gemini(
        prompt="Reply with exactly: 'Gemini API connection successful.'",
        temperature=0.0,
        max_tokens=20
    )
    if result["status"] == "success":
        print(f"[Gemini] ✓ Connected. Model: {result['model']}")
        print(f"[Gemini]   Response: {result['text'].strip()}")
        return True
    else:
        print(f"[Gemini] ✗ Connection failed: {result['error']}")
        return False
