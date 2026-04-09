from datetime import datetime
import numpy as np
import random
import torch
import requests
from time import sleep
from openai import OpenAI
import httpx
import os

CLIENT = None

def set_random_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def init_openai_clients():
    """
    Initialize global OpenAI clients with aggressive keep-alive settings.
    Safe to call multiple times.
    """
    global CLIENT

    timeout = httpx.Timeout(
        connect=10.0,
        read=30.0,
        write=30.0,
        pool=10.0,
    )

    limits = httpx.Limits(
        max_connections=1,
        max_keepalive_connections=1,
        keepalive_expiry=86400,  # 24 hours
    )

    http_client = httpx.Client(
        timeout=timeout,
        limits=limits,
        http2=True,
    )

    if CLIENT is None:
        CLIENT = OpenAI(
            base_url=os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1"),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            http_client=http_client,
        )


def get_model_response(messages, temperature=0.0, max_new_tokens=150, model=None):
    """
    Generate a response using the loaded local model.
    Acts as the generation backend for Expert, User, and Critic.
    """
    
    global CLIENT
    init_openai_clients()
    for attempt in range(3):
        try:
            response = CLIENT.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_new_tokens,
                stream=False,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                reasoning_effort="minimal" if model.startswith("gemini") else None,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            CLIENT.close()
            CLIENT = None
            sleep(2)
            init_openai_clients()
            print(f"Request failed with error {e}. Retrying...")


def con_to_text(conversation, question=""):
    """Convert conversation list to text context."""
    if question != "":
        context = f"target info: {question}\nconversation:\n"
    else:
        context = ""
    for msg in conversation:
        role = msg["role"]
        content = msg["content"]
        if role == "user":
            context += f"user: {content}\n"
        elif role == "assistant":
            context += f"assistant: {content}\n"
    return context.strip()
