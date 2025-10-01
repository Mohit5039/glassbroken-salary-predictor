import os
from openai import OpenAI

# Initialize Hugging Face LLaMA inference client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

# System prompt for conversational behavior
system_prompt = """
You are CareerBuddy, a friendly and conversational career guidance assistant part of GlassBroken: AI Career Buddy.
- Ask questions first to understand the user's background: education, experience, interests, and career goals.
- Only provide advice or roadmap after gathering enough information.
- Keep responses concise, structured, and career-focused.
- Politely refuse unrelated questions.
- Once giving advice, you may present it in a short list style with a brief disclaimer at the end.
"""

# Conversation history
conversation_history = []

def ask_career_bot(user_input):
    """
    Takes user input, appends it to conversation history, calls LLaMA API,
    returns CareerBuddy response while preserving multi-turn context.
    """
    # Append user input to history
    conversation_history.append({"role": "user", "content": user_input})

    # Prepare full message sequence for multi-turn
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)

    # Call the LLaMA inference API
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct:fireworks-ai",
        messages=messages,
    )

    # Get bot response and append to history
    bot_response = completion.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": bot_response})

    return bot_response

# ---------------- Terminal test ----------------
if __name__ == "__main__":
    print("CareerBuddy Test (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = ask_career_bot(user_input)
        print("CareerBuddy:", response, "\n")
