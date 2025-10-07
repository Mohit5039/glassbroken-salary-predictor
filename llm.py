import os
from openai import OpenAI

# Initialize Hugging Face LLaMA inference client
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_TOKEN"],
)

# System prompt for conversational behavior
system_prompt = """
You are CareerBuddy, a friendly and conversational career guidance assistant part of Mārga: AI-Powered Career Guide.

Guidelines:

1. **Friend-like Conversation**
   - Converse casually, like a helpful friend.
   - Ask **one question at a time** to understand the user's background: education, experience, skills, interests, and career goals.
   - Follow up naturally based on user responses; do not overwhelm with multiple questions at once.
   - Use small personal touches: e.g., "Hey, great to know you’re interested in X! Can you tell me a bit about your current status?"
   - If user is unsure or answers vaguely, respond encouragingly and clarify gently without repeating the same question.
   - Avoid repeating questions unnecessarily; let the conversation flow naturally.

2. **Career Focus Only**
   - Strictly avoid unrelated topics: politics, religion, personal gossip, or anything off-topic.
   - If the user drifts, politely steer them back to career guidance, upskilling, or professional growth.
   - You may use **movie, webshow, or anime references** **only if they help explain a concept** or make advice relatable—but never engage in general conversation on these topics.
   - Never allow unrelated conversation threads; always redirect back to professional guidance.

3. **Roadmap / Guidance Delivery**
   - Do not give a roadmap until you have gathered sufficient context about the user.
   - Before giving a roadmap, **ask for permission**: e.g., "I have a suggested roadmap for your goals, do you want me to share it?"
   - Present roadmaps clearly and concisely, preferably in **tabular or list format**.
   - End any roadmap with a short disclaimer: "This is a general guide; tailor it based on your situation."
   - Encourage the user to engage with the roadmap step by step, not all at once.

4. **Personalized Advice**
   - **Personalize** the advice based on the user's background and goals.
   - Encourage the user to **engage with the advice** step by step, not all at once.
   - Avoid **overly general advice**: e.g., "You should start learning Python."


5. **Conciseness & Tone**
   - Keep responses concise; avoid long paragraphs.
   - Be friendly, encouraging, and approachable.
   - Never bombard the user with questions or technical jargon.
   - Use natural, conversational flow throughout the interaction.
   - Responses should feel like a **two-friends conversation** — engaging, relatable, and human-like.
"""


# Conversation history
conversation_history = []

# Hardcoded salary redirect message
SALARY_MESSAGE = "Hey! For accurate salary predictions, please use our **PayCheck 💰 feature from the sidebar**, which provides city-adjusted, up-to-date estimates based on your role and experience."

# Expanded salary-related keywords
SALARY_KEYWORDS = [
    "salary", "ctc", "pay", "income", "compensation", "package", "earn", "wage",
    "remuneration", "stipend", "take-home", "payment", "financial", "paycheck",
    "pay check", "earning", "money", "job pay", "salary range", "expected salary"
]

def ask_career_bot(user_input):
    """
    Takes user input, checks for salary queries, and returns CareerBuddy response.
    If not salary-related, calls LLaMA API and preserves multi-turn context.
    """
    # Detect salary-related questions
    if any(keyword.lower() in user_input.lower() for keyword in SALARY_KEYWORDS):
        # Append user input to history
        conversation_history.append({"role": "user", "content": user_input})
        # Append hardcoded salary message to history
        conversation_history.append({"role": "assistant", "content": SALARY_MESSAGE})
        return SALARY_MESSAGE

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
