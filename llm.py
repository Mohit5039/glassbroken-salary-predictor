from huggingface_hub import InferenceClient
import os

# Initialize Hugging Face Inference Client
client = InferenceClient(os.environ["HF_TOKEN"])

# ---------------- System prompt, salary message, keywords ----------------
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

SALARY_MESSAGE = "Hey! For accurate salary predictions, please use our **PayCheck 💰 feature from the sidebar**, which provides city-adjusted, up-to-date estimates based on your role and experience."

SALARY_KEYWORDS = [
    "salary", "ctc", "pay", "income", "compensation", "package", "earn", "wage",
    "remuneration", "stipend", "take-home", "payment", "financial", "paycheck",
    "pay check", "earning", "money", "job pay", "salary range", "expected salary"
]

# ---------------- Conversation history ----------------
conversation_history = []

# ---------------- CareerBuddy function ----------------
def ask_career_bot(user_input: str) -> str:
    """
    Takes user input, checks for salary queries, and returns CareerBuddy response.
    If not salary-related, calls LLaMA API and preserves multi-turn context.
    """
    # Check for salary-related queries first
    if any(keyword.lower() in user_input.lower() for keyword in SALARY_KEYWORDS):
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": SALARY_MESSAGE})
        return SALARY_MESSAGE

    # Append user input
    conversation_history.append({"role": "user", "content": user_input})

    # Prepare messages with system prompt + conversation history
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(conversation_history)

    # Call the Hugging Face LLaMA model
    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:fireworks-ai",
            messages=messages,
        )
        bot_response = completion.choices[0].message.content
    except Exception as e:
        bot_response = f"⚠️ Sorry, something went wrong with CareerBuddy: {e}"

    # Append bot response to history
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
