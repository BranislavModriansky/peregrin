"""
Lightweight chatbot for chatting with a Polars DataFrame using Groq's free API.

Setup:
    pip install polars groq
    Get a free API key at https://console.groq.com/keys
    Set it as an environment variable:
        Windows (PowerShell):  $env:GROQ_API_KEY="your_key_here"
"""

import os
import polars as pl
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])
for m in client.models.list().data:
    print(m.id)


def build_context(df: pl.DataFrame, max_rows: int = 20) -> str:
    """Create a compact text summary of the DataFrame for the model."""
    schema = "\n".join(f"  - {name}: {dtype}" for name, dtype in df.schema.items())
    preview = df.head(max_rows).to_pandas().to_csv(index=False)
    stats = ""
    try:
        stats = df.describe().to_pandas().to_string(index=False)
    except Exception:
        pass

    return (
        f"DataFrame shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
        f"Columns:\n{schema}\n\n"
        f"First {min(max_rows, df.shape[0])} rows (CSV):\n{preview}\n"
        f"Summary statistics:\n{stats}\n"
    )


def chat_with_dataframe(df: pl.DataFrame, model: str = "groq/compound"):
    """Start an interactive chat session about the given DataFrame."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Set the GROQ_API_KEY environment variable first.")

    client = Groq(api_key=api_key)
    context = build_context(df)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful data analyst. Answer questions about the "
                "user's Polars DataFrame using the provided context. Be concise. "
                "If asked for code, prefer Polars syntax.\n\n"
                f"DataFrame context:\n{context}"
            ),
        }
    ]

    print("Chat with your DataFrame! Type 'exit' or 'quit' to stop.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.3,
        )
        reply = response.choices[0].message.content
        messages.append({"role": "assistant", "content": reply})
        print(f"\nBot: {reply}\n")


if __name__ == "__main__":
    # Example DataFrame — replace with your own data
    df = pl.DataFrame(
        {
            "name": ["Alice", "Bob", "Carol", "Dan", "Eve"],
            "age": [29, 41, 35, 23, 50],
            "city": ["Paris", "London", "Paris", "Berlin", "London"],
            "salary": [55000, 72000, 68000, 40000, 90000],
        }
    )

    chat_with_dataframe(df)