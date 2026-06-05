import streamlit as st
import google.generativeai as genai

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv
from sentence_transformers import (
    SentenceTransformer,
    CrossEncoder
)


@st.cache_resource
def load_reranker():

    print("Loading reranker...")

    return CrossEncoder(
        "BAAI/bge-reranker-base"
    )

reranker = load_reranker()
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(
        "BAAI/bge-small-en-v1.5"
    )

embedding_model = load_embedding_model()



load_dotenv()


genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

gemini = genai.GenerativeModel(
    "gemini-2.5-flash"
)



client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)


st.set_page_config(
    page_title="Yann LeCun Digital Twin",
    layout="wide"
)

st.title("Yann LeCun Digital Twin")




if "messages" not in st.session_state:
    st.session_state.messages = []




with st.sidebar:

    st.header("Controls")

    if st.button("New Chat"):
        st.session_state.messages = []
        st.rerun()




def build_context(points):

    chunks = []

    for point in points:

        chunks.append(
            f"""
Title: {point.payload.get('title', 'Unknown')}
Type: {point.payload.get('source_type', 'Unknown')}
Year: {point.payload.get('year', 'Unknown')}

{point.payload['text']}
"""
        )

    return "\n\n---\n\n".join(chunks)


def build_history(messages, max_turns=6):

    history = []

    for msg in messages[-max_turns:]:

        history.append(
            f"{msg['role']}: {msg['content']}"
        )

    return "\n".join(history)


def retrieve(query):

    query_vector = embedding_model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = client.query_points(
        collection_name="Twin",
        query=query_vector,
        limit=40
    ).points

    pairs = [
        (
            query,
            f"Title: {point.payload.get('title', '')}\n\n"
            f"{point.payload['text']}"
        )
        for point in results
    ]

    rerank_scores = reranker.predict(
        pairs
    )

    reranked = sorted(
        zip(results, rerank_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return reranked[:10]


def answer_question(question, history):

    reranked = retrieve(question)

    points = [
        point
        for point, score in reranked
    ]

    context = build_context(points)

    prompt = f"""
You are a digital twin of Yann LeCun.

Your purpose is to answer questions as Yann LeCun would, using the retrieved papers, interviews, talks, and transcripts provided as context.

Core principles frequently reflected in Yann LeCun's work and public statements:

* Intelligence requires learning predictive models of the world.
* Self-supervised learning is a key path toward more general AI.
* World models, planning, and reasoning are essential components of intelligent systems.
* Purely autoregressive language modeling is not sufficient for human-level intelligence.
* Representations should capture abstract, predictive structure rather than merely memorizing observations.
* Scientific skepticism and evidence-based reasoning are preferred over hype and speculation.

Behavior guidelines:

1. Ground answers primarily in the retrieved context.
2. When the answer is directly supported by the context, answer confidently.
3. When the answer is not explicitly addressed but can be inferred from Yann LeCun's established views, state the inference clearly.
4. Never invent papers, experiments, quotes, or opinions.
5. If the retrieved context does not provide enough information, say so.
6. Prefer technical precision over marketing language.
7. Avoid exaggeration, hype, or sensational claims.
8. Explain ideas using Yann's recurring concepts such as:

   * self-supervised learning
   * predictive representations
   * world models
   * planning
   * reasoning
   * representation learning

Style guidelines:

* Write in a concise, technical, and analytical style.
* Be direct and informative.
* Avoid excessive verbosity.
* Avoid sounding like a generic chatbot.
* Do not mention that you are an AI assistant.
* Speak in first person when appropriate, as if answering as Yann LeCun.

Reasoning guidelines:

* First determine whether the answer is explicitly present in the retrieved context.
* If it is, answer using that evidence.
* If not, determine whether a reasonable inference can be made from Yann LeCun's documented views.
* Clearly distinguish between direct evidence and inference.
* If neither is possible, respond that you do not have enough information.
Evidence Policy:

* Treat the retrieved context as the primary source of truth.
* Do not contradict the retrieved context.
* When making an inference, explicitly indicate that it is an inference rather than a direct statement from the sources.
* Never present an inference as a confirmed opinion or quote.
* If the context is insufficient, say so instead of speculating.
* Distinguish between:

  * Directly supported by the retrieved sources.
  * Reasonable inference from Yann LeCun's documented views.
  * Unknown or unsupported.
Communication Style:

* Prefer scientific reasoning over authority.
* Be skeptical of hype and grand claims.
* Favor mechanistic explanations.
* Emphasize learning, prediction, and representation building.
* Avoid certainty when evidence is limited.
* Critique ideas based on technical limitations rather than dismissive language.
Source awareness:
*Pay attention to the source type and year of retrieved documents.
*Papers typically contain technical details, methods, and research results.
*Transcripts typically contain opinions, predictions, explanations, and informal discussions.
*When discussing how views evolved over time, use the years provided in the retrieved context.
Your goal is not merely to answer questions, but to faithfully represent Yann LeCun's technical knowledge, reasoning style, and perspective while remaining grounded in the provided sources.

Keep the answer under 200 words.

Conversation History:
{history}

Context:
{context}

Question:
{question}
You are Yann LeCun
"""

    response = gemini.generate_content(
        prompt
    )

    initial_answer = response.text

    verifier_prompt = f"""
You are a Hallucination Verifier for a Retrieval-Augmented Generation (RAG) system simulating Yann LeCun.
Your task is to analyze the generated answer against the retrieved context to verify if the answer contains any hallucinations, factual errors, or statements unsupported by the context.

Retrieved Context:
{context}

User Question:
{question}

Generated Answer:
{initial_answer}

Analyze the Generated Answer. Follow these steps:
1. Identify any claims, facts, papers, quotes, or opinions in the Generated Answer that are NOT supported by the Retrieved Context or contradict it. Note that the Generated Answer should not invent papers, experiments, quotes, or opinions.
2. Determine if the Generated Answer contains any hallucinations (i.e. information not grounded in the Retrieved Context).
3. If there are hallucinations or unsupported statements, rewrite the Generated Answer to correct or remove them. The rewritten answer must strictly follow the original guidelines:
   - Maintain the persona of Yann LeCun (concise, technical, direct, first-person when appropriate).
   - Do NOT invent papers, experiments, quotes, or opinions.
   - Keep the answer under 200 words.
4. If the Generated Answer is fully grounded in the retrieved context and has no hallucinations, output the exact original Generated Answer.

Respond with a JSON object containing the following keys:
- "is_hallucination": boolean (true if any hallucination or unsupported claim was found, false otherwise)
- "explanation": string (brief explanation of what was hallucinated, or why it is clean)
- "final_answer": string (the corrected/rewritten answer, or the original answer if no hallucinations were found)
"""

    try:
        verification_response = gemini.generate_content(
            verifier_prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        import json
        result = json.loads(verification_response.text)
        is_hallucination = result.get("is_hallucination", False)
        explanation = result.get("explanation", "")
        final_answer = result.get("final_answer", initial_answer)
    except Exception as e:
        is_hallucination = False
        explanation = f"Error during verification: {str(e)}"
        final_answer = initial_answer

    hallucination_info = {
        "is_hallucination": is_hallucination,
        "explanation": explanation,
        "original_answer": initial_answer
    }

    return (
        final_answer,
        points,
        reranked,
        hallucination_info
    )


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])

        if "sources" in message:

            with st.expander("Sources"):

                for source in message["sources"]:
                    st.write(source)

        if "hallucination_info" in message:
            info = message["hallucination_info"]
            if info.get("is_hallucination"):
                with st.expander("⚠️ Hallucination Corrected"):
                    st.markdown(f"**Explanation:** {info.get('explanation')}")
                    st.markdown("**Original Response:**")
                    st.write(info.get("original_answer"))




question = st.chat_input(
    "Ask Yann something..."
)

if question:

    

    with st.chat_message("user"):
        st.write(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    

    history = build_history(
        st.session_state.messages
    )

    # Generate Answer

    with st.spinner("Thinking..."):

        answer, points, reranked, hallucination_info = answer_question(
                    question,
                    history
                )
    # -----------------------------
# RETRIEVAL DEBUG
# -----------------------------

    with st.sidebar:

        st.header("Hallucination Check")
        if hallucination_info["is_hallucination"]:
            st.warning("⚠️ Hallucination Detected & Corrected!")
            with st.expander("Verification Details"):
                st.markdown(f"**Explanation:** {hallucination_info['explanation']}")
                st.markdown("**Original Response:**")
                st.write(hallucination_info["original_answer"])
        else:
            st.success("✅ No Hallucination Detected")
            with st.expander("Verification Details"):
                st.markdown(f"**Explanation:** {hallucination_info['explanation']}")

        st.divider()

        st.header("Retrieval Debug")

        paper_count = sum(
            1
            for p in points
            if p.payload["source_type"] == "paper"
        )

        transcript_count = sum(
            1
            for p in points
            if p.payload["source_type"] == "transcript"
        )

        st.metric(
            "Papers Retrieved",
            paper_count
        )

        st.metric(
            "Transcripts Retrieved",
            transcript_count
        )

        st.divider()

        for i, (point, rerank_score) in enumerate(reranked):

            title = point.payload.get(
                "title",
                "Unknown"
            )

            with st.expander(
                f"{i+1}. {title}"
            ):

                st.write(
                    f"Rerank Score: {float(rerank_score):.4f}"
                )

                if hasattr(point, "score"):
                    st.write(
                        f"Vector Score: {point.score:.4f}"
                    )

                st.write(
                    f"Type: {point.payload.get('source_type', 'N/A')}"
                )

                st.write(
                    f"Year: {point.payload.get('year', 'N/A')}"
                )

                st.code(
                            point.payload["text"]
                        )

        st.divider()

        if st.checkbox("Show Full Context"):

            full_context = build_context(
                points
            )

            st.text_area(
                "Context Sent To Gemini",
                full_context,
                height=500
            )
            
    

   

    sources = list(
        dict.fromkeys(
            point.payload["title"]
            for point in points
        )
    )

   

    with st.chat_message("assistant"):

        st.write(answer)

        with st.expander("Sources"):

            for source in sources:
                st.write(source)

        if hallucination_info.get("is_hallucination"):
            with st.expander("⚠️ Hallucination Corrected"):
                st.markdown(f"**Explanation:** {hallucination_info.get('explanation')}")
                st.markdown("**Original Response:**")
                st.write(hallucination_info.get("original_answer"))

 

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "hallucination_info": hallucination_info
        }
    )