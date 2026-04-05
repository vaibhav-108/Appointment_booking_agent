import streamlit as st
import sys
import os
import time

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessageChunk,AIMessage
from dental_agent.agent import dental_graph
from dental_agent.config import settings
from dental_agent.utils import sanitize_messages


#Page Configuration
st.set_page_config(
    page_title="Dental appointment agent",
    layout="centered",
    page_icon="🤖"
)

#simple css to center content
st.markdown(
    """
    <style>
    st.Button > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight:bold;
    }
    <style>
    """,unsafe_allow_html=True
)

def init_session_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "history" not in st.session_state:
        st.session_state.history = []          # list of LangChain BaseMessages
    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []     # list of (role, content) for UI


@st.cache_resource
def initialize_rag():
    return dental_graph                        # already compiled, don't call it


def stream_response(user_input: str) -> str:
    """Stream dental_graph response, return full response string."""
    st.session_state.history.append(HumanMessage(content=user_input))

    full_response = ""
    placeholder = st.empty()

    try:
        for event_type, data in dental_graph.stream(
            {"messages": sanitize_messages(st.session_state.history)},
            stream_mode=["messages", "values"],
            config={"recursion_limit": 20},
        ):
            if event_type == "messages":
                chunk, meta = data
                if (
                    isinstance(chunk, AIMessageChunk)
                    and chunk.content
                    and not getattr(chunk, "tool_calls", None)
                ):
                    full_response += chunk.content
                    placeholder.markdown(full_response + "▌")   # streaming cursor

            elif event_type == "values":
                final_messages = data.get("messages", [])
                # Pick up final_response if agent set it explicitly
                last = final_messages[-1] if final_messages else None
                if isinstance(last, AIMessage) and last.content and not full_response:
                    full_response = last.content

        placeholder.markdown(full_response)    # remove cursor on finish

    except Exception as exc:
        placeholder.error(f"Error: {exc}")
        st.session_state.history.pop()         # remove HumanMessage on failure
        return ""

    # Append assistant reply to history for next turn
    st.session_state.history.append(AIMessage(content=full_response))
    return full_response


def main():
    init_session_state()

    st.title("Dental Appointment Booking Agent 🦷")
    st.markdown("Book your dental appointment with ease! Powered by LangGraph + ChatOpenAI")

    # Initialize once
    if not st.session_state.initialized:
        with st.spinner("Loading system..."):
            book_system = initialize_rag()
            if book_system:
                st.session_state.initialized = True
                st.success("✅ System ready! You can now book your appointment.")
    st.markdown("---")

    # Render chat history
    for role, content in st.session_state.chat_display:
        with st.chat_message(role):
            st.markdown(content)

    # Chat input (replaces st.form — better UX for chat)
    question = st.chat_input("Which appointment would you like to book?")

    if question and st.session_state.initialized:
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(question)

        # Stream assistant response
        with st.chat_message("assistant"):
            response = stream_response(question)

        # Save to display history
        if response:
            st.session_state.chat_display.append(("user", question))
            st.session_state.chat_display.append(("assistant", response))


if __name__ == "__main__":
    main()
