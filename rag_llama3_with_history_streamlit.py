import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag_llama3_with_history import RAG


if __name__ == '__main__':
    st.set_page_config(page_title="DermAssist", page_icon="🤖")
    st.title("DermAssist")

    rag = RAG()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.markdown(message.content)
        else:
            with st.chat_message("AI"):
                st.markdown(message.content)

    user_input = st.chat_input("Your message")
    if user_input is not None and user_input != "":
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with st.chat_message("Human"):
            st.markdown(user_input)

        with st.chat_message("AI"):
            rag_response = st.write_stream(rag.generate_response_streamlit(user_input, st.session_state.chat_history))

        st.session_state.chat_history.append(AIMessage(content=rag_response))
