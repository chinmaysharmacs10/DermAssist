import os
import streamlit as st
from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage
from rag_llama3_with_history import RAG


if __name__ == '__main__':
    st.set_page_config(page_title="DermAssist", page_icon="⚕️")
    st.title("DermAssist")

    rag = RAG()

    uploaded_file = st.file_uploader("Upload an image of the affected area", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        image_save_dir = "/Users/chinmaysharma/Documents/DermAssist/images"
        if not os.path.exists(image_save_dir):
            os.makedirs(image_save_dir)

        image_save_path = os.path.join(image_save_dir, uploaded_file.name)
        with open(image_save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Uploaded Image", use_column_width=True)
    else:
        st.info("Please upload an image of the affected area to perform diagnosis and conversation.")
        st.stop()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [AIMessage(content="You are suffering with acne")]

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
