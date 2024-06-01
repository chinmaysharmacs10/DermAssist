import os
import streamlit as st
from PIL import Image
from langchain_core.messages import HumanMessage, AIMessage
from rag_system import RAG


class DermAssist:
    def __init__(self, image_save_dir, dermassist_logo):
        self.dermassist_logo = dermassist_logo
        self.uploaded_file = None
        self.image = None
        self.image_save_dir = image_save_dir
        self.image_save_path = None
        self.rag = RAG()

    def create_directory(self):
        if not os.path.exists(self.image_save_dir):
            os.makedirs(self.image_save_dir)

    def save_image(self):
        if self.image:
            self.create_directory()
            self.image_save_path = os.path.join(self.image_save_dir, self.uploaded_file.name)
            with open(self.image_save_path, "wb") as f:
                f.write(self.uploaded_file.getbuffer())

    def display_image(self):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(self.image, caption="Uploaded Image", use_column_width=True)

    def upload_image(self):
        self.uploaded_file = st.file_uploader("Upload an image of the affected area", type=["jpg", "jpeg", "png"])
        if self.uploaded_file is not None:
            self.image = Image.open(self.uploaded_file)
            self.save_image()
            return True
        return False

    @staticmethod
    def initialize_chat_history(skin_disease):
        if "chat_history" not in st.session_state:
            initial_context = f"You are suffering with {skin_disease}"
            st.session_state.chat_history = [AIMessage(content=initial_context)]

    @staticmethod
    def display_chat():
        for message in st.session_state.chat_history:
            if isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.markdown(message.content)
            else:
                with st.chat_message("AI"):
                    st.markdown(message.content)

    def handle_user_input(self):
        user_input = st.chat_input("Ask a question")
        if user_input is not None and user_input != "":
            st.session_state.chat_history.append(HumanMessage(content=user_input))
            with st.chat_message("Human"):
                st.markdown(user_input)

            with st.chat_message("AI"):
                rag_response = st.write_stream(
                    self.rag.generate_response_streamlit(user_input, st.session_state.chat_history))

            st.session_state.chat_history.append(AIMessage(content=rag_response))

    def setup_page(self):
        st.set_page_config(page_title="DermAssist", page_icon="⚕️")

        left, center, right = st.columns(3)
        logo = Image.open(self.dermassist_logo)
        with center:
            st.image(logo, width=200)

        st.markdown(
            """
            <h1 style="text-align: center;">
                <span style="color: white;">Derm</span>
                <span style="color: red;">Assist</span>
            </h1>
            <h3 style="text-align: center;">
                Your AI Assistant for Skin Problems
            </h3>
            <br>
            """,
            unsafe_allow_html=True
        )

    def run(self):
        self.setup_page()

        if self.upload_image():
            self.display_image()
        else:
            st.info("Please upload an image of the affected area to perform diagnosis and ask questions.")
            st.stop()

        # TODO: call vision model with self.image_save_path as input
        # vision_model_response = self.vision_model(self.image_save_path)
        # self.initialize_chat_history(skin_disease=vision_model_response)

        self.initialize_chat_history(skin_disease="acne")
        self.display_chat()
        self.handle_user_input()


if __name__ == '__main__':
    images_folder = "/Users/chinmaysharma/Documents/DermAssist/images"
    dermassist_logo = "/Users/chinmaysharma/Documents/DermAssist/derm_assist_logo.png"
    dermassist = DermAssist(image_save_dir=images_folder, dermassist_logo=dermassist_logo)
    dermassist.run()
