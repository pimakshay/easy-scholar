import streamlit as st
import sys, os
import base64
from omegaconf import OmegaConf
from notebooks.pdf_query import PDFQuery

def display_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="800" height="800" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

# set system path
CURR_DIR = os.path.dirname('__file__')
ROOT_DIR=os.path.join(os.getcwd() ,'../..')
sys.path.append(ROOT_DIR)

# load configs
config = OmegaConf.load(os.path.join(ROOT_DIR,'src/configs/config_pdfquery.yaml'))

# setup streamlit app
st.set_page_config(layout="wide")

st.title("Easy-Scholar App")

# file upload section
uploaded_file = st.file_uploader("Upload a research paper", type="pdf")

if uploaded_file is not None:
    # save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    chatbot = PDFQuery(file_path="temp.pdf")
    chatbot.setup_embeddings(model_name=config.model.embedding_model)
    chatbot.load_and_process_pdf()

    inferred_model = chatbot.infer_model(model_name=config.model.llm_model)
    st.write(f"Inferred model: {inferred_model}")

    chatbot.setup_llm(model_name=inferred_model)
    chatbot.setup_qa()
    col1, col2 = st.columns((12,8))

    with col1:
        with st.expander("View Uploaded Paper"):
            # Read PDF content
            display_pdf("temp.pdf")

    with col2:
        # Chat section
        st.subheader("Chat with Paper")
        user_question = st.text_input("Enter your question.")

        if user_question:
            answer, source_docs = chatbot.chat_with_paper(user_question)
            st.write("Answer:", answer)
            st.write("Sources:")
            for i, doc in enumerate(source_docs):
                st.write(f"Source {i+1}:", doc.page_content[:200] + "...")
