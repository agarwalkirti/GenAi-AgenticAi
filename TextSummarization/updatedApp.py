import streamlit as st
import validators
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
import whisper
from pytube import YouTube
from langchain.schema import Document
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from streamlit_mic_recorder import mic_recorder
import tempfile
import os

# Page config
st.set_page_config(page_title="LangChain: Summarize YT, Website or Audio", page_icon="🦜")
st.title("🦜 LangChain: Summarizer")
st.subheader("Summarize YouTube, websites, or audio")

# Sidebar for API key
with st.sidebar:
    groq_api_key = st.text_input("Groq API Key", value="", type="password")

# LLM
llm = ChatGroq(model="Gemma-7b-It", groq_api_key=groq_api_key)

# Prompt
prompt_template = """
Provide a concise summary of the following content in 300 words:

Content:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

# Whisper model (loads once & cached)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")   # options: tiny, base, small, medium, large

whisper_model = load_whisper_model()

# Tabs
tab1, tab2, tab3 = st.tabs(["🎥 YouTube", "🌐 Website", "🎤 Audio"])

# --- Tab 1: YouTube ---
with tab1:
    yt_url = st.text_input("Enter YouTube URL")
    if st.button("Summarize YouTube"):
        if not groq_api_key.strip() or not yt_url.strip():
            st.error("Please provide API key and YouTube URL")
        elif not validators.url(yt_url):
            st.error("Invalid YouTube URL")
        else:
            with st.spinner("Loading YouTube transcript..."):
                docs = []
                ## loading the website or youtube video data
                if "youtube.com" in yt_url:
                    docs = []
                    try:
                        loader = YoutubeLoader.from_youtube_url(yt_url, add_video_info=True)
                        docs = loader.load()
                    except (TranscriptsDisabled, NoTranscriptFound, Exception) as yt_err:
                        st.warning(f"Transcript not available: {yt_err}")
                        st.info("Falling back to Whisper transcription...")

                        try:
                            yt = YouTube(yt_url)
                            audio_path = yt.streams.filter(only_audio=True).first().download(filename="audio.mp4")
                            model = whisper.load_model("base")
                            result = model.transcribe("audio.mp4")
                            docs = [Document(page_content=result["text"])]
                        except Exception as w_err:
                            st.error(f"Whisper transcription also failed: {w_err}")
                            docs = []

                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output = chain.run(docs)
                st.success(output)

# --- Tab 2: Website ---
with tab2:
    web_url = st.text_input("Enter Website URL")
    if st.button("Summarize Website"):
        if not groq_api_key.strip() or not web_url.strip():
            st.error("Please provide API key and Website URL")
        elif not validators.url(web_url):
            st.error("Invalid Website URL")
        else:
            with st.spinner("Loading website content..."):
                loader = UnstructuredURLLoader(
                    urls=[web_url],
                    ssl_verify=False,
                    headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                )
                docs = loader.load()
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output = chain.run(docs)
                st.success(output)

# --- Tab 3: Audio ---
with tab3:
    st.write("Upload an audio file or record using microphone")

    audio_file = st.file_uploader("Upload Audio (mp3/wav/m4a)", type=["mp3", "wav", "m4a"])
    audio_recorded = mic_recorder(start_prompt="🎙️ Start recording", stop_prompt="⏹ Stop recording")

    if st.button("Summarize Audio"):
        if not groq_api_key.strip():
            st.error("Please provide API key")
        else:
            transcript_text = ""
            if audio_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                    tmp.write(audio_file.read())
                    tmp_path = tmp.name
                result = whisper_model.transcribe(tmp_path)
                transcript_text = result["text"]
                os.remove(tmp_path)

            elif audio_recorded:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(audio_recorded["bytes"])
                    tmp_path = tmp.name
                result = whisper_model.transcribe(tmp_path)
                transcript_text = result["text"]
                os.remove(tmp_path)

            else:
                st.error("Please upload or record an audio file")

            if transcript_text:
                st.info("Transcript: " + transcript_text[:200] + "...")
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                output = chain.run([transcript_text])
                st.success(output)