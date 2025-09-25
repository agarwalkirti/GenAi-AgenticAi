import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain.schema import Document

# Optional: for Whisper fallback
from pytube import YouTube
import whisper
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

## streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YouTube or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YouTube or Website")
st.subheader('Summarize below provided URL')

## Get the Groq API Key and url(YouTube or website)to be summarized
with st.sidebar:
    groq_api_key=st.text_input("Groq API Key",value="",type="password")

generic_url=st.text_input("URL",label_visibility="collapsed")

## Llama Model Using Groq API
llm =ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt=PromptTemplate(input_variables = ["text"], template = prompt_template)

final_summary_prompt_template = """
Provide the final summary of the entire content with these important points:
Add a title. Start the precise summary with an introduction and provide the summary in number points for the content.
Content:{text}
"""
final_summary_prompt = PromptTemplate(
                        input_variables=['text'],
                        template=final_summary_prompt_template
                    )

if st.button("Summarize"): #  the Content from YouTube or Website
    ## Validate all the inputs, strip - removes empty characters
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
    
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YouTube video url or website url")

    else:
        try:
            with st.spinner("Waiting..."):
                docs = []
                ## loading the website or youtube video data
                if "youtube.com" in generic_url:
                    docs = []
                    try:
                        loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                        docs = loader.load()
                    except (TranscriptsDisabled, NoTranscriptFound, Exception) as yt_err:
                        st.warning(f"Transcript not available: {yt_err}")
                        st.info("Falling back to Whisper transcription...")

                        try:
                            yt = YouTube(generic_url)
                            audio_path = yt.streams.filter(only_audio=True).first().download(filename="audio.mp4")
                            model = whisper.load_model("base")
                            result = model.transcribe("audio.mp4")
                            docs = [Document(page_content=result["text"])]
                        except Exception as w_err:
                            st.error(f"Whisper transcription also failed: {w_err}")
                            docs = []

                else:
                    loader = UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                    docs=loader.load()

                ## Chain For Summarization
                # chain = load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                summary_chain = load_summarize_chain(
                                    llm = llm,
                                    chain_type='map_reduce', 
                                    map_prompt = prompt, 
                                    combine_prompt = final_summary_prompt,
                                    verbose=True
                                )
                output_summary = summary_chain.run(docs)

                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")
                    