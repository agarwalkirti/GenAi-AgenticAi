#### creating our custom gpt like application on codellama (Multilanguage code assistant-codeguru)
#codeguru and ollama are running in backend, Now we will craete our API
import requests
import json
import gradio as gr

#Ollama exposes a REST API locally on port 11434. we are POSTing prompts to it.
url="http://localhost:11434/api/generate"

headers={

    'Content-Type':'application/json'
}

history=[]

# back-end (Fast API simulation using requests)
#It sends the chat history + new question to Ollama’s API.
#Ollama (running codeguru) generates an answer.
#Answer is returned back.

def generate_response(prompt):
    history.append(prompt) # stores chat history
    final_prompt="\n".join(history) # concatenates all past messages

    data={
        "model":"codeguru", # <- our custom Ollama model
        "prompt":final_prompt, 
        "stream":False  # request full response at once
    }

    response=requests.post(url,headers=headers,data=json.dumps(data))

    if response.status_code==200:
        response=response.text
        data=json.loads(response)
        actual_response=data['response'] #extract reply
        return actual_response
    else:
        print("error:",response.text)

# front-end (Gradio app)
#Wraps your backend (generate_response) inside a web UI. We type a question in the text box → goes to backend → hits Ollama → returns answer. 
# We get a simple local chatbot app in our browser.
interface=gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=4,placeholder="Enter your Prompt"),
    outputs=gr.Textbox(lines=15, label="CodeGuru Response")
)
interface.launch()
