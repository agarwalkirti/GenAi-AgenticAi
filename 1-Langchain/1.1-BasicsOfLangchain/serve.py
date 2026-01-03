#langserve helps developers deploy langchain and chains as a REST API. Integrated with fastapi and used pydantic for data valiation.
from fastapi import FastAPI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

import os
from dotenv import load_dotenv
from pydantic import BaseModel
from langserve import validation

import pydantic
print(pydantic.VERSION)

# 1. Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# 2. Define model
model = ChatGroq(model="gemma:2b", groq_api_key=groq_api_key)

# 3. Prompt + chain
system_template = "Translate the following into {language}:"
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("user", "{text}")
    ]
)
parser = StrOutputParser()
chain = prompt_template | model | parser

# 4. FastAPI app
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="LangServe API with FastAPI + Groq",
    openapi_url=None,   # disables schema generation
    docs_url=None       # disables Swagger UI
)

# 5. Add routes
add_routes(app, chain, path="/chain")

# 6. Fix Pydantic incomplete models issue
for subclass in BaseModel.__subclasses__():
    try:
        subclass.model_rebuild(force=True)
    except Exception as e:
        print(f"Skipping rebuild for {subclass}: {e}")

# 7. Root route
@app.get("/")
def read_root():
    return {"message": "LangServe is running. See /docs for API routes."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)