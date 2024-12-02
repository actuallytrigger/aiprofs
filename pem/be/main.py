from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
from functools import lru_cache

app = FastAPI()
load_dotenv()

# Configure CORS for Vercel frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://aiprofs.vercel.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatMessage(BaseModel):
    message: str

class GeminiWrapper:
    def __init__(self):
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        
        self.model = genai.GenerativeModel(
            model_name="learnlm-1.5-pro-experimental",
            generation_config=self.generation_config,
            system_instruction="You are a helpful assistant designed to help college students study for their exams. They will ask various questions regarding principles of economics and management, and your job is to answer them. You are given context of 2 different past year question papers."
        )
        
        self.processed_files = None
        self.chat_session = None

    @lru_cache(maxsize=1)
    def upload_and_process_files(self):
        """Upload files once and cache the result"""
        if self.processed_files is None:
            files = [
                self._upload_to_gemini("march24.pdf", "application/pdf"),
                self._upload_to_gemini("nov23.pdf", "application/pdf")
            ]
            self._wait_for_files_active(files)
            self.processed_files = files
            # Initialize single chat session
            self.chat_session = self.model.start_chat(
                history=[{
                    "role": "user",
                    "parts": files,
                }]
            )
        return self.processed_files

    def _upload_to_gemini(self, path, mime_type=None):
        """Upload a single file to Gemini"""
        file = genai.upload_file(path, mime_type=mime_type)
        print(f"Uploaded file '{file.display_name}' as: {file.uri}")
        return file

    def _wait_for_files_active(self, files):
        """Wait for files to be processed"""
        print("Waiting for file processing...")
        for name in (file.name for file in files):
            file = genai.get_file(name)
            while file.state.name == "PROCESSING":
                print(".", end="", flush=True)
                time.sleep(10)
                file = genai.get_file(name)
            if file.state.name != "ACTIVE":
                raise Exception(f"File {file.name} failed to process")
        print("...all files ready")
        print()

    def get_response(self, message: str):
        """Get response from the chat session"""
        if self.chat_session is None:
            self.upload_and_process_files()
        return self.chat_session.send_message(message)

# Initialize wrapper
gemini_wrapper = GeminiWrapper()

@app.on_event("startup")
async def startup_event():
    """Process files when server starts"""
    gemini_wrapper.upload_and_process_files()

@app.post("/chat")
async def chat(message: ChatMessage):
    try:
        response = gemini_wrapper.get_response(message.message)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)