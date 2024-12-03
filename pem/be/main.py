from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
from functools import lru_cache
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        print("Starting file processing...")
        gemini_wrapper.upload_and_process_files()
        print("File processing completed successfully")
    except Exception as e:
        print(f"Error during startup: {str(e)}")
    yield
    # Shutdown (if needed)

app = FastAPI(lifespan=lifespan)
load_dotenv()

# Get port from environment variable for Render compatibility
PORT = int(os.getenv("PORT", 8000))

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
        api_key = os.getenv("GEMINI_API_KEY")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
            
        genai.configure(api_key=api_key)
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
        self.file_paths = [
            os.path.join(script_dir, "march24.pdf"),
            os.path.join(script_dir, "nov23.pdf")
        ]

    @lru_cache(maxsize=1)
    def upload_and_process_files(self):
        """Upload files once and cache the result"""
        try:
            if self.processed_files is None:
                # Verify files exist before uploading
                for path in self.file_paths:
                    if not os.path.exists(path):
                        raise FileNotFoundError(f"Required file not found: {path}")
                
                files = [
                    self._upload_to_gemini(path, "application/pdf")
                    for path in self.file_paths
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
        except Exception as e:
            print(f"Error in upload_and_process_files: {str(e)}")
            raise

    def _upload_to_gemini(self, path, mime_type=None):
        """Upload a single file to Gemini"""
        try:
            file = genai.upload_file(path, mime_type=mime_type)
            print(f"Uploaded file '{file.display_name}' as: {file.uri}")
            return file
        except Exception as e:
            print(f"Error uploading file {path}: {str(e)}")
            raise

    def _wait_for_files_active(self, files):
        """Wait for files to be processed"""
        print("Waiting for file processing...")
        max_retries = 30  # 5 minutes maximum wait
        retry_count = 0
        
        for name in (file.name for file in files):
            file = genai.get_file(name)
            while file.state.name == "PROCESSING" and retry_count < max_retries:
                print(".", end="", flush=True)
                time.sleep(10)
                file = genai.get_file(name)
                retry_count += 1
                
            if file.state.name != "ACTIVE":
                raise Exception(f"File {file.name} failed to process. Current state: {file.state.name}")
        print("...all files ready")
        print()

    def get_response(self, message: str):
        """Get response from the chat session"""
        try:
            if self.chat_session is None:
                self.upload_and_process_files()
            return self.chat_session.send_message(message)
        except Exception as e:
            print(f"Error getting response: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing message: {str(e)}"
            )

# Initialize wrapper
gemini_wrapper = GeminiWrapper()

@app.post("/chat")
async def chat(message: ChatMessage):
    try:
        response = gemini_wrapper.get_response(message.message)
        return {"response": response.text}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)