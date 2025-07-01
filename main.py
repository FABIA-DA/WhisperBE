from fastapi import FastAPI, UploadFile, File, HTTPException
import whisper
import tempfile
import os

app = FastAPI()
model = whisper.load_model("base")

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename.endswith(('.mp3', '.wav', '.m4a', '.flac')):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path, language="de")
        return {"text": result["text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)