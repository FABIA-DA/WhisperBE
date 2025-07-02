from fastapi import FastAPI, UploadFile, HTTPException
import whisper
import tempfile
import os

app = FastAPI()
model = whisper.load_model("turbo")

@app.post("/transcribe")
async def transcribe(file: UploadFile | None = None):
    if file is None:
        raise HTTPException(status_code=400, detail="No file provided")

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = model.transcribe(audio=tmp_path, language="de")
        return {"text": result["text"]}
    except RuntimeError as e:
        raise HTTPException(status_code=415, detail="Unsupported audio format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.remove(tmp_path)

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok"}