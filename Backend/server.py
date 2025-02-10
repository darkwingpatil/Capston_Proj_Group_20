import os
import shutil
import uvicorn
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.customImageCaptionModel import generate_caption
from models.gpt2VitImageCaptioning import generate_caption_gpt2vit
from utils.saveFiles import saveImage
app = FastAPI()
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:5173"  # add any other allowed origins if needed
]


# Add CORS middleware to the app
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # Allows requests from these origins
    allow_credentials=True,         # Allows cookies and authentication headers
    allow_methods=["*"],            # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],            # Allows all headers
)
# Define the directory where images will be saved
UPLOAD_DIR = "uploaded/images"

# Create the directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload/i2c/custom")
async def upload_image(image: UploadFile = File(...)):
    """
    Endpoint to receive an image file and store it in the UPLOAD_DIR.
    The client should send the file with the form field name 'image'.
    """
    try:
        # Build the file path to save the uploaded image.
        # Using the original filename from the uploaded file.
        # file_location = os.path.join(UPLOAD_DIR, image.filename)
        # # model = image_to_cap_custom_model()
        # # model.load_state_dict(torch.load(os.path.join('Models', 'image_captioning_modelTmv2.pth')))  # Load the saved parameters
        # # model.to(device) 

        # # Open a file in write-binary mode and copy the contents of the uploaded file.
        # with open(file_location, "wb") as buffer:
        #     shutil.copyfileobj(image.file, buffer)
        file_location = saveImage(image)
        # print ({"info": f"File '{image.filename}' saved at '{file_location}'"})
        decoded_caption = generate_caption(file_location)
        generate_caption_gpt2vit(file_location)

        return {"caption": decoded_caption}
    except Exception as e:
        # Raise an HTTP exception if something goes wrong
        raise HTTPException(status_code=500, detail=f"An error occurred while saving the file: {str(e)}")
    
@app.post("/upload/i2c/gpt2vit")
async def upload_image(image: UploadFile = File(...)):
    """
    Endpoint to receive an image file and store it in the UPLOAD_DIR.
    The client should send the file with the form field name 'image'.
    """
    try:
        file_location = saveImage(image)
        decoded_caption = generate_caption_gpt2vit(file_location)

        return {"caption": decoded_caption[0]}
    except Exception as e:
        # Raise an HTTP exception if something goes wrong
        raise HTTPException(status_code=500, detail=f"An error occurred while saving the file: {str(e)}")
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)


# uvicorn server:app --reload
