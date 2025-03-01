import os
import shutil
import uvicorn
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Body
import uuid
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from models.customImageCaptionModel import generate_caption
from models.gpt2VitImageCaptioning import generate_caption_gpt2vit
from models.retrieval.script import process_flicker8k_dataset
# from models.fluxDiffusion import retrive_image_From_caption_using_flux
from models.retrieval.retrieval import retrieve_image_from_text, generate_caption_from_image
from utils.saveFiles import saveImage, saveGeneratedImage
from common.types import DATATSETTYPE
app = FastAPI()
origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://localhost:5173"  # add any other allowed origins if needed
]

@app.on_event("startup")
async def startup_event():
    # This function is executed before any routes are handled
    process_flicker8k_dataset()

# Mount the static directory to serve images
app.mount("/images", StaticFiles(directory="generated_images"), name="images")

app.mount("/dImages", StaticFiles(directory="models/retrieval/Flickr8k_Images/Flicker8k_Dataset"), name="dImages")

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

@app.post("/upload/i2c/8k")
async def upload_image(image: UploadFile = File(...)):
    """
    Endpoint to receive an image file and store it in the UPLOAD_DIR.
    The client should send the file with the form field name 'image'.
    """
    try:
        file_location = saveImage(image)
        print("file saved successfully", file_location)
        decoded_caption = generate_caption_from_image(image.filename)
        print(decoded_caption)
        return {"caption": decoded_caption}
    except Exception as e:
        # Raise an HTTP exception if something goes wrong
        raise HTTPException(status_code=500, detail=f"An error occurred while saving the file: {str(e)}")    


@app.post("/upload/c2i/8k")
async def upload_caption_fluxDiff(caption: str = Body(..., embed=True)):
    """
    Endpoint to receive an text file
    The client should send the valid caption with proper details.
    """
    try:
        print(caption,"logg the caption")
        decoded_image = retrieve_image_from_text(caption, DATATSETTYPE.FLIICKER_8K_DATASET )
        print(decoded_image)
        return {"image_url": f"http://127.0.0.1:8000/dImages/{decoded_image}.jpg"}
    except Exception as e:
        # Raise an HTTP exception if something goes wrong
        raise HTTPException(status_code=500, detail=f"An error occurred while saving the file: {str(e)}")
    
       
# @app.post("/upload/c2i/fluxDiffuser")
# async def upload_caption_fluxDiff(caption: str = Body(..., embed=True)):
#     """
#     Endpoint to receive an text file
#     The client should send the valid caption with proper details.
#     """
#     try:
#         print(caption,"logg the caption")
#         decoded_image = retrive_image_From_caption_using_flux(caption, uuid.uuid4() )
#         static_img_url = saveGeneratedImage(decoded_image)
#         return {"image_url": static_img_url}
#     except Exception as e:
#         # Raise an HTTP exception if something goes wrong
#         raise HTTPException(status_code=500, detail=f"An error occurred while saving the file: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)


# uvicorn server:app --reload
