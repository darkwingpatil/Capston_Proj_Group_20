import os
import shutil
UPLOAD_DIR = "uploaded/images"
GEN_IMG_DIR = "generated_images"

# Create the directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

def saveImage(image):
            # Build the file path to save the uploaded image.
        # Using the original filename from the uploaded file.
        file_location = os.path.join(UPLOAD_DIR, image.filename)
        # model = image_to_cap_custom_model()
        # model.load_state_dict(torch.load(os.path.join('Models', 'image_captioning_modelTmv2.pth')))  # Load the saved parameters
        # model.to(device) 

        # Open a file in write-binary mode and copy the contents of the uploaded file.
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        print ({"info": f"File '{image.filename}' saved at '{file_location}'"})

        return file_location

def saveGeneratedImage(image):
      # Build the file path to save the uploaded image.
        # Using the original filename from the uploaded file.
        file_location = os.path.join(GEN_IMG_DIR, image.filename)
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        
        return "http://localhost:8080/images/" + image.filename
        # model = image_to_cap_custom_model()
        # model.load_state_dict(torch.load(os.path.join('Models', 'image_captioning_modelTmv2.pth')))  # Load the saved parameters
        # model.to(device)