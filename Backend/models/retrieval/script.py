import os
import zipfile
import requests
from tqdm import tqdm
import torch
from PIL import Image
from common.types import DATATSETTYPE
from transformers import CLIPProcessor, CLIPModel
from torch.nn import functional as F
from torch.nn.functional import cosine_similarity
from torch.optim import Adam
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# Load the pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)
# Dictionary to store image captions
# Note decide later on where to store this.

image_captions = {}
image_embedding_8k =[]
image_embedding_30k = []
image_path_8k = []
image_path_30k = []
caption_embedding_8k=[]
caption_embedding_30k=[]
caption_8k_list=[]
caption_30k_list=[]

def get_image_embedding(imageType: DATATSETTYPE):
    if imageType == DATATSETTYPE.FLIICKER_8K_DATASET and len(image_embedding_8k) > 0:
        return image_embedding_8k, image_path_8k
    elif imageType == DATATSETTYPE.FLIKER_30K_DATASET and len(image_embedding_8k) > 0:
        return image_embedding_30k, image_path_30k
    print("No image embedding found have been intialized")
    return []

def get_text_embedding(imageType: DATATSETTYPE):
    if imageType == DATATSETTYPE.FLIICKER_8K_DATASET and len(caption_embedding_8k) > 0:
        return caption_embedding_8k, caption_8k_list
    elif imageType == DATATSETTYPE.FLIKER_30K_DATASET and len(caption_embedding_30k) > 0:
        return caption_embedding_30k, caption_30k_list
    print("No text embedding found have been intialized")
    return []

def download_file(url, filename):
    # Skip download if file exists
    if os.path.exists(filename):
        print(f"File {filename} already exists. Skipping download.")
        return
        
    print(f"Downloading {filename} from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    # Get total file size from headers
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192  # 8KB chunks
    
    # Create progress bar
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                progress_bar.update(len(chunk))
                f.write(chunk)
    progress_bar.close()
    print(f"\nDownloaded {filename} successfully.")

def extract_with_progress(zip_path, extract_dir):
    # Create directory if needed
    os.makedirs(extract_dir, exist_ok=True)
    
    # Get list of files to extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        file_list = zip_ref.infolist()
        
        # Set up progress bar
        progress_bar = tqdm(file_list, desc=f"Extracting {os.path.basename(zip_path)}")
        
        for file in progress_bar:
            zip_ref.extract(file, extract_dir)
            progress_bar.set_postfix(file=file.filename)

def createEmb_dir(img_emb_name, txt_emb_name):
    # Create the directory if it doesn't exist
    img_embedding = img_emb_name
    txt_embedding = txt_emb_name
    os.makedirs(img_embedding, exist_ok=True)
    os.makedirs(txt_embedding, exist_ok=True)

def gen_image_embedding(image_folder, output_folder):
    for image_name in os.listdir(image_folder):
        # Check if the file is an image (add more extensions if needed)
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_folder, image_name)

            try:
                # Load and preprocess the image
                image = Image.open(image_path).convert("RGB")
                inputs = processor(images=image, return_tensors="pt", padding=True).to(device)

                # Extract image embeddings
                with torch.no_grad():
                    image_features = model.get_image_features(**inputs)

                # Normalize the embeddings
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True).to(device)

                # Save the embeddings
                embedding_path = os.path.join(output_folder, f"{os.path.splitext(image_name)[0]}.pt")
                torch.save(image_features, embedding_path)

                print(f"Processed and saved: {embedding_path}")

            except Exception as e:
                print(f"Failed to process {image_name}: {e}")

def gen_text_Embedding(captions_file_path, output_dir):
    # Read the captions file
    with open(captions_file_path, "r") as f:
        lines = f.readlines()
        num_lines = len(lines)
        print(f"Number of lines in {captions_file_path}: {num_lines}")
    
    for line in lines:
        image_name, caption = line.strip().split("\t")
        # Remove the #index at the end of the image name
        base_name = image_name.split(".")[0]
        if base_name == "2258277193_586949ec62":
            continue
        if base_name not in image_captions:
            image_captions[base_name] = []
        image_captions[base_name].append(caption)
    
    image_captions_iter = iter(image_captions.items())
    print(len(image_captions))
    print (next(image_captions_iter))

    # Generate and save text embeddings
    for image_name, captions in image_captions.items():
        for i, caption in enumerate(captions):
            suffix = i + 1
            #image_name = image_name.split(".")[0]
            embedding_name = f"{image_name}_{suffix}.pt"

            # Preprocess the caption
            inputs = processor(text=[caption], return_tensors="pt", padding=True).to(device)

            # Extract text embeddings
            with torch.no_grad():
                text_features = model.get_text_features(**inputs)

            # Normalize the embeddings
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True).to(device)

            # Save the embeddings to the specified folder
            torch.save(text_features, os.path.join(output_dir, embedding_name))

            print(f"Caption is: {caption} Saved embedding: {embedding_name}")

def store_embeddings_in_list(image_embeddings, image_paths,img_embedding_folder):    
    # Load all the embeddings from .pt files in the folder
    for filename in os.listdir(img_embedding_folder):
        if filename.endswith(".pt"):
            # Construct the full file path
            file_path = os.path.join(img_embedding_folder, filename)

            # Load the embedding from the .pt file
            embedding = torch.load(file_path, weights_only=True)  # Shape: [embedding_dim]

            # Assuming the filename (without extension) is the image name
            image_name = filename.split('.')[0]  # Remove .pt extension to get image name

            # Append the embedding and image path to the lists
            image_embeddings.append(embedding)
            image_paths.append(image_name)

    # Convert the list of embeddings to a tensor and move to the same device
    image_embedds = torch.stack(image_embeddings).to(device)  # Shape: [num_images, embedding_dim]
    image_embedds = image_embedds.squeeze(1)

    # Normalize image embeddings
    image_embedds = F.normalize(image_embedds, p=2, dim=-1)

    return image_embedds, image_paths

def store_caption_embeddings_in_list(caption_embedding_folder,caption_embeddings_lst,captions_lst):

    for filename in os.listdir(caption_embedding_folder):
        if filename.endswith(".pt"):
            file_path = os.path.join(caption_embedding_folder, filename)
            embedding = torch.load(file_path, weights_only=True).to(device)
            embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
            caption_embeddings_lst.append(embedding)
            captions_lst.append(filename)
    
    return caption_embeddings_lst, captions_lst

def process_flicker8k_dataset():
    # Path configuration
    print("Intializing 8k dataset started")
    images_zip = os.path.join("models/retrieval/Flickr8k_Dataset.zip")
    captions_zip = os.path.join("models/retrieval/Flickr8k_Captions.zip")
    images_dir = os.path.join("models/retrieval/Flickr8k_Images")
    captions_dir = os.path.join("models/retrieval/Flickr8k_Captions")
    images_embedd_dir_8k = os.path.join("models/retrieval/img_embedding_8k")
    txt_embedd_dir_8k = os.path.join("models/retrieval/txt_embedding_8k")
    
    # URLs
    dataset_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    captions_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"
    
    # Download files with progress
    download_file(dataset_url, images_zip)
    download_file(captions_url, captions_zip)
    
    # Extract files with progress
    if not os.path.exists(images_dir):
        extract_with_progress(images_zip, images_dir)
    else:
        print(f"{images_dir} already exists. Skipping extraction.")
    
    if not os.path.exists(captions_dir):
        extract_with_progress(captions_zip, captions_dir)
    else:
        print(f"{captions_dir} already exists. Skipping extraction.")


    # create Embedding directory
    # createEmb_dir(images_embedd_dir_8k,txt_embedd_dir_8k)

    # generate image embedding: 
    # if not os.path.exists(images_embedd_dir_8k):
    #     os.makedirs(images_embedd_dir_8k,exist_ok=True)
    #     image_lists = os.path.join("models/retrieval/Flickr8k_Images/Flicker8k_Dataset")
    #     gen_image_embedding(image_lists, images_embedd_dir_8k)
    
    # if not os.path.exists(txt_embedd_dir_8k):
    #     os.makedirs(txt_embedd_dir_8k,exist_ok=True)
    #     captions_file_path = os.path.join("models/retrieval/Flickr8k_Captions/Flickr8k.token.txt")  # Replace with the actual path
    #     gen_text_Embedding(captions_file_path, txt_embedd_dir_8k)
    global image_embedding_8k
    global image_path_8k
    global caption_embedding_8k
    global caption_8k_list
    if  len(image_embedding_8k) == 0:
        image_embedd, image_path= store_embeddings_in_list(image_embedding_8k, image_path_8k, images_embedd_dir_8k)
        image_embedding_8k = image_embedd
        image_path_8k = image_path
    else:
        print("Image embeddings list already exist.")
    
    if len(caption_embedding_8k) == 0:
        caption_embedd, caption_list = store_caption_embeddings_in_list(txt_embedd_dir_8k,caption_embedding_8k,caption_8k_list)
        caption_embedding_8k = caption_embedd
        caption_8k_list = caption_list
    else:
        print("Caption embeddings list already exist.")
    
    print("All operations completed.")


# Execute the dataset preparation
# flicker8k_dataset()
# createEmb_dir("img_embedding_8k","txt_embedding_8k")

# image_folder = "/models/retrieval/Flickr8k_Images/Flicker8k_Dataset"
# output_folder = "/models/retrieval/img_embedding_8k"
