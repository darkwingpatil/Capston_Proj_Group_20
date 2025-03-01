import os
import torch
from PIL import Image
import numpy 
from transformers import CLIPProcessor, CLIPModel, CLIPFeatureExtractor
from common.types import DATATSETTYPE
from models.retrieval.script import  get_image_embedding, get_text_embedding
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
feature_extractor = CLIPFeatureExtractor.from_pretrained(
    "openai/clip-vit-base-patch32",
    do_resize=True,        # Force resizing
    size=224,               # Model expects 224x224
    do_center_crop=True,    # Center crop to square
    do_normalize=True       # Apply CLIP-specific normalization
)
model.to(device)


def retrieve_image_from_text(text_query, dataSetType: DATATSETTYPE):
    # Encode the text query
    try:

        text_inputs = processor(text=[text_query], return_tensors="pt", padding=True).to(device)#, truncation=True)

        # Extract text embeddings
        with torch.no_grad():
            text_embeddings = model.get_text_features(**text_inputs)

        # Normalize text embeddings
        text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True).to(device)

        print(f"Text Query: {text_query}")


        all_img_embd,image_paths = get_image_embedding(dataSetType)

        # Calculate cosine similarity between the text and image embeddings
        similarity_scores = torch.mm(text_embeddings, all_img_embd.T)  # Shape: [1, num_images]

        # Get the index of the most similar image
        most_similar_idx = torch.argmax(similarity_scores, dim=1).item()

        # Retrieve the corresponding image path
        retrieved_image_name = image_paths[most_similar_idx]
        retrieved_image_name = retrieved_image_name.replace(" ", "")
        #retrieved_score = similarity_scores[0][most_similar_idx].item()

        return retrieved_image_name
    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_third_part(caption_heading):
    """
    Opens a file, finds a specific line, splits it, and returns the third part.


    Args:
        filename (str): The name of the file to open. Defaults to "captions.txt".

    Returns:
        str: The third part of the split line, or None if the line is not found.
    """
    caption_idx = int(caption_heading[-1])

    caption_n_idx = f".jpg#{caption_idx - 1}"

    capt_first = caption_heading.rsplit("_", 1)[0]

    cap_heading_final = capt_first + caption_n_idx
    #print(f"Caption heading is : {cap_heading_final}")
    captions_file = os.path.join("models/retrieval/Flickr8k_Captions/Flickr8k.token.txt")
    try:
        with open(captions_file, "r") as file:  # Open the file in read-only mode
            for line in file:  # Iterate through each line
              image_name, caption = line.strip().split("\t")
              if image_name == cap_heading_final:
                print(f"Retrived Caption is : {caption}")
                return caption

    except FileNotFoundError:
        print(f"Image file name {caption_heading} not found.")
        return None

def generate_caption_from_image(filename):
    try:
        # Load and preprocess the image
        file_location = os.path.join("uploaded/images", filename)
        print(file_location,"file_location in retrivier")
        image = Image.open(file_location).convert("RGB")
        preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            ),
        ])

        image = preprocess(image).unsqueeze(0).to(device)  # Add batch dimension
        inputs = {"pixel_values": image}
        print("inputs formed successfully")
        # Extract image features
        with torch.no_grad():
            image_embed = model.get_image_features(**inputs)
        # Normalize the embeddings
        image_embed = image_embed / image_embed.norm(p=2, dim=-1, keepdim=True).to(device)
        #print(f"Image Embeddings Shape: {image_embed.shape}")

        caption_embeddings_lst, captions_lst = get_text_embedding(DATATSETTYPE.FLIICKER_8K_DATASET)

        # Caption embeddings to tensors
        print(len(caption_embeddings_lst))
        print(caption_embeddings_lst[0])
        caption_embeddings_stk = torch.stack(caption_embeddings_lst).to(device)
        caption_embeddings_stk = caption_embeddings_stk.squeeze(1)
        #print(f"Caption Embeddings Shape: {caption_embeddings_stk.shape}")


        # Compute similarity scores (cosine similarity)
        similarity_scores = torch.mm(image_embed, caption_embeddings_stk.T)  # Shape: [1, num_templates]
        best_caption_idx = torch.argmax(similarity_scores, dim=1).item()
        #print(f"Best Similarity score is : ", torch.argmax(similarity_scores, dim=1))
        extracted_img_idx = captions_lst[best_caption_idx]
        extracted_img_idx = extracted_img_idx.replace(".pt","")
        #print(extracted_img_idx)
        extracted_caption = extract_third_part(extracted_img_idx)
        # Return the best caption for the image
        return extracted_caption
    except Exception as e:
        print(f"Error: {e}")
        return None



def generate_Image_from_Image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", padding=True).to(device)

    # Extract image features
    with torch.no_grad():
        image_embed = model.get_image_features(**inputs)
    # Normalize the embeddings
    image_embed = image_embed / image_embed.norm(p=2, dim=-1, keepdim=True).to(device)

    similarity_scores = torch.mm(image_embed, all_img_embd.T)  # Shape: [1, num_images]

    # Get the index of the most similar image
    most_similar_idx = torch.argmax(similarity_scores, dim=1).item()

    # Retrieve the corresponding image path
    retrieved_image_name = image_paths[most_similar_idx]
    retrieved_image_name = retrieved_image_name.replace(" ", "")
    #retrieved_score = similarity_scores[0][most_similar_idx].item()

    return retrieved_image_name