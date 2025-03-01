import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the pre-trained CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def retrieve_image_from_text(text_query, all_img_embd):
    # Encode the text query
    text_inputs = processor(text=[text_query], return_tensors="pt", padding=True).to(device)#, truncation=True)

    # Extract text embeddings
    with torch.no_grad():
      text_embeddings = model.get_text_features(**text_inputs)

    # Normalize text embeddings
    text_embeddings = text_embeddings / text_embeddings.norm(p=2, dim=-1, keepdim=True).to(device)

    print(f"Text Query: {text_query}")

    # Calculate cosine similarity between the text and image embeddings
    similarity_scores = torch.mm(text_embeddings, all_img_embd.T)  # Shape: [1, num_images]

    # Get the index of the most similar image
    most_similar_idx = torch.argmax(similarity_scores, dim=1).item()

    # Retrieve the corresponding image path
    retrieved_image_name = image_paths[most_similar_idx]
    retrieved_image_name = retrieved_image_name.replace(" ", "")
    #retrieved_score = similarity_scores[0][most_similar_idx].item()

    return retrieved_image_name()