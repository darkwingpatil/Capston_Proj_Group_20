# Encoder: CLIP
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class CLIPEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super(CLIPEncoder, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def forward(self, images):
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(images)
        return image_features
    

# Attention Mechanism for Encoder-Decoder
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]

        hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)  # [batch_size, seq_len, hidden_size]

        combined = torch.cat((hidden, encoder_outputs), dim=2)

        energy = torch.tanh(self.attn(combined))  # [batch_size, seq_len, hidden_size]
        attention = self.v(energy).squeeze(2)  # [batch_size, seq_len]

        return F.softmax(attention, dim=1)
    

# Decoder: LSTM with Encoder-Decoder Attention
class DecoderWithAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=1):
        super(DecoderWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)

        hidden_states = []
        output, (hidden, cell) = self.lstm(embeddings)

        context = []
        for t in range(output.size(1)):
            attn_weights = self.attention(hidden[-1], features.unsqueeze(1))  # features expanded for batch processing
            context_vector = torch.bmm(attn_weights.unsqueeze(1), features.unsqueeze(1)).squeeze(1)
            context.append(context_vector)

        context = torch.stack(context, dim=1)  # [batch_size, seq_len, hidden_size]

        combined = torch.cat((output, context), dim=2)  # [batch_size, seq_len, hidden_size * 2]

        outputs = self.fc(combined)
        return outputs


# Combined Model
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = CLIPEncoder()
        self.decoder = DecoderWithAttention(embed_dim, hidden_dim, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


model = ImageCaptioningModel(512 , 512, 49408)
model_path= os.path.join('Models', 'image_captioning_modelTmv2.pth')
print(model_path,"Path of pkl file")
model.load_state_dict(torch.load(model_path))  # Load the saved parameters
model.to(device) 
tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").tokenizer

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
def gen_image_tensors(img_path):
    images_tensor=[]
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    images_tensor.append(img)
    images_tensor = torch.stack(images_tensor).to(device)
    return images_tensor

def generate_caption(img_path):
    try:
        image_tensor = gen_image_tensors(img_path)
        print(image_tensor,"image_tensor")
        image_tensor = image_tensor.squeeze(0)
        model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(device)
            caption = [1]  # Assuming <SOS> token is 1
            for _ in range(20):  # Maximum caption length
                caption_tensor = torch.tensor(caption).unsqueeze(0).to(device)
                output = model(image_tensor, caption_tensor)
                next_word = output.argmax(2)[:, -1].item()
                if next_word == 2:  # Assuming <EOS> token is 2
                    break
                caption.append(next_word)
            decoded_caption = tokenizer.decode(caption, skip_special_tokens=True)
            print(decoded_caption,"Logging the decoded_caption")
            return decoded_caption
    except Exception as e:
            print(f"Error generating caption: {e}")
            return None
    
