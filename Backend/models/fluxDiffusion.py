# import torch
# import os
# from diffusers import FluxPipeline
# from huggingface_hub import login
# from dotenv import load_dotenv

# env_path = ".env"  # Update this path as per your setup
# load_dotenv(env_path)

# HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# if not HUGGINGFACE_TOKEN:
#     raise ValueError("HUGGINGFACE_TOKEN not found. Ensure it's set in the .env file and accessible.")
# print(HUGGINGFACE_TOKEN,"HUGGINGFACE_TOKEN")
# login(HUGGINGFACE_TOKEN)

# def retrive_image_From_caption_using_flux(caption, id):
#     pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
#     pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

#     prompt = caption
#     image = pipe(
#         prompt,
#         height=1024,
#         width=1024,
#         guidance_scale=3.5,
#         num_inference_steps=50,
#         max_sequence_length=512,
#         generator=torch.Generator("cpu").manual_seed(0)
#     ).images[0]
#     image_name = id.toString() + ".png"
#     image.save(image_name)
#     return image
