from huggingface_hub import hf_hub_download

model_id = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
filename = "mistral-7b-instruct-v0.2.Q4_0.gguf"
# model_id = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
# filename = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

model_path = hf_hub_download(
    repo_id=model_id,
    filename=filename,
    cache_dir="models/"
)

print(f"Model downloaded at: {model_path}")