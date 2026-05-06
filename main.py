import os
import sys
import requests
from dotenv import load_dotenv
from datetime import datetime

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

HF_TOKEN = os.getenv("HF_API_KEY")

# Models available - SDXL is highest quality
MODELS = {
    "1": ("stabilityai/stable-diffusion-xl-base-1.0", "SDXL - Best quality"),
    "2": ("stabilityai/stable-diffusion-2-1", "SD 2.1 - Fast"),
    "3": ("runwayml/stable-diffusion-v1-5", "SD 1.5 - Fastest"),
}

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


def generate_image(prompt: str, model: str = DEFAULT_MODEL, negative_prompt: str = "") -> bytes:
    """Generate image via Hugging Face Inference API."""
    url = f"https://router.huggingface.co/hf-inference/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    payload = {
        "inputs": prompt,
        "parameters": {
            "negative_prompt": negative_prompt or "blurry, low quality, distorted, ugly, bad anatomy",
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
        }
    }

    print(f"\n🎨 Generating image...")
    print(f"   Model: {model}")
    print(f"   Prompt: {prompt}\n")

    response = requests.post(url, headers=headers, json=payload, timeout=120)

    if response.status_code == 503:
        print("⏳ Model is loading, retrying in 20 seconds...")
        import time
        time.sleep(20)
        response = requests.post(url, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        raise Exception(f"API error {response.status_code}: {response.text}")

    return response.content


def save_image(image_bytes: bytes, filename: str = None) -> str:
    """Save image bytes to file."""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output/image_{timestamp}.png"

    os.makedirs("output", exist_ok=True)

    with open(filename, "wb") as f:
        f.write(image_bytes)

    print(f"✅ Saved to: {filename}")
    return filename


def interactive_mode():
    """Interactive CLI loop."""
    print("=" * 50)
    print("  Hugging Face Image Generator (SDXL)")
    print("=" * 50)
    print("Commands: 'quit' to exit, 'model' to switch model\n")

    current_model = DEFAULT_MODEL

    while True:
        prompt = input("Enter prompt: ").strip()

        if not prompt:
            continue
        if prompt.lower() == "quit":
            print("Goodbye!")
            break
        if prompt.lower() == "model":
            print("\nAvailable models:")
            for k, (m, desc) in MODELS.items():
                print(f"  {k}. {desc}")
            choice = input("Choose (1-3): ").strip()
            if choice in MODELS:
                current_model = MODELS[choice][0]
                print(f"Switched to: {current_model}\n")
            continue

        negative = input("Negative prompt (optional, press Enter to skip): ").strip()

        try:
            image_bytes = generate_image(prompt, model=current_model, negative_prompt=negative)
            path = save_image(image_bytes)
            print(f"🖼️  Open: {os.path.abspath(path)}\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        image_bytes = generate_image(prompt)
        save_image(image_bytes)
    else:
        interactive_mode()
