# Stable Diffusion 3 Medium Demo

This repository contains a demo application for generating images using the Stable Diffusion 3 Medium model from Stability AI. The demo is built using Gradio, making it easy to interact with the model through a web interface.

## Overview

Stable Diffusion 3 Medium is a state-of-the-art text-to-image generation model that can create highly detailed and visually appealing images from textual prompts. This demo allows you to experiment with the model by entering your own prompts and adjusting various settings to customize the output.

![Stable Diffusion](https://stability.ai/assets/images/news/stable-diffusion-3.jpg)

## Features

- **Text-to-Image Generation:** Generate images based on custom text prompts.
- **Negative Prompts:** Specify negative prompts to guide the model away from certain concepts.
- **Customizable Settings:** Adjust seed, image dimensions, guidance scale, and number of inference steps.
- **Random Seed Option:** Randomize the seed for generating different variations of the same prompt.
- **Examples:** Pre-defined examples to quickly test the model's capabilities.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/TheCleverIdiott/stable-diffusion-3-medium-demo.git
    cd stable-diffusion-3-medium-demo
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

### Usage

1. Run the application:
    ```sh
    python app.py
    ```

2. Open your browser and go to `http://localhost:7860` to access the demo interface.

![Gradio Interface](https://user-images.githubusercontent.com/32692812/135695839-2b6ab9a4-5e5d-4b7f-961b-47e1e2a03aa6.png)

### Configuration

You can configure various settings in the demo interface:

- **Prompt:** Enter the text prompt to generate the image.
- **Negative Prompt:** Enter a negative prompt to avoid certain concepts.
- **Seed:** Set a specific seed or randomize it for different variations.
- **Width and Height:** Adjust the dimensions of the generated image.
- **Guidance Scale:** Control the adherence to the prompt.
- **Number of Inference Steps:** Set the number of steps for the diffusion process.

![Settings](https://user-images.githubusercontent.com/32692812/135695885-2cbbd2b4-c18f-42ae-8429-2ec63cd4341d.png)

## Examples

Here are some example prompts you can try:

1. Astronaut in a jungle, cold color palette, muted colors, detailed, 8k
2. An astronaut riding a green horse
3. A delicious ceviche cheesecake slice

## Code Explanation

The main components of the code are as follows:

1. **Imports and Initialization:**
    ```python
    import gradio as gr
    import numpy as np
    import random
    import torch
    from diffusers import StableDiffusion3Pipeline
    import spaces

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    repo = "stabilityai/stable-diffusion-3-medium-diffusers"
    pipe = StableDiffusion3Pipeline.from_pretrained(repo, torch_dtype=torch.float16).to(device)
    ```

2. **Inference Function:**
    ```python
    @spaces.GPU
    def infer(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps, progress=gr.Progress(track_tqdm=True)):
        if randomize_seed:
            seed = random.randint(0, np.iinfo(np.int32).max)
        generator = torch.Generator().manual_seed(seed)
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            generator=generator
        ).images[0]
        return image, seed
    ```

3. **Gradio Interface:**
    ```python
    with gr.Blocks(css=css) as demo:
        with gr.Column(elem_id="col-container"):
            gr.Markdown("...")
            prompt = gr.Text(...)
            run_button = gr.Button("Run", scale=0)
            result = gr.Image(label="Result", show_label=False)
            with gr.Accordion("Advanced Settings", open=False):
                negative_prompt = gr.Text(...)
                seed = gr.Slider(...)
                randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
                width = gr.Slider(...)
                height = gr.Slider(...)
                guidance_scale = gr.Slider(...)
                num_inference_steps = gr.Slider(...)
            gr.Examples(...)
        gr.on(...)
    demo.launch()
    ```

---
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
license: mit
---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3)
- [Gradio](https://www.gradio.app/)

## Contact

For any questions or suggestions, please open an issue or contact the repository maintainer.

---

Feel free to replace the placeholder image URLs with actual URLs for the images you want to include.


