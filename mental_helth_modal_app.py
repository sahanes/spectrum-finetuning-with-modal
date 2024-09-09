# #mental_health_app_modal.py

import modal
from fastapi import FastAPI, HTTPException
from starlette.responses import FileResponse 
from pathlib import Path
from dataclasses import dataclass
import gradio as gr
from gradio.routes import mount_gradio_app
import yaml
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = modal.App(name="spectrum-finetunning-deployment-inference-mental_health_support",
                secrets=[modal.Secret.from_name("my-huggingface-secret")])

default_system_prompt = """Instruction: Provide mental health consultancy support for the following mental health issue. 
"""

web_app = FastAPI()
assets_path = Path(__file__).parent / "assets"

@dataclass
class AppConfig:
    """Configuration information for inference."""
    num_inference_steps: int = 50
    guidance_scale: float = 6

volume = modal.Volume.from_name("example-runs-vol")
MODEL_DIR = "/mnt/data"

image = modal.Image.debian_slim().pip_install(["torch", "transformers", "fastapi", "gradio", "pydantic", "PyYAML"])

@app.cls(
    image=image,
    gpu="A100",
    volumes={MODEL_DIR: volume},
    concurrency_limit=1,
    allow_concurrent_inputs=1000,
)
class Model:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        config_path = f"{MODEL_DIR}/axo-2024-09-04-16-44-01-d1ae/config.yml"
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        self.tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
        self.model = AutoModelForCausalLM.from_pretrained(
            config['base_model'],
            torch_dtype=torch.bfloat16
        ).to("cuda")

    @modal.method()
    def inference(self, system_prompt: str, input_text: str, config: AppConfig) -> str:
        full_prompt = f"{system_prompt}\n\nInput: {input_text}\n\nOutput:"
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to("cuda")
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=config.num_inference_steps,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.6
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text.split("Output:")[-1].strip()
        return response

@app.function(
    image=image,
    concurrency_limit=1,
    allow_concurrent_inputs=1000,
    mounts=[modal.Mount.from_local_dir(assets_path, remote_path="/assets")],
)
@modal.asgi_app()
def fastapi_app():
    config = AppConfig()

    def go(input_text="", system_prompt=default_system_prompt):
        if not input_text:
            input_text = example_prompts[0]['input']
        return Model().inference.remote(system_prompt, input_text, config)

    example_prompts = [
        {
            "system_prompt": default_system_prompt,
            "input": "I don't know what to say. I have never really known who I am."
        },
        {
            "system_prompt": default_system_prompt,
            "input": "My partner is struggling with addiction and I don't know how to support them while also taking care of myself."
        },
        {
            "system_prompt": default_system_prompt,
            "input": "I'm a teenager and I get these really intense mood swings. My mood will be really high and I'll think of something that I want to do. When I start to make it happen I get irritated by other people if they intervene. Then if the thing I wanted to do doesn't work out, I have these tendencies to blame other people for it not working out. Can you explain what's going on?"
        },
    ]

    description = "Provide compassionate and practical mental health consultancy support for the following issues. Our responses aim to be empathetic, insightful, and offer constructive advice."

    @web_app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        return FileResponse("/assets/favicon.svg")

    @web_app.get("/assets/background.svg", include_in_schema=False)
    async def background():
        return FileResponse("/assets/background.svg")

    with open("/assets/index.css") as f:
        css = f.read()

    theme = gr.themes.Default(
        primary_hue="green", secondary_hue="emerald", neutral_hue="neutral"
    )

    with gr.Blocks(theme=theme, css=css, title="Mental Health Consultancy Support") as interface:
        gr.Markdown(f"# Mental Health Consultancy Support\n\n{description}")
        with gr.Row():
            system_prompt_input = gr.Textbox(
                label="System Prompt",
                value=default_system_prompt,
                lines=3
            )
        with gr.Row():
            inp = gr.Textbox(
                label="Describe your mental health concern",
                placeholder="Enter your concern here",
                lines=5
            )
        out = gr.Textbox(
            lines=10,
            label="Consultancy Response",
            elem_id="output"
        )
        with gr.Row():
            btn = gr.Button("Get Support", variant="primary")
            btn.click(fn=go, inputs=[inp, system_prompt_input], outputs=out)

            gr.Button(
                "⚡️ Powered by Modal",
                variant="secondary",
                link="https://modal.com",
            )

        with gr.Column(variant="compact"):
            for prompt in example_prompts:
                btn = gr.Button(prompt['input'][:50] + "...", variant="secondary")
                btn.click(
                    fn=lambda p=prompt: (p['input'], p['system_prompt']), 
                    outputs=[inp, system_prompt_input]
                )

    return mount_gradio_app(
        app=web_app,
        blocks=interface,
        path="/",
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(fastapi_app(), host="127.0.0.1", port=8000)

# ! modal deploy mental_health_modal_app.py
# ! modal serve mental_health_modal_app.py
