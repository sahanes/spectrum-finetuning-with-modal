# #mental_health_app_modal.py

# from fastapi import FastAPI, HTTPException
# from starlette.responses import FileResponse 
# from pathlib import Path
# from dataclasses import dataclass
# import modal
# import gradio as gr
# from gradio.routes import mount_gradio_app
# import yaml
# import os

# app = modal.App(name="spectrum-finetunning-deployment-inference-mental_health")
# web_app = FastAPI()
# assets_path = Path(__file__).parent / "assets"
# #----------------------------------------------------------
# #  Use a relative path or environment variable for assets
# #----------------------------------------------------------
# # ASSETS_DIR = os.environ.get('ASSETS_DIR', './assets')
# # assets_path = Path(ASSETS_DIR).resolve()

# #----------------------------------------------------------
# # Ensure the assets directory exists
# #----------------------------------------------------------

# os.makedirs(assets_path, exist_ok=True)


# @dataclass
# class AppConfig:
#     """Configuration information for inference."""
#     num_inference_steps: int = 50
#     guidance_scale: float = 6

# #----------------------------------------------------------
# # Create a Modal volume
# #----------------------------------------------------------

# volume = modal.Volume.from_name("example-runs-vol")
# MODEL_DIR = "/mnt/data"

# image = modal.Image.debian_slim().pip_install(["torch", "transformers", "fastapi", "gradio", "pydantic"])

# @app.function(
#     image=image,
#     gpu="A100",
#     volumes={MODEL_DIR: volume},
#     concurrency_limit=1,
#     allow_concurrent_inputs=1000,
#     # mounts=[modal.Mount.from_local_dir(assets_path, remote_path="/assets")],
#     mounts=[modal.Mount.from_local_dir(assets_path, remote_path="/assets")]
# )

# @app.cls()
# class Model:
#     def __init__(self):
#         self.model = None
#         self.tokenizer = None

#     @modal.enter()
#     def load_model(self):
#         import torch
#         from transformers import AutoModelForCausalLM, AutoTokenizer
        
#         config_path = f"{MODEL_DIR}/axo-2024-09-04-16-44-01-d1ae/config.yaml"
#         with open(config_path, 'r') as file:
#             config = yaml.safe_load(file)
        
#         self.tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
#         self.model = AutoModelForCausalLM.from_pretrained(
#             config['base_model'],
#             torch_dtype=torch.bfloat16
#         ).to("cuda")

#     @modal.method()
#     def inference(self, input_text: str, config: AppConfig) -> str:
#         inputs = self.tokenizer(input_text, return_tensors="pt").to("cuda")
        
#         outputs = self.model.generate(
#             **inputs,
#             max_length=config.num_inference_steps,
#             num_return_sequences=1,
#             do_sample=True,
#             temperature=config.guidance_scale
#         )
        
#         generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         return generated_text

# #----------------------------------------------------------
# # Modal stub for deployment
# #----------------------------------------------------------

# # @app.function(
# #     image=image,
# #     concurrency_limit=1,
# #     allow_concurrent_inputs=1000,
# #     mounts=[modal.Mount.from_local_dir(assets_path, remote_path="/assets")],
# # )
# # from fastapi.staticfiles import StaticFiles

# @modal.asgi_app()
# # @modal.web_endpoint()
# def fastapi_app():
#     # Call out to the inference in a separate Modal environment with a GPU
#     def go(text=""):
#         if not text:
#             text = example_prompts[0]
#         return Model().inference.remote(text, config)

#     # set up AppConfig
#     config = AppConfig()

#     system_prompt = "Instruction: Provide mental health consultancy support for the following mental health issue."

#     example_prompts = [
#         f"{system_prompt} I don't know what to say. I have never really known who I am.",
#         f"{system_prompt} Or how to send him somewhere that can help him, something like The Baker Act.",
#         f"{system_prompt} I'm a teenager and I get these really intense mood swings. My mood will be really high and I'll think of something that I want to do. When I start to make it happen I get irritated by other people if they intervene. Then if the thing I wanted to do doesn't work out, I have these tendencies to blame other people for it not working out. Can you explain what's going on?",
#     ]

#     description = "Instruction: Provide mental health consultancy support for the following mental health issue. Be Caring."

    
#     #----------------------------------------------------------
#     # Custom styles
#     #----------------------------------------------------------
#     @web_app.get("/favicon.ico", include_in_schema=False)
#     async def favicon():
#         favicon_path = Path("/assets/favicon.svg")
#         if favicon_path.exists():
#             return FileResponse(favicon_path)
#         raise HTTPException(status_code=404, detail="Favicon not found")
    
#     @web_app.get("/assets/background.svg", include_in_schema=False)
#     async def background():
#         bg_path = Path("/assets/background.svg")
#         if bg_path.exists():
#             return FileResponse(bg_path)
#         raise HTTPException(status_code=404, detail="Background image not found")

#     #----------------------------------------------------------
#     # Check if CSS file exists before trying to open it
#     #----------------------------------------------------------
#     css_path = Path("/assets/index.css")
#     if css_path.exists():
#         with open(css_path) as f:
#             css = f.read()
#     else:
#         css = ""  # Provide a default or empty CSS if file not found

#     theme = gr.themes.Default(
#         primary_hue="green", secondary_hue="emerald", neutral_hue="neutral"
#     )

#     # @staticmethod
#     # def lookup(
#     #     app_name: str,
#     #     tag: Optional[str] = None,
#     #     namespace=api_pb2.DEPLOYMENT_NAMESPACE_WORKSPACE,
#     #     client: Optional[_Client] = None,
#     #     environment_name: Optional[str] = None,
#     #     workspace: Optional[str] = None,
#     # ) -> "_Cls":
#           # will run with an A100 GPU


#     #---------------------------------------------------------------------------------------------
#     # Gradio UI
#     #---------------------------------------------------------------------------------------------
#     with gr.Blocks(theme=theme, css=css, title="Mental Health Consultancy Support") as interface:
#         gr.Markdown(f"# {system_prompt}\n\n{description}")
#         with gr.Row():
#             inp = gr.Textbox(
#                 label="",
#                 placeholder="Describe your mental health concern",
#                 lines=10,
#             )
#             out = gr.Textbox(
#                 lines=10,
#                 height=512, width=512, label="", min_width=512, elem_id="output"
#             )
#         with gr.Row():
#             btn = gr.Button("Get Support", variant="primary", scale=2)
#             btn.click(fn=go, inputs=inp, outputs=out)

#             gr.Button(
#                 "⚡️ Powered by Modal",
#                 variant="secondary",
#                 link="https://modal.com",
#             )

#         with gr.Column(variant="compact"):
#             for ii, prompt in enumerate(example_prompts):
#                 btn = gr.Button(prompt, variant="secondary")
#                 btn.click(fn=lambda idx=ii: example_prompts[idx], outputs=inp)

#         mount_gradio_app(
#         app=web_app,
#         blocks=interface,
#         path="/",    
#         )
#         return web_app
    
# #-------------------------------------------
# if __name__ == "__main__":
#     # import uvicorn
#     # uvicorn.run(fastapi_app(), host="127.0.0.1", port=8000)
#     Model = modal.Cls.lookup("spectrum-finetunning-deployment-inference-mental_health", "Model")
#     ModelUsingGPU = Model.with_options(gpu="A100")
#     ModelUsingGPU().generate.remote(42)
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
Be empathetic, understanding, and offer practical advice. Acknowledge the person's feelings, highlight positive aspects if any, and suggest constructive steps forward."""

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
