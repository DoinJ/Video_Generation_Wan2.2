import argparse
import gc
import logging
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import torch

from wan import WanTI2V
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, WAN_CONFIGS
from wan.utils.utils import save_video


APP_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = APP_ROOT / "outputs" / "ti2v_demo"
DEFAULT_MODEL_DIR = Path(
    os.getenv("WAN_TI2V_CKPT_DIR", "/mnt/nas10_shared/models/Wan2.2-TI2V-5B")
)
DEFAULT_PROMPT = (
    "A cinematic medium shot of a red panda barista working behind a brass espresso "
    "machine in a rain-soaked neon cafe, steam curling through warm backlight, "
    "camera drifting forward with rich reflections and shallow depth of field."
)
NONE_OPTION = "Auto / leave unspecified"
DEFAULT_BUILDER_SCENE = DEFAULT_PROMPT
PROMPT_TEMPLATES = {
        "Neon cafe push-in": DEFAULT_PROMPT,
        "Golden field portrait": (
                "A young girl sits in a field of tall grass beside two fluffy donkeys, "
                "turning toward the camera while the grass sways and the animals shift their weight."
        ),
        "Candlelit close-up": (
                "A woman seated at a dark wooden table speaks softly and gestures with one hand while candles flicker in the foreground."
        ),
        "Surf cat tracking shot": (
                "A white cat wearing sunglasses balances on a surfboard as water sprays around it and the camera tracks alongside the wave."
        ),
}
PROMPT_GUIDE = {
        "Time": ["Day time", "Night time", "Dawn time", "Sunrise time"],
        "Light source": [
                "Daylight",
                "Artificial lighting",
                "Moonlight",
                "Practical lighting",
                "Firelight",
                "Fluorescent lighting",
                "Overcast lighting",
                "Sunny lighting",
        ],
        "Light quality": ["Soft lighting", "Hard lighting"],
        "Light angle": ["Top lighting", "Side lighting", "Underlighting", "Edge lighting"],
        "Color tone": ["Warm colors", "Cool colors", "Mixed colors"],
        "Shot size": [
                "Medium shot",
                "Medium close-up shot",
                "Wide shot",
                "Medium wide shot",
                "Close-up shot",
                "Extreme close-up shot",
                "Extreme wide shot",
        ],
        "Camera angle": [
                "Over-the-shoulder shot",
                "Low angle shot",
                "High angle shot",
                "Dutch angle shot",
                "Aerial shot",
                "Overhead shot",
        ],
        "Composition": [
                "Center composition",
                "Balanced composition",
                "Right-heavy composition",
                "Left-heavy composition",
                "Symmetrical composition",
                "Short-side composition",
        ],
        "Camera movement": [
                "pushes forward",
                "pulls back",
                "moves from left to right",
                "moves from right to left",
                "tilts upward",
                "tilts downward",
                "drifts alongside the subject",
        ],
        "Motion cue": [
                "The subject turns toward the camera and shifts posture.",
                "Wind moves through the environment while background details stay active.",
                "Hands and facial expression change over time as the subject performs the action.",
                "Small environmental motion keeps the frame alive, such as drifting steam, moving leaves, or water ripples.",
        ],
}
CSS = """
:root {
        --page-glow: radial-gradient(circle at 10% 10%, rgba(58, 255, 146, 0.18), transparent 24%),
                                                         radial-gradient(circle at 88% 8%, rgba(13, 212, 120, 0.12), transparent 18%),
                                                         radial-gradient(circle at 50% 100%, rgba(28, 255, 163, 0.10), transparent 30%),
                                                         linear-gradient(180deg, #060908 0%, #08110d 45%, #030504 100%);
        --panel-bg: rgba(7, 15, 11, 0.84);
        --panel-border: rgba(69, 255, 155, 0.16);
        --text-main: #e9fff2;
        --text-muted: #9ccab0;
        --accent: #26ff91;
        --accent-deep: #0bc46e;
        --accent-soft: rgba(38, 255, 145, 0.12);
        --teal-soft: rgba(38, 255, 145, 0.10);
        --surface-soft: rgba(12, 25, 18, 0.82);
        --surface-deep: rgba(5, 10, 8, 0.92);
}

body, .gradio-container {
  background: var(--page-glow);
  color: var(--text-main);
    font-family: "Space Grotesk", "IBM Plex Sans", "Segoe UI", sans-serif;
}

.app-shell {
    max-width: 1440px;
  margin: 0 auto;
}

.hero, .panel, .stat-card {
  background: var(--panel-bg);
  border: 1px solid var(--panel-border);
    border-radius: 24px;
        box-shadow: 0 24px 70px rgba(0, 0, 0, 0.34);
        backdrop-filter: blur(12px);
}

.hero {
    padding: 28px 30px 18px 30px;
    position: relative;
    overflow: hidden;
}

.panel {
    padding: 12px;
}

.hero::after {
    content: "";
    position: absolute;
    inset: auto -60px -70px auto;
    width: 240px;
    height: 240px;
    border-radius: 999px;
    background: radial-gradient(circle, rgba(38, 255, 145, 0.18), transparent 62%);
}

.hero-grid {
    display: grid;
    grid-template-columns: 2.2fr 1fr;
    gap: 18px;
    align-items: start;
}

.hero-kicker {
    display: inline-block;
    padding: 6px 10px;
    border-radius: 999px;
    background: var(--accent-soft);
    color: #08110d;
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
    background: linear-gradient(135deg, var(--accent) 0%, #7effc2 100%);
}

.hero h1 {
    font-family: "JetBrains Mono", "Space Grotesk", sans-serif;
    font-size: 44px;
    line-height: 1.02;
    margin: 14px 0 12px;
}

.hero p {
    font-size: 16px;
    line-height: 1.6;
    margin: 0 0 10px;
}

.stat-stack {
    display: grid;
    gap: 12px;
}

.stat-card {
    padding: 16px 18px;
}

.stat-label {
    color: var(--text-muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.stat-value {
    display: block;
    margin-top: 8px;
    font-size: 22px;
    font-weight: 700;
}

.stat-note {
    display: block;
    margin-top: 6px;
    color: var(--text-muted);
    line-height: 1.45;
}

.playbook {
    display: grid;
    gap: 12px;
}

.playbook-card {
    border: 1px solid var(--panel-border);
    background: linear-gradient(180deg, rgba(11, 24, 17, 0.95), rgba(7, 14, 10, 0.92));
    border-radius: 18px;
    padding: 14px 16px;
}

.playbook-card h3 {
    margin: 0 0 8px;
    font-size: 15px;
}

.playbook-card p, .playbook-card li {
    margin: 0;
    color: var(--text-muted);
    line-height: 1.5;
}

.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 10px;
}

.chip {
    display: inline-flex;
    align-items: center;
    padding: 6px 10px;
    border-radius: 999px;
    background: var(--teal-soft);
    color: #082013;
    font-size: 12px;
    font-weight: 600;
    background: linear-gradient(135deg, rgba(38, 255, 145, 0.92), rgba(18, 214, 117, 0.9));
}

.hero h1, .hero p, .hero li {
  color: var(--text-main);
}

.hero p, .hero li {
  color: var(--text-muted);
}

.gr-button-primary {
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-deep) 100%) !important;
    border: none !important;
    color: #041009 !important;
    box-shadow: 0 0 18px rgba(38, 255, 145, 0.22);
}

.gradio-container .gr-box,
.gradio-container .gr-form,
.gradio-container .gr-panel,
.gradio-container .gr-accordion,
.gradio-container .gr-tab,
.gradio-container .gradio-group {
    background: var(--surface-soft) !important;
    border-color: var(--panel-border) !important;
}

.gradio-container textarea,
.gradio-container input,
.gradio-container select {
    background: var(--surface-deep) !important;
    color: var(--text-main) !important;
    border-color: rgba(69, 255, 155, 0.18) !important;
}

.gradio-container label,
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .prose strong,
.gradio-container .prose h1,
.gradio-container .prose h2,
.gradio-container .prose h3 {
    color: var(--text-main) !important;
}

.gradio-container .prose code,
.builder-preview textarea {
    font-family: "JetBrains Mono", monospace !important;
}

.builder-preview textarea {
    background: linear-gradient(180deg, rgba(6, 16, 10, 0.96), rgba(4, 10, 7, 0.98)) !important;
    border: 1px solid rgba(69, 255, 155, 0.24) !important;
    box-shadow: inset 0 0 0 1px rgba(38, 255, 145, 0.05);
}

.section-note {
    color: var(--text-muted);
    font-size: 13px;
    line-height: 1.5;
}

@media (max-width: 980px) {
    .hero-grid {
        grid-template-columns: 1fr;
    }
}
"""

_PIPELINE_LOCK = threading.Lock()
_GENERATION_LOCK = threading.Lock()
_PIPELINE = None
_PIPELINE_KEY = None


def _sanitize_filename(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip())
    cleaned = cleaned.strip("._")
    return cleaned[:64] or "ti2v"


def _build_guide_markdown() -> str:
    sections = []
    for title, values in PROMPT_GUIDE.items():
        sections.append(f"- **{title}**: {', '.join(values)}")
    return "\n".join(sections)


def _load_template(template_name: str) -> str:
    return PROMPT_TEMPLATES.get(template_name, DEFAULT_PROMPT)


def _compose_builder_prompt(
    scene_idea: str,
    time_of_day: str,
    light_source: str,
    light_quality: str,
    light_angle: str,
    color_tone: str,
    shot_size: str,
    camera_angle: str,
    composition: str,
    camera_move: str,
    motion_cue: str,
    extra_direction: str,
) -> str:
    selected_tags = [
        value
        for value in [
            time_of_day,
            light_source,
            light_quality,
            light_angle,
            color_tone,
            shot_size,
            camera_angle,
            composition,
        ]
        if value and value != NONE_OPTION
    ]
    fragments = []
    if selected_tags:
        fragments.append(", ".join(selected_tags) + ".")

    prompt_body = scene_idea.strip()
    if prompt_body:
        fragments.append(prompt_body)
    else:
        fragments.append(
            "Describe the subject, visible action, environment, and the key visual moment."
        )

    if camera_move and camera_move != NONE_OPTION:
        fragments.append(f"The camera {camera_move}.")
    if motion_cue and motion_cue != NONE_OPTION:
        fragments.append(motion_cue)
    if extra_direction.strip():
        fragments.append(extra_direction.strip())

    return " ".join(fragments).strip()


def _load_builder_template(template_name: str):
    template = _load_template(template_name)
    return template, template


def _apply_builder_prompt(prompt_from_builder: str) -> str:
    return prompt_from_builder.strip()


def _append_builder_prompt(existing_prompt: str, prompt_from_builder: str) -> str:
    existing_prompt = existing_prompt.strip()
    prompt_from_builder = prompt_from_builder.strip()
    if not existing_prompt:
        return prompt_from_builder
    if not prompt_from_builder:
        return existing_prompt
    return f"{existing_prompt} {prompt_from_builder}".strip()


def _reset_builder():
    return (
        DEFAULT_BUILDER_SCENE,
        DEFAULT_BUILDER_SCENE,
        NONE_OPTION,
        NONE_OPTION,
        NONE_OPTION,
        NONE_OPTION,
        NONE_OPTION,
        NONE_OPTION,
        NONE_OPTION,
        NONE_OPTION,
        NONE_OPTION,
        NONE_OPTION,
        "",
    )


def _resolve_model_dir(model_dir: str) -> Path:
    resolved = Path(model_dir).expanduser().resolve()
    if not resolved.exists():
        raise gr.Error(f"Checkpoint directory does not exist: {resolved}")
    required_files = [
        "config.json",
        "Wan2.2_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
    ]
    missing = [name for name in required_files if not (resolved / name).exists()]
    if missing:
        raise gr.Error(
            f"Checkpoint directory is missing required files: {', '.join(missing)}"
        )
    return resolved


def _get_pipeline(model_dir: str, device_id: int, t5_cpu: bool, convert_model_dtype: bool):
    global _PIPELINE, _PIPELINE_KEY

    cfg = WAN_CONFIGS["ti2v-5B"]
    resolved_model_dir = _resolve_model_dir(model_dir)
    key = (str(resolved_model_dir), device_id, t5_cpu, convert_model_dtype)

    with _PIPELINE_LOCK:
        if _PIPELINE is not None and _PIPELINE_KEY == key:
            return _PIPELINE, cfg, resolved_model_dir

        _PIPELINE = None
        _PIPELINE_KEY = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logging.info("Loading WanTI2V from %s on cuda:%s", resolved_model_dir, device_id)
        _PIPELINE = WanTI2V(
            config=cfg,
            checkpoint_dir=str(resolved_model_dir),
            device_id=device_id,
            rank=0,
            t5_cpu=t5_cpu,
            convert_model_dtype=convert_model_dtype,
        )
        _PIPELINE_KEY = key
        return _PIPELINE, cfg, resolved_model_dir


def generate_video(
    prompt: str,
    image,
    negative_prompt: str,
    model_dir: str,
    device_id: int,
    size_preset: str,
    frame_num: int,
    sampling_steps: int,
    guide_scale: float,
    sample_shift: float,
    sample_solver: str,
    seed: int,
    offload_model: bool,
    t5_cpu: bool,
    convert_model_dtype: bool,
    progress=gr.Progress(track_tqdm=True),
):
    del progress

    if not torch.cuda.is_available():
        raise gr.Error("CUDA is required for this demo.")
    if image is None and not prompt.strip():
        raise gr.Error("Enter a prompt for text-to-video, or provide an input image.")
    if size_preset not in SIZE_CONFIGS:
        raise gr.Error(f"Unsupported size preset: {size_preset}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    mode = "i2v" if image is not None else "t2v"
    resolved_seed = int(seed)
    negative_prompt = negative_prompt.strip()

    with _GENERATION_LOCK:
        pipeline, cfg, resolved_model_dir = _get_pipeline(
            model_dir=model_dir,
            device_id=int(device_id),
            t5_cpu=t5_cpu,
            convert_model_dtype=convert_model_dtype,
        )
        result = pipeline.generate(
            input_prompt=prompt.strip(),
            img=image.convert("RGB") if image is not None else None,
            size=SIZE_CONFIGS[size_preset],
            max_area=MAX_AREA_CONFIGS[size_preset],
            frame_num=int(frame_num),
            shift=float(sample_shift),
            sample_solver=sample_solver,
            sampling_steps=int(sampling_steps),
            guide_scale=float(guide_scale),
            n_prompt=negative_prompt,
            seed=resolved_seed,
            offload_model=offload_model,
        )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prompt_stub = _sanitize_filename(prompt or mode)
        output_path = OUTPUT_DIR / f"wan_ti2v_{mode}_{size_preset}_{prompt_stub}_{timestamp}.mp4"
        save_video(
            tensor=result[None],
            save_file=str(output_path),
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        del result

        torch.cuda.synchronize(int(device_id))
        torch.cuda.empty_cache()

    if not output_path.exists():
        raise gr.Error("Generation finished, but the output video was not written to disk.")

    elapsed = time.time() - start_time
    status = (
        f"Generated {mode} video in {elapsed:.1f}s\\n"
        f"Checkpoint: {resolved_model_dir}\\n"
        f"Output: {output_path}\\n"
        f"Seed: {resolved_seed}"
    )
    return str(output_path), status


def _build_demo(default_model_dir: Path, default_device: int) -> gr.Blocks:
    example_image = APP_ROOT / "examples" / "i2v_input.JPG"
    guide_markdown = _build_guide_markdown()
    examples = [
        [
            DEFAULT_PROMPT,
            None,
            "",
            str(default_model_dir),
            str(default_device),
            "1280*704",
            121,
            50,
            5.0,
            5.0,
            "unipc",
            -1,
            False,
            False,
            False,
        ]
    ]
    if example_image.exists():
        examples.append(
            [
                "A handheld camera tracks a surfer cat balancing easily as the board cuts across sparkling water, sea spray catching the afternoon sun.",
                str(example_image),
                "",
                str(default_model_dir),
                str(default_device),
                "1280*704",
                121,
                50,
                5.0,
                5.0,
                "unipc",
                -1,
                False,
                False,
                False,
            ]
        )

    with gr.Blocks(css=CSS, title="Wan2.2 TI2V Demo") as demo:
        with gr.Column(elem_classes=["app-shell"]):
            gr.HTML(
                f"""
                <div class="hero">
                  <div class="hero-grid">
                    <div>
                      <span class="hero-kicker">Wan2.2 TI2V Studio</span>
                      <h1>Design the shot, not just the prompt.</h1>
                      <p>This demo runs the repo's native TI2V pipeline against your local Wan2.2-TI2V-5B weights. It supports plain text-to-video and image-guided video generation from the same interface.</p>
                      <div class="chip-row">
                        <span class="chip">720P TI2V</span>
                        <span class="chip">Lighting-aware prompts</span>
                        <span class="chip">Camera-motion hints</span>
                        <span class="chip">Local checkpoint</span>
                      </div>
                    </div>
                    <div class="stat-stack">
                      <div class="stat-card">
                        <span class="stat-label">Default model</span>
                        <span class="stat-value">Wan2.2-TI2V-5B</span>
                        <span class="stat-note">Uses {default_model_dir}</span>
                      </div>
                      <div class="stat-card">
                        <span class="stat-label">Prompting tip</span>
                        <span class="stat-value">Use 2-4 cinematic labels</span>
                        <span class="stat-note">The repo prompt system explicitly emphasizes lighting, composition, camera angle, shot size, color tone, and motion cues.</span>
                      </div>
                    </div>
                  </div>
                </div>
                """
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=5, elem_classes=["panel"]):
                    with gr.Tabs():
                        with gr.Tab("Prompt Builder"):
                            gr.HTML(
                                '<div class="section-note">Build a shot in the model\'s own vocabulary, preview it, then push it into the final prompt editor.</div>'
                            )
                            starter_prompt = gr.Dropdown(
                                label="Starter Prompt",
                                choices=list(PROMPT_TEMPLATES.keys()),
                                value="Neon cafe push-in",
                            )
                            with gr.Row():
                                load_template_button = gr.Button("Load Into Builder")
                                reset_builder_button = gr.Button("Reset Builder")
                            builder_scene = gr.Textbox(
                                label="Scene Idea",
                                lines=4,
                                value=DEFAULT_BUILDER_SCENE,
                                placeholder="Write the subject, action, environment, and any important visible details.",
                            )
                            with gr.Row():
                                time_of_day = gr.Dropdown(
                                    label="Time",
                                    choices=[NONE_OPTION] + PROMPT_GUIDE["Time"],
                                    value=NONE_OPTION,
                                )
                                shot_size = gr.Dropdown(
                                    label="Shot Size",
                                    choices=[NONE_OPTION] + PROMPT_GUIDE["Shot size"],
                                    value=NONE_OPTION,
                                )
                                composition = gr.Dropdown(
                                    label="Composition",
                                    choices=[NONE_OPTION] + PROMPT_GUIDE["Composition"],
                                    value=NONE_OPTION,
                                )
                            with gr.Row():
                                light_source = gr.Dropdown(
                                    label="Light Source",
                                    choices=[NONE_OPTION] + PROMPT_GUIDE["Light source"],
                                    value=NONE_OPTION,
                                )
                                light_quality = gr.Dropdown(
                                    label="Light Quality",
                                    choices=[NONE_OPTION] + PROMPT_GUIDE["Light quality"],
                                    value=NONE_OPTION,
                                )
                                light_angle = gr.Dropdown(
                                    label="Light Angle",
                                    choices=[NONE_OPTION] + PROMPT_GUIDE["Light angle"],
                                    value=NONE_OPTION,
                                )
                            with gr.Row():
                                color_tone = gr.Dropdown(
                                    label="Color Tone",
                                    choices=[NONE_OPTION] + PROMPT_GUIDE["Color tone"],
                                    value=NONE_OPTION,
                                )
                                camera_angle = gr.Dropdown(
                                    label="Camera Angle",
                                    choices=[NONE_OPTION] + PROMPT_GUIDE["Camera angle"],
                                    value=NONE_OPTION,
                                )
                                camera_move = gr.Dropdown(
                                    label="Camera Movement",
                                    choices=[NONE_OPTION] + PROMPT_GUIDE["Camera movement"],
                                    value=NONE_OPTION,
                                )
                            motion_cue = gr.Dropdown(
                                label="Motion Cue",
                                choices=[NONE_OPTION] + PROMPT_GUIDE["Motion cue"],
                                value=NONE_OPTION,
                            )
                            extra_direction = gr.Textbox(
                                label="Extra Direction",
                                lines=2,
                                placeholder="Optional: add one short instruction for expression, pacing, or environment motion.",
                            )
                            builder_preview = gr.Textbox(
                                label="Builder Preview",
                                lines=7,
                                value=DEFAULT_BUILDER_SCENE,
                                elem_classes=["builder-preview"],
                            )
                            with gr.Row():
                                apply_builder_button = gr.Button("Use Builder As Prompt", variant="primary")
                                append_builder_button = gr.Button("Append Builder To Prompt")

                        with gr.Tab("Prompt Editor"):
                            gr.HTML(
                                '<div class="section-note">This is the final prompt sent to WanTI2V. You can type here directly or fill it from the builder tab.</div>'
                            )
                            prompt = gr.Textbox(
                                label="Prompt",
                                lines=8,
                                value=DEFAULT_PROMPT,
                                placeholder="Describe the subject, visible action, camera movement, environment, and 2-4 cinematic labels.",
                            )
                    image = gr.Image(label="Input Image (Optional)", type="pil")
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt (Optional)",
                        lines=2,
                        placeholder="Elements to avoid. Leave empty to use the model default.",
                    )
                    with gr.Row():
                        model_dir = gr.Textbox(
                            label="Checkpoint Directory",
                            value=str(default_model_dir),
                        )
                        device_id = gr.Dropdown(
                            label="GPU",
                            choices=[str(index) for index in range(max(torch.cuda.device_count(), 1))],
                            value=str(default_device),
                        )
                    with gr.Row():
                        size_preset = gr.Dropdown(
                            label="Size",
                            choices=["1280*704", "704*1280"],
                            value="1280*704",
                        )
                        frame_num = gr.Slider(
                            label="Frames",
                            minimum=81,
                            maximum=161,
                            step=4,
                            value=121,
                        )
                    with gr.Accordion("Advanced", open=False):
                        with gr.Row():
                            sampling_steps = gr.Slider(
                                label="Sampling Steps",
                                minimum=20,
                                maximum=80,
                                step=1,
                                value=50,
                            )
                            guide_scale = gr.Slider(
                                label="Guidance Scale",
                                minimum=1.0,
                                maximum=9.0,
                                step=0.1,
                                value=5.0,
                            )
                        with gr.Row():
                            sample_shift = gr.Slider(
                                label="Sample Shift",
                                minimum=1.0,
                                maximum=8.0,
                                step=0.1,
                                value=5.0,
                            )
                            sample_solver = gr.Dropdown(
                                label="Solver",
                                choices=["unipc", "dpm++"],
                                value="unipc",
                            )
                        with gr.Row():
                            seed = gr.Number(label="Seed", value=-1, precision=0)
                            offload_model = gr.Checkbox(
                                label="Offload Model",
                                value=False,
                                info="Enable this on smaller GPUs to reduce VRAM pressure.",
                            )
                        with gr.Row():
                            t5_cpu = gr.Checkbox(label="Run T5 on CPU", value=False)
                            convert_model_dtype = gr.Checkbox(
                                label="Convert Model DType",
                                value=False,
                            )
                    with gr.Row():
                        generate_button = gr.Button("Generate Video", variant="primary")

                with gr.Column(scale=4, elem_classes=["panel"]):
                    video_output = gr.Video(label="Generated Video", autoplay=False)
                    status_output = gr.Textbox(label="Run Log", lines=6)
                    clear_button = gr.ClearButton(
                        [prompt, builder_scene, builder_preview, extra_direction, image, negative_prompt, video_output, status_output],
                        value="Clear Inputs And Output",
                    )
                    gr.HTML(
                        """
                        <div class="playbook">
                          <div class="playbook-card">
                            <h3>Prompt Recipe</h3>
                            <p>Start with the subject and visible action, then add 2-4 cinematic labels, then one clear camera move if you want motion in the framing.</p>
                          </div>
                          <div class="playbook-card">
                            <h3>What Wan explicitly responds to</h3>
                            <p>The repo prompt rules call out lighting, composition, color tone, shot size, camera angle, and camera movement as useful controllable labels.</p>
                          </div>
                        </div>
                        """
                    )
                    with gr.Accordion("Cinematic Keyword Guide", open=True):
                        gr.Markdown(guide_markdown)
                    with gr.Accordion("Prompting Notes", open=False):
                        gr.Markdown(
                            """
- Use **2-4 labels**, not every possible label.
- If you already describe camera motion in the prompt, avoid adding a conflicting camera angle.
- For image-guided mode, keep the prompt focused on motion, expression, and camera behavior rather than re-describing every static detail.
- The repo README notes that Wan2.2 was trained with detailed labels for **lighting, composition, contrast, and color tone**.
                            """
                        )

            gr.Examples(
                examples=examples,
                inputs=[
                    prompt,
                    image,
                    negative_prompt,
                    model_dir,
                    device_id,
                    size_preset,
                    frame_num,
                    sampling_steps,
                    guide_scale,
                    sample_shift,
                    sample_solver,
                    seed,
                    offload_model,
                    t5_cpu,
                    convert_model_dtype,
                ],
                label="Examples",
            )

        builder_inputs = [
            builder_scene,
            time_of_day,
            light_source,
            light_quality,
            light_angle,
            color_tone,
            shot_size,
            camera_angle,
            composition,
            camera_move,
            motion_cue,
            extra_direction,
        ]

        load_template_button.click(
            fn=_load_builder_template,
            inputs=[starter_prompt],
            outputs=[builder_scene, builder_preview],
        )

        for builder_component in builder_inputs:
            builder_component.change(
                fn=_compose_builder_prompt,
                inputs=builder_inputs,
                outputs=[builder_preview],
            )

        apply_builder_button.click(
            fn=_apply_builder_prompt,
            inputs=[builder_preview],
            outputs=[prompt],
        )

        append_builder_button.click(
            fn=_append_builder_prompt,
            inputs=[prompt, builder_preview],
            outputs=[prompt],
        )

        reset_builder_button.click(
            fn=_reset_builder,
            outputs=[
                builder_scene,
                builder_preview,
                time_of_day,
                light_source,
                light_quality,
                light_angle,
                color_tone,
                shot_size,
                camera_angle,
                composition,
                camera_move,
                motion_cue,
                extra_direction,
            ],
        )

        generate_button.click(
            fn=generate_video,
            inputs=[
                prompt,
                image,
                negative_prompt,
                model_dir,
                device_id,
                size_preset,
                frame_num,
                sampling_steps,
                guide_scale,
                sample_shift,
                sample_solver,
                seed,
                offload_model,
                t5_cpu,
                convert_model_dtype,
            ],
            outputs=[video_output, status_output],
        )

    return demo.queue(max_size=4, default_concurrency_limit=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Wan2.2 TI2V Gradio demo")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--ckpt-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--share", action="store_true", default=False)
    parser.add_argument("--no-open-browser", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the TI2V demo.")
    if args.device < 0 or args.device >= torch.cuda.device_count():
        raise RuntimeError(
            f"Invalid GPU index {args.device}. Found {torch.cuda.device_count()} CUDA device(s)."
        )

    demo = _build_demo(Path(args.ckpt_dir).expanduser(), args.device)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_open_browser,
    )


if __name__ == "__main__":
    main()