#GW
import datetime
print(datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S"), "Starting DeepSeek-over-Juice multi-modal demo (model=Janus-Pro-1B). Please wait...")
VERBOSE=False
if VERBOSE: print("before imports", datetime.datetime.now())
last_time = datetime.datetime.now()
#GW

#GW
import warnings
warnings.filterwarnings("ignore")
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import collections
from transformers.utils import logging
logging.set_verbosity_error() 
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
#GW 

import gradio as gr
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextStreamer
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image

import numpy as np
#GW import os
import time

#GW
import warnings
warnings.filterwarnings("ignore")
import traceback
from threading import Thread
from transformers import (
    TextIteratorStreamer
)
#GW

# import spaces  # Import spaces for ZeroGPU compatibility

#GW
if VERBOSE: print("after imports", datetime.datetime.now())
this_time = datetime.datetime.now()
print(this_time.strftime("%Y/%m/%d %H:%M:%S"), "Done importing Python packages.", "It took %.1f" % (this_time-last_time).total_seconds(),"seconds. Please wait...")
last_time = this_time
#GW

# Load model and processor
#GW model_path = "deepseek-ai/Janus-Pro-7B"
model_path = "deepseek-ai/Janus-Pro-1B"

config = AutoConfig.from_pretrained(model_path)
language_config = config.language_config
language_config._attn_implementation = 'eager'

#GW
if VERBOSE: print()
a = datetime.datetime.now()
if VERBOSE: print("automodel before from_pretrained", a)
#GW

vl_gpt = AutoModelForCausalLM.from_pretrained(model_path,
                                             language_config=language_config,
                                             trust_remote_code=True)

#GW
b = datetime.datetime.now()
if VERBOSE: print("automodel after from_pretrained", b, (b-a).total_seconds())
#GW

if torch.cuda.is_available():
    #GW
    c = datetime.datetime.now()
    if VERBOSE: print("cuda is available", c, (c-b).total_seconds())
    #GW

    #GW vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
    vl_gpt = vl_gpt.to(torch.bfloat16)

    d = datetime.datetime.now()
    if VERBOSE: print("to blfoat16", d, (d-c).total_seconds())

    vl_gpt = vl_gpt.cuda()

    e = datetime.datetime.now()
    if VERBOSE: print("to cuda", e, (e-d).total_seconds(), (e-a).total_seconds())
    print(e.strftime("%Y/%m/%d %H:%M:%S"), "Done initializing model.", "It took %.1f" % (e-last_time).total_seconds(),"seconds. Please wait...")
    last_time = e
    #GW

else:
    if VERBOSE: print("cuda is not available", datetime.datetime.now())

    vl_gpt = vl_gpt.to(torch.float16)

    #GW
    c = datetime.datetime.now()
    if VERBOSE: print("model to float16", (c-b).total_seconds(), (c-a).total_seconds())
    #GW

# GW
if VERBOSE: 
    print()
    print("processor before from_pretrained", datetime.datetime.now())
#GW
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)

if VERBOSE: print("processor after from_pretrained", datetime.datetime.now())

tokenizer = vl_chat_processor.tokenizer
cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'

#GW
if VERBOSE: print("cuda device", cuda_device, datetime.datetime.now())
global_thread = None
global_time = None
#GW

@torch.inference_mode()
# @spaces.GPU(duration=120) 
# Multimodal Understanding function
#GW def multimodal_understanding(image, question, seed, top_p, temperature):
def multimodal_understanding(image, question, seed, top_p, temperature,streamer, vl_gpt):
    if VERBOSE: print("start understanding", type(streamer), datetime.datetime.now())
#GW

    # Clear CUDA cache before generating
    torch.cuda.empty_cache()
   
    if VERBOSE: print("after empty cache", datetime.datetime.now())

    # set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image_placeholder>\n{question}",
            "images": [image],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
   
    if VERBOSE: print("before prepare image", datetime.datetime.now())
    pil_images = [Image.fromarray(image)]
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(cuda_device, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16)
    if VERBOSE: print("after prepare image", datetime.datetime.now())
    
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

#GW
    if VERBOSE:
        print("after embeds", datetime.datetime.now())
        print("inputs=", inputs_embeds, inputs_embeds.shape)
        print()
    b = datetime.datetime.now()
    if VERBOSE: print("before generate", b)
#GW

    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False if temperature == 0 else True,
        use_cache=True,
        temperature=temperature,
        top_p=top_p,
#GW
        streamer = streamer
#GW
    )

#GW
    if VERBOSE: print("outputs=", outputs, outputs.shape)
    a = datetime.datetime.now()
    if VERBOSE:
        print("after generate tokens len-", len( tokenizer.decode(outputs[0].cpu().tolist() )) , a, (a-b).total_seconds())
        print()
#GW
    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    if VERBOSE: print("after decode", datetime.datetime.now())
    return answer

#GW
def multimodal_understanding_thread(image, question, seed, top_p, temperature):
    try:
        global global_time
        global_time = datetime.datetime.now()
        print(global_time.strftime("%Y/%m/%d %H:%M:%S"), "Preparing model for decoding the image and prompt.  Please wait...")
        tok = AutoTokenizer.from_pretrained(model_path)
        if VERBOSE: print("after tokenzer create", datetime.datetime.now())
        streamer = TextIteratorStreamer(
            tokenizer=tok, timeout=60.0, skip_prompt=True, skip_special_tokens=True
        )
        if VERBOSE: print("created tokenzier and streamer", type(streamer), datetime.datetime.now())
        global global_thread
        if global_thread:
            if VERBOSE: print("killing current thread")
            try:
                global_thread.stop()
            except:
                traceback.print_exc()

        # launch new thread
        if VERBOSE: print("launching decode thread...")
        global_thread = Thread(target=multimodal_understanding, args=[image, question, seed, top_p, temperature, streamer, vl_gpt])
        global_thread.start()

        generated_text = ""
        count = 0
        last_time = None
        for i in streamer:
            generated_text += i
            if count==0:
                count += 1
                last_time = datetime.datetime.now()
                print(last_time.strftime("%Y/%m/%d %H:%M:%S"),"Model is ready to decode.", "Model prep took %.1f" % (last_time - global_time).total_seconds(),"seconds. Decode commencing...")
            else:
                count += 1
                this_time = datetime.datetime.now()
                token_rate = count*1.0 / ( (this_time - last_time).total_seconds() )
                print(this_time.strftime("%Y/%m/%d %H:%M:%S"), "Model is decoding! Current cecode stats: total tokens=", count, "token rate=", "%.1f" % token_rate,"...")

            yield generated_text

    except:
        traceback.print_exc()
#GW

def generate(input_ids,
             width,
             height,
             temperature: float = 1,
             parallel_size: int = 5,
             cfg_weight: float = 5,
             image_token_num_per_image: int = 576,
             patch_size: int = 16):
    if VERBOSE: print("generate top", datetime.datetime.now())
    # Clear CUDA cache before generating
    torch.cuda.empty_cache()
    if VERBOSE: print("generate after empty cache", datetime.datetime.now())
    
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(cuda_device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(cuda_device)

    pkv = None
    for i in range(image_token_num_per_image):
        if VERBOSE: print("for loop top", i, datetime.datetime.now())
        with torch.no_grad():
            if VERBOSE: print("before model", i, datetime.datetime.now())
            outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds,
                                                use_cache=True,
                                                past_key_values=pkv)
            if VERBOSE: print("after model", i, datetime.datetime.now())
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)

            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)
            if VERBOSE: print("for loop bottom", i, datetime.datetime.now())

    if VERBOSE: print("before decode", datetime.datetime.now())
    patches = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                                 shape=[parallel_size, 8, width // patch_size, height // patch_size])
    if VERBOSE: print("after decode", datetime.datetime.now())

    return generated_tokens.to(dtype=torch.int), patches

def unpack(dec, width, height, parallel_size=5):
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return visual_img



@torch.inference_mode()
# @spaces.GPU(duration=120)  # Specify a duration to avoid timeout
def generate_image(prompt,
                   seed=None,
                   guidance=5,
                   t2i_temperature=1.0):
    # Clear CUDA cache and avoid tracking gradients
    torch.cuda.empty_cache()
    # Set the seed for reproducible results
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
    width = 384
    height = 384
    parallel_size = 5
    
    with torch.no_grad():
        messages = [{'role': '<|User|>', 'content': prompt},
                    {'role': '<|Assistant|>', 'content': ''}]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(conversations=messages,
                                                                   sft_format=vl_chat_processor.sft_format,
                                                                   system_prompt='')
        text = text + vl_chat_processor.image_start_tag
        
        input_ids = torch.LongTensor(tokenizer.encode(text))
        output, patches = generate(input_ids,
                                   width // 16 * 16,
                                   height // 16 * 16,
                                   cfg_weight=guidance,
                                   parallel_size=parallel_size,
                                   temperature=t2i_temperature)
        images = unpack(patches,
                        width // 16 * 16,
                        height // 16 * 16,
                        parallel_size=parallel_size)

        return [Image.fromarray(images[i]).resize((768, 768), Image.LANCZOS) for i in range(parallel_size)]
        

# Gradio interface
#GW with gr.Blocks() as demo:
with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown(value="# Multimodal Understanding: DeepSeek JanusPro1B via JUICE")
    with gr.Row():
        image_input = gr.Image()
        with gr.Column():
            question_input = gr.Textbox(label="Question")
            und_seed_input = gr.Number(label="Seed", precision=0, value=42)
            top_p = gr.Slider(minimum=0, maximum=1, value=0.95, step=0.05, label="top_p")
            temperature = gr.Slider(minimum=0, maximum=1, value=0.1, step=0.05, label="temperature")
        
    understanding_button = gr.Button("Chat")
    understanding_output = gr.Textbox(label="Response")

    examples_inpainting = gr.Examples(
        label="Multimodal Understanding examples",
        examples=[
            [
                "explain this meme",
                "images/doge.png",
            ],
            [
                "Convert the formula into latex code.",
                "images/equation.png",
            ],
        ],
        inputs=[question_input, image_input],
    )
    
        
    gr.Markdown(value="# Text-to-Image Generation")

    
    
    with gr.Row():
        cfg_weight_input = gr.Slider(minimum=1, maximum=10, value=5, step=0.5, label="CFG Weight")
        t2i_temperature = gr.Slider(minimum=0, maximum=1, value=1.0, step=0.05, label="temperature")

    prompt_input = gr.Textbox(label="Prompt. (Prompt in more detail can help produce better images!)")
    seed_input = gr.Number(label="Seed (Optional)", precision=0, value=12345)

    generation_button = gr.Button("Generate Images")

    image_output = gr.Gallery(label="Generated Images", columns=2, rows=2, height=300)

    examples_t2i = gr.Examples(
        label="Text to image generation examples.",
        examples=[
            "Master shifu racoon wearing drip attire as a street gangster.",
            "The face of a beautiful girl",
            "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
            "A glass of red wine on a reflective surface.",
            "A cute and adorable baby fox with big brown eyes, autumn leaves in the background enchanting,immortal,fluffy, shiny mane,Petals,fairyism,unreal engine 5 and Octane Render,highly detailed, photorealistic, cinematic, natural colors.",
            "The image features an intricately designed eye set against a circular backdrop adorned with ornate swirl patterns that evoke both realism and surrealism. At the center of attention is a strikingly vivid blue iris surrounded by delicate veins radiating outward from the pupil to create depth and intensity. The eyelashes are long and dark, casting subtle shadows on the skin around them which appears smooth yet slightly textured as if aged or weathered over time.\n\nAbove the eye, there's a stone-like structure resembling part of classical architecture, adding layers of mystery and timeless elegance to the composition. This architectural element contrasts sharply but harmoniously with the organic curves surrounding it. Below the eye lies another decorative motif reminiscent of baroque artistry, further enhancing the overall sense of eternity encapsulated within each meticulously crafted detail. \n\nOverall, the atmosphere exudes a mysterious aura intertwined seamlessly with elements suggesting timelessness, achieved through the juxtaposition of realistic textures and surreal artistic flourishes. Each component\u2014from the intricate designs framing the eye to the ancient-looking stone piece above\u2014contributes uniquely towards creating a visually captivating tableau imbued with enigmatic allure.",
        ],
        inputs=prompt_input,
    )
    
    understanding_button.click(
        multimodal_understanding_thread, #GW multimodal_understanding
        inputs=[image_input, question_input, und_seed_input, top_p, temperature],
        outputs=understanding_output
    )
    
    generation_button.click(
        fn=generate_image,
        inputs=[prompt_input, seed_input, cfg_weight_input, t2i_temperature],
        outputs=image_output
    )

if VERBOSE: print("before demo launch...", datetime.datetime.now())
#GW 
# demo.launch(share=True)
demo.queue()
demo.launch(server_name="0.0.0.0", server_port=7860, quiet=True)
#GW 
# demo.queue(concurrency_count=1, max_size=10).launch(server_name="0.0.0.0", server_port=37906, root_path="/path")
