import html
import os
if os.getenv('LANGFUSE_SECRET_KEY'):
    from langfuse.openai import OpenAI
else:
    from openai import OpenAI
import gradio as gr
import dotenv

dotenv.load_dotenv(override=True)

selected_model = "/root/deepseek-r1-distill-qwen-32b"
input_openai_api_base = os.getenv("OPENAI_API_BASE")
def predict(message, history, system_prompt):
    client = OpenAI(            
                api_key=os.getenv("OPENAI_API_KEY"),
            base_url=input_openai_api_base,  
                )
    history_openai_format = [{"role": "system", "content": system_prompt}]
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human })
        history_openai_format.append({"role": "assistant", "content":assistant})
    history_openai_format.append({"role": "user", "content": message})
  
    response = client.chat.completions.create(model=selected_model,
    messages= history_openai_format,
    temperature=1.0,
    stream=True)

    partial_message = ""
    in_thinking_tag = True
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
              content = chunk.choices[0].delta.content
              if in_thinking_tag:
                  content = html.escape(content)
              if chunk.choices[0].delta.content == "</think>":
                  in_thinking_tag = False
              partial_message = partial_message + content
              yield partial_message
              
        
with gr.Blocks(css="footer {display: none} .contain { height: 100vh !important } .gap { height: 90% !important }") as demo:
    # with gr.Accordion("Advanced options"):
    #     selected_model = gr.Radio(label="model", choices=["llama-3.2-90b-text-preview", "llama-3.1-sonar-large-128k-online"], value="llama-3.2-90b-text-preview", interactive=True)
    #     input_openai_api_base = gr.Textbox(label="openai api base", placeholder=input_openai_api_base)
    
    with gr.Accordion("System Prompt", open=False):
        system_prompt = gr.Textbox(label="system", placeholder="system prompt here...")
        gr.Examples(
            examples=[
                "针砭时弊，微言大义",
                "你是一位文采斐然的文学家",
            ],
            inputs=[system_prompt],
        )
    chatbot = gr.ChatInterface(
        predict,
        additional_inputs=[system_prompt],
        fill_height=True,
    )

demo.launch(server_port=18080)


