import os
if os.getenv('LANGFUSE_SECRET_KEY'):
    from langfuse.openai import OpenAI
else:
    from openai import OpenAI
import gradio as gr
import dotenv

dotenv.load_dotenv(override=True)

selected_model = "llama-3.2-90b-text-preview"
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
    for chunk in response:
        if chunk.choices[0].delta.content is not None:
              partial_message = partial_message + chunk.choices[0].delta.content
              yield partial_message
CSS ="""
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""
with gr.Blocks(css=CSS) as demo:
    # with gr.Accordion("Advanced options"):
    #     selected_model = gr.Radio(label="model", choices=["llama-3.2-90b-text-preview", "llama-3.1-sonar-large-128k-online"], value="llama-3.2-90b-text-preview", interactive=True)
    #     input_openai_api_base = gr.Textbox(label="openai api base", placeholder=input_openai_api_base)
    
    system_prompt = gr.Textbox(label="system", placeholder="system prompt here...")
    gr.Examples(
        examples=[
            "翻译用户提供的文案为中文，翻译时考虑上下文。删除括号内参考文献文字内容。",
            "作为一名人工智能行业内专家，找出用户提供的文案中不正确的信息，并进行修正。",
        ],
        inputs=[system_prompt],
    )
    chatbot = gr.ChatInterface(
        predict,
        additional_inputs=[system_prompt],
    )

demo.launch(server_port=18080)


