from openai import OpenAI

client = OpenAI(api_key="132455",base_url="http://127.0.0.1:6006/v1")

stream = client.chat.completions.create(
    model="/root/autodl-tmp/models",
    messages=[{"role": "user", "content": "使用html5做一个随机的经典小游戏"}],
    stream=True,
)
for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
                                                                         