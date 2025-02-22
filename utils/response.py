from openai import OpenAI
import tiktoken
import requests

class InductionResponse:
    def __init__(self, config, model):
        self.config = config
        self.model = model
        if self.model in ["deepseek", "gpt-4o", "gpt-4o-mini"]:
            self.client = OpenAI(
                api_key=self.config[model]["api_key"], 
                base_url=self.config[model]["base_url"]
            )
        else:
            #TODO other models
            self.client = None

    def generate_response(self, prompt, temperature=0.0, retry=10, system_message="You are an expert Python programmer."):
        for i in range(retry):
            try:
                response = self.client.chat.completions.create(
                    model=self.config[self.model]["model"],
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature,
                    max_tokens=4096
                )
                return response.choices[0].message.content
            except Exception as e:
                if i == retry - 1:
                    raise e
                print(e)
                continue


if __name__ == "__main__":
    induct = InductionResponse(None, None)
    prompt = "Say this is a test"
    response = induct.generate_response(prompt)
    print(response)