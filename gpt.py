import openai
import time


class GPTQuery:
    def __init__(self, api_key, model="gpt-3.5-turbo",):
        openai.api_key = api_key
        self.model = model
        self.examples = ""
        self.no = 0
    def query_with_examples(self, query, examples):
        messages = []
        if self.no == 0:
            if len(examples) > 0:
                for example in examples:
                    user_query = example[0]  # The question part of the example
                    assistant_response = example[1]  # The response part of the example
                    messages.append({"role": "user", "content": user_query})
                    messages.append({"role": "assistant", "content": assistant_response})
        messages.append({"role": "user", "content": query})
        messages.append({"role": "system", "content": "Please keep all answers within 200 characters or 30 tokens."},)
        while True:
            try:
                completion = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    request_timeout=10
                )
                return completion['choices'][0]['message']['content']
            except Exception as e:
                print(e)
                print("retry!")
                time.sleep(5)

    def query_without_examples(self, query):
        response = openai.Completion.create(
            model=self.model,
            prompt=query,
            max_tokens=50
        )
        return response.choices[0].text.strip()
