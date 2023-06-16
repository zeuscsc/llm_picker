from .llm import LLM_Base,ON_TOKENS_OVERSIZED
import openai
from time import sleep
import os

GPT3_MODEL = "gpt-3.5-turbo"
GPT4_MODEL = "gpt-4"
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
TECKY_API_KEY = os.environ.get('TECKY_API_KEY')

ON_RESULT_FILTERED="on_result_filtered"

class GPT(LLM_Base):
    gpt_error_delay=2

    def switch2tecky():
        openai.api_key = TECKY_API_KEY
        openai.api_base = "https://api.gpt.tecky.ai/v1"
    def switch2openai():
        openai.api_key = OPENAI_API_KEY
        openai.api_base = "https://api.openai.com/v1"

    def model_picker():
        if TECKY_API_KEY is not None and TECKY_API_KEY != "":
            return GPT4_MODEL
        elif OPENAI_API_KEY is not None and OPENAI_API_KEY != "":
            return GPT3_MODEL
        else:
            return None

    def get_model_name(self):
        return GPT.model_picker()
    def get_response(self,system,assistant,user):
        model=self.get_model_name()
        if model is None:
            raise Exception("No API key found for OpenAI or Tecky")
        response_cache=LLM_Base.load_response_cache(model,system,assistant,user)
        if response_cache is not None:
            if "choices" in response_cache and len(response_cache["choices"])>0 and "message" in response_cache["choices"][0] and \
                "content" in response_cache["choices"][0]["message"]:
                return response_cache["choices"][0]["message"]["content"]
            elif ON_TOKENS_OVERSIZED in response_cache:
                e=response_cache[ON_TOKENS_OVERSIZED]
                return self.instant.on_tokens_oversized(e,system,assistant,user)
            elif ON_RESULT_FILTERED in response_cache:
                return None
            else:
                completion=response_cache
                if (len(completion.choices)==0 or
                    "message" not in completion.choices[0] or
                    "content" not in completion.choices[0].message):
                    LLM_Base.delete_response_cache(model,system,assistant,user)
        if model=="gpt-3.5-turbo":
            GPT.switch2openai()
        elif model=="gpt-4":
            GPT.switch2tecky()
        print(f"Connecting to {model} model...")
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[
                        {"role": "system","content": system},
                        {"role": "assistant","content": assistant},
                        {"role": "user","content": user}
                    ],
                max_tokens=2048
                )
            LLM_Base.save_response_cache(model,system,assistant,user,completion)
            if len(completion.choices)==0:
                raise Exception("Invalid Output: No choices found in completion")
            elif "message" not in completion.choices[0]:
                raise Exception("Invalid Output: No message found in completion")
            elif "content" not in completion.choices[0].message:
                raise Exception("Invalid Output: No content found in completion")
            return completion.choices[0].message.content
        except Exception as e:
            print(e)
            if LLM_Base.detect_if_tokens_oversized(e):
                LLM_Base.save_response_cache(model,system,assistant,user,{ON_TOKENS_OVERSIZED:str(e)})
                return self.instant.on_tokens_oversized(e,system,assistant,user)
            elif LLM_Base.detect_if_result_filtered(e):
                LLM_Base.save_response_cache(model,system,assistant,user,{ON_RESULT_FILTERED:str(e)})
                return None
            else:
                sleep(self.gpt_error_delay)
                self.gpt_error_delay=self.gpt_error_delay*2
                return self.instant.get_response(system,assistant,user)
    pass