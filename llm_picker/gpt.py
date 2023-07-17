from .llm import LLM_Base,ON_TOKENS_OVERSIZED,CallStack
import openai
from time import sleep
import os
import re

GPT3_MODEL = "gpt-3.5-turbo"
GPT4_MODEL = "gpt-4"
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
TECKY_API_KEY = os.environ.get('TECKY_API_KEY')

ON_RESULT_FILTERED="on_result_filtered"

def detect_if_result_filtered(e):
    return re.search(r"The response was filtered due to the prompt triggering Azure OpenAI’s content management policy.", str(e)) is not None

class GPT(LLM_Base):
    gpt_error_delay=2
    temperature=0

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
        if self.model_name is None:
            self.model_name = GPT.model_picker()
        return self.model_name
    def set_model_name(self, name):
        self.model_name = name
    def detect_if_tokens_oversized(self,e):
        return (re.search(r"This model's maximum context length is", str(e)) is not None and \
            re.search(r"tokens", str(e)) is not None and \
            re.search(r"Please reduce the length of the messages.", str(e)) is not None) or \
            (re.search(r"HTTP code 413 from API", str(e)) is not None and \
                re.search(r"PayloadTooLargeError: request entity too large", str(e)) is not None)
    def get_response(self,system,assistant,user):
        model=self.get_model_name()
        if model is None:
            raise Exception("No API key found for OpenAI or Tecky")
        response_cache=LLM_Base.load_response_cache(model,system,assistant,user)
        if response_cache is not None and self.use_cache:
            if "choices" in response_cache and len(response_cache["choices"])>0 and "message" in response_cache["choices"][0] and \
                "content" in response_cache["choices"][0]["message"]:
                response_content=response_cache["choices"][0]["message"]["content"]
                return response_content
            elif ON_TOKENS_OVERSIZED in response_cache:
                e=response_cache[ON_TOKENS_OVERSIZED]
                return self.instant.on_tokens_oversized(e,system,assistant,user)
            elif ON_RESULT_FILTERED in response_cache:
                return None
            else:
                if (len(response_cache["choices"])==0 or
                    "message" not in response_cache["choices"][0] or
                    "content" not in response_cache["choices"][0]["message"]):
                    LLM_Base.delete_response_cache(model,system,assistant,user)
        print(f"Connecting to {model} model...")
        try:
            completion = openai.ChatCompletion.create(
                model=model,
                messages=[
                        {"role": "system","content": system},
                        {"role": "assistant","content": assistant},
                        {"role": "user","content": user}
                    ],
                temperature=self.temperature,
                # max_tokens=2048
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
            if self.detect_if_tokens_oversized(e):
                LLM_Base.save_response_cache(model,system,assistant,user,{ON_TOKENS_OVERSIZED:str(e)})
                return self.instant.on_tokens_oversized(e,system,assistant,user)
            elif detect_if_result_filtered(e):
                LLM_Base.save_response_cache(model,system,assistant,user,{ON_RESULT_FILTERED:str(e)})
                return None
            elif re.search(r"Invalid Output: ", str(e)) is not None:
                return None
            else:
                print(f"Retrying in {self.gpt_error_delay} seconds...")
                sleep(self.gpt_error_delay)
                self.gpt_error_delay=self.gpt_error_delay*2
                return self.instant.get_response(system,assistant,user)
    def get_streaming_response(self, system, assistant, user,on_receive_callback) -> str:
        """
        Get a streaming response from the GPT model
        :param system: The system message
        :param assistant: The assistant message
        :param user: The user message
        :param on_receive_callback: The callback function to receive the streaming response
            callback function signature: on_receive_callback(response_chunk)
        :return: The response
        """
        model=self.get_model_name()
        if model is None:
            raise Exception("No API key found for OpenAI or Tecky")
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                        {"role": "system","content": system},
                        {"role": "assistant","content": assistant},
                        {"role": "user","content": user}
                    ],
                temperature=0,
                stream=True  
            )

            for chunk in response:
                on_receive_callback(chunk)
        except Exception as e:
            print(e)
            pass
        return
    pass