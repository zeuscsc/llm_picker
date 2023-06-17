import os
import json
import datetime
import hashlib
import glob
from abc import ABC, abstractmethod
from typing import Type
from .folders import LLM_RESPONSE_CACHE_FOLDER

ON_TOKENS_OVERSIZED="on_tokens_oversized"

def calculate_md5(string:str):
    md5_hash = hashlib.md5(string.encode()).hexdigest()
    return md5_hash

class CallStack:
    system:str
    assistant:str
    user:str
    response:str
    def __init__(self,system,assistant,user,response) -> None:
        self.system=system
        self.assistant=assistant
        self.user=user
        self.response=response
        pass
    pass
class _LLM_Base(ABC):
    # separator = ""
    # model_name:str=None
    # save_call_history:bool=False
    # responses_calls_history:list[Type[CallStack]] = []
    def load_response_cache(model,system,assistant,user):
        try:
            hashed_request=calculate_md5(f"{model}{system}{assistant}{user}")
            print(f"Loading response cache for {model} model with id: {hashed_request}...")
            matching_files = glob.glob(f"{LLM_RESPONSE_CACHE_FOLDER}/{hashed_request}/*.json")
            if len(matching_files)>0:
                with open(matching_files[-1], "r",encoding="utf8") as chat_cache_file:
                    chat_cache = json.load(chat_cache_file)
                    return chat_cache
        except Exception as e:
            print(e)
        return None
    def save_response_cache(model,system,assistant,user,chat_cache):
        hashed_request=calculate_md5(f"{model}{system}{assistant}{user}")
        now = datetime.datetime.now()
        time_string = now.strftime("%Y%m%d%H%M%S")
        os.makedirs(f"{LLM_RESPONSE_CACHE_FOLDER}/{hashed_request}", exist_ok=True)
        with open(f"{LLM_RESPONSE_CACHE_FOLDER}/{hashed_request}/{time_string}.json", "w",encoding="utf8") as temp_file:
            json.dump(chat_cache, temp_file, indent=4, ensure_ascii=False)
    def delete_response_cache(model,system,assistant,user):
        hashed_request=calculate_md5(f"{model}{system}{assistant}{user}")
        matching_files = glob.glob(f"{LLM_RESPONSE_CACHE_FOLDER}/{hashed_request}/*.json")
        for file in matching_files:
            os.remove(file)
    def split_text_in_half_if_too_large(text:str,max_tokens=10000):
        words = text.split()
        results = []
        
        if len(words) <= max_tokens:
            results.append(' '.join(words))
        else:
            half = len(words) // 2
            results.extend(_LLM_Base.split_text_in_half_if_too_large(' '.join(words[:half])))
            results.extend(_LLM_Base.split_text_in_half_if_too_large(' '.join(words[half:])))
        return results
    def split_text_in_half(text:str):
        words = text.split()
        results = []
        half = len(words) // 2
        results.extend(_LLM_Base.split_text_in_half_if_too_large(' '.join(words[:half])))
        results.extend(_LLM_Base.split_text_in_half_if_too_large(' '.join(words[half:])))
        return results

    @abstractmethod
    def get_model_name(self):
        pass
    @abstractmethod
    def set_model_name(self,name):
        pass
    @abstractmethod
    def detect_if_tokens_oversized(self,e):
        pass
    @abstractmethod
    def get_response(self,system,assistant,user):
        pass

    def on_tokens_oversized(self,e,system,assistant,user):
        if self.detect_if_tokens_oversized(e):
            print("Splitting text in half...")
            chunks = []
            chunks.extend(_LLM_Base.split_text_in_half(user))
            responses=""
            for chunk in chunks:
                try:
                    response=self.get_response(system,assistant,chunk)
                    # if self.save_call_history:
                    #     self.responses_calls_history.append(CallStack(system,assistant,chunk,response))
                except Exception as e:
                    print(e)
                    continue
                if response is not None:
                    responses+=response+self.separator
            return responses
    
    pass


class LLM_Base(_LLM_Base):
    def __init__(self,instant:_LLM_Base) -> None:
        self.instant=instant
        self.separator:str = ""
        self.model_name:str=None
        self.save_call_history:bool=False
        self.responses_calls_history:list[Type[CallStack]] = []
        pass
    pass
class LLM:
    def __init__(self,ModelClass:Type[LLM_Base],separator="",save_call_history=False) -> None:
        self.model_class=ModelClass(self)
        self.separator=separator
        self.save_call_history=save_call_history
        self.model_class.save_call_history=save_call_history
        pass
    def get_model_name(self):
        return self.model_class.get_model_name()
    def get_response(self,system,assistant,user):
        response=self.model_class.get_response(system,assistant,user)
        if self.save_call_history:
            self.model_class.responses_calls_history.append(CallStack(system,assistant,user,response))
        return response
    def get_called_history(self):
        return self.model_class.responses_calls_history
    def on_tokens_oversized(self,e,system,assistant,user):
        return self.model_class.on_tokens_oversized(e,system,assistant,user)
    pass

def get_best_available_llm(separator="",save_call_history=False):
    from .gpt import GPT
    model=GPT.model_picker()
    if model is not None:
        instant=LLM(GPT,separator,save_call_history)
        return instant
    from .llama import LLaMA
    instant=LLM(LLaMA,separator,save_call_history)
    return instant