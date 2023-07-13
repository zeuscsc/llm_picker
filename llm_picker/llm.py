import os
import json
import datetime
import hashlib
import glob
from abc import ABC, abstractmethod
from typing import Any, Callable, Type
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
    def __init__(self) -> None:
        self.model_name:str=None
        self.use_cache:bool=True
        self.on_each_response:Type[object]=None
        pass
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
    def get_response(self,system,assistant,user)->str:
        pass
    @abstractmethod
    def get_streaming_response(self,system,assistant,user,on_receive_callback)->str:
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
                except Exception as e:
                    print(e)
                    continue
                if response is not None:
                    if self.on_each_response is None:
                        responses+=response
                    else:
                        responses=self.on_each_response(system,assistant,chunk,responses,response)
            return responses
    
    pass


class LLM_Base(_LLM_Base):
    def __init__(self,instant:_LLM_Base) -> None:
        self.instant=instant
        pass
    pass
class LLM:
    def __init__(self,ModelClass:Type[LLM_Base],use_cache:bool=True,on_each_response:Callable[[str,str,str,str,str], str]=None) -> None:
        """
        Constructor for LLM class
        :param ModelClass: LLM_Base The class of the model to be used
        :param use_cache: bool Whether to use cache or not
        :param on_each_response: Callable[[str,str,str,str,str], str] A function to be called on each response
            - This function should take 5 string from get_response or in between on_tokens_oversized: 
                system,assistant,user,responses,response, and the return value should be a string
            - for get_response, the responses is None and the response is the current response.
            - for on_tokens_oversized, the responses is the concatenated string of previous responses and the response is the current response.
                because the input is too large, on tokens oversized is called recursively, 
                it has become complicated to get back the concatenated responses.
            - for system,assistant,user, it will send the string needed to generate current response
        """
        self.model_class=ModelClass(self)
        self.use_cache=use_cache
        self.on_each_response=on_each_response

        self.model_class.use_cache=self.use_cache
        self.model_class.on_each_response=self.on_each_response
        pass
    def get_model_name(self):
        return self.model_class.get_model_name()
    def get_response(self,system,assistant,user):
        if self.on_each_response is None:
            return self.model_class.get_response(system,assistant,user)
        else:
            return self.on_each_response(system,assistant,user,None,self.model_class.get_response(system,assistant,user))
    def on_tokens_oversized(self,e,system,assistant,user):
        return self.model_class.on_tokens_oversized(e,system,assistant,user)
    pass

def get_best_available_llm(use_cache:bool=True,on_each_response:Callable[[str,str,str,str,str], str]=None):
    """
    Constructor for LLM class
    :param ModelClass: LLM_Base The class of the model to be used
    :param use_cache: bool Whether to use cache or not
    :param on_each_response: Callable[[str,str,str,str,str], str] A function to be called on each response
        - This function should take 5 string from get_response or in between on_tokens_oversized: 
            system,assistant,user,responses,response, and the return value should be a string
        - for get_response, the responses is None and the response is the current response.
        - for on_tokens_oversized, the responses is the concatenated string of previous responses and the response is the current response.
            because the input is too large, on tokens oversized is called recursively, 
            it has become complicated to get back the concatenated responses.
        - for system,assistant,user, it will send the string needed to generate current response
    """
    from .gpt import GPT
    model=GPT.model_picker()
    if model is not None:
        instant=LLM(GPT,use_cache,on_each_response)
        return instant
    # from .llama import LLaMA
    # instant=LLM(LLaMA,use_cache,on_each_response)
    return instant