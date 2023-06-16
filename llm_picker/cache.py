from .llm import LLM_Base
from .gpt import ON_TOKENS_OVERSIZED,ON_RESULT_FILTERED

class Cache(LLM_Base):
    model_name:str

    def __init__(self,model_name):
        self.model_name=model_name
        pass
    
    def get_model_name(self):
        return self.model_name
    def get_response(self,system,assistant,user):
        model=self.get_model_name()
        if model is None:
            raise Exception("Model name is None")
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
                if (len(response_cache["choices"])==0 or
                    "message" not in response_cache["choices"][0] or
                    "content" not in response_cache["choices"][0]["message"]):
                    return None
        return None
    pass