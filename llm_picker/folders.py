
def reset_root_folder(path:str):
    global ROOT_FOLDER,\
        LLM_RESPONSE_CACHE_FOLDER,\
        LLM_FOLDER
    ROOT_FOLDER=path
    LLM_RESPONSE_CACHE_FOLDER = f"{ROOT_FOLDER}llm_response_cache"
    LLM_FOLDER = f"{ROOT_FOLDER}llm"

reset_root_folder("./")