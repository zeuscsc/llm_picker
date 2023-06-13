
def reset_root_folder(path:str):
    global ROOT_FOLDER,\
        LLM_FOLDER
    ROOT_FOLDER=path
    LLM_FOLDER = f"{ROOT_FOLDER}llm"

reset_root_folder("./")