# LLM Picker
Just a simple picker for different LLM models

## Features
- [x] GPT-3.5
- [x] GPT-4
- [x] Self pre-train LLM without GPT

## Quick Usage
~~~shell
# Install llm_picker with pip
pip install git+https://github.com/zeuscsc/llm_picker.git
~~~
## Set Environment Variables
Unix:
~~~shell Unix
export OPENAI_API_KEY="your openai key" (Optional)
export TECKY_API_KEY="your tecky key" (Optional)
# Example
export OPENAI_API_KEY=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
export TECKY_API_KEY=12a34567-bc89-1011-12de-1234567x1234
~~~
Windows:
~~~shell Windows
$ENV:OPENAI_API_KEY="your tecky key" (Optional)
$ENV:TECKY_API_KEY="your openai key" (Optional)
# Example
$ENV:OPENAI_API_KEY="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"
$ENV:TECKY_API_KEY="12a34567-bc89-1011-12de-1234567x1234"
~~~