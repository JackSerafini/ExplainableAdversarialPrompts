import os
from dotenv import load_dotenv

from huggingface_hub import login

from TokenSHAP.token_shap.base import LocalModel, HuggingFaceEmbeddings
from TokenSHAP.token_shap.token_shap import StringSplitter, TokenSHAP

load_dotenv()

hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
if not hf_api_key:
    raise RuntimeError("Missing HUGGINGFACE_API_KEY. Set it in your environment or .env file.")

login(hf_api_key)

model_path = "meta-llama/Llama-3.2-1B-Instruct"
local_model = LocalModel(model_name=model_path, max_new_tokens=10)
hf_embedding = HuggingFaceEmbeddings()
splitter = StringSplitter()
token_shap_local = TokenSHAP(model=local_model, splitter=splitter, vectorizer=hf_embedding, debug=True)

prompt4 = "What are the symptoms of COVID-19?"
df_local = token_shap_local.analyze(prompt4, sampling_ratio=0.0, print_highlight_text=True)
token_shap_local.print_colored_text()

print(df_local)
token_shap_local.highlight_text_background()
token_shap_local.print_colored_text()
token_shap_local.plot_colored_text()