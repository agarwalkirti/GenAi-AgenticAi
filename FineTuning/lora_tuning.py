#!pip install -q -U keras-nlp
#!pip install -q -U keras>=3
#!pip install -q keras-nlp

import keras_nlp
print(keras_nlp.__version__)

#downloading dataset
#!wget -O databricks-dolly-15k.jsonl https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl # Changed URL to download raw file

import os
from google.colab import userdata
from datasets import load_dataset
from peft import LoraConfig
import keras
import keras_nlp
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, GemmaTokenizer

#userdata.get is a Colab API. If not using colab set env variables in system
os.environ["HUGGING_FACE_API_KEY"] = userdata.get('HUGGING_FACE_API_KEY')
os.environ["GOOGLE_API_KEY"] = userdata.get('GOOGLE_API_KEY')
os.environ["KAGGLE_USERNAME"] = userdata.get('KAGGLE_USERNAME')
os.environ["KAGGLE_KEY"] = userdata.get('KAGGLE_KEY')

#using keras framework we can run workflows on one of the three backends: jax,tensorflow or pytorch
os.environ["KERAS_BACKEND"] = "jax" #selecting jax as backend
#avoid memory fragmemtation on jax backend
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"

import json

data = []
with open("databricks-dolly-15k.jsonl", 'r') as file:
    for line in file:
      features = json.loads(line)
      # filter out examples with cotext, to keep it simple.
      if features['context']:
        continue
      # format the entire example as a single string
      # Define the prompt structure
      template = "Instruction:\n{instruction}\n\nResponse:\n{response}"#\n\nOutput:\n{output}
      data.append(template.format(**features))

#only use 1000 training examples, to keep it fast
data = data[:1000]

data

#loading model
gemma_lm = keras_nlp.models.GemmaCausalLM.from_preset("gemma_2b_en")
gemma_lm.summary()

sampler = keras_nlp.samplers.TopKSampler(k=5,seed=2)

#europe trip prompt
prompt = template.format(
    instruction="What should I do on a trip to Europe?",
    response="",
)
gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=256))

#EL15 photosynthesis prompt
prompt = template.format(
    instruction="Explain the process of photosynthesis in a way that a child could understand.",
    response="",
)

print(gemma_lm.generate(prompt, max_length=256))

#enable lora for the model and set the lora rank to 4
gemma_lm.backbone.enable_lora(rank=4)
gemma_lm.summary()

#limit the input sequence length to 512(to control memory usage).
gemma_lm.preprocessor.sequence_length = 512
# use adamW(a common optimizer for transformer models)
optimizer = keras.optimizers.AdamW(
    learning_rate=5e-5,
    weight_decay=0.01,
    #epsilon=1e-05,
    #amsgrad=False,
    #jit_compile=True
)

#exclude layernorm and bias terms from decay.
optimizer.exclude_from_weight_decay(var_names=['bias','scale'])

gemma_lm.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=optimizer,
    weighted_metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
)
gemma_lm.fit(data, epochs=1,batch_size=1)

#Mixed precision fine-tuning on nvidia GPUs

#inference after fine-tuning
#europe trip prompt
prompt = template.format(
    instruction="What should I do on a trip to Europe?",
    response="",
)
sampler = keras_nlp.samplers.TopKSampler(k=5,seed=2)
gemma_lm.compile(sampler=sampler)
print(gemma_lm.generate(prompt, max_length=256))

#EL15 photosynthesis prompt
prompt = template.format(
    instruction="Explain the process of photosynthesis in a way that a child could understand.",
    response="",
)

print(gemma_lm.generate(prompt, max_length=256))

