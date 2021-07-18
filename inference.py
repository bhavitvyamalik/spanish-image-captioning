import jax
from PIL import Image
import numpy as np
import requests
from transformers import CLIPProcessor
from models.flax_clip_vision_marian.modeling_clip_vision_marian import FlaxCLIPVisionMarianMT
from transformers import MarianTokenizer, HfArgumentParser, TrainingArguments, is_tensorboard_available, set_seed

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="np", padding=True)
model= FlaxCLIPVisionMarianMT.from_clip_vision_marian_pretrained('openai/clip-vit-base-patch32', 'Helsinki-NLP/opus-mt-en-es')


def shift_tokens_right(input_ids: np.array, pad_token_id: int):  # TODO would this be required?
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = np.zeros(input_ids.shape, dtype=np.int64)
    shifted_input_ids[:, 1:] = input_ids[:, :-1]
    shifted_input_ids[:, 0] = pad_token_id
    return shifted_input_ids

tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
with tokenizer.as_target_tokenizer():
    input_token = tokenizer("2 cats sleeping in a bed", max_length=64, padding="max_length", return_tensors="np", truncation=True)

decoder_input_ids = shift_tokens_right(input_token["input_ids"], model.config.marian_config.pad_token_id)   #TODO change pad_token_id

outputs = model(inputs['pixel_values'], input_token['input_ids'], input_token['attention_mask'], decoder_input_ids)

# print(outputs["logits"])
# print(outputs.keys())

output_ids = model.generate(input_ids=inputs['pixel_values'], max_length=64, num_beams=4, early_stopping=True)
# print(output_ids)
# print(output_ids)
print(tokenizer.batch_decode(output_ids[0], skip_special_tokens=True, max_length=64))  # here 0 but change later