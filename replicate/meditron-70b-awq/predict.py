# Prediction interface for Cog
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path, ConcatenateIterator
import transformers
import torch
from threading import Thread

MODEL_NAME = "TheBloke/meditron-70B-AWQ"

PROMPT_TEMPLATE="""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>question
{prompt}<|im_end|>
<|im_start|>answer
"""

PROMPT_TEMPLATE_2="""<|system|>
{system_message}</s>
<|user|>
{prompt}</s>
<|assistant|>
"""

HF_TOKEN = "hf_xyz"

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        self.model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME, token=HF_TOKEN)
        self.model.to(self.device)

    def predict(
        self,
        prompt: str = Input(description="Prompt"),
        prompt_template: str = Input(description="Prompt template", default=PROMPT_TEMPLATE),
        system_message: str = Input(description="System message", default="You are a helpful AI assistant trained in the medical domain"),
        max_new_tokens: int = Input(description="The maximum number of tokens the model should generate as output.", default=512),
        temperature: float = Input(description="Model temperature", default=0.2),
        top_p: float = Input(description="Top P", default=0.95),
        top_k: int = Input(description="Top K", default=50),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""
        streamer = transformers.TextIteratorStreamer(self.tokenizer, skip_prompt=True, timeout=20.0, skip_special_tokens=True)
        input_ids = self.tokenizer(prompt_template.format(prompt=prompt, system_message=system_message), return_tensors="pt").input_ids.to(self.device)

        with torch.inference_mode():
            kwargs = dict(input_ids=input_ids, streamer=streamer,
                # max_length=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample = True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens
            )
            thread = Thread(target=self.model.generate, kwargs=kwargs)
            thread.start()
            for new_text in streamer:
                yield new_text

            thread.join()
