import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from logging import getLogger

logger = getLogger(__name__)


class ModelLoader:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1", quantize=False):
        self.model_name = model_name
        self.quantize = quantize
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None
        self.generation_config = None

    def _get_device(self):
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _configure_quantization(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

    def load_model(self):
        try:
            # 量化配置
            quantization_config = self._configure_quantization() if self.quantize else None

            # 加载Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True
            )

            # 动态设备映射
            if self.device == "cuda":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    quantization_config=quantization_config,
                    attn_implementation="flash_attention_2"  # 使用Flash Attention加速
                )
            else:
                with init_empty_weights():
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        trust_remote_code=True
                    )
                self.model = load_checkpoint_and_dispatch(
                    self.model,
                    checkpoint=self.model_name,
                    device_map="auto",
                    no_split_module_classes=["DeepseekBlock"]
                )

            # 设置生成参数
            self.generation_config = GenerationConfig(
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            logger.info(f"Successfully loaded model on {self.device}")
            return self.model, self.tokenizer

        except Exception as e:
            logger.error(f"Model loading failed: {str(e)}")
            raise RuntimeError("Failed to initialize model") from e

    def get_generation_config(self):
        return self.generation_config
