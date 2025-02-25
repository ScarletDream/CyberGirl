import torch
from typing import Dict, Optional
from logging import getLogger
from .model_loader import ModelLoader

logger = getLogger(__name__)


class ResponseGenerator:
    def __init__(self, model_loader: ModelLoader, character_prompt: str):
        self.model, self.tokenizer = model_loader.load_model()
        self.character_prompt = character_prompt
        self.generation_config = model_loader.get_generation_config()

        # 缓存系统
        self.response_cache = {}
        self.cache_size = 100  # 缓存最近100条对话

    def _build_prompt(self, user_input: str, memory_context: Optional[Dict] = None) -> str:
        base_prompt = f"""<｜begin▁of▁sentence｜>
[角色设定]
{self.character_prompt}

[记忆上下文]
{self._format_memory(memory_context)}

[当前对话]
用户：{user_input}
助手："""

        return base_prompt.strip()

    def _format_memory(self, memory: Optional[Dict]) -> str:
        if not memory:
            return "暂无记忆"
        return "\n".join([f"- {k}: {v}" for k, v in memory.items()])

    def generate_response(self, user_input: str, memory_context: Optional[Dict] = None) -> str:
        try:
            # 检查缓存
            cache_key = hash(user_input)
            if cache_key in self.response_cache:
                return self.response_cache[cache_key]

            # 构建prompt
            full_prompt = self._build_prompt(user_input, memory_context)

            # Tokenization
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.model.device)

            # 生成响应
            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config
                )

            # 解码并后处理
            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            ).strip()

            # 缓存结果
            self._update_cache(cache_key, response)

            return response

        except torch.cuda.OutOfMemoryError:
            logger.warning("CUDA OOM detected, clearing cache")
            torch.cuda.empty_cache()
            return "请再说一遍好吗？我刚刚有点分心了..."

        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            return "哎呀，我的大脑突然短路了，能再重复一次吗？"

    def _update_cache(self, key: int, response: str):
        if len(self.response_cache) >= self.cache_size:
            # 移除最旧的条目
            self.response_cache.pop(next(iter(self.response_cache)))
        self.response_cache[key] = response

    def update_character_prompt(self, new_prompt: str):
        self.character_prompt = new_prompt
        logger.info("Character prompt updated")
