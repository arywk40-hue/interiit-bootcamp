"""Small byte-level multi-task network trained from random initialization.

The module contains no checkpoint loader. UTF-8 bytes keep spelling variants,
emoji, and punctuation without learning a subword vocabulary from private text.
"""
from dataclasses import asdict, dataclass
import unicodedata

import torch
from torch import nn
from torch.nn import functional as F


PAD, CLS, SEP, MASK = 0, 1, 2, 3
TASK_IDS = {"sentiment": 4, "intent": 5, "qa": 6, "summarization": 7}
BYTE_OFFSET = 8
VOCAB_SIZE = BYTE_OFFSET + 256


def _symbol_class(char):
    group = unicodedata.category(char)[0]
    return {"L": 1, "N": 2, "Z": 3, "P": 4, "S": 5, "M": 6}.get(group, 7)


def _append_text(ids, symbols, segments, text, segment, budget):
    for char in unicodedata.normalize("NFC", text):
        encoded = char.encode("utf-8")
        if len(ids) + len(encoded) > budget:
            break
        category = _symbol_class(char)
        ids.extend(BYTE_OFFSET + byte for byte in encoded)
        symbols.extend([category] * len(encoded))
        segments.extend([segment] * len(encoded))


def encode_bytes(text, task, max_length=512, context=""):
    """Encode one input without lowercasing or discarding Unicode symbols."""
    if task not in TASK_IDS:
        raise ValueError(f"unknown task: {task}")
    if max_length < 8:
        raise ValueError("max_length must be at least 8")
    ids, symbols, segments = [CLS, TASK_IDS[task]], [0, 0], [0, 0]
    # Keep room for QA context even when a question is unexpectedly long.
    text_budget = (min(max_length - 3, 2 + min(128, max_length // 3))
                   if task == "qa" else max_length - 1)
    _append_text(ids, symbols, segments, text, 0, text_budget)
    ids.append(SEP); symbols.append(0); segments.append(0)
    context_start = len(ids)
    if task == "qa":
        _append_text(ids, symbols, segments, context, 1, max_length - 1)
        ids.append(SEP); symbols.append(0); segments.append(1)
    return {"input_ids": ids, "symbol_ids": symbols, "segment_ids": segments,
            "attention_mask": [1] * len(ids), "context_start": context_start}


def collate_bytes(rows):
    """Pad encoded dictionaries into batch-first integer tensors."""
    width = max(len(row["input_ids"]) for row in rows)
    batch = {}
    for key in ("input_ids", "symbol_ids", "segment_ids", "attention_mask"):
        batch[key] = torch.tensor([row[key] + [0] * (width - len(row[key])) for row in rows],
                                  dtype=torch.long)
    return batch


@dataclass
class ByteMultiTaskConfig:
    d_model: int = 256
    layers: int = 6
    heads: int = 8
    feed_forward: int = 768
    dropout: float = 0.1
    max_positions: int = 512
    intent_labels: int = 57
    downsample_stages: int = 2


class ByteMultiTaskModel(nn.Module):
    """Shared byte encoder with classification and extractive task heads."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or ByteMultiTaskConfig()
        c, d = self.config, self.config.d_model
        if d % c.heads:
            raise ValueError("d_model must be divisible by heads")
        self.byte_embedding = nn.Embedding(VOCAB_SIZE, d, padding_idx=PAD)
        self.symbol_embedding = nn.Embedding(8, d, padding_idx=0)
        self.segment_embedding = nn.Embedding(2, d)
        self.position_embedding = nn.Embedding(c.max_positions, d)
        self.input_norm = nn.LayerNorm(d)
        self.stem = nn.ModuleList([
            nn.Sequential(nn.Conv1d(d, d, 5, stride=2, padding=2, groups=d),
                          nn.Conv1d(d, d, 1), nn.GELU())
            for _ in range(c.downsample_stages)
        ])
        layer = nn.TransformerEncoderLayer(d, c.heads, c.feed_forward, c.dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, c.layers,
                                             norm=nn.LayerNorm(d),
                                             enable_nested_tensor=False)
        self.sentiment_head = nn.Linear(d, 3)
        self.intent_head = nn.Linear(d, c.intent_labels)
        self.summary_head = nn.Linear(d, 1)
        self.qa_head = nn.Linear(2 * d, 2)
        self.answerable_head = nn.Linear(d, 1)
        self.mlm_bias = nn.Parameter(torch.zeros(VOCAB_SIZE))

    def _encode(self, input_ids, symbol_ids, segment_ids, attention_mask):
        length = input_ids.shape[1]
        if length > self.config.max_positions:
            raise ValueError("input exceeds configured maximum")
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0)
        raw = (self.byte_embedding(input_ids) + self.symbol_embedding(symbol_ids)
               + self.segment_embedding(segment_ids)
               + self.position_embedding(positions))
        raw = self.input_norm(raw)
        raw = raw * attention_mask.unsqueeze(-1)
        hidden, mask = raw.transpose(1, 2), attention_mask.float().unsqueeze(1)
        for stem in self.stem:
            hidden = stem(hidden)
            mask = F.max_pool1d(mask, 5, stride=2, padding=2)
        hidden = hidden.transpose(1, 2)
        valid = mask.squeeze(1).bool()
        hidden = self.encoder(hidden, src_key_padding_mask=~valid)
        pooled = (hidden * valid.unsqueeze(-1)).sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        return raw, hidden, pooled

    def forward(self, task, input_ids, symbol_ids, segment_ids, attention_mask):
        raw, hidden, pooled = self._encode(input_ids, symbol_ids, segment_ids,
                                           attention_mask)
        if task == "sentiment":
            return {"logits": self.sentiment_head(pooled)}
        if task == "intent":
            return {"logits": self.intent_head(pooled)}
        if task == "summarization":
            return {"logits": self.summary_head(pooled).squeeze(-1)}
        if task == "qa":
            contextual = F.interpolate(hidden.transpose(1, 2), size=input_ids.shape[1],
                                       mode="linear", align_corners=False).transpose(1, 2)
            span = self.qa_head(torch.cat((raw, contextual), dim=-1))
            context_mask = (segment_ids == 1) & attention_mask.bool()
            span = span.masked_fill(~context_mask.unsqueeze(-1), -1e4)
            return {"start_logits": span[..., 0], "end_logits": span[..., 1],
                    "answerable_logits": self.answerable_head(pooled).squeeze(-1)}
        if task == "masked_byte":
            contextual = F.interpolate(hidden.transpose(1, 2), size=input_ids.shape[1],
                                       mode="linear", align_corners=False).transpose(1, 2)
            return {"logits": F.linear(contextual, self.byte_embedding.weight, self.mlm_bias)}
        raise ValueError(f"unknown task: {task}")

    def parameter_report(self):
        unique = sum(parameter.numel() for parameter in self.parameters())
        return {"unique_parameters": unique,
                "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "config": asdict(self.config), "random_initialization": True}
