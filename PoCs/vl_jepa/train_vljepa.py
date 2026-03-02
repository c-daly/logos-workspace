#!/usr/bin/env python3
"""VL-JEPA Training with Real Models"""
import torch
from transformers import AutoModel, AutoImageProcessor, LlamaForCausalLM
print("Loading models from HuggingFace...")
