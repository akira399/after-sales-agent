"""DashScope 连通性测试：验证 API Key 是否可用。

不依赖任何第三方 LLM 框架，仅用 dashscope SDK 直连测试 chat/embedding。
用法：python scripts/check_api.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dotenv import load_dotenv
load_dotenv()

import dashscope
from dashscope import TextEmbedding, Generation

key = os.getenv("DASHSCOPE_API_KEY", "").strip()
if not key:
    print("[FAIL] 未在 .env 中找到 DASHSCOPE_API_KEY")
    sys.exit(1)

print(f"[INFO] key 前缀: {key[:10]}... 长度: {len(key)}")
dashscope.api_key = key

# 1. Chat 连通性（qwen 模型）
print("\n=== 1. Chat 测试 ===")
resp = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "用一句话自我介绍"}],
)
print(f"status_code = {resp.status_code}")
if resp.status_code == 200:
    print(f"回复: {resp.output.text[:80]}")
    print(f"用量: {getattr(resp, 'usage', None)}")
else:
    print(f"错误: {resp.message}")
    print("[HINT] 若提示需开通/额度不足 → 未充值或未开通模型服务")

# 2. Embedding 连通性（text-embedding-v4，RAG 向量检索必需）
print("\n=== 2. Embedding 测试 ===")
emb = TextEmbedding.call(model="text-embedding-v4", input="售后政策测试文本")
print(f"status_code = {emb.status_code}")
if emb.status_code == 200:
    dim = len(emb.output["embeddings"][0]["embedding"])
    print(f"OK, 维度 = {dim}")
else:
    print(f"错误: {emb.message}")

# 3. Rerank 连通性（qwen3-rerank，精排可选依赖）
print("\n=== 3. Rerank 测试 ===")
rr = dashscope.TextReRank.call(
    model="qwen3-rerank",
    query="七天无理由退货",
    documents=["规则A：支持七天无理由退货", "规则B：耳机进水不保修"],
    top_n=2,
)
print(f"status_code = {rr.status_code}")
if rr.status_code == 200:
    results = rr.output.get("results") or []
    print(f"OK, 结果数 = {len(results)}, 排序: {[r['index'] for r in results]}")
else:
    print(f"错误: {rr.message}")
