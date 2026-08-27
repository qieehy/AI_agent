"""D20/D21 验收：上传 PDF 问内容，回答带引用来源。

用法：
    python rag_demo.py <pdf路径>

流程：加载 PDF -> 切块 -> 双路召回（BGE 向量 + BM25 关键词）-> RRF 融合
      -> cross-encoder 精排 -> LLM 问答。
依赖全部由工厂组装（组合根），demo 不 new 任何具体类。
回答中的 [编号] 是 LLM 引用的上下文块，对应的来源脚注打印在下方（D21 citation）。
来源里的 score 是精排分（logits，只有相对大小有意义）。
"""
import asyncio
import sys

from llm import AsyncLLMClient
from rag import (
    BM25Index,
    create_embedding_service,
    create_rag_pipeline,
    create_reranker,
    create_vector_store,
)

# 中文 Windows 管道 stdout 默认 GBK，LLM 回答可能含 GBK 编不了的字符（如 U+202F），
# 统一重配为 UTF-8；errors="replace" 兜底避免任何字符让脚本崩溃。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> None:
    if len(sys.argv) != 2:
        print("用法: python rag_demo.py <pdf路径>")
        sys.exit(1)

    # 组合根：embedder / store / llm 互不知晓，pipeline 只认识接口
    embedding_service = create_embedding_service()
    vector_store = create_vector_store()
    llm_client = AsyncLLMClient()
    pipeline = create_rag_pipeline(
        embedding_service,
        vector_store,
        llm_client,
        keyword_index=BM25Index(),
        reranker=create_reranker(),
    )

    path = sys.argv[1]
    count = pipeline.index_pdf(path)
    print(f"已索引 {path}：{count} 块")
    print("输入问题提问，输入 /quit 退出。")

    while True:
        try:
            question = input("\n问: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question == "/quit":
            break
        answer = asyncio.run(pipeline.ask(question))
        print(f"答: {answer.text}")

        # 来源脚注：[n] 与 LLM 回答里的引用编号一一对应
        if answer.sources:
            print("\n来源:")
            for source in answer.sources:
                preview = source.text.replace("\n", " ")[:80]
                print(
                    f"  [{source.number}] {source.source} 第 {source.page} 页 | {preview}..."
                )


if __name__ == "__main__":
    main()
