from llama_index.core.postprocessor import LLMRerank
from loguru import logger


# Utility function for common retrieval logic
def retrieve_common(index, query, top_k=3, with_reranker=False, **kwargs):
    if not index:
        logger.error("Index is not initialized. Please build or load an index first.")
        raise ValueError("Index is not initialized.")

    retriever = index.as_retriever(similarity_top_k=(top_k * 3 if with_reranker else top_k), **kwargs)

    nodes = retriever.retrieve(query)
    logger.info(f"Retrieved {len(nodes)} nodes.")

    if with_reranker:
        ranker = LLMRerank(choice_batch_size=top_k, top_n=top_k)
        ranked_nodes = ranker.postprocess_nodes(nodes, query_str=query)
        logger.info(f"Reranked nodes to {len(ranked_nodes)}")
        return ranked_nodes
    else:
        return nodes
