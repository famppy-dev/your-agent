from typing import List
from pydantic import ConfigDict
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore, QueryBundle
from FlagEmbedding import FlagReranker


class BGERerankPostprocessor(BaseNodePostprocessor):

    model_config = ConfigDict(
        extra="allow",  # self.reranker 같은 동적 속성 허용
        arbitrary_types_allowed=True,  # FlagReranker 타입 허용
    )

    def __init__(
        self,
        top_k: int = 3,
        use_fp16: bool = True,
        reranker_model_path="BAAI/bge-reranker-v2-m3",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reranker_model = FlagReranker(
            reranker_model_path, trust_remote_code=True, use_fp16=use_fp16
        )
        self.top_k = top_k

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: QueryBundle | None = None,
    ):
        if not nodes:
            return nodes

        texts = [node.get_content() for node in nodes]
        pairs = [[query_bundle.query_str, text] for text in texts]
        scores = self.reranker_model.compute_score(pairs)

        for node, score in zip(nodes, scores):
            node.score = score
        return sorted(nodes, key=lambda x: x.score or 0, reverse=True)[:5]
