from typing import Optional

from openai._base_client import make_request_options

from ...types import QueryResult
from ..adapters.nvidia_adapter import NvidiaAdapter, TempBaseURL
from .._base_components._base_reranker import Reranker

# Point of this is to have a castable type subclassing OpenAI's BaseModel so it does not
# raise TypeError("Pydantic models must subclass our base model type, e.g. `from openai import BaseModel`")
from openai import BaseModel 
class NvidiaRerankItem(BaseModel):
    index: int
    logit: float

class NvidiaRerankResponse(BaseModel):
    rankings: list[NvidiaRerankItem]


class NvidiaReranker(NvidiaAdapter, Reranker):
    provider = "nvidia"    
    default_reranking_model = "nv-rerank-qa-mistral-4b:1"
  
    reranking_models = {
        "nvidia/nv-rerankqa-mistral-4b-v3",
        "nv-rerank-qa-mistral-4b:1"        
    }


    def _get_rerank_response(
        self,
        query: str,
        query_result: QueryResult,
        model: str,
        top_n: Optional[int] = None,               
        **kwargs
        ) -> NvidiaRerankResponse:

        assert query_result.texts, "No documents to rerank"
        body = {
            "model": model,
            "query": {"text": query},
            "passages": [{"text": document} for document in query_result.texts],
        }

        options = {}
        if (
            (extra_headers := kwargs.get("extra_headers"))
            or (extra_query := kwargs.get("extra_query"))
            or (extra_body := kwargs.get("extra_body"))
            or (timeout := kwargs.get("timeout"))
        ):
            options["options"] = make_request_options(
                extra_headers=extra_headers, 
                extra_query=extra_query, 
                extra_body=extra_body, 
                timeout=timeout
            )

        # Use the reranking model specific base URL (always required)
        model_base_url = self.model_base_urls.get(model)
        with TempBaseURL(self.client, model_base_url, self.default_base_url):
            return self.client.post(
                "/reranking",
                body=body,
                **options,
                cast_to=NvidiaRerankResponse, # See above
                stream=False,
                stream_cls=None,
            )

    def _extract_similarity_scores(
        self,
        response: NvidiaRerankResponse,
        **kwargs
        ) -> list[float]:
        return [item.logit for item in sorted(response.rankings, key=lambda item: item.index)]
    
    # def _extract_reranked_order(
    #     self,
    #     response,
    #     top_n: Optional[int] = None,
    #     **kwargs
    #     ) -> list[int]:
    #     return [item.index for item in response.rankings]    