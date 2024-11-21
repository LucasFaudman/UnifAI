from typing import Optional

from openai._base_client import make_request_options

from ...types import QueryResult
from ..base_adapters.nvidia_base import NvidiaAdapter, TempBaseURL
from ._base_reranker import Reranker

# Entire point of this is to have a castable type subclassing OpenAI's BaseModel so it does not
# raise TypeError("Pydantic models must subclass our base model type, e.g. `from openai import BaseModel`")
# and ensure it is only created once and at runtime (not recreated every call and only created if needed)
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

        assert query_result.documents, "No documents to rerank"
        body = {
            "model": model,
            "query": {"text": query},
            "passages": [{"text": document} for document in query_result.documents],
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


    def _extract_reranked_order(
        self,
        response,
        top_n: Optional[int] = None,
        **kwargs
        ) -> list[int]:
        return [item.index for item in response.rankings]
  