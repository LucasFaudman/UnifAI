from typing import Optional, Union, Sequence, Any, Literal, Mapping, Iterator, Generator

# import google.generativeai as genai
  
from unifai.exceptions import ( 
    ProviderUnsupportedFeatureError,
    ModelUnsupportedFeatureError
)
from unifai.types import (
    Embeddings,
    ResponseInfo, 
    Usage,
)
from ..base_adapters.google_base import GoogleAdapter
from ._base_embedder import Embedder, EmbeddingTaskTypeInput


GoogleEmbeddingTaskType = Literal[
    "RETRIEVAL_QUERY",      # Specifies the given text is a query in a search or retrieval setting.
    "RETRIEVAL_DOCUMENT",   # Specifies the given text is a document in a search or retrieval setting.
    "SEMANTIC_SIMILARITY",  # Specifies the given text is used for Semantic Textual Similarity (STS).
    "CLASSIFICATION",       # Specifies that the embedding is used for classification.
    "CLUSTERING",	        # Specifies that the embedding is used for clustering.
    "QUESTION_ANSWERING",	# Specifies that the query embedding is used for answering questions. Use RETRIEVAL_DOCUMENT for the document side.
    "FACT_VERIFICATION",	# Specifies that the query embedding is used for fact verification.
    "CODE_RETRIEVAL_QUERY"  # Specifies that the query embedding is used for code retrieval for Java and Python. 
]

class GoogleEmbedder(GoogleAdapter, Embedder):
    provider = "google"
    default_embedding_model = "text-embedding-004"
        
    model_embedding_dimensions = {
        "textembedding-gecko@001": 768,
        "textembedding-gecko-multilingual@001": 768,
        "textembedding-gecko@003": 768,
        "text-multilingual-embedding-002": 768,
        "text-embedding-004	": 768,
        "text-embedding-preview-0815": 768,
    }

    # Embeddings
    def validate_task_type(self,
                            model: str,
                            task_type: Optional[EmbeddingTaskTypeInput] = None,
                            task_type_not_supported: Literal["use_closest_supported", "raise_error"] = "use_closest_supported"
                            ) -> Optional[GoogleEmbeddingTaskType]:
        if task_type is None:
            return None
        if "@001" in model:
            raise ModelUnsupportedFeatureError(
                f"Model {model} does not support task_type specification for embeddings. "
                "See: https://cloud.google.com/vertex-ai/generative-ai/docs/embeddings/task-types"
            )

        if (_task_type := task_type.upper()) != "IMAGE":
            return _task_type # GoogleAI supports all input types except "image" # type: ignore
        if task_type_not_supported == "use_closest_supported":
            return "RETRIEVAL_DOCUMENT"
        raise ProviderUnsupportedFeatureError(
            f"Embedding task_type={task_type} is not supported by Google. "
             "Supported input types are 'retreival_query', 'retreival_document', "
             "'semantic_similarity', 'classification', 'clustering', 'question_answering', "
             "'fact_verification', and 'code_retreival_query'. Use 'retreival_document' for images with Google. "
             "Or use Cohere which supports embedding 'image' task_type."
        )


    def _get_embed_response(
            self,            
            input: Sequence[str],
            model: str,
            dimensions: Optional[int] = None,
            task_type: Optional[GoogleEmbeddingTaskType] = None,
            input_too_large: Literal[
                "truncate_end", 
                "truncate_start", 
                "raise_error"
                ] = "truncate_end",
            **kwargs
            ) -> Any:
        
        return self.client.embed_content(
            content=input,
            model=self.format_model_name(model),
            output_dimensionality=dimensions,
            task_type=task_type,
            # TODO - Preform this logic only when using Vertex AI models
            # https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/text-embeddings-api#parameter-list
            # auto_truncate=(input_too_large != "raise_error"),
            **kwargs
        )
    

    def _extract_embeddings(
            self,            
            response: Mapping,
            model: str,
            **kwargs
            ) -> Embeddings:
        
        return Embeddings(
            root=response["embedding"],
            response_info=ResponseInfo(
                model=model, 
                usage=Usage()
            )
        ) 