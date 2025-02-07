

#whisk run -n nats://nats.playground.kitchenai.dev:4222 app:kitchen

from whisk.kitchenai_sdk.kitchenai import KitchenAIApp
from whisk.kitchenai_sdk.schema import (
    WhiskQuerySchema,
    WhiskQueryBaseResponseSchema,
    WhiskStorageSchema,
    WhiskStorageResponseSchema,
    TokenCountSchema
)
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
import tiktoken

try:
    from llama_index.llms.openai import OpenAI
except ImportError:
    raise ImportError("Please install llama-index to use this example")

import logging
token_counter = TokenCountingHandler(
    tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode
)
# Initialize LLM and embeddings
llm = OpenAI(model="gpt-3.5-turbo")

kitchen = KitchenAIApp(namespace="my-remote-client")

# pip install llama-index
logger = logging.getLogger(__name__)


@kitchen.query.handler("query-2")
async def query_handler(data: WhiskQuerySchema) -> WhiskQueryBaseResponseSchema:
    """Query handler with RAG"""
    # Create filters from metadata if provided
    try:
        # Execute query using llm.acomplete
        response = await llm.acomplete(data.query)
        print("IM Making a change here")

        # Get token counts
        token_counts = {
            "llm_prompt_tokens": token_counter.prompt_llm_token_count,
            "llm_completion_tokens": token_counter.completion_llm_token_count,
            "total_llm_tokens": token_counter.total_llm_token_count
        }
        token_counter.reset_counts()

        return WhiskQueryBaseResponseSchema.from_llm_invoke(
            data.query,
            response.text,
            token_counts=TokenCountSchema(**token_counts),
            metadata={"token_counts": token_counts, **data.metadata} if data.metadata else {"token_counts": token_counts}
        )
    except Exception as e:
        logger.error(f"Error in query handler: {str(e)}")
        raise


@kitchen.query.handler("query")
async def query_handler(data: WhiskQuerySchema) -> WhiskQueryBaseResponseSchema:
    """Query handler"""

    response = await llm.acomplete(data.query)

    print(response)
    print("heree")


    return WhiskQueryBaseResponseSchema.from_llm_invoke(
        data.query,
        response.text,
    )


@kitchen.storage.handler("storage")
async def storage_handler(data: WhiskStorageSchema) -> WhiskStorageResponseSchema:
    """Storage handler"""
    print("storage handler")

    return WhiskStorageResponseSchema(
        id=data.id,
        data=data.data,
        metadata=data.metadata,
    )

@kitchen.storage.on_delete("storage")
async def storage_delete_handler(data: WhiskStorageSchema) -> None:
    """Storage delete handler"""
    print("storage delete handler")
    print(data)

