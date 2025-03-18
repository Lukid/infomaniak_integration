from typing import List, Optional, Type, Any, Mapping
from cat.mad_hatter.decorators import tool, hook, plugin
from pydantic import BaseModel, ConfigDict, SecretStr
from datetime import datetime, date
from cat.factory.llm import LLMSettings
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.messages import AIMessage
from langchain.embeddings.base import Embeddings
import requests
import numpy as np

class InfomaniakChatModel(BaseChatModel):
    """Infomaniak Chat model."""
    
    api_key: str
    product_id: str
    model: str = "llama3"
    temperature: float = 0.7
    base_url: str = "https://api.infomaniak.com/1/ai"
    streaming: bool = True

    @property
    def _llm_type(self) -> str:
        return "infomaniak"

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[Any] = None, **kwargs: Any) -> ChatResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Convert messages to the format expected by Infomaniak API
        formatted_messages = []
        for message in messages:
            # Ensure content is a string
            content = str(message.content) if message.content is not None else ""
            
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                # Skip unknown message types
                continue
                
            formatted_messages.append({
                "role": role,
                "content": content
            })

        # Ensure we have at least one message
        if not formatted_messages:
            raise ValueError("No valid messages to send to the API")

        try:
            url = f"{self.base_url}/{self.product_id}/openai/chat/completions"
            
            # Debug print
            print(f"Sending request to {url}")
            print(f"Messages: {formatted_messages}")
            
            response = requests.post(
                url,
                json={
                    "messages": formatted_messages,
                    "model": self.model,
                    "temperature": self.temperature,
                    "stream": self.streaming
                },
                headers=headers
            )
            
            # Debug print
            print(f"Response status: {response.status_code}")
            print(f"Response text: {response.text}")
            
            if response.status_code != 200:
                raise ValueError(f"API returned status code {response.status_code}: {response.text}")
            
            response_json = response.json()
            message_content = response_json['choices'][0]['message']['content']
            
            # Create a ChatGeneration object
            message = AIMessage(content=message_content)
            generation = ChatGeneration(message=message)
            
            # Return a ChatResult
            return ChatResult(generations=[generation])
            
        except Exception as e:
            raise ValueError(f"Error calling Infomaniak API: {str(e)}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "base_url": self.base_url,
            "product_id": self.product_id,
            "model": self.model,
            "temperature": self.temperature
        }

class InfomaniakConfig(LLMSettings):
    """The configuration for the Infomaniak plugin."""

    api_key: SecretStr
    product_id: str
    model: str = "llama3"
    # temperature: float = 0.7
    base_url: str = "https://api.infomaniak.com/1/ai"
    # streaming: bool = True
    
    _pyclass: Type = InfomaniakChatModel

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Infomaniak AI",
            "description": "Configuration for Infomaniak AI language models",
            "link": "https://www.infomaniak.com/en/ai",
        }
    )

class InfomaniakEmbeddings(Embeddings):
    """Infomaniak Embeddings model."""
    
    api_key: str
    product_id: str
    model: str = "bge_multilingual_gemma2"  # Default model for embeddings
    base_url: str = "https://api.infomaniak.com/1/ai"

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = []
        for text in texts:
            embedding = self.embed_query(text)
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a query."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        url = f"{self.base_url}/{self.product_id}/openai/v1/embeddings"
        
        try:
            response = requests.post(
                url,
                json={
                    "input": text,
                    "model": self.model
                },
                headers=headers
            )
            
            if response.status_code != 200:
                raise ValueError(f"API returned status code {response.status_code}: {response.text}")
            
            response_json = response.json()
            return response_json['data'][0]['embedding']
            
        except Exception as e:
            raise ValueError(f"Error calling Infomaniak Embeddings API: {str(e)}")

class InfomaniakEmbeddingsConfig(BaseModel):
    """Configuration for Infomaniak Embeddings."""
    api_key: SecretStr
    product_id: str
    model: str = "bge_multilingual_gemma2"
    base_url: str = "https://api.infomaniak.com/1/ai"

    model_config = ConfigDict(
        json_schema_extra={
            "humanReadableName": "Infomaniak Embeddings",
            "description": "Configuration for Infomaniak AI embeddings models",
            "link": "https://www.infomaniak.com/en/ai",
        }
    )

@hook
def factory_allowed_llms(allowed, cat) -> List:
    allowed.append(InfomaniakConfig)
    return allowed

@hook
def factory_allowed_embedders(allowed, cat) -> List:
    """Add Infomaniak embeddings to allowed embedders."""
    allowed.append(InfomaniakEmbeddingsConfig)
    return allowed 