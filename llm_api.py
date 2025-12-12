from loguru import logger
from pydantic import BaseModel
from app.config.config import settings
from z import h

class AI:
    def __init__(
        self,
        llm_provider: str = settings.llm_provider,
        llm_api_key: str = settings.llm_api_key,
        llm_base_url: str | None = None,
        llm_model_name: str | None = settings.llm_model_name,
    ):
        self.llm_provider = llm_provider
        self.llm_api_key = llm_api_key
        self.llm_base_url = llm_base_url
        self.llm_model_name = llm_model_name

        # Initialize the appropriate client based on provider
        if self.llm_provider == "gigachat":
            try:
                from langchain_gigachat import GigaChat # type: ignore

                self.ai_client = GigaChat(
                    credentials=llm_api_key, verify_ssl_certs=False
                )
            except ImportError:
                raise ImportError(
                    "langchain-gigachat is not installed. Run: uv add langchain-gigachat"
                )

        elif self.llm_provider == "openai":
            try:
                from langchain_openai import ChatOpenAI  # type: ignore

                self.ai_client = ChatOpenAI(api_key=llm_api_key, base_url=llm_base_url)
            except ImportError:
                raise ImportError(
                    "langchain-openai is not installed. Run: uv add langchain-openai"
                )

        elif self.llm_provider == "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore

                self.ai_client = ChatGoogleGenerativeAI(
                    google_api_key=llm_api_key,
                    base_url=llm_base_url,
                    model=llm_model_name,
                )
            except ImportError:
                raise ImportError(
                    "langchain-google-genai is not installed. Run: uv add langchain-google-genai"
                )

        else:
            raise ValueError(f"Unsupported LLM provider: {llm_provider}")

    async def structured_response(
        self,
        system_prompt: str,
        structured_output: BaseModel,
        messages: list[dict],
        temperature: float = 1,
        max_tokens: int = 1000,
    ) -> BaseModel:
        try:
            structured_llm = self.ai_client.with_structured_output(
                structured_output, method="json_schema"
            )

            input_data = [{"role": "system", "content": system_prompt}, *messages]

            response = await structured_llm.ainvoke(
                input=input_data,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            return response

        except Exception as e:
            logger.exception(f"Error in structured_response: {e}")
