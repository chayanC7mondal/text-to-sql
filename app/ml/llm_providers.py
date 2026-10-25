from abc import ABC, abstractmethod
from typing import Tuple, Optional
import logging

from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2048
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    async def generate_sql(
        self,
        question: str,
        schema_context: str
    ) -> Tuple[str, float]:
        """
        Generate SQL from natural language question.

        Args:
            question: Natural language question
            schema_context: Database schema context

        Returns:
            Tuple of (generated_sql, confidence_score)
        """
        pass

    def _build_prompt(
        self,
        question: str,
        schema_context: str
    ) -> str:
        """Build the prompt for SQL generation."""
        return f"""You are an expert SQL query generator for PostgreSQL. Your task is to convert natural language questions into accurate SQL queries.

Database Schema:
{schema_context}

User Question:
{question}

Instructions:
1. Generate only the SQL query, no explanations
2. Use proper PostgreSQL syntax
3. Only use SELECT statements (no INSERT, UPDATE, DELETE, DROP, etc.)
4. Use appropriate JOINs when needed
5. Handle date filters correctly
6. Use proper aggregation functions
7. Return the SQL query as a single line

SQL Query:"""


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider for SQL generation."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.1,
        max_tokens: int = 2048
    ):
        super().__init__(model, temperature, max_tokens)
        self.api_key = api_key
        self._client = None

    async def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def generate_sql(
        self,
        question: str,
        schema_context: str
    ) -> Tuple[str, float]:
        """Generate SQL using OpenAI GPT."""
        try:
            client = await self._get_client()
            prompt = self._build_prompt(question, schema_context)

            response = await client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert SQL query generator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            sql = response.choices[0].message.content.strip()
            confidence = min(response.choices[0].finish_reason == "stop", 0.95)

            # Clean up the SQL
            sql = self._clean_sql(sql)

            logger.info(f"OpenAI generated SQL: {sql[:100]}...")
            return sql, confidence

        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}", exc_info=True)
            raise

    def _clean_sql(self, sql: str) -> str:
        """Clean up generated SQL."""
        # Remove markdown code blocks if present
        if sql.startswith("```"):
            sql = sql.split("```")[1]
            if sql.startswith("sql"):
                sql = sql[3:]
        sql = sql.strip()
        # Remove trailing semicolon if present
        if sql.endswith(";"):
            sql = sql[:-1]
        return sql


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider for SQL generation."""

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-opus-20240229",
        temperature: float = 0.1,
        max_tokens: int = 2048
    ):
        super().__init__(model, temperature, max_tokens)
        self.api_key = api_key
        self._client = None

    async def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            import anthropic
            self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
        return self._client

    async def generate_sql(
        self,
        question: str,
        schema_context: str
    ) -> Tuple[str, float]:
        """Generate SQL using Anthropic Claude."""
        try:
            client = await self._get_client()
            prompt = self._build_prompt(question, schema_context)

            response = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            sql = response.content[0].text.strip()
            confidence = min(response.stop_reason == "end_turn", 0.95)

            # Clean up the SQL
            sql = self._clean_sql(sql)

            logger.info(f"Anthropic generated SQL: {sql[:100]}...")
            return sql, confidence

        except Exception as e:
            logger.error(f"Anthropic generation failed: {e}", exc_info=True)
            raise

    def _clean_sql(self, sql: str) -> str:
        """Clean up generated SQL."""
        # Remove markdown code blocks if present
        if sql.startswith("```"):
            sql = sql.split("```")[1]
            if sql.startswith("sql"):
                sql = sql[3:]
        sql = sql.strip()
        # Remove trailing semicolon if present
        if sql.endswith(";"):
            sql = sql[:-1]
        return sql


class LocalProvider(BaseLLMProvider):
    """Local provider using smaller models for faster, cheaper generation."""

    def __init__(
        self,
        model: str = "distilbert-base-uncased",
        temperature: float = 0.1,
        max_tokens: int = 2048
    ):
        super().__init__(model, temperature, max_tokens)
        self._model = None
        self._tokenizer = None

    async def _load_model(self):
        """Lazy load the local model."""
        if self._model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                logger.info("Loading local model...")

                # Use a smaller model for local generation
                model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

                self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    load_in_8bit=True
                )

                logger.info("Local model loaded successfully")

            except Exception as e:
                logger.error(f"Failed to load local model: {e}", exc_info=True)
                raise RuntimeError("Local model not available")

    async def generate_sql(
        self,
        question: str,
        schema_context: str
    ) -> Tuple[str, float]:
        """Generate SQL using local model."""
        try:
            import torch
            await self._load_model()

            prompt = self._build_prompt(question, schema_context)

            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id
                )

            generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            sql = generated_text.split("SQL Query:")[-1].strip()
            confidence = 0.85  # Lower confidence for local model

            # Clean up the SQL
            sql = self._clean_sql(sql)

            logger.info(f"Local model generated SQL: {sql[:100]}...")
            return sql, confidence

        except Exception as e:
            logger.error(f"Local model generation failed: {e}", exc_info=True)
            raise

    def _clean_sql(self, sql: str) -> str:
        """Clean up generated SQL."""
        sql = sql.strip()
        # Remove trailing semicolon if present
        if sql.endswith(";"):
            sql = sql[:-1]
        return sql
