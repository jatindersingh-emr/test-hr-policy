
# config.py

import os
from typing import Optional, Dict

class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass

class Config:
    """
    Centralized configuration management for HR Policy Support Agent.
    Handles environment variable loading, API key management, LLM config,
    domain-specific settings, validation, error handling, and default values.
    """

    # Required environment variables for Azure AI Search and OpenAI
    REQUIRED_ENV_VARS = [
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
    ]

    # Domain-specific settings
    DOMAIN = "general"
    AGENT_NAME = "HR Policy Support Agent"
    POLICY_CACHE_TTL = 600  # seconds
    POLICY_CACHE_MAXSIZE = 100

    # LLM configuration
    LLM_CONFIG = {
        "provider": "openai",
        "model": "gpt-4.1",
        "temperature": 0.7,
        "max_tokens": 2000,
        "system_prompt": (
            "You are an expert HR Support Agent for the company. Your primary responsibility is to answer employee HR-related questions strictly using official company HR policies and documentation retrieved via Azure AI Search. Do not guess, fabricate, or provide information not explicitly supported by company HR policies. If the answer to a query is missing from the available policy documents or involves sensitive/confidential information, politely instruct the user to contact the HR department for further assistance. Always maintain a formal, professional tone and ensure responses are clear, accurate, and policy-compliant."
        ),
        "user_prompt_template": (
            "Please enter your HR-related question. Responses will be based strictly on company HR policies. If your question cannot be answered, you will be guided to contact HR."
        ),
        "few_shot_examples": [
            "Q: What is the company's leave policy? A: According to the company's HR policy, leave entitlements are outlined in Section 4 of the Employee Handbook. For detailed information, please refer to the official policy document or contact HR.",
            "Q: Can I get information about another employee's salary? A: I am unable to provide information regarding other employees' compensation. For sensitive or confidential queries, please contact the HR department directly."
        ]
    }

    # RAG pipeline settings
    RAG_CONFIG = {
        "enabled": True,
        "retrieval_service": "azure_ai_search",
        "embedding_model": "text-embedding-ada-002",
        "top_k": 5,
        "search_type": "vector_semantic"
    }

    @classmethod
    def load_env(cls) -> Dict[str, str]:
        """
        Load and validate required environment variables.
        Raises ConfigError if any required variable is missing.
        """
        env_vars = {}
        missing = []
        for key in cls.REQUIRED_ENV_VARS:
            value = os.getenv(key)
            if not value:
                missing.append(key)
            env_vars[key] = value
        if missing:
            raise ConfigError(f"Missing required environment variables: {', '.join(missing)}")
        return env_vars

    @classmethod
    def get_env(cls, key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        """
        Get an environment variable with optional default and required flag.
        Raises ConfigError if required and missing.
        """
        value = os.getenv(key, default)
        if required and not value:
            raise ConfigError(f"Missing required environment variable: {key}")
        return value

    @classmethod
    def get_llm_config(cls) -> Dict[str, any]:
        """Return LLM configuration dictionary."""
        return cls.LLM_CONFIG.copy()

    @classmethod
    def get_rag_config(cls) -> Dict[str, any]:
        """Return RAG pipeline configuration dictionary."""
        return cls.RAG_CONFIG.copy()

    @classmethod
    def get_domain(cls) -> str:
        """Return domain setting."""
        return cls.DOMAIN

    @classmethod
    def get_agent_name(cls) -> str:
        """Return agent name."""
        return cls.AGENT_NAME

    @classmethod
    def get_policy_cache_settings(cls) -> Dict[str, int]:
        """Return policy cache settings."""
        return {
            "ttl": cls.POLICY_CACHE_TTL,
            "maxsize": cls.POLICY_CACHE_MAXSIZE
        }

    @classmethod
    def validate(cls) -> None:
        """
        Validate all required configuration.
        Raises ConfigError if validation fails.
        """
        try:
            cls.load_env()
        except ConfigError as e:
            # Commented: raise for production, print for dev
            # raise
            print(f"Configuration validation error: {e}")

    @classmethod
    def get_default(cls, key: str) -> Optional[str]:
        """Return default value for a config key if available."""
        defaults = {
            "AZURE_SEARCH_ENDPOINT": "",
            "AZURE_SEARCH_API_KEY": "",
            "AZURE_SEARCH_INDEX_NAME": "",
            "AZURE_OPENAI_ENDPOINT": "",
            "AZURE_OPENAI_API_KEY": "",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "",
        }
        return defaults.get(key)

# Example usage:
# Config.validate()
# env = Config.load_env()
# llm_cfg = Config.get_llm_config()
# rag_cfg = Config.get_rag_config()
