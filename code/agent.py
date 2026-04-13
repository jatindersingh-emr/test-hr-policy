

class Response:
    def __init__(self, **kwargs):
        self.status_code = 200
        self._data = kwargs
    def json(self):
        return self._data
try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {'check_credentials_output': True,
 'check_jailbreak': True,
 'check_output': True,
 'check_pii_input': False,
 'check_toxic_code_output': True,
 'check_toxicity': True,
 'content_safety_enabled': True,
 'content_safety_severity_threshold': 3,
 'runtime_enabled': True,
 'sanitize_pii': False}


import os
import time as _time
import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from functools import lru_cache

from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, ValidationError, Field
from cachetools import TTLCache
from loguru import logger

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
import openai
import jwt

# Observability wrappers are injected by runtime; do not import or decorate @trace_agent manually.

# ------------------- Configuration Management -------------------

class Config:
    """Centralized configuration loader for environment variables."""
    @staticmethod
    def get(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
        value = os.getenv(key, default)
        if required and not value:
            raise ValueError(f"Missing required environment variable: {key}")
        return value

    @classmethod
    def validate(cls) -> None:
        """Validate all required environment variables for Azure AI Search and OpenAI."""
        required_keys = [
            "AZURE_SEARCH_ENDPOINT",
            "AZURE_SEARCH_API_KEY",
            "AZURE_SEARCH_INDEX_NAME",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_EMBEDDING_DEPLOYMENT"
        ]
        missing = [k for k in required_keys if not os.getenv(k)]
        if missing:
            raise EnvironmentError(f"Missing required environment variables: {missing}")

# ------------------- Logging Configuration -------------------

logger.remove()
logger.add(lambda msg: print(msg, end=""), level="INFO")

# ------------------- Base Component -------------------

class BaseComponent:
    """Base class for all components, providing logging and error handling."""
    def __init__(self):
        self.logger = logger

    @trace_agent(agent_name='HR Policy Support Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def log_info(self, message: str, **kwargs):
        self.logger.info(f"{self.__class__.__name__}: {message} | {kwargs}")

    @trace_agent(agent_name='HR Policy Support Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def log_error(self, message: str, **kwargs):
        self.logger.error(f"{self.__class__.__name__}: {message} | {kwargs}")

    @trace_agent(agent_name='HR Policy Support Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def log_warning(self, message: str, **kwargs):
        self.logger.warning(f"{self.__class__.__name__}: {message} | {kwargs}")

# ------------------- Input/Output Models -------------------

class UserQueryModel(BaseModel):
    user_input: str = Field(..., max_length=50000)
    user_context: Optional[Dict[str, Any]] = None

    @field_validator("user_input")
    @classmethod
    def validate_user_input(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Input cannot be empty.")
        if len(v) > 50000:
            raise ValueError("Input exceeds maximum allowed length (50,000 characters).")
        return v.strip()

class AgentResponseModel(BaseModel):
    success: bool
    response: Optional[str] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    tips: Optional[str] = None

# ------------------- Utility: PII Masking -------------------

@with_content_safety(config=GUARDRAILS_CONFIG)
def mask_pii(text: str) -> str:
    # Simple PII masking (expand as needed)
    import re
    # Mask emails
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[REDACTED_EMAIL]', text)
    # Mask phone numbers
    text = re.sub(r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b', '[REDACTED_PHONE]', text)
    # Mask SSN
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_SSN]', text)
    return text

# ------------------- Authentication Service -------------------

class AuthenticationService(BaseComponent):
    """Performs SSO authentication, manages tokens, enforces session timeouts."""
    SECRET_KEY = Config.get("AUTH_SECRET_KEY", "supersecretkey")
    ALGORITHM = "HS256"
    SESSION_TIMEOUT = 3600  # seconds

    def authenticate(self, user_credentials: Dict[str, Any]) -> str:
        """Authenticate user and return JWT token."""
        with trace_step_sync(
            "authenticate", step_type="process",
            decision_summary="Authenticate user and issue token",
            output_fn=lambda r: f"token_issued={bool(r)}"
        ) as step:
            try:
                # For demo: accept any username/password, issue JWT
                username = user_credentials.get("username")
                if not username:
                    raise ValueError("Missing username.")
                payload = {
                    "sub": username,
                    "iat": int(_time.time()),
                    "exp": int(_time.time()) + self.SESSION_TIMEOUT
                }
                token = jwt.encode(payload, self.SECRET_KEY, algorithm=self.ALGORITHM)
                step.capture(token)
                return token
            except Exception as e:
                self.log_error("Authentication failed", error=str(e))
                step.capture(None)
                raise HTTPException(status_code=401, detail="Authentication failed.")

    def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token and return payload."""
        with trace_step_sync(
            "validate_token", step_type="process",
            decision_summary="Validate JWT token",
            output_fn=lambda r: f"valid={bool(r)}"
        ) as step:
            try:
                payload = jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
                step.capture(payload)
                return payload
            except jwt.ExpiredSignatureError:
                self.log_warning("Token expired")
                step.capture(None)
                raise HTTPException(status_code=401, detail="Session expired.")
            except Exception as e:
                self.log_error("Token validation failed", error=str(e))
                step.capture(None)
                raise HTTPException(status_code=401, detail="Invalid token.")

# ------------------- Embedding Service -------------------

class EmbeddingService(BaseComponent):
    """Generates embeddings for user queries using Azure OpenAI embedding model."""
    def __init__(self):
        super().__init__()
        self._client = None

    def _get_client(self):
        if not self._client:
            api_key = Config.get("AZURE_OPENAI_API_KEY", required=True)
            endpoint = Config.get("AZURE_OPENAI_ENDPOINT", required=True)
            self._client = openai.AzureOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for user query."""
        retries = 0
        max_retries = 3
        delay = 1
        model = Config.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        while retries < max_retries:
            async with trace_step(
                "embed_text", step_type="tool_call",
                decision_summary="Generate embedding for user query",
                output_fn=lambda r: f"embedding_dim={len(r) if r else 0}"
            ) as step:
                try:
                    client = self._get_client()
                    resp = await client.embeddings.create(
                        input=text,
                        model=model
                    )
                    embedding = resp.data[0].embedding
                    step.capture(embedding)
                    return embedding
                except Exception as e:
                    self.log_warning(f"Embedding API error: {e}")
                    retries += 1
                    await asyncio.sleep(delay * (2 ** retries))
        self.log_error("Failed to generate embedding after retries.")
        raise HTTPException(status_code=500, detail="Embedding service unavailable.")

# ------------------- Retrieval Service -------------------

class RetrievalService(BaseComponent):
    """Queries Azure AI Search for relevant HR policy content using semantic search."""
    def __init__(self):
        super().__init__()
        self._client = None

    def _get_client(self):
        if not self._client:
            endpoint = Config.get("AZURE_SEARCH_ENDPOINT", required=True)
            index_name = Config.get("AZURE_SEARCH_INDEX_NAME", required=True)
            api_key = Config.get("AZURE_SEARCH_API_KEY", required=True)
            self._client = SearchClient(
                endpoint=endpoint,
                index_name=index_name,
                credential=AzureKeyCredential(api_key)
            )
        return self._client

    async def retrieve_policy_content(self, query_embedding: List[float], user_query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-K relevant policy chunks from Azure AI Search."""
        retries = 0
        max_retries = 3
        delay = 1
        while retries < max_retries:
            async with trace_step(
                "retrieve_policy_content", step_type="tool_call",
                decision_summary="Retrieve policy content from Azure AI Search",
                output_fn=lambda r: f"chunks={len(r) if r else 0}"
            ) as step:
                try:
                    client = self._get_client()
                    vector_query = VectorizedQuery(
                        vector=query_embedding,
                        k_nearest_neighbors=top_k,
                        fields="vector"
                    )
                    results = client.search(
                        search_text=user_query,
                        vector_queries=[vector_query],
                        top=top_k,
                        select=["chunk", "title"]
                    )
                    chunks = []
                    for r in results:
                        if r.get("chunk"):
                            chunks.append({
                                "chunk": r["chunk"],
                                "title": r.get("title", "")
                            })
                    step.capture(chunks)
                    return chunks
                except Exception as e:
                    self.log_warning(f"Azure Search error: {e}")
                    retries += 1
                    await asyncio.sleep(delay * (2 ** retries))
        self.log_error("Failed to retrieve policy content after retries.")
        return []

# ------------------- Policy Cache -------------------

class PolicyCache(BaseComponent):
    """Caches frequently accessed policy documents for performance optimization."""
    def __init__(self, maxsize: int = 100, ttl: int = 600):
        super().__init__()
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)

    @trace_agent(agent_name='HR Policy Support Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def get_cached_policy(self, query: str) -> Optional[List[Dict[str, Any]]]:
        with trace_step_sync(
            "get_cached_policy", step_type="process",
            decision_summary="Check cache for policy content",
            output_fn=lambda r: f"cached={bool(r)}"
        ) as step:
            result = self.cache.get(query)
            step.capture(result)
            return result

    @trace_agent(agent_name='HR Policy Support Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def update_cache(self, query: str, policy_content: List[Dict[str, Any]]) -> None:
        with trace_step_sync(
            "update_cache", step_type="process",
            decision_summary="Update policy cache",
            output_fn=lambda r: f"updated={bool(r)}"
        ) as step:
            try:
                self.cache[query] = policy_content
                step.capture(True)
            except Exception as e:
                self.log_warning("Cache update error", error=str(e))
                step.capture(False)

# ------------------- Business Rules Engine -------------------

class BusinessRulesEngine(BaseComponent):
    """Applies HR policy validation, escalation, and mapping rules to retrieved content."""
    def __init__(self):
        super().__init__()

    def validate_policy_answer(self, user_query: str, policy_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check if answer is directly supported by policy content."""
        with trace_step_sync(
            "validate_policy_answer", step_type="process",
            decision_summary="Validate if policy content supports answer",
            output_fn=lambda r: f"valid={r.get('valid', False)}"
        ) as step:
            if not policy_content or all(not c.get("chunk") for c in policy_content):
                step.capture({"valid": False, "reason": "HR_POLICY_NOT_FOUND"})
                return {"valid": False, "reason": "HR_POLICY_NOT_FOUND"}
            # Simple validation: check if any chunk contains keywords from query
            query_terms = set(user_query.lower().split())
            for chunk in policy_content:
                chunk_text = chunk.get("chunk", "").lower()
                if any(term in chunk_text for term in query_terms):
                    step.capture({"valid": True, "reason": None})
                    return {"valid": True, "reason": None}
            step.capture({"valid": False, "reason": "HR_POLICY_NOT_FOUND"})
            return {"valid": False, "reason": "HR_POLICY_NOT_FOUND"}

    def check_sensitivity(self, user_query: str, policy_content: List[Dict[str, Any]]) -> bool:
        """Detect if query or content is sensitive/confidential."""
        with trace_step_sync(
            "check_sensitivity", step_type="process",
            decision_summary="Check for sensitive/confidential information",
            output_fn=lambda r: f"sensitive={r}"
        ) as step:
            sensitive_keywords = [
                "salary", "compensation", "ssn", "social security", "medical", "confidential", "address", "bank", "account", "disciplinary"
            ]
            text = user_query.lower() + " " + " ".join([c.get("chunk", "").lower() for c in policy_content])
            for keyword in sensitive_keywords:
                if keyword in text:
                    step.capture(True)
                    return True
            step.capture(False)
            return False

# ------------------- LLM Service -------------------

class LLMService(BaseComponent):
    """Formats prompts, invokes LLM (GPT-4.1), and generates policy-compliant responses."""
    def __init__(self):
        super().__init__()
        self._client = None
        self.model = "gpt-4.1"
        self.temperature = 0.7
        self.max_tokens = 2000
        self.system_prompt = (
            "You are an expert HR Support Agent for the company. Your primary responsibility is to answer employee HR-related questions strictly using official company HR policies and documentation retrieved via Azure AI Search. Do not guess, fabricate, or provide information not explicitly supported by company HR policies. If the answer to a query is missing from the available policy documents or involves sensitive/confidential information, politely instruct the user to contact the HR department for further assistance. Always maintain a formal, professional tone and ensure responses are clear, accurate, and policy-compliant."
        )
        self.fallback_response = (
            "I am unable to find the requested information in the available HR policy documents. For further assistance, please contact the HR department."
        )

    def _get_client(self):
        if not self._client:
            api_key = Config.get("AZURE_OPENAI_API_KEY", required=True)
            endpoint = Config.get("AZURE_OPENAI_ENDPOINT", required=True)
            self._client = openai.AsyncOpenAI(
                api_key=api_key,
                api_version="2024-02-01",
                azure_endpoint=endpoint
            )
        return self._client

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def generate_response(
        self,
        system_prompt: str,
        user_prompt: str,
        context_chunks: List[Dict[str, Any]],
        fallback_response: Optional[str] = None
    ) -> str:
        """Format prompt and call LLM for response generation."""
        retries = 0
        max_retries = 3
        delay = 1
        context_text = "\n\n".join(
            f"Document: {c.get('title','')}\n{c.get('chunk','')}" for c in context_chunks if c.get("chunk")
        )
        prompt = (
            f"{user_prompt}\n\n"
            f"Relevant HR Policy Excerpts:\n{context_text if context_text else '[No relevant policy excerpts found]'}"
        )
        while retries < max_retries:
            async with trace_step(
                "generate_response", step_type="llm_call",
                decision_summary="Call LLM to generate HR policy-compliant response",
                output_fn=lambda r: f"length={len(r) if r else 0}"
            ) as step:
                try:
                    client = self._get_client()
                    _t0 = _time.time()
                    response = await client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    content = response.choices[0].message.content
                    try:
                        trace_model_call(
                            provider="openai",
                            model_name=self.model,
                            prompt_tokens=response.usage.prompt_tokens,
                            completion_tokens=response.usage.completion_tokens,
                            latency_ms=int((_time.time() - _t0) * 1000),
                            response_summary=content[:200] if content else ""
                        )
                    except Exception:
                        pass
                    step.capture(content)
                    return content
                except Exception as e:
                    self.log_warning(f"LLM API error: {e}")
                    retries += 1
                    await asyncio.sleep(delay * (2 ** retries))
        self.log_error("Failed to generate LLM response after retries.")
        return fallback_response or self.fallback_response

# ------------------- Audit Logger -------------------

class AuditLogger(BaseComponent):
    """Logs all user interactions, escalations, and errors for compliance and monitoring."""
    def __init__(self):
        super().__init__()

    def log_event(self, event_type: str, details: Dict[str, Any]) -> str:
        with trace_step_sync(
            "log_event", step_type="process",
            decision_summary=f"Log event: {event_type}",
            output_fn=lambda r: f"log_id={r}"
        ) as step:
            try:
                log_id = f"{event_type}-{int(_time.time()*1000)}"
                self.logger.info(f"AuditLog [{log_id}]: {event_type} | {details}")
                step.capture(log_id)
                return log_id
            except Exception as e:
                self.logger.error(f"Audit logging failed: {e}")
                step.capture("log_failed")
                # Alert admin if critical
                if event_type in ("error", "escalation"):
                    # Placeholder: send alert
                    pass
                return "log_failed"

# ------------------- Application Controller -------------------

class ApplicationController(BaseComponent):
    """Coordinates query processing, error handling, and escalation logic."""
    def __init__(
        self,
        retrieval_service: RetrievalService,
        embedding_service: EmbeddingService,
        business_rules_engine: BusinessRulesEngine,
        llm_service: LLMService,
        policy_cache: PolicyCache,
        audit_logger: AuditLogger
    ):
        super().__init__()
        self.retrieval_service = retrieval_service
        self.embedding_service = embedding_service
        self.business_rules_engine = business_rules_engine
        self.llm_service = llm_service
        self.policy_cache = policy_cache
        self.audit_logger = audit_logger

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_query(self, user_query: str, user_context: Optional[Dict[str, Any]] = None) -> str:
        """Coordinates retrieval, rule validation, LLM call, and response formatting."""
        async with trace_step(
            "process_query", step_type="process",
            decision_summary="End-to-end HR policy query processing",
            output_fn=lambda r: f"response_length={len(r) if r else 0}"
        ) as step:
            try:
                sanitized_query = mask_pii(user_query)
                self.audit_logger.log_event("query_received", {"user_query": sanitized_query, "user_context": user_context})

                # Check cache
                cached = self.policy_cache.get_cached_policy(sanitized_query)
                if cached:
                    policy_content = cached
                    self.log_info("Cache hit for policy content.")
                else:
                    # Embed query
                    embedding = await self.embedding_service.embed_text(sanitized_query)
                    # Retrieve policy content
                    policy_content = await self.retrieval_service.retrieve_policy_content(
                        query_embedding=embedding,
                        user_query=sanitized_query,
                        top_k=5
                    )
                    self.policy_cache.update_cache(sanitized_query, policy_content)

                # Business rules: validate and check sensitivity
                validation = self.business_rules_engine.validate_policy_answer(sanitized_query, policy_content)
                is_sensitive = self.business_rules_engine.check_sensitivity(sanitized_query, policy_content)

                # Decision table logic
                if not validation.get("valid") or is_sensitive:
                    # Escalate to HR
                    self.audit_logger.log_event("escalation", {
                        "reason": validation.get("reason") or "SENSITIVE_INFORMATION_ESCALATION",
                        "user_query": sanitized_query
                    })
                    response = self.llm_service.fallback_response
                else:
                    # Generate LLM response
                    response = await self.llm_service.generate_response(
                        system_prompt=self.llm_service.system_prompt,
                        user_prompt=sanitized_query,
                        context_chunks=policy_content,
                        fallback_response=self.llm_service.fallback_response
                    )
                self.audit_logger.log_event("response_generated", {
                    "user_query": sanitized_query,
                    "response": response
                })
                step.capture(response)
                return response
            except Exception as e:
                self.audit_logger.log_event("error", {"error": str(e), "user_query": user_query})
                self.log_error("Error in process_query", error=str(e))
                step.capture(self.llm_service.fallback_response)
                return self.llm_service.fallback_response

# ------------------- User Interface Handler -------------------

class UserInterfaceHandler(BaseComponent):
    """Handles user input/output via chat and email; manages session state."""
    def __init__(self, authentication_service: AuthenticationService, application_controller: ApplicationController):
        super().__init__()
        self.authentication_service = authentication_service
        self.application_controller = application_controller

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def submit_query(self, user_input: str, user_context: Optional[Dict[str, Any]] = None) -> str:
        """Accepts user input and initiates query processing."""
        async with trace_step(
            "submit_query", step_type="process",
            decision_summary="Handle user query submission",
            output_fn=lambda r: f"response_length={len(r) if r else 0}"
        ) as step:
            try:
                # Session validation (if token in context)
                token = None
                if user_context and "token" in user_context:
                    token = user_context["token"]
                    self.authentication_service.validate_token(token)
                # Sanitize input
                sanitized_input = mask_pii(user_input)
                # Process query
                response = await self.application_controller.process_query(sanitized_input, user_context)
                step.capture(response)
                return response
            except HTTPException as e:
                self.log_warning("Session or authentication error", error=str(e.detail))
                step.capture("Session expired or authentication failed.")
                raise
            except Exception as e:
                self.log_error("Error in submit_query", error=str(e))
                step.capture("Internal error.")
                raise

    def receive_response(self, agent_output: str) -> str:
        """Handles agent output (for UI/email)."""
        with trace_step_sync(
            "receive_response", step_type="format",
            decision_summary="Format agent output for user",
            output_fn=lambda r: f"output_length={len(r) if r else 0}"
        ) as step:
            # For now, just return as-is
            step.capture(agent_output)
            return agent_output

# ------------------- Main Agent Class -------------------

class HRPolicySupportAgent(BaseComponent):
    """Main agent class composing all supporting components."""
    def __init__(self):
        super().__init__()
        # Compose all components
        self.authentication_service = AuthenticationService()
        self.embedding_service = EmbeddingService()
        self.retrieval_service = RetrievalService()
        self.business_rules_engine = BusinessRulesEngine()
        self.llm_service = LLMService()
        self.policy_cache = PolicyCache()
        self.audit_logger = AuditLogger()
        self.application_controller = ApplicationController(
            retrieval_service=self.retrieval_service,
            embedding_service=self.embedding_service,
            business_rules_engine=self.business_rules_engine,
            llm_service=self.llm_service,
            policy_cache=self.policy_cache,
            audit_logger=self.audit_logger
        )
        self.user_interface_handler = UserInterfaceHandler(
            authentication_service=self.authentication_service,
            application_controller=self.application_controller
        )

    @trace_agent(agent_name='HR Policy Support Agent')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def handle_user_query(self, user_input: str, user_context: Optional[Dict[str, Any]] = None) -> str:
        """Entry point for handling user queries."""
        async with trace_step(
            "handle_user_query", step_type="final",
            decision_summary="Top-level agent query handler",
            output_fn=lambda r: f"response_length={len(r) if r else 0}"
        ) as step:
            response = await self.user_interface_handler.submit_query(user_input, user_context)
            step.capture(response)
            return response

# ------------------- FastAPI App -------------------

app = FastAPI(
    title="HR Policy Support Agent",
    description="An expert HR Support Agent answering employee HR-related questions strictly using official company HR policies and documentation retrieved via Azure AI Search.",
    version="1.0.0"
)

# CORS (allow all origins for demo; restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = HRPolicySupportAgent()

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error_type": "ValidationError",
            "error_message": str(exc),
            "tips": "Check your input for missing or malformed fields. Ensure all required fields are present and properly formatted."
        }
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP error: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error_type": "HTTPException",
            "error_message": exc.detail,
            "tips": "Check your authentication/session or input formatting."
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_type": "InternalServerError",
            "error_message": "An unexpected error occurred.",
            "tips": "Please try again later or contact support if the issue persists."
        }
    )

@app.post("/api/ask", response_model=AgentResponseModel)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def ask_agent(query: UserQueryModel):
    """
    Endpoint to submit HR-related questions to the agent.
    """
    try:
        # Input validation and sanitization handled by Pydantic
        user_input = query.user_input
        user_context = query.user_context or {}
        response = await agent.handle_user_query(user_input, user_context)
        return AgentResponseModel(success=True, response=response)
    except ValidationError as ve:
        logger.warning(f"Malformed JSON: {ve}")
        return AgentResponseModel(
            success=False,
            error_type="MalformedJSON",
            error_message="Malformed JSON in request body.",
            tips="Ensure your JSON is properly formatted with correct quotes, commas, and field names."
        )
    except HTTPException as he:
        logger.warning(f"HTTP error: {he.detail}")
        return AgentResponseModel(
            success=False,
            error_type="HTTPException",
            error_message=he.detail,
            tips="Check your authentication/session or input formatting."
        )
    except Exception as e:
        logger.error(f"Unhandled error: {e}")
        return AgentResponseModel(
            success=False,
            error_type="InternalServerError",
            error_message="An unexpected error occurred.",
            tips="Please try again later or contact support if the issue persists."
        )

@app.post("/api/authenticate")
async def authenticate(credentials: Dict[str, Any]):
    """
    Endpoint to authenticate user and return JWT token.
    """
    try:
        token = agent.authentication_service.authenticate(credentials)
        return {"success": True, "token": token}
    except HTTPException as he:
        return {"success": False, "error_type": "AuthError", "error_message": he.detail}
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return {"success": False, "error_type": "InternalServerError", "error_message": str(e)}

@app.get("/api/health")
async def health_check():
    return {"success": True, "status": "ok"}

# ------------------- Main Entry Point -------------------



async def _run_with_eval_service():
    """Entrypoint: initialises observability then runs the agent."""
    import logging as _obs_log
    _obs_logger = _obs_log.getLogger(__name__)
    # ── 1. Observability DB schema ─────────────────────────────────────
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401 – register ORM models
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
    except Exception as _e:
        _obs_logger.warning('Observability DB init skipped: %s', _e)
    # ── 2. OpenTelemetry tracer ────────────────────────────────────────
    try:
        from observability.instrumentation import initialize_tracer
        initialize_tracer()
    except Exception as _e:
        _obs_logger.warning('Tracer init skipped: %s', _e)
    # ── 3. Evaluation background worker ───────────────────────────────
    _stop_eval = None
    try:
        from observability.evaluation_background_service import (
            start_evaluation_worker as _start_eval,
            stop_evaluation_worker as _stop_eval_fn,
        )
        await _start_eval()
        _stop_eval = _stop_eval_fn
    except Exception as _e:
        _obs_logger.warning('Evaluation worker start skipped: %s', _e)
    # ── 4. Run the agent ───────────────────────────────────────────────
    try:
        import uvicorn
        logger.info("Starting HR Policy Support Agent API...")
        uvicorn.run("agent:app", host="0.0.0.0", port=8080, reload=False)
        pass  # TODO: run your agent here
    finally:
        if _stop_eval is not None:
            try:
                await _stop_eval()
            except Exception:
                pass


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(_run_with_eval_service())