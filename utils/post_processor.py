import logging
from langchain_openai import ChatOpenAI

from .span_grounding_check import SpanGroundingChecker
from .hallucination_taxonomy import SpanTaxonomyClassifier

logger = logging.getLogger(__name__)


class PostProcessor(object):
    def __init__(self, llm_model: str = "gpt-4.1", temperature: float = 0.0):
        """
        Initialize the PostProcessor class.

        Args:
            llm_model: Model to use for answer extraction
            temperature: Temperature for LLM calls
        """
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)

    def _get_prompt(self, is_llm_response: bool) -> str:
        if is_llm_response:
            prompt = """
                You are an advanced assistant operating in strict extraction mode.  
                Your mission is **extremely important**: extract **only** the **direct, final answer** to the user's query, based solely on the provided response.
        
                ## Rules (non-negotiable):
                - Do **not** explain, paraphrase, summarize, or add any context.
                - Return **only** the final answer — nothing else.
        
                ## Query: 
                {}
        
                ## Response:
                {}
        
                Now return the single, most accurate answer to the query.
            """
        else:
            prompt = """
                You are an advanced assistant operating in strict extraction mode.  
                Your mission is **extremely important**: extract **only** the **direct, final answer** to the user's query, based solely on the provided list of documents. Each document includes a `URL` and `Content`.

                ## Rules (non-negotiable):
                - Do **not** explain, paraphrase, summarize, or add any context.
                - Return **only** the final answer — nothing else.
                - If multiple documents suggest different answers, choose the one from the **most reliable URL** (e.g., Wikipedia, .gov, .edu, official sources).

                ## Query: 
                {}

                ## Documents list:
                {}

                Now return the single, most accurate answer to the query.
            """

        return prompt

    def extract_answer(
        self, query: str, is_llm_response: bool, search_result: str
    ) -> str:
        """Extract a concise answer from an LLM response based on the query.

        Args:
            query: The original user query
            is_llm_response: Whether the search results includes answer already
            search_result: String representing the result from search

        Returns:
            str: A concise, focused answer extracted from the LLM response
        """
        logger.info(f"Extracting answer for query: {query}")

        prompt = self._get_prompt(is_llm_response).format(query, search_result)

        try:
            result = self.llm.invoke(prompt)
            answer = result.content
            logger.info("Successfully extracted answer")
            return answer
        except Exception as e:
            logger.error(f"Error extracting answer: {str(e)}")
            return "Sorry, I couldn't process the answer properly."

    def check_answer_grounding(
        self,
        query: str,
        answer: str,
        search_result: str,
        model: str = None,
        temperature: float = None,
    ) -> dict:
        """Score how much of ``answer`` is unsupported by the retrieved evidence.

        Thin wrapper over :class:`~utils.span_grounding_check.SpanGroundingChecker`
        so the SimpleQA loop can request a span-level grounding / hallucination
        score from the same place it already extracts answers. Mirrors the
        ``(context, question, answer)`` contract of the grounding checker: the
        post-processed ``search_result`` is the context, ``query`` is the
        question, and ``answer`` is the extracted prediction to audit.

        Args:
            query: The original user query.
            answer: The extracted predicted answer to audit.
            search_result: The retrieved evidence the answer was drawn from.
            model: Grounding-judge model. When ``None``, the checker's fallback
                (``GROUNDING_CHECK_DEFAULT_CONFIG``) is used. Live values come
                from configs/config.json's ``grounding_check`` block.
            temperature: Grounding-judge sampling temperature (see ``model``).

        Returns:
            dict with ``hallucination_score`` (0..1), ``ungrounded_spans``, and
            ``grounded`` (see ``SpanGroundingChecker.check``), enriched with the
            paper's span-severity taxonomy: ``contradiction_score`` (0..1),
            ``unverifiable_score``, ``contradicted_spans``, and
            ``unverifiable_spans`` (see ``SpanTaxonomyClassifier.classify``).
        """
        logger.info(f"Checking answer grounding for query: {query}")
        # Judge settings are threaded in from config (falling back to the
        # checkers' GROUNDING_CHECK_DEFAULT_CONFIG when not supplied) rather
        # than hardcoded here.
        checker_kwargs = {}
        if model is not None:
            checker_kwargs["llm_model"] = model
        if temperature is not None:
            checker_kwargs["temperature"] = temperature
        grounding = SpanGroundingChecker(**checker_kwargs).check(
            context=search_result, question=query, answer=answer
        )
        # Split the detected ungrounded spans into contradicted (actively wrong)
        # vs merely unverifiable, per the paper's span-severity taxonomy.
        taxonomy = SpanTaxonomyClassifier(**checker_kwargs).classify(
            context=search_result,
            question=query,
            answer=answer,
            ungrounded_spans=grounding["ungrounded_spans"],
        )
        grounding.update(taxonomy)
        return grounding
