"""
Handler for Claude Agent SDK with custom Tavily tools.

Uses the Claude Agent SDK with a custom MCP server for Tavily search and extract.
Replaces native WebSearch with tavily_search and WebFetch with tavily_extract.

Defaults:
- search_depth: "advanced", max_results: 10
- extract_depth: "advanced", chunks_per_source: 5
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional

from tavily import AsyncTavilyClient
from claude_agent_sdk import query, ClaudeAgentOptions, tool, create_sdk_mcp_server

logger = logging.getLogger(__name__)

# Default parameters (can be overridden via config)
DEFAULT_SEARCH_DEPTH = "advanced"
DEFAULT_MAX_RESULTS = 10
DEFAULT_EXTRACT_DEPTH = "advanced"
DEFAULT_CHUNKS_PER_SOURCE = 5

# System prompt to guide tool usage
DEFAULT_SYSTEM_PROMPT = """You are a research assistant with web search capabilities. Your job is to find accurate, up-to-date information to answer questions.

## Your Tools

**tavily_search** - Search the web for current information
**tavily_extract** - Get full content from URLs when you need deeper details

## Your Approach

Always search first before answering. The web has information you don't - prices, dates, statistics, recent events, niche details. A quick search takes seconds and dramatically improves accuracy.

## When to Search (Almost Always!)

- Specific facts: prices, dates, statistics, measurements
- Current information: news, stock prices, weather, schedules  
- Niche topics: local businesses, specific products, regional details
- Verification: even if you think you know, search confirms it
- Anything the user is asking about that exists in the real world

## Example Workflows

**Specific data**: "What's the price of X at Y location?"
→ Search for the specific business/product, extract menu or pricing page

**Current events**: "What happened with [recent news]?"  
→ Search for recent articles, extract for full details

**Factual verification**: "When was [thing] built/founded/released?"
→ Search to confirm exact date with authoritative sources

## Tips

- Start with a search - it's fast and catches things you'd miss
- Use extract to get full details from promising URLs
- Try multiple search queries if the first doesn't find what you need
- Be specific in searches: include names, locations, dates when relevant

Never say "I don't have information" without searching first. The answer is usually out there - go find it!"""


def create_tavily_search_tool(search_depth: str = DEFAULT_SEARCH_DEPTH, max_results: int = DEFAULT_MAX_RESULTS):
    """Create a Tavily search tool with configurable defaults."""
    
    @tool(
        "tavily_search",
        f"""Search the web using Tavily's AI-powered search API. Returns relevant search results with content snippets.

This tool uses search_depth="{search_depth}" and max_results={max_results} by default.

Parameters:
- query: Search query (keep under 400 chars, think search query not long prompt)""",
        {
            "query": str,
        }
    )
    async def tavily_search(args: dict[str, Any]) -> dict[str, Any]:
        """Execute a Tavily search query using the official SDK."""
        try:
            client = AsyncTavilyClient()
            search_query = args["query"]

            if len(search_query) > 400:
                search_query = search_query[:400]

            search_kwargs = {
                "query": search_query,
                "search_depth": search_depth,
                "max_results": max_results,
            }

            logger.info("[tavily_search] ✓ TAVILY SEARCH API CALLED")

            response = await client.search(**search_kwargs)

            results = response.get("results", [])
            result_text = [f"Search Results for: {search_query}\n"]
            result_text.append(f"Total results: {len(results)}\n")

            for i, r in enumerate(results, 1):
                result_text.append(f"\n{i}. {r.get('title', 'No title')}")
                result_text.append(f"   URL: {r.get('url', 'No URL')}")
                content = r.get("content", "No content")
                if content:
                    result_text.append(f"   Content: {content[:500]}")

            return {
                "content": [{
                    "type": "text",
                    "text": "\n".join(result_text)
                }]
            }

        except Exception as e:
            logger.error(f"[tavily_search] Error: {str(e)}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error executing Tavily search: {str(e)}"
                }]
            }

    return tavily_search


def create_tavily_extract_tool(extract_depth: str = DEFAULT_EXTRACT_DEPTH):
    """Create a Tavily extract tool with configurable defaults."""
    
    @tool(
        "tavily_extract",
        f"""Extract clean content from URLs using Tavily's extract API. Use this to fetch and read web page content.

This tool uses extract_depth="{extract_depth}" by default (handles JS-rendered pages, tables, complex content).

Parameters:
- urls: List of URLs to extract content from
- query: Optional - reranks extracted chunks by relevance to this query (recommended for focused extraction)
- chunks_per_source: Number of relevant chunks per URL (1-5, max 500 chars each). Only works with query.""",
        {
            "urls": list,
            "query": str,
            "chunks_per_source": int,
        }
    )
    async def tavily_extract(args: dict[str, Any]) -> dict[str, Any]:
        """Extract content from URLs using the official Tavily SDK."""
        try:
            client = AsyncTavilyClient()

            urls = args.get("urls", [])
            if not urls:
                return {
                    "content": [{
                        "type": "text",
                        "text": "Error: No URLs provided for extraction"
                    }]
                }

            # Handle case where urls is passed as a string instead of list
            if isinstance(urls, str):
                if urls.startswith("["):
                    try:
                        urls = json.loads(urls)
                    except:
                        pass
                
                if isinstance(urls, str):
                    urls = [u.strip().strip('"').strip("'") for u in urls.split(",")]
            
            # Filter to valid URLs only
            valid_urls = [url for url in urls if isinstance(url, str) and url.startswith(("http://", "https://"))]
            
            if not valid_urls:
                return {
                    "content": [{
                        "type": "text",
                        "text": f"Error: No valid URLs provided. Received: {urls[:3]}"
                    }]
                }
            
            urls = valid_urls

            extract_kwargs = {
                "urls": urls,
                "extract_depth": extract_depth,
            }

            if args.get("query"):
                extract_kwargs["query"] = args["query"]
                chunks = args.get("chunks_per_source", DEFAULT_CHUNKS_PER_SOURCE)
                extract_kwargs["chunks_per_source"] = min(max(chunks, 1), 5)

            logger.info("[tavily_extract] ✓ TAVILY EXTRACT API CALLED")

            response = await client.extract(**extract_kwargs)

            results = response.get("results", [])
            result_text = [f"Extracted Content from {len(results)} URLs:\n"]

            for result in results:
                result_text.append(f"\n--- URL: {result.get('url', 'Unknown')} ---")
                content = result.get("raw_content", "No content extracted")
                if len(content) > 3000:
                    content = content[:3000] + "... [truncated]"
                result_text.append(content)

            failed = response.get("failed_results", [])
            if failed:
                result_text.append("\n\nFailed URLs:")
                for f in failed:
                    result_text.append(f"  - {f.get('url')}: {f.get('error')}")

            return {
                "content": [{
                    "type": "text",
                    "text": "\n".join(result_text)
                }]
            }

        except Exception as e:
            logger.error(f"[tavily_extract] Error: {str(e)}")
            return {
                "content": [{
                    "type": "text",
                    "text": f"Error executing Tavily extract: {str(e)}"
                }]
            }

    return tavily_extract


class ClaudeCodeTavilyHandler:
    """Handler for Claude Agent SDK with custom Tavily MCP for search and extract."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        token_model: str = "gpt-4.1"
    ):
        """Initialize the Claude Code Tavily handler.

        Args:
            params: Configuration parameters including:
                - model: Claude model to use
                - timeout: Request timeout in seconds (default: 120)
                - search_depth: Tavily search depth (default: "advanced")
                - max_results: Max search results (default: 10)
                - extract_depth: Tavily extract depth (default: "advanced")
                - chunks_per_source: Chunks per URL for extract (default: 5)
                - system_prompt: Custom system prompt (optional)
            token_model: Model name for token counting (not used)
        """
        self.params = params or {}
        self.token_model = token_model
        self.is_llm_response = True

        self.model = self.params.get("model", None)
        self.timeout = self.params.get("timeout", 120)
        
        self.search_depth = self.params.get("search_depth", DEFAULT_SEARCH_DEPTH)
        self.max_results = self.params.get("max_results", DEFAULT_MAX_RESULTS)
        self.extract_depth = self.params.get("extract_depth", DEFAULT_EXTRACT_DEPTH)
        self.chunks_per_source = self.params.get("chunks_per_source", DEFAULT_CHUNKS_PER_SOURCE)
        self.system_prompt = self.params.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

        self._tavily_search_tool = create_tavily_search_tool(
            search_depth=self.search_depth,
            max_results=self.max_results
        )
        self._tavily_extract_tool = create_tavily_extract_tool(
            extract_depth=self.extract_depth
        )

        self._mcp_server = create_sdk_mcp_server(
            name="tavily",
            version="1.0.0",
            tools=[self._tavily_search_tool, self._tavily_extract_tool]
        )

    async def _message_generator(self, prompt_text: str):
        """Async generator for prompt - required for MCP tools in Claude Agent SDK."""
        yield {
            "type": "user",
            "message": {
                "role": "user",
                "content": prompt_text
            }
        }

    async def search(self, query_text: str) -> Dict[str, Any]:
        """Run a search query through Claude Agent SDK using custom Tavily MCP.

        Args:
            query_text: The question to answer using web search

        Returns:
            Dictionary containing 'answer', 'raw_output', and 'error'
        """
        messages = []
        tool_calls = []
        final_result = None
        
        try:
            options = ClaudeAgentOptions(
                system_prompt=self.system_prompt,
                mcp_servers={"tavily": self._mcp_server},
                allowed_tools=[
                    "mcp__tavily__tavily_search",
                    "mcp__tavily__tavily_extract",
                ],
                permission_mode="bypassPermissions",
            )

            if self.model:
                options.model = self.model

            async def run_agent():
                async for message in query(
                    prompt=self._message_generator(query_text),
                    options=options
                ):
                    msg_info = {
                        "type": getattr(message, "type", None),
                        "subtype": getattr(message, "subtype", None),
                    }

                    if hasattr(message, "tool_name"):
                        tool_call = {
                            "tool_name": message.tool_name,
                            "tool_input": getattr(message, "tool_input", None),
                        }
                        tool_calls.append(tool_call)
                        msg_info["tool_name"] = message.tool_name

                    if hasattr(message, "result"):
                        nonlocal final_result
                        final_result = message.result
                        msg_info["result"] = message.result

                    messages.append(msg_info)

            await asyncio.wait_for(run_agent(), timeout=self.timeout)

            answer = final_result if final_result else ""
            raw_output = json.dumps({
                "messages": messages,
                "tool_calls": tool_calls,
                "result": final_result
            }, indent=2, default=str)

            return {
                "answer": answer,
                "raw_output": raw_output,
                "error": None
            }

        except asyncio.TimeoutError:
            return {
                "answer": "ERROR: Timeout",
                "raw_output": "",
                "error": "Request timed out"
            }
        except Exception as e:
            logger.error(f"[claude_code_tavily] Error: {str(e)}")
            return {
                "answer": f"ERROR: {str(e)}",
                "raw_output": "",
                "error": str(e)
            }

    async def post_process(self, search_result: dict, **kwargs) -> tuple:
        """Post-process the search result.

        For Claude Code, the response is already processed by the LLM,
        so we just return the answer directly.

        Returns:
            Tuple of (answer, token_count, token_avg)
        """
        return search_result.get("answer", ""), 0, 0
