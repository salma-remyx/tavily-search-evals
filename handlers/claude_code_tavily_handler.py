"""
Handler for Claude Code CLI with Tavily MCP for web search.

Uses --allowedTools to restrict Claude to only use mcp__tavily__tavily_search.
Requires Tavily MCP to be configured locally via: claude mcp add
"""
import os
import asyncio
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


class ClaudeCodeTavilyHandler:
    """Handler for invoking Claude Code CLI with Tavily MCP for search."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        token_model: str = "gpt-4.1"
    ):
        """Initialize the Claude Code Tavily handler.

        Args:
            params: Configuration parameters (e.g., model, timeout)
            token_model: Model name for token counting (not used for Claude Code)
        """
        self.params = params or {}
        self.token_model = token_model
        self.is_llm_response = True  # Claude Code returns LLM responses directly

        self.model = self.params.get("model", None)  # Use default if not specified
        self.timeout = self.params.get("timeout", 120)  # Timeout in seconds

    async def search(self, query: str) -> Dict[str, Any]:
        """Run a search query through Claude Code CLI using Tavily MCP.

        Args:
            query: The question to answer using web search

        Returns:
            Dictionary containing 'answer' and 'raw_output'
        """
        try:
            # Build the prompt that FORCES Tavily MCP usage
            prompt = f"""CRITICAL INSTRUCTION: You MUST call mcp__tavily__tavily_search tool FIRST before answering.

DO NOT answer from your internal knowledge. You MUST search first.

Question: {query}

REQUIRED STEPS:
1. IMMEDIATELY call mcp__tavily__tavily_search with an appropriate search query
2. Wait for search results

If you answer without searching first, your response is INVALID."""

            # Build the claude command
            # --allowedTools restricts to ONLY mcp__tavily__tavily_search
            # This ensures Claude MUST use Tavily MCP and cannot use anything else
            cmd = [
                "claude",
                "--print",
                "--dangerously-skip-permissions",
                "--output-format", "json",
                "--allowedTools", "mcp__tavily__tavily_search"
            ]

            # Add model if specified
            if self.model:
                cmd.extend(["--model", self.model])

            logger.info(f"[claude_code_tavily] Running query: {query[:100]}...")
            logger.info(f"[claude_code_tavily] Command: {' '.join(cmd)}")

            # Create subprocess - pass prompt via stdin
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ}
            )

            # Wait for completion with timeout, passing prompt via stdin
            stdout, stderr = await asyncio.wait_for(
                process.communicate(input=prompt.encode('utf-8')),
                timeout=self.timeout
            )

            output = stdout.decode('utf-8').strip()
            error = stderr.decode('utf-8').strip()

            if process.returncode != 0:
                logger.error(f"[claude_code_tavily] Command failed with error: {error}")
                return {
                    "answer": f"ERROR: {error or 'Unknown error'}",
                    "raw_output": output,
                    "error": error
                }

            # Parse JSON output to extract tool usage info
            try:
                result_json = json.loads(output)
                answer = result_json.get("result", output)

                # Log tool usage from modelUsage
                model_usage = result_json.get("modelUsage", {})
                for model_name, usage in model_usage.items():
                    web_search_requests = usage.get("webSearchRequests", 0)
                    logger.info(f"[claude_code_tavily] Model {model_name}: webSearchRequests={web_search_requests}")

                # Log the number of turns (tool calls)
                num_turns = result_json.get("num_turns", 0)
                logger.info(f"[claude_code_tavily] Number of turns: {num_turns}")

                # Log session_id for debugging
                session_id = result_json.get("session_id", "unknown")
                logger.info(f"[claude_code_tavily] Session ID: {session_id}")

                # Save raw output for debugging
                debug_file = f"/tmp/claude_code_tavily_debug_{session_id}.json"
                with open(debug_file, 'w') as f:
                    f.write(output)
                logger.info(f"[claude_code_tavily] Raw output saved to: {debug_file}")

                # Check if Tavily MCP was used
                # Since --allowedTools only allows mcp__tavily__tavily_search:
                # - num_turns > 1 means a tool was called
                # - webSearchRequests=0 confirms it wasn't native WebSearch
                # Therefore, the only tool that could have been called is mcp__tavily__tavily_search
                if num_turns > 1 and all(u.get("webSearchRequests", 0) == 0 for u in model_usage.values()):
                    logger.info("[claude_code_tavily] ✓ Tavily MCP was used (num_turns=%d, no native WebSearch)", num_turns)
                elif num_turns == 1:
                    logger.warning("[claude_code_tavily] ⚠ No tool was called (num_turns=1) - answered from memory")
                else:
                    logger.warning("[claude_code_tavily] ⚠ Unexpected state: num_turns=%d", num_turns)

            except json.JSONDecodeError:
                logger.warning("[claude_code_tavily] Could not parse JSON output, using raw output")
                answer = output

            return {
                "answer": answer,
                "raw_output": output,
                "error": None
            }

        except asyncio.TimeoutError:
            return {
                "answer": "ERROR: Timeout",
                "raw_output": "",
                "error": "Command timed out"
            }
        except Exception as e:
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
