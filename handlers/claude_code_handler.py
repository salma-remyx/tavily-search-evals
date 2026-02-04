"""
Handler for Claude Code CLI with native WebSearch tool.
"""
import subprocess
import os
import asyncio
from typing import Dict, Any, Optional


class ClaudeCodeHandler:
    """Handler for invoking Claude Code CLI for search queries using WebSearch."""

    def __init__(
        self,
        params: Optional[Dict[str, Any]] = None,
        token_model: str = "gpt-4.1"
    ):
        """Initialize the Claude Code handler.

        Args:
            params: Configuration parameters (e.g., model, allowed_tools)
            token_model: Model name for token counting (not used for Claude Code)
        """
        self.params = params or {}
        self.token_model = token_model
        self.is_llm_response = True  # Claude Code returns LLM responses directly

        # Configure which tools to allow - default to WebSearch only
        self.allowed_tools = self.params.get("allowed_tools", ["WebSearch","WebFetch"])
        self.model = self.params.get("model", None)  # Use default if not specified
        self.timeout = self.params.get("timeout", 120)  # Timeout in seconds

    async def search(self, query: str) -> Dict[str, Any]:
        """Run a search query through Claude Code CLI.

        Args:
            query: The question to answer using web search

        Returns:
            Dictionary containing 'answer' and 'raw_output'
        """
        # Build the prompt that instructs Claude Code to use web search
        prompt = f"""Answer this factual question using the WebSearch tool.
Be concise and provide only the factual answer - no extra explanation needed.

Question: {query}

Use WebSearch to find current, accurate information, then provide a brief answer."""

        # Build the claude command
        cmd = ["claude", "--print", "--dangerously-skip-permissions"]

        # Add allowed tools
        if self.allowed_tools:
            for tool in self.allowed_tools:
                cmd.extend(["--allowedTools", tool])

        # Add model if specified
        if self.model:
            cmd.extend(["--model", self.model])

        try:
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
                return {
                    "answer": f"ERROR: {error or 'Unknown error'}",
                    "raw_output": output,
                    "error": error
                }

            return {
                "answer": output,
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
