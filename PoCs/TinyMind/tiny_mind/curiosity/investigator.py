"""
Investigator - pursues curiosity goals using available tools.

Uses web search and fetch to gather information, then extracts
and integrates knowledge into the graph.
"""

import os
import json
from dataclasses import dataclass
from typing import Optional, Callable
from datetime import datetime

from tiny_mind.substrate.graph import KnowledgeGraph
from tiny_mind.substrate.source import Source, SourceType

from .goals import CuriosityGoal, GoalType, InvestigationResult


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str


class Investigator:
    """
    Pursues curiosity goals using web search and other tools.
    
    Primary workflow:
    1. Convert goal to search query
    2. Search web for relevant information
    3. Fetch content from top results
    4. Extract knowledge using Extractor
    5. Integrate into graph
    """
    
    def __init__(
        self,
        extractor,  # Extractor instance
        llm_provider: str = "openai",
        model: str = "gpt-4o",
        max_search_results: int = 3,
        max_fetch_chars: int = 5000,
    ):
        self.extractor = extractor
        self.llm_provider = llm_provider
        self.model = model
        self.max_search_results = max_search_results
        self.max_fetch_chars = max_fetch_chars
        
        self._client = None
    
    def _get_client(self):
        """Get or create LLM client for processing."""
        if self._client is None:
            if self.llm_provider == "openai":
                from openai import OpenAI
                self._client = OpenAI()
            else:
                raise ValueError(f"Unsupported provider: {self.llm_provider}")
        return self._client
    
    def investigate(
        self,
        goal: CuriosityGoal,
        graph: KnowledgeGraph,
        verbose: bool = True,
    ) -> InvestigationResult:
        """
        Pursue a curiosity goal.
        
        Returns InvestigationResult with what was learned.
        """
        if verbose:
            print(f"\nInvestigating: {goal.question}")
        
        result = InvestigationResult(goal=goal, success=False)
        
        try:
            # Step 1: Generate search query
            query = goal.to_search_query()
            if verbose:
                print(f"  Searching: {query}")
            
            # Step 2: Search web
            search_results = self._web_search(query)
            if not search_results:
                result.errors.append("No search results found")
                return result
            
            if verbose:
                print(f"  Found {len(search_results)} results")
            
            # Step 3: Fetch and process top results
            all_content = []
            for sr in search_results[:self.max_search_results]:
                try:
                    content = self._web_fetch(sr.url)
                    if content:
                        all_content.append({
                            "url": sr.url,
                            "title": sr.title,
                            "content": content[:self.max_fetch_chars],
                        })
                        result.sources.append(sr.url)
                        if verbose:
                            print(f"  Fetched: {sr.title[:50]}...")
                except Exception as e:
                    result.errors.append(f"Fetch error for {sr.url}: {str(e)[:50]}")
            
            if not all_content:
                result.errors.append("Could not fetch any content")
                return result
            
            # Step 4: Extract knowledge from fetched content
            nodes_before = len(list(graph.nodes()))
            edges_before = len(list(graph.edges()))
            
            for item in all_content:
                try:
                    # Create a focused extraction prompt based on goal
                    focused_content = self._focus_content(goal, item["content"])
                    
                    # Extract using existing extractor
                    extraction = self.extractor.extract(focused_content, graph)
                    
                    # Integrate into graph
                    self.extractor.integrate(extraction, graph, focused_content)
                    
                except Exception as e:
                    result.errors.append(f"Extraction error: {str(e)[:50]}")
            
            # Calculate what was learned
            result.nodes_added = len(list(graph.nodes())) - nodes_before
            result.edges_added = len(list(graph.edges())) - edges_before
            
            # Step 5: For verification goals, check if claim was verified
            if goal.type == GoalType.VERIFICATION:
                result.verified = self._verify_claim(goal, all_content)
                if verbose:
                    status = "VERIFIED" if result.verified else "NOT VERIFIED"
                    print(f"  Claim {status}")
            
            # Update target node confidence if we found supporting info
            if result.nodes_added > 0 or result.edges_added > 0:
                self._update_target_confidence(goal, graph, increase=True)
            
            # Generate summary
            result.summary = self._summarize_findings(goal, all_content, result)
            result.success = True
            
            if verbose:
                print(f"  Learned: +{result.nodes_added} nodes, +{result.edges_added} edges")
                if result.summary:
                    print(f"  Summary: {result.summary[:100]}...")
            
        except Exception as e:
            result.errors.append(f"Investigation error: {str(e)}")
        
        return result
    
    def _web_search(self, query: str) -> list[SearchResult]:
        """
        Search the web for information.

        Uses DuckDuckGo search (no API key needed) or falls back to
        HTML-based search.
        """
        results = []

        # Try duckduckgo-search library first
        try:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from duckduckgo_search import DDGS

                with DDGS() as ddgs:
                    for r in ddgs.text(query, max_results=self.max_search_results + 2):
                        url = r.get("href", "")
                        url = self._extract_real_url(url)
                        if url:
                            results.append(SearchResult(
                                title=r.get("title", ""),
                                url=url,
                                snippet=r.get("body", ""),
                            ))

                if results:
                    return results
        except ImportError:
            pass
        except Exception:
            pass

        # Fallback: use HTML-based search
        return self._simple_search(query)
    
    def _extract_real_url(self, url: str) -> str:
        """Extract real URL from DuckDuckGo redirect or fix malformed URLs."""
        import urllib.parse
        
        if not url:
            return ""
        
        # Handle DuckDuckGo redirect URLs
        if "duckduckgo.com/l/" in url or url.startswith("//duckduckgo"):
            # Extract the uddg parameter which contains the real URL
            try:
                if url.startswith("//"):
                    url = "https:" + url
                parsed = urllib.parse.urlparse(url)
                params = urllib.parse.parse_qs(parsed.query)
                if "uddg" in params:
                    return urllib.parse.unquote(params["uddg"][0])
            except Exception:
                pass
        
        # Fix URLs missing scheme
        if url.startswith("//"):
            url = "https:" + url
        elif not url.startswith("http"):
            url = "https://" + url
        
        return url
    
    def _simple_search(self, query: str) -> list[SearchResult]:
        """Simple search fallback using requests + regex."""
        try:
            import requests
            import re

            # Use DuckDuckGo HTML search
            url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
            headers = {"User-Agent": "TinyMind/1.0"}

            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            # Parse with regex - more reliable than HTMLParser for this page
            results = []

            # Find all result links (class="result__a" href="...")
            link_pattern = r'<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([^<]+)</a>'
            matches = re.findall(link_pattern, response.text)

            for href, title in matches[:self.max_search_results]:
                # Extract actual URL from DuckDuckGo redirect
                real_url = self._extract_real_url(href)
                if real_url:
                    results.append(SearchResult(
                        title=title.strip(),
                        url=real_url,
                        snippet="",
                    ))

            return results

        except Exception as e:
            print(f"  Simple search error: {e}")
            return []
    
    def _web_fetch(self, url: str) -> Optional[str]:
        """Fetch content from a URL."""
        try:
            import requests
            from bs4 import BeautifulSoup
            
            headers = {"User-Agent": "TinyMind/1.0"}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Parse HTML and extract text
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator="\n")
            
            # Clean up whitespace
            lines = [line.strip() for line in text.splitlines()]
            text = "\n".join(line for line in lines if line)
            
            return text
            
        except ImportError:
            print("  Note: Install beautifulsoup4 for better content extraction")
            return self._simple_fetch(url)
        except Exception as e:
            print(f"  Fetch error: {e}")
            return None
    
    def _simple_fetch(self, url: str) -> Optional[str]:
        """Simple fetch without BeautifulSoup."""
        try:
            import requests
            import re
            
            headers = {"User-Agent": "TinyMind/1.0"}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            # Very basic HTML stripping
            text = re.sub(r'<script[^>]*>.*?</script>', '', response.text, flags=re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()
            
        except Exception as e:
            return None
    
    def _focus_content(self, goal: CuriosityGoal, content: str) -> str:
        """Focus content on what the goal is asking about."""
        # Prepend context about what we're looking for
        context = f"[Context: Looking for information about '{goal.target}' to answer: {goal.question}]\n\n"
        
        # Truncate content if too long
        if len(content) > self.max_fetch_chars:
            content = content[:self.max_fetch_chars] + "..."
        
        return context + content
    
    def _verify_claim(self, goal: CuriosityGoal, content_items: list[dict]) -> bool:
        """Check if fetched content supports or refutes a claim."""
        try:
            client = self._get_client()
            
            # Combine content
            combined = "\n\n---\n\n".join([
                f"Source: {item['title']}\n{item['content'][:1000]}"
                for item in content_items
            ])
            
            prompt = f"""Based on the following sources, does the evidence support or refute this claim?

Claim to verify: {goal.question}
Target concept: {goal.target}

Sources:
{combined}

Answer with just "SUPPORTED", "REFUTED", or "INCONCLUSIVE" followed by a one-sentence explanation."""
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            
            answer = response.choices[0].message.content.strip().upper()
            return answer.startswith("SUPPORTED")
            
        except Exception as e:
            return None  # Inconclusive
    
    def _update_target_confidence(self, goal: CuriosityGoal, graph: KnowledgeGraph, increase: bool):
        """Update confidence of target node based on investigation results."""
        for node_id in goal.related_nodes:
            node = graph.get_node(node_id)
            if node:
                if increase:
                    # Found supporting information
                    old_conf = node.confidence
                    node.confidence = old_conf + (1 - old_conf) * 0.15
                else:
                    # Found contradicting information
                    node.confidence = max(0.1, node.confidence * 0.8)
    
    def _summarize_findings(
        self,
        goal: CuriosityGoal,
        content_items: list[dict],
        result: InvestigationResult,
    ) -> str:
        """Generate a brief summary of what was learned."""
        try:
            client = self._get_client()
            
            # Build context
            learned = f"Added {result.nodes_added} new concepts and {result.edges_added} relationships."
            
            combined = "\n".join([
                f"- {item['title']}: {item['content'][:200]}..."
                for item in content_items[:2]
            ])
            
            prompt = f"""Summarize in one sentence what was learned about '{goal.target}' from these sources:

{combined}

{learned}

One-sentence summary:"""
            
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Investigated {goal.target}, added {result.nodes_added} nodes."
