import os
import unittest
from unittest.mock import MagicMock, patch

from agents.performance_agent import PerformanceAgent


class DummyDoc:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}


class PerformanceAgentRetrievalTests(unittest.TestCase):
    def test_retriever_initializes_when_source_documents_exist(self):
        agent = PerformanceAgent()
        source_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "performance_sources")

        if not os.path.isdir(source_dir):
            self.skipTest("performance_sources directory not present")

        self.assertIsNotNone(agent.retriever)

    def test_retrieve_evidence_returns_actual_document_content(self):
        agent = PerformanceAgent()
        source_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "performance_sources")

        if not os.path.isdir(source_dir):
            self.skipTest("performance_sources directory not present")

        evidence = agent.retrieve_evidence("The system uses FAISS and embedding-based local retrieval.")

        self.assertIn("FAISS", evidence)
        self.assertIn("HuggingFace", evidence)
        self.assertIn("[SOURCE: performance_sources/", evidence)
        self.assertNotIn("Simulated evidence document", evidence)
        self.assertNotIn("NO_EVIDENCE", evidence)

    def test_local_evidence_is_used_before_web_fallback(self):
        agent = PerformanceAgent()
        agent.retriever = MagicMock()
        agent.retriever.invoke.return_value = [
            DummyDoc("The project uses FAISS for real local retrieval.", {"source": "performance_sources/local.txt"})
        ]
        agent.web_retriever = MagicMock()

        evidence = agent.retrieve_evidence("The project uses FAISS for retrieval.")

        self.assertNotIn("[WEB SOURCE:", evidence)
        self.assertIn("[SOURCE: performance_sources/local.txt]", evidence)

    def test_web_fallback_used_when_local_evidence_is_insufficient(self):
        agent = PerformanceAgent()
        agent.retriever = MagicMock()
        agent.retriever.invoke.return_value = [
            DummyDoc("A completely unrelated text about penguins.", {"source": "performance_sources/other.txt"})
        ]

        agent.web_retriever = MagicMock()
        agent.web_retriever.invoke.return_value = [{
            "title": "Example Web Result",
            "url": "https://example.com/fallback",
            "content": "The project uses FAISS for document retrieval.",
        }]

        evidence = agent.retrieve_evidence("The project uses FAISS for retrieval.")

        self.assertIn("[WEB SOURCE: Example Web Result]", evidence)
        self.assertIn("https://example.com/fallback", evidence)

    def test_local_failure_uses_web_fallback(self):
        agent = PerformanceAgent()
        agent.retriever = MagicMock()
        agent.retriever.invoke.side_effect = RuntimeError("local retrieval failed")
        agent.web_retriever = MagicMock()
        agent.web_retriever.invoke.return_value = [{
            "title": "Recovered Web Result",
            "url": "https://example.com/recovered",
            "content": "Recovered content from the web.",
        }]

        evidence = agent.retrieve_evidence("Recovered claim")

        self.assertIn("[WEB SOURCE: Recovered Web Result]", evidence)
        self.assertNotIn("Simulated evidence document", evidence)

    def test_no_evidence_returns_no_evidence(self):
        agent = PerformanceAgent()
        agent.retriever = None
        agent.web_retriever = None
        self.assertEqual(agent.retrieve_evidence("Any claim"), "NO_EVIDENCE")

    def test_tavily_key_absent_does_not_crash(self):
        agent = PerformanceAgent()
        agent.retriever = None
        agent.web_retriever = None

        with patch.dict(os.environ, {}, clear=False):
            if "TAVILY_API_KEY" in os.environ:
                os.environ.pop("TAVILY_API_KEY")
            self.assertEqual(agent.retrieve_evidence("Any claim"), "NO_EVIDENCE")

    def test_unrelated_local_documents_trigger_web_fallback(self):
        agent = PerformanceAgent()
        agent.retriever = MagicMock()
        agent.retriever.invoke.return_value = [
            DummyDoc("A note about penguins and Arctic weather.", {"source": "performance_sources/irrelevant.txt"})
        ]
        agent.web_retriever = MagicMock()
        agent.web_retriever.invoke.return_value = [{
            "title": "Relevant Web Source",
            "url": "https://example.com/relevant",
            "content": "The system uses FAISS for retrieval.",
        }]

        evidence = agent.retrieve_evidence("The system uses FAISS for retrieval.")

        self.assertIn("[WEB SOURCE: Relevant Web Source]", evidence)
        self.assertNotIn("[SOURCE: performance_sources/irrelevant.txt]", evidence)

    def test_mock_external_rag_retrieval_is_backward_compatible(self):
        agent = PerformanceAgent()
        agent.retriever = MagicMock()
        agent.retriever.invoke.return_value = [
            DummyDoc("Legacy compatibility evidence.", {"source": "performance_sources/legacy.txt"})
        ]

        evidence = agent.mock_external_rag_retrieval("Legacy call")

        self.assertIn("Legacy compatibility evidence", evidence)
        self.assertIn("[SOURCE: performance_sources/legacy.txt]", evidence)


if __name__ == "__main__":
    unittest.main()
