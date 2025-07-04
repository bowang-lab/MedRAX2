from langchain.tools import BaseTool
from medrax.rag.rag import RAGConfig, CohereRAG
from langchain.chains import RetrievalQA
from typing import Dict, Tuple, Any


class RAGTool(BaseTool):
    """Tool for answering medical questions using RAG with medical knowledge base.

    Args:
        config (RAGConfig): Configuration for the RAG system
        use_medrag_textbooks (bool): Whether to use MedRAG textbooks dataset
        use_local_docs (bool): Whether to use local documents from docs_dir
        docs_dir (str, optional): Directory containing local medical documents
    """

    name: str = "medical_knowledge_rag"
    description: str = """
    Use this tool to answer medical questions using a knowledge base of medical documents.
    Input should be a medical question in natural language.
    """
    rag: CohereRAG = None
    chain: RetrievalQA = None

    def __init__(
        self,
        config: RAGConfig,
    ):
        """Initialize RAG tool with config and documents.

        Args:
            config (RAGConfig): Configuration for the RAG system
        """
        super().__init__()
        self.rag = CohereRAG(config)
        self.chain = self.rag.initialize_rag(with_memory=True)

    def _run(self, query: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Execute the RAG tool with the given query.

        Args:
            query (str): Medical question to answer

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Output dictionary and metadata dictionary
        """
        try:
            result = self.chain.invoke({"query": query})
            
            output = {
                "answer": result["result"],
                "source_documents": [
                    {
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in result.get("source_documents", [])
                ]
            }
            
            metadata = {
                "query": query,
                "analysis_status": "completed",
                "num_sources": len(result.get("source_documents", [])),
                "model": self.rag.config.model,
                "embedding_model": self.rag.config.embedding_model,
            }
            
            return output, metadata
            
        except Exception as e:
            output = {"error": str(e)}
            metadata = {
                "query": query,
                "analysis_status": "failed",
                "error_details": str(e),
            }
            return output, metadata

    async def _arun(self, query: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Async version of _run.

        Args:
            query (str): Medical question to answer

        Returns:
            Tuple[Dict[str, Any], Dict[str, Any]]: Output dictionary and metadata dictionary

        Raises:
            NotImplementedError: Async not implemented yet
        """
        return self._run(query)
