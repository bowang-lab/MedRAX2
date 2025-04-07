from langchain.tools import BaseTool
from medrax.rag.rag import RAGConfig, CohereRAG
from langchain.chains import RetrievalQA


class RAGTool(BaseTool):
    """Tool for answering medical questions using RAG with medical knowledge base.

    Args:
        config (RAGConfig): Configuration for the RAG system
        docs_dir (str): Directory containing medical documents. Defaults to "medrax/rag/docs"
    """

    name: str = "medical_knowledge_rag"
    description: str = """
    Use this tool to answer medical questions using a knowledge base of medical documents.
    Input should be a medical question in natural language.
    """
    rag: CohereRAG = None
    chain: RetrievalQA = None
    docs_dir: str = "medrax/rag/docs"

    def __init__(
        self,
        config: RAGConfig,
        docs_dir: str = "medrax/rag/docs",
    ):
        """Initialize RAG tool with config and documents."""
        super().__init__()
        self.rag = CohereRAG(config)
        self.docs_dir = docs_dir

        # Initialize vectorstore if empty
        if self.rag.vectorstore is None:
            docs = self.rag.load_directory(self.docs_dir)
            self.rag.create_or_update_vectorstore(docs)

        self.chain = self.rag.initialize_rag(with_memory=True)

    def _run(self, query: str) -> str:
        """Execute the RAG tool with the given query.

        Args:
            query (str): Medical question to answer

        Returns:
            str: Generated answer from the RAG system
        """
        result = self.chain.invoke({"query": query})
        return result["result"]

    async def _arun(self, query: str) -> str:
        """Async version of _run.

        Args:
            query (str): Medical question to answer

        Returns:
            str: Generated answer from the RAG system

        Raises:
            NotImplementedError: Async not implemented yet
        """
        raise NotImplementedError("Async not implemented")
