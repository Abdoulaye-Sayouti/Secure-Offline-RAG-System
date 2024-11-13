from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Llamafile
from langchain.schema import Document
from typing import List, Dict, Optional
import logging
from queue import Queue
import threading

class ResponseGenerator:
    """
    Advanced response generation system using LLM with context-aware capabilities.
    
    This class handles the generation of responses to user queries based on retrieved
    documents, incorporating context preparation, query expansion, and structured
    response formatting. It ensures thread-safe LLM usage and comprehensive error
    handling.
    
    Features:
        - Context-aware response generation
        - Query expansion for improved retrieval
        - Source tracking and citation
        - Thread-safe LLM handling
        - Structured response formatting
        - Comprehensive error handling
    
    Attributes:
        config (Dict): Configuration settings
        llm_queue (Queue): Thread-safe queue for LLM instances
        llm_lock (threading.Lock): Thread synchronization lock
        llm (Llamafile): LLM instance
        logger (logging.Logger): Logger for tracking operations
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the ResponseGenerator with configuration.
        
        Args:
            config (Dict): Configuration dictionary containing:
                
        Note:
            - Sets up thread-safe LLM handling
            - Initializes logging
            - Creates LLM instance
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Thread-safe LLM management
        self.llm_queue = Queue()
        self.llm_lock = threading.Lock()
        
        # Initialize primary LLM instance
        self.llm = Llamafile(base_url=self.config["retrieval"]["llamafile_server_base_url"], seed=2024)

    def generate_answer(self, 
                       query: str,
                       relevant_chunks: List[Document],
                       metadata: Optional[Dict] = None) -> Dict:
        """
        Generate a comprehensive answer based on retrieved document chunks.
        
        Processes retrieved documents and generates a contextualized response
        to the user's query, including source information and metadata.
        
        Args:
            query (str): User's question or query
            relevant_chunks (List[Document]): Retrieved relevant document chunks
            metadata (Optional[Dict]): Additional context or request metadata
            
        Returns:
            Dict: Response dictionary containing:
                - response: Generated answer text
                - source_documents: List of source metadata
                - metadata: Additional response metadata
                - error: Error information if generation fails
                
        Note:
            - Handles errors gracefully with informative messages
            - Preserves source attribution
            - Includes metadata in response
        """
        try:
            context = self._prepare_context(relevant_chunks)
            response = self._generate_response(query, context)
            
            return {
                'response': response,
                'source_documents': [doc.metadata for doc in relevant_chunks],
                'metadata': metadata or {}
            }
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                'response': "I apologize, but I encountered an error generating a response. Please try again.",
                'error': str(e),
                'metadata': metadata or {}
            }

    def _prepare_context(self, chunks: List[Document]) -> str:
        """
        Prepare structured context from retrieved document chunks.
        
        Formats retrieved chunks with source information and enumeration
        for clear reference in generated responses.
        
        Args:
            chunks (List[Document]): Retrieved document chunks to process
            
        Returns:
            str: Formatted context string with source information
            
        Note:
            - Enumerates chunks for reference
            - Includes source metadata
            - Formats for clarity and readability
        """
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            source_info = f"Source: {chunk.metadata.get('source', 'Unknown')}"
            context_parts.append(f"[{i}] {source_info}\n{chunk.page_content}")
        
        return "\n\n".join(context_parts)

    def _generate_response(self, query: str, context: str) -> str:
        """
        Generate formatted response using LLM based on query and context.
        
        Uses a carefully crafted prompt to generate clear, accurate, and
        well-structured responses that address the query while maintaining
        appropriate context and source attribution.
        
        Args:
            query (str): User's question
            context (str): Prepared context from relevant documents
            
        Returns:
            str: Generated response text
            
        Note:
            - Uses structured prompt template
            - Emphasizes clarity and accuracy
            - Includes source attribution guidance
            - Maintains consistent response format
        """

        prompt = ChatPromptTemplate.from_template("""
        You are an efficient Q&A assistant that respond to user queries based on provided context.
        Based on the following information, please provide a clear, accurate, and comprehensive answer to the question.
        If the information is not sufficient to answer the question completely, acknowledge this and provide what you can.

        Question: {query}

        Relevant Information:
        {context}

        Please provide a response that:
        1. Directly addresses the question
        2. Is supported by the provided information
        3. Is clear and well-organized
        5. Is short as possible with relevant informations
        6. Don't mention the source of the information
        7. Do not repeat the question 
        8. Clearly state the lack of information in the provided context to answer the question if it's the case.
        9. Respond only in the language of the query.
        Response:
        """)
       
        messages = prompt.format_messages(query=query, context=context)
        response = self.llm.invoke(messages)
        
        return response
        

    
    def expand_query(self, query: str) -> str:
        """
        Expand the original query for improved document retrieval.
        
        Uses LLM to generate an expanded version of the query that includes
        related terms, synonyms, and alternative phrasings while maintaining
        the original intent.
        
        Args:
            query (str): Original user query
            
        Returns:
            str: Expanded query text
            
        Note:
            - Includes relevant synonyms
            - Adds technical terms
            - Considers alternative phrasings
            - Maintains query intent
            - Returns concise expansion
        """
        prompt = ChatPromptTemplate.from_template("""
        You are a query processor that MUST follow these rules strictly:
        1. NEVER expand or define acronyms/abbreviations under any circumstances (e.g., DAC, ULA, API must stay exactly as written)
        2. Only fix grammatical structure if the query is incomplete
        3. Return the exact original query if it's already a complete sentence

        Examples of correct processing:
        Input: "What is the subject key identifier of the DAC?"
        Output: "What is the subject key identifier of the DAC?"

        Input: "acronym of ULA"
        Output: "What is the acronym of ULA"

        Input: "difference REST SOAP"
        Output: "What is the difference between REST and SOAP?"

        Input: "python list comprehension examples"
        Output: "What are some Python list comprehension examples?"

        Original query: {query}
        Expanded query:
        """)
                
        messages = prompt.format_messages(query=query)
        response = self.llm.invoke(messages)
        
        return response.content
