# streamlit_rag.py
import streamlit as st
import logging
from typing import Dict
import random
from initialize_rag import RAGInitializer


def process_query(query: str,
                 retriever,
                 reranker,
                 response_generator,
                 process_config: Dict,
                 send_nb_chunks_to_llm=1) -> Dict:
    """
    Process a single query through the complete RAG pipeline.
    
    This function orchestrates the query processing workflow:
    1. Optional query expansion
    2. Document retrieval
    3. Optional result reranking
    4. Response generation
    
    Args:
        query (str): The user's query
        retriever: Document retrieval component
        reranker: Result reranking component
        response_generator: Response generation component
        process_config (Dict): Processing configuration
        send_nb_chunks_to_llm (int): Number of chunks to send to LLM
        
    Returns:
        Dict: Processing results containing:
            - Query: Original query
            - Response: Generated response
            - Score: Best retrieval/reranking score
            
    Note:
        The function handles errors gracefully, returning an error message
        in the response if any step fails.
    """
    try:
        # Expand query if configured
        if process_config['retrieval']['use_query_expansion']:
            expanded_query = response_generator.expand_query(query)
            logging.info(f"Expanded query: {expanded_query}")
        else:
            expanded_query = query
            
        # Retrieve relevant documents using expanded or original query
        if process_config['retrieval']['use_bm25']:
            retrieved_results = retriever.retrieve_with_method(
                expanded_query,
                method="hybrid",
                top_k=process_config['retrieval']['top_k']
            )
        else:
            retrieved_results = retriever.retrieve_with_method(
                expanded_query,
                method="vector",
                top_k=process_config['retrieval']['top_k']
            )
        logging.info(f"Retrieved {len(retrieved_results)} documents")
        
        # Apply reranking if configured
        if process_config['retrieval']['use_reranking']:
            reranked_results = reranker.rerank(
                query,
                [r.document for r in retrieved_results],
                top_k=send_nb_chunks_to_llm
            )
            relevant_docs = [r.document for r in reranked_results]
            best_score = reranked_results[0].score if reranked_results else 0.0
            logging.info(f"Reranked results. Best score: {best_score}")
        else:
            relevant_docs = [r.document for r in retrieved_results]
            best_score = retrieved_results[0].score if retrieved_results else 0.0
            logging.info(f"Using retrieval scores. Best score: {best_score}")
        
        # Generate final response using selected documents
        response_data = response_generator.generate_answer(
            query,
            relevant_docs,
            metadata={'retrieval_score': best_score}
        )
        
        return {
            'Query': query,
            'Response': response_data['response'][:-4],
            'Score': best_score,
            'Sources': relevant_docs
        }
        
    except Exception as e:
        logging.error(f"Error processing query: {str(e)}")
        return {
            'Query': query,
            'Response': "An error occurred processing your query.",
            'Score': 0.0,
            'Sources': []
        }

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'rag_components' not in st.session_state:
        st.session_state.rag_components = None

def display_chat_message(role: str, content: str, sources: list = None, score: float = None):
    """Display a chat message with optional sources and confidence score."""
    with st.chat_message(role):
        # Convert the text to proper Markdown format by:
        # 1. Preserving line breaks
        # 2. Adding proper list formatting
        formatted_content = (
            content
            .replace("\n-", "\n\n-")  # Add extra newline before list items
            .replace("\n ", "\n")      # Remove extra spaces at line starts
            .strip()                   # Remove extra whitespace
        )
        
        # Use markdown to render the formatted text
        st.markdown(formatted_content)
        
        if sources:
            with st.expander("View Sources Used"):
                for idx, source in enumerate(sources, 1):
                    st.markdown(f"**Source {idx}:**")
                    # Create a scrollable text area with fixed height
                    st.text_area(
                        label=f"Source {idx} content",
                        value="From : "+source.metadata["source"]+"\n\nContent : \n"+source.page_content,
                        height=200,
                        label_visibility="collapsed",
                        key=f"source_{role}_{idx}_{hash(source.page_content+str(random.random()*1000000))}",
                        disabled=False
                    )
        if score is not None:
            # Normalize score to be between 0 and 1
            normalized_score = max(0.0, min(abs(score), 1.0))
            st.progress(normalized_score, text=f"Confidence: {normalized_score:.2%}")
            
            # If the original score was outside [0,1], show the actual value
            if score < 0 or score > 1:
                st.caption(f"Original confidence score: {score:.2f}")

def main():
    st.title("RAG Chat System")
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize RAG components if not already done
    if st.session_state.rag_components is None:
        with st.spinner("Initializing RAG system..."):
            try:
                initializer = RAGInitializer(
                    "config/init_config.yaml",
                    "config/process_config.yaml"
                )
                components = initializer.initialize()

                components.retriever.initialize(components.original_chunks)
    
                st.session_state.rag_components = components
                st.success("RAG system initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing RAG system: {str(e)}")
                return
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(
            role=message["role"],
            content=message["content"],
            sources=message.get("sources"),
            score=message.get("score")
        )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Display user message
        display_chat_message("user", prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Process query and display response
        with st.spinner("Processing your question..."):
            result = process_query(
                prompt,
                st.session_state.rag_components.retriever,
                st.session_state.rag_components.reranker,
                st.session_state.rag_components.response_generator,
                st.session_state.rag_components.process_config,
                st.session_state.rag_components.process_config['retrieval']['send_nb_chunks_to_llm']
            )
            
            display_chat_message(
                role="assistant",
                content=result['Response'],
                sources=result['Sources'],
                score=result['Score']
            )
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": result['Response'],
                "sources": result['Sources'],
                "score": result['Score']
            })

    # Sidebar with system information
    with st.sidebar:
        st.header("System Information")
        st.write("Retrieval Settings:")
        st.write("- Top N:", st.session_state.rag_components.process_config['retrieval']['send_nb_chunks_to_llm'])
        st.write("- Reranking:", st.session_state.rag_components.process_config['retrieval']['use_reranking'])
        
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()