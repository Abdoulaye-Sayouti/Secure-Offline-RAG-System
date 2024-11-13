from typing import List, Tuple, Dict, Any, Optional
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, Language,RecursiveCharacterTextSplitter,PythonCodeTextSplitter
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from ..cache.cache import CacheManager
import re
import copy
from .jupyter_and_code_spliter import CodeSplitter, JupyterNotebookSplitter




class ModifiedMarkdownHeaderTextSplitter(MarkdownHeaderTextSplitter):
    """
    Enhanced version of LangChain's MarkdownHeaderTextSplitter with improved handling
    of nested headers and empty sections.
    
    This class extends the base MarkdownHeaderTextSplitter to provide better support for:
    - Nested header hierarchies
    - Empty sections handling
    - Metadata inheritance
    - Header-based document filtering
    
    Features:
        - Maintains header hierarchy in metadata
        - Properly handles empty sections
        - Supports nested header levels
        - Filters redundant empty documents
        - Preserves header relationships
    
    Note:
        Header format should be specified in headers_to_split_on parameter
        during initialization, sorted by header level (e.g., #, ##, ###)
    """
    
    def split_text(self, text: str) -> List[Document]:
        """
        Split markdown text into documents based on header hierarchy.
        
        This method processes markdown text and creates separate documents based on
        headers while maintaining the header hierarchy in metadata.
        
        Args:
            text (str): Markdown text to split
            
        Returns:
            List[Document]: List of documents with appropriate metadata
            
        Note:
            - Headers are processed in order of length to handle nested structures
            - Empty sections are handled appropriately
            - Metadata inheritance is maintained through header levels
        """
        # Sort headers by length for proper nesting handling
        sorted_headers = sorted(self.headers_to_split_on, 
                              key=lambda x: len(x[0]), 
                              reverse=True)
        
        # Create regex pattern for all headers
        headers_to_check = [re.escape(header) for header, _ in sorted_headers]
        header_pattern = f"({'|'.join(headers_to_check)})"
        header_regex = re.compile(f"^{header_pattern}\\s+(.+?)$", re.MULTILINE)
        
        # Find all headers in text
        matches = list(header_regex.finditer(text))
        if not matches:
            return [Document(page_content=text, metadata={})]
        
        docs = []
        current_metadata = {}
        
        # Process each header and its content
        for i, match in enumerate(matches):
            header_text = match.group(1)
            header_value = match.group(2)
            
            # Process header hierarchy
            current_header_level = None
            for header, name in sorted_headers:
                if header == header_text:
                    current_header_level = len(header)
                    # Clean metadata for current level
                    current_metadata = {k: v for k, v in current_metadata.items() 
                                     if len(self._get_header_for_name(k)) < current_header_level}
                    current_metadata[name] = header_value.strip()
                    
                    # Create empty document for header
                    docs.append(Document(page_content="", 
                                      metadata=current_metadata.copy()))
                    break
            
            # Extract content between headers
            if i < len(matches) - 1:
                content_start = match.end() + 1
                content_end = matches[i + 1].start()
                content = text[content_start:content_end].strip()
            else:
                content_start = match.end() + 1
                content = text[content_start:].strip()
            
            # Process content based on settings
            if content:
                if self.return_each_line:
                    for line in content.split('\n'):
                        line = line.strip()
                        if line:
                            docs.append(Document(
                                page_content=line,
                                metadata=current_metadata.copy()
                            ))
                else:
                    docs.append(Document(
                        page_content=content,
                        metadata=current_metadata.copy()
                    ))
        
        # Filter out redundant empty documents
        filtered_docs = []
        for i, doc in enumerate(docs):
            if doc.page_content or i == len(docs) - 1 or \
               docs[i + 1].metadata != doc.metadata:
                filtered_docs.append(doc)
                
        return filtered_docs
    
    def _get_header_for_name(self, name: str) -> str:
        """
        Get the markdown header symbol for a given header name.
        
        Args:
            name (str): Header name to look up
            
        Returns:
            str: Corresponding header symbol or empty string if not found
        """
        for header, header_name in self.headers_to_split_on:
            if header_name == name:
                return header
        return ""

class DataPreprocessor:
    """
    Advanced document preprocessing system with parallel processing, caching, and
    specialized handling for different document types.
    
    This class handles document chunking and preprocessing with support for:
    - Markdown documents with header-based splitting
    - CSV files with custom column handling
    - Regular text documents with recursive splitting
    
    Features:
        - Parallel document processing
        - Intelligent caching system
        - Header-based markdown splitting
        - CSV-specific processing
        - Configurable chunk sizes
        - Progress tracking
        - Memory-efficient operation
    
    Attributes:
        config (Dict): Configuration dictionary
        chunk_size (int): Size of text chunks
        chunk_overlap (int): Overlap between chunks
        cache_dir (str): Directory for caching
        num_processes (int): Number of parallel processes
        logger: Logger instance
        cache_manager: Cache management system
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the DataPreprocessor with configuration.
        
        Args:
            config (Dict): Configuration dictionary containing:
                - processing: Chunking parameters and header configurations
                - paths: Directory configurations
        """
        self.config = config
        self.chunk_size = config['processing']['chunk_size']
        self.chunk_overlap = config['processing']['chunk_overlap']
        self.cache_dir = config['paths']['cache_dir']
        self.num_processes = cpu_count() - 1 or 1
        self.logger = logging.getLogger(__name__)
        self.cache_manager = CacheManager(self.config['paths']['cache_dir'])


    def generate_code_and_jupyter_chunks(self, docs, splitter):
        """
        Generate chunks from code or Jupyter notebook documents with improved parsing.
        
        Args:
            docs (List[Document]): List of documents to process
            splitter: Instance of CodeSplitter or JupyterNotebookSplitter
            
        Returns:
            List[Document]: Processed chunks with metadata
        """
        final_docs = []
        for doc in docs:
            try:
                content = doc.page_content
                
                # Check if this is a notebook (contains 'cell' indicators)
                if isinstance(splitter, JupyterNotebookSplitter):
                    # Extract and process cells
                    cells = []
                    current_cell = ""
                    current_output = ""
                    
                    # Split content into lines
                    lines = content.split('\n')
                    for line in lines:
                        line = line.strip()
                        if "'code' cell: '" in line or "'markdown' cell: '" in line:
                            # Save previous cell if exists
                            if current_cell:
                                cells.append({
                                    'cell_type': 'code' if "'code' cell: '" in line else 'markdown',
                                    'source': current_cell,
                                    'outputs': [current_output] if current_output else []
                                })
                            current_cell = ""
                            current_output = ""
                        elif " with output: '" in line:
                            current_output = line.split(" with output: '")[1].rstrip("']")
                        else:
                            # Clean up the line (remove leading/trailing quotes and brackets)
                            cleaned_line = line.strip("[]'")
                            if cleaned_line:
                                current_cell += cleaned_line + "\n"
                    
                    # Add the last cell
                    if current_cell:
                        cells.append({
                            'cell_type': 'code',  # default to code for last cell
                            'source': current_cell,
                            'outputs': [current_output] if current_output else []
                        })
                    
                    # Process each cell
                    for i, cell in enumerate(cells):
                        chunk_content = f"Cell {i+1} ({cell['cell_type']}):\n{cell['source']}"
                        if cell['outputs']:
                            chunk_content += f"\nOutput:\n{cell['outputs'][0]}"
                        
                        final_docs.append(
                            Document(
                                page_content=chunk_content,
                                metadata={
                                    **doc.metadata,
                                    'cell_type': cell['cell_type'],
                                    'cell_index': i,
                                    'has_output': bool(cell['outputs'])
                                }
                            )
                        )
                else:
                    # Handle regular code files
                    code_chunks = splitter.split_text(content)
                    for chunk in code_chunks:
                        final_docs.append(
                            Document(
                                page_content=f"start_line: {chunk['metadata']['start_line']}\nend_line: {chunk['metadata']['end_line']}\ncode:\n{chunk['text']}",
                                metadata=doc.metadata | chunk['metadata']
                            )
                        )
                        
            except Exception as e:
                self.logger.error(f"Error processing document from {doc.metadata.get('source', 'unknown source')}: {str(e)}")
                # Fall back to basic text splitting
                try:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )
                    basic_chunks = text_splitter.create_documents(
                        texts=[doc.page_content],
                        metadatas=[doc.metadata]
                    )
                    final_docs.extend(basic_chunks)
                    self.logger.info(f"Successfully fell back to basic text splitting for document from {doc.metadata.get('source', 'unknown source')}")
                except Exception as fallback_error:
                    self.logger.error(f"Fallback processing failed for document from {doc.metadata.get('source', 'unknown source')}: {str(fallback_error)}")
                    
        return final_docs
    
    

    def generate_csv_chunks(self, docs):
        chunk_size = self.chunk_size
        chunk_overlap = self.chunk_overlap
        text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
        
        final_docs = []
        for doc in docs:
            chunks = text_splitter.create_documents(
                    texts=[doc.page_content]
                )
            
            for chunk in chunks:
                final_docs.append(
                    Document(
                        page_content=chunk.page_content,
                        metadata=doc.metadata
                    )
                )

        return final_docs


    def generate_markdown_chunks(self, markdown_text: str, doc_path: str, source_type:str) -> List[Document]:
        """
        Generate chunks from markdown text using a two-level splitting approach.
        
        This method processes markdown text by:
        1. Splitting based on headers specified in config
        2. Further splitting large sections into smaller chunks
        3. Maintaining header hierarchy in chunk metadata
        
        Args:
            markdown_text (str): Markdown content to process
            doc_path (str): Source document path
            
        Returns:
            List[Document]: Processed document chunks with metadata
        """
        # Remove image placeholders
        markdown_text = markdown_text.replace("<!-- image -->\n\n", "")
        if source_type == "is_markdown":
            headers_to_split_on = self.config["processing"]["headers_to_split_on"]
        else:
            headers_to_split_on = self.config["processing"]["headers_to_split_on_for_urls"]

        # First level split by headers
        level_1_text_splitter = ModifiedMarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        chunks_level_1 = level_1_text_splitter.split_text(markdown_text)

        # Configure second level splitter for large sections
        level_2_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        final_docs = []
        for level_1_idx, level_1_doc in tqdm(enumerate(chunks_level_1)):
            # Split large sections into smaller chunks
            chunks_level_2 = level_2_text_splitter.create_documents(
                texts=[level_1_doc.page_content]
            )
            
            # Process metadata
            toc_text = self.config["processing"]["TOC_text"]
            final_metadata = copy.deepcopy(level_1_doc.metadata)
            
            # Handle TOC metadata
            if toc_text in final_metadata.keys() and len(final_metadata) > 1:
                del final_metadata[toc_text]
            elif toc_text in final_metadata.keys() and len(final_metadata) == 1:
                final_metadata[toc_text] = toc_text

            # Handle single chunk case
            if len(chunks_level_2) <= 1:
                final_header = self._build_header(final_metadata, level_1_idx, level_1_doc)
                final_metadata["source"] = doc_path
                final_docs.append(Document(
                    page_content=final_header + level_1_doc.page_content,
                    metadata=final_metadata
                ))
            else:
                # Handle multiple chunks case
                headers_keys = self.config["processing"]["headers_keys"]
                for k, level_2_doc in enumerate(chunks_level_2):
                    final_header = ""
                    for h in headers_keys.keys():
                        if h in final_metadata.keys() and final_metadata[h] != "":
                            final_header += headers_keys[h] + final_metadata[h] + f" Part ({k+1})\n"

                    final_metadata["source"] = doc_path
                    final_docs.append(Document(
                        page_content=final_header + level_2_doc.page_content,
                        metadata=final_metadata
                    ))

        return final_docs

    def _build_header(self, metadata: Dict, idx: int, doc: Document) -> str:
        """
        Build header string from metadata and document information.
        
        Args:
            metadata (Dict): Document metadata
            idx (int): Document index
            doc (Document): Document object
            
        Returns:
            str: Formatted header string
        """
        final_header = ""
        headers_keys = self.config["processing"]["headers_keys"]
        for h in headers_keys.keys():
            if h in metadata.keys() and metadata[h] != "":
                final_header += headers_keys[h] + metadata[h] + "\n"

        if idx == 0 and doc.page_content == "":
            final_header = final_header[:-1] + " (Title)"

        return final_header

    def process_documents(self, doc_contents: Dict[str, str]) -> List[Document]:
        """
        Process multiple documents into chunks using parallel processing.
        
        Args:
            doc_contents (Dict[str, str]): Mapping of document paths to contents
            
        Returns:
            List[Document]: Combined list of processed document chunks
        """
        self.logger.info("Processing documents into chunks...")
        
        # Process in parallel with progress tracking
        with Pool(processes=self.num_processes) as pool:
            results = list(tqdm(
                pool.imap(self.process_single_document, doc_contents.items()),
                total=len(doc_contents),
                desc="Chunking documents"
            ))
        
        # Combine all chunks
        all_chunks = []
        for _, chunks in results:
            all_chunks.extend(chunks)
            
        self.logger.info(f"Created {len(all_chunks)} chunks from {len(doc_contents)} documents")
        return all_chunks

    def process_single_document(self, args: Tuple[str, str]) -> Tuple[str, List[Document]]:
        """
        Process a single document into chunks with caching support.
        
        Args:
            args (Tuple[str, str]): Document path and content
            
        Returns:
            Tuple[str, List[Document]]: Document path and processed chunks
        """

        
        doc_path, docs = args
        content = "\n\n".join([doc.page_content for doc in docs])
        # Check cache
        cached_content = self.cache_manager.get_cached(doc_path, "chunks")
        if cached_content:
            return cached_content

        # Process based on document type
        if "is_markdown" in docs[0].metadata.keys() and docs[0].metadata["is_markdown"]:
            chunks = self.generate_markdown_chunks(content, doc_path, "is_markdown")
        elif "is_url" in docs[0].metadata.keys() and docs[0].metadata["is_url"]:
            chunks = self.generate_markdown_chunks(content, doc_path, "is_url")
        elif "is_csv" in docs[0].metadata.keys() and docs[0].metadata["is_csv"]:
           chunks = self.generate_csv_chunks(docs)
        elif "is_code" in docs[0].metadata.keys() and docs[0].metadata["is_code"]:
             code_splitter = CodeSplitter(chunk_size = self.chunk_size, chunk_overlap = self.chunk_overlap,language=docs[0].metadata["language"])
             chunks = self.generate_code_and_jupyter_chunks(docs, code_splitter)
        elif "is_notebook" in docs[0].metadata.keys() and docs[0].metadata["is_notebook"]: 
            notebook_splitter = JupyterNotebookSplitter(chunk_size = self.chunk_size, chunk_overlap = self.chunk_overlap)           
            chunks = self.generate_code_and_jupyter_chunks(docs, notebook_splitter)
        else:
            # Default text splitting
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            chunks = text_splitter.create_documents(
                texts=[content],
                metadatas=[{"source": doc_path}]
            )
        
        # Cache and return results
        result = (doc_path, chunks)
        self.cache_manager.cache_content(doc_path, result, "chunks")
        return result