from typing import Dict, Type, Callable, List, Any, Union
from pathlib import Path
import logging
from langchain.docstore.document import Document
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    JSONLoader,
    UnstructuredExcelLoader,
    BSHTMLLoader,
    UnstructuredEmailLoader,
    NotebookLoader
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.language.language_parser import LanguageParser
from .base import BaseLoader
from docling.document_converter import DocumentConverter
from .file_converter import OfficeConverter
from bs4 import BeautifulSoup
import re
import os
import pandas as pd

class FileLoader(BaseLoader):
    """
    Enhanced file loader supporting multiple file formats using LangChain document loaders
    and Docling for improved PDF handling.
    
    This class provides a unified interface for loading various file types including:
    - Text files (.txt)
    - Markdown files (.md)
    - PDF documents (.pdf) - Using Docling for enhanced conversion
    - Microsoft Office documents (.docx, .doc, .ppt, .pptx, .xlsx, .xls)
    - Data files (.csv, .json)
    - Web content (.html, .htm)
    - Email files (.eml, .msg)
    - Source code files (multiple programming languages)
    - Jupyter notebooks (.ipynb)
    
    The loader automatically detects the file type and uses the appropriate loader
    implementation, with special handling for PDFs using Docling.
    """
    
    def __init__(self, config):
        """Initialize the FileLoader with logging setup and loader mappings."""
        self.logger = logging.getLogger(__name__)
        self.docling_converter = DocumentConverter()
        self.file_converter = OfficeConverter()
        self.config = config
        os.environ["OMP_NUM_THREADS"] = str(self.config["processing"]["OMP_NUM_THREADS"] if self.config["processing"]["OMP_NUM_THREADS"] else max(1, os.cpu_count() -1))
        self._initialize_loaders()
    
    def _load_code_file(self, file_path: str) -> List[Document]:
        """
        Enhanced code file loader with improved metadata handling.
        
        Args:
            file_path (str): Path to the code file
            
        Returns:
            List[Document]: List of documents containing the parsed code
        """
        try:
            loader = GenericLoader.from_filesystem(file_path, parser=LanguageParser())
            documents = loader.load()
            
            # Enhance metadata for code files
            for doc in documents:
                doc.metadata.update({
                    "is_code": True,
                    "file_extension": Path(file_path).suffix,
                    "line_count": len(doc.page_content.splitlines())
                })
            
            return documents
        except Exception as e:
            self.logger.error(f"Error loading code file {file_path}: {str(e)}")
            raise

    def _load_jupyter_notebook(self, file_path: str) -> List[Document]:
        """
        Load and process Jupyter notebooks with cell-type awareness.
        
        Args:
            file_path (str): Path to the Jupyter notebook
            
        Returns:
            List[Document]: List of documents containing notebook cells
        """
        try:
            print(file_path)
            loader = NotebookLoader(
                file_path,
                include_outputs=True,
                max_output_length=8192,
                remove_newline=False
            )
            documents = loader.load()
            
            # Enhance metadata for notebook cells
            for doc in documents:
                doc.metadata.update({
                    "is_notebook": True,
                    "notebook_path": file_path
                })
            
            return documents
        except Exception as e:
            self.logger.error(f"Error loading notebook {file_path}: {str(e)}")
            raise
    
    def _initialize_loaders(self):
        """Initialize the mapping between file extensions and their corresponding loader classes."""
        self.document_loaders: Dict[str, Union[Type, Callable]] = {
            # Text and Markdown
            '.txt': TextLoader,
            '.md': self._load_pdf_or_markdonw,
            
            # PDF Documents
            '.pdf': self._load_pdf_or_markdonw,
            
            # Office Documents
            '.docx': self._convert_and_load,
            '.doc': self._convert_and_load,
            '.ppt': self._convert_and_load,
            '.pptx': self._convert_and_load,
            '.xlsx': self._load_excel,
            '.xls': self._load_excel,
            '.odt': self._convert_and_load,
            
            # Data Files
            '.csv': self._load_csv,
            '.json': JSONLoader,
            
            # Web Content
            '.html': BSHTMLLoader,
            '.htm': BSHTMLLoader,
            
            # Email
            '.eml': UnstructuredEmailLoader,
            '.msg': UnstructuredEmailLoader,
            
            # Jupyter Notebooks
            '.ipynb': self._load_jupyter_notebook,
            
            # Code Files - Common Programming Languages
            '.py': self._load_code_file,
            '.js': self._load_code_file,
            '.jsx': self._load_code_file,
            '.ts': self._load_code_file,
            '.tsx': self._load_code_file,
            '.java': self._load_code_file,
            '.cpp': self._load_code_file,
            '.c': self._load_code_file,
            '.h': self._load_code_file,
            '.hpp': self._load_code_file,
            '.cs': self._load_code_file,
            '.rs': self._load_code_file,
            '.go': self._load_code_file,
            '.rb': self._load_code_file,
            '.php': self._load_code_file,
            '.swift': self._load_code_file,
            '.kt': self._load_code_file,
            '.scala': self._load_code_file,
            '.r': self._load_code_file,
            '.m': self._load_code_file,
            '.sql': self._load_code_file,
            '.sh': self._load_code_file,
            '.bash': self._load_code_file,
            '.ps1': self._load_code_file,
            '.vue': self._load_code_file,
            '.dart': self._load_code_file,
            '.lua': self._load_code_file,
            '.pl': self._load_code_file,
            '.ex': self._load_code_file,
            '.exs': self._load_code_file,
            '.elm': self._load_code_file,
            '.f90': self._load_code_file,
            '.f95': self._load_code_file,
            '.f03': self._load_code_file,
            '.ml': self._load_code_file,
            '.mli': self._load_code_file,
        }


    def html_table_to_markdown(self, html_string):
        """
        Convert an HTML table string to Markdown format.
        
        Args:
            html_string (str): String containing HTML table markup
            
        Returns:
            str: Markdown formatted table string
        """
        # Parse HTML using BeautifulSoup
        soup = BeautifulSoup(html_string, 'html.parser')
        table = soup.find('table')
        
        if not table:
            return "No table found in HTML string"
        
        markdown_lines = []
        
        # Extract and process headers
        headers = []
        header_row = table.find('thead')
        if header_row:
            headers = [th.get_text().strip() for th in header_row.find_all(['th', 'td'])]
        else:
            # Try to get headers from first row if no thead
            first_row = table.find('tr')
            if first_row:
                headers = [th.get_text().strip() for th in first_row.find_all(['th', 'td'])]
        
        if headers:
            # Create markdown table header and separator
            markdown_lines.append('| ' + ' | '.join(headers) + ' |')
            markdown_lines.append('| ' + ' | '.join(['---'] * len(headers)) + ' |')
        
        # Process table body
        rows = table.find_all('tr')
        start_index = 1 if (headers and not header_row) else 0
        
        for row in rows[start_index:]:
            cells = row.find_all(['td', 'th'])
            if cells:
                cell_contents = []
                for cell in cells:
                    # Clean cell content
                    content = cell.get_text().strip()
                    content = re.sub(r'\s+', ' ', content)
                    content = content.replace('|', '\\|')
                    cell_contents.append(content)
                
                markdown_lines.append('| ' + ' | '.join(cell_contents) + ' |')
        
        return '\n'.join(markdown_lines)

    def _load_csv(self, file_path: str) -> List[Document]:
        """
        Load and process CSV files using specified content columns.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            List[Document]: List of processed documents from CSV
            
        Raises:
            Exception: If any error occurs during loading
        """
        try:
            if self.config["ingestion"]["ignore_columns"]:
                ignore_columns = self.config["ingestion"]["ignore_columns"]
                content_columns = list(set(pd.read_csv(file_path).columns).difference(set(ignore_columns)))
                loader = CSVLoader(file_path=file_path, content_columns=content_columns)
            else:
                loader = CSVLoader(file_path=file_path)

            docs = loader.load()

            documents = []
            for doc in docs:
                # Update metadata and clean content
                initial_metadata = doc.metadata
                initial_metadata.update({"is_csv": True})
                page_content = doc.page_content
                # for column in content_columns:
                #     page_content = page_content.replace(column+":", "")

                documents.append(
                    Document(
                        page_content=page_content,
                        metadata=initial_metadata
                    )
                )
            return documents
        
        except Exception as e:
            self.logger.error(f"Error loading File {file_path}: {str(e)}")
            raise

    def _load_excel(self, file_path: str) -> List[Document]:
        """
        Load and process Excel files, converting tables to markdown format.
        
        Args:
            file_path (str): Path to the Excel file
            
        Returns:
            List[Document]: List of processed documents from Excel
            
        Raises:
            Exception: If any error occurs during loading
        """
        try:
            documents = []
            loader = UnstructuredExcelLoader(file_path, mode="elements")
            docs = loader.load() 
            for doc in docs:
                initial_metadata = doc.metadata
                if "text_as_html" in initial_metadata:
                    metadata = {
                        "source": file_path,
                        "page_name": initial_metadata["page_name"],
                        "page_number": initial_metadata["page_number"],
                        "is_markdown": True
                    }
                    documents.append(
                        Document(
                            page_content=self.html_table_to_markdown(doc.metadata["text_as_html"]),
                            metadata=metadata
                        )
                    )
            return [documents]
            
        except Exception as e:
            self.logger.error(f"Error loading File {file_path}: {str(e)}")
            raise

    def _convert_and_load(self, file_path: str) -> List[Document]:
        """
        Convert Office documents to PDF and then load them.
        
        Args:
            file_path (str): Path to the Office document
            
        Returns:
            List[Document]: List containing the converted and processed document
            
        Raises:
            Exception: If any error occurs during conversion or loading
        """
        try:
            # Convert to PDF
            pdf_file_path = self.file_converter.convert_to_pdf(
                file_path,
                output_file="temp_file",
                output_dir="./temp_folder/"
            ) 

            markdown_result = self._load_pdf_or_markdonw(pdf_file_path)[0]

            document = Document(
                page_content=markdown_result.page_content,
                metadata=markdown_result.metadata
            )
            
            return [document]
            
        except Exception as e:
            self.logger.error(f"Error loading File {file_path}: {str(e)}")
            raise

    def _load_pdf_or_markdonw(self, file_path: str) -> List[Document]:
        """
        Load and convert PDF or Markdown files to markdown format.
        
        Args:
            file_path (str): Path to the PDF or Markdown file
            
        Returns:
            List[Document]: List containing the converted document
            
        Raises:
            Exception: If any error occurs during loading or conversion
        """
        try:
            result = self.docling_converter.convert(file_path)
            markdown_result = result.document.export_to_markdown().replace("<!-- image -->\n\n", "")
            metadata = {
                "source": file_path,
                "is_markdown": True
            }
            
            document = Document(
                page_content=markdown_result,
                metadata=metadata
            )
            
            return [document]
            
        except Exception as e:
            self.logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise
    
    def load(self, source: Union[str, Path]) -> List[Any]:
        """
        Load and parse a document from the given file source.
        
        This method serves as the main entry point for loading documents. It automatically
        detects the file type and uses the appropriate loader.
        
        Args:
            source (Union[str, Path]): Path to the file to be loaded
            
        Returns:
            List[Any]: List of loaded and parsed documents
            
        Raises:
            ValueError: If the file type is not supported
            Exception: If any error occurs during loading process
            
        Example:
            ```python
            loader = FileLoader(config)
            documents = loader.load("path/to/document.pdf")
            ```
        """
        source_path = Path(source)
        loader_class = self.document_loaders.get(source_path.suffix.lower())
        
        if not loader_class:
            raise ValueError(f"Unsupported file type: {source_path.suffix}")
            
        try:
            if callable(loader_class) and not isinstance(loader_class, type):
                # Handle custom loaders (PDF) and code file loaders
                return loader_class(str(source_path))
            else:
                # Handle standard loaders
                loader = loader_class(str(source_path))
                return loader.load()
                
        except Exception as e:
            self.logger.error(f"Error loading {source}: {str(e)}")
            raise