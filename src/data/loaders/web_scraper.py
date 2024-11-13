from typing import List, Set
import logging
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langchain.docstore.document import Document
from .base import BaseLoader
from docling.document_converter import DocumentConverter
import os
from tqdm  import tqdm


class EnhancedWebLoader(BaseLoader):
    """
    Advanced web content loader that asynchronously crawls and processes web pages.
    
    This loader provides sophisticated web content extraction capabilities, including
    table of contents (TOC) traversal, content hierarchy preservation, and markdown
    conversion. It uses asynchronous operations for efficient content loading and
    implements smart content parsing strategies.
    
    Features:
        - Asynchronous content loading using aiohttp
        - Intelligent table of contents (TOC) detection and traversal
        - Domain-restricted crawling for security
        - Markdown conversion of HTML content
        - Comprehensive error handling and logging
        - Configurable TOC selectors with fallback options
        - URL normalization and validation
    
    Attributes:
        base_url (str): The starting URL for content loading
        domain (str): Domain name extracted from base_url for restricted crawling
        toc_selector (str): CSS selector pattern for identifying TOC links
        traverse_toc (bool): Flag to enable/disable TOC link traversal
        logger: Logger instance for tracking operations
        converter: DocumentConverter instance for content processing
    
    Example:
        ```python
        loader = EnhancedWebLoader(
            base_url="https://example.com/docs",
            toc_selector=".documentation-nav a",
            traverse_toc=True
        )
        documents = loader.load()
        ```
    """
    
    def __init__(self, 
                 config, 
                 base_url: str, 
                 toc_selector: str = ".table-of-contents a, .toc a, nav a",
                 traverse_toc: bool = True):
        """
        Initialize the EnhancedWebLoader with the specified configuration.
        
        Args:
            base_url (str): The starting URL for content loading
            toc_selector (str, optional): CSS selector pattern for identifying TOC links.
                Defaults to common TOC selectors.
            traverse_toc (bool, optional): Whether to follow and process TOC links.
                Defaults to True.
        
        Note:
            The loader automatically extracts the domain from the base_url and
            restricts crawling to that domain for security purposes.
        """
        self.config = config
        self.base_url = base_url
        self.domain = urlparse(base_url).netloc
        self.toc_selector = toc_selector
        self.traverse_toc = traverse_toc
        self.logger = logging.getLogger(__name__)
        # Set optimal thread count for parallel processing
        os.environ["OMP_NUM_THREADS"] = str(self.config["processing"]["OMP_NUM_THREADS"] if self.config["processing"]["OMP_NUM_THREADS"] else max(1, os.cpu_count() -1))
        self.converter = DocumentConverter()

    def _is_valid_url(self, url: str) -> bool:
        """
        Validate a URL based on domain restrictions and file type filtering.
        
        Args:
            url (str): The URL to validate
            
        Returns:
            bool: True if the URL is valid for processing, False otherwise
        
        Validation criteria:
            - URL must be within the same domain as base_url
            - File extensions like .pdf, .zip, .jpg, .png are excluded
            - URLs containing fragments (#) are excluded
        """
        parsed = urlparse(url)
        return (
            parsed.netloc == self.domain and  # Domain check
            not parsed.path.endswith(('.pdf', '.zip', '.jpg', '.png')) and  # File type check
            '#' not in url  # Fragment check
        )

    def _normalize_url(self, url: str) -> str:
        """
        Normalize a URL by standardizing its format and removing unnecessary components.
        
        Args:
            url (str): The URL to normalize
            
        Returns:
            str: The normalized URL
            
        Normalizations applied:
            - Remove trailing slashes
            - Remove fragments
            - Preserve query parameters
            - Standardize scheme and domain format
        """
        parsed = urlparse(url)
        # Construct normalized URL with optional query parameters
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path.rstrip('/')}{'?' + parsed.query if parsed.query else ''}"

    async def get_toc_links(self, session: aiohttp.ClientSession, url: str) -> Set[str]:
        """
        Extract and validate links from the table of contents using multiple selector strategies.
        
        This method implements a fallback mechanism for TOC link extraction, trying multiple
        common CSS selectors if the primary selector fails to find links.
        
        Args:
            session (aiohttp.ClientSession): Active aiohttp session for making requests
            url (str): The URL to extract TOC links from
            
        Returns:
            Set[str]: Set of normalized, valid URLs found in the TOC
            
        Note:
            The method includes error handling for failed requests and parsing issues,
            logging warnings and errors as appropriate.
        """
        try:
            # Attempt to fetch the page
            async with session.get(url) as response:
                if response.status != 200:
                    self.logger.warning(f"Failed to fetch {url}: {response.status}")
                    return set()

                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                all_links = set()
                # List of CSS selectors to try, in order of preference
                selectors = [
                    self.toc_selector,  # User-provided selector
                    "nav a",            # Common navigation selectors
                    ".nav-links a",
                    ".menu a",
                    ".sidebar a",
                    "aside a",
                    ".navigation a"
                ]

                # Try each selector until we find links
                for selector in selectors:
                    links = soup.select(selector)
                    if links:
                        for link in links:
                            if 'href' in link.attrs:
                                href = link['href']
                                full_url = urljoin(url, href)  # Convert relative URLs to absolute
                                normalized_url = self._normalize_url(full_url)
                                if self._is_valid_url(normalized_url):
                                    all_links.add(normalized_url)
                        if all_links:  # If we found links, stop trying other selectors
                            break

                return all_links

        except Exception as e:
            self.logger.error(f"Error processing {url}: {str(e)}")
            return set()

    async def scrape_all(self) -> List[Document]:
        """
        Asynchronously scrape all relevant pages and convert them to markdown documents.
        
        This method coordinates the entire scraping process:
        1. Establishes an HTTP session
        2. Collects TOC links if enabled
        3. Processes each URL to create markdown documents
        
        Returns:
            List[Document]: List of processed documents with markdown content
            
        Note:
            The method includes comprehensive error handling at both the TOC
            discovery and document processing levels.
        """
        async with aiohttp.ClientSession() as session:
            # Initialize with base URL
            urls_to_scrape = {self.base_url}
            
            # Add TOC links if traversal is enabled
            if self.traverse_toc:
                try:
                    toc_links = await self.get_toc_links(session, self.base_url)
                    urls_to_scrape.update(toc_links)
                except Exception as e:
                    self.logger.error(f"Error getting TOC links: {str(e)}")
            
            try:
                # Process each URL and convert to markdown
                documents = []
                processed_page = []
                for url in tqdm(urls_to_scrape, desc="Processing Web pages"):
                    result = self.converter.convert(url)
                    markdown_output = result.document.export_to_markdown()

                    if markdown_output not in processed_page:
                        processed_page.append(markdown_output)
                        # Create document with metadata
                        documents.append(
                            Document(
                                page_content=markdown_output,
                                metadata={
                                    "source": url,
                                    "is_url": True
                                }
                            )
                        )
                
                return documents
                
            except Exception as e:
                self.logger.error(f"Error in document processing: {str(e)}")
                return []

    def load(self) -> List[Document]:
        """
        Main entry point for loading and processing web content.
        
        This method provides a synchronous interface to the asynchronous scraping
        functionality, handling the event loop management automatically.
        
        Returns:
            List[Document]: List of processed documents with markdown content
            
        Example:
            ```python
            loader = EnhancedWebLoader("https://example.com")
            documents = loader.load()
            for doc in documents:
                print(f"Processed {doc.metadata['sources']}")
            ```
        """
        try:
            return asyncio.run(self.scrape_all())
        except Exception as e:
            self.logger.error(f"Error in load method: {str(e)}")
            return []