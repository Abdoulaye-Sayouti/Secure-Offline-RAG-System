from typing import List, Tuple, Dict, Any, Optional
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter, Language,RecursiveCharacterTextSplitter,PythonCodeTextSplitter
import logging
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from ..cache.cache import CacheManager
import re


import json
import nbformat
from dataclasses import dataclass
from abc import ABC, abstractmethod


@dataclass
class SplitMetadata:
    """Metadata for text splits"""
    start_line: int
    end_line: int
    type: str
    parent_type: Optional[str] = None
    language: Optional[str] = None

class BaseSplitter(ABC):
    """Abstract base class for specialized splitters"""
    @abstractmethod
    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        pass


class CodeSplitter(BaseSplitter):
    def __init__(
        self,
        language: str = "python",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_lines_per_chunk: int = 100,
    ):
        self.language = language
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_lines_per_chunk = max_lines_per_chunk
        
        # Extended language-specific settings
        self.language_settings = {
            "python": {
                "splitter": RecursiveCharacterTextSplitter.from_language(
                    language=Language.PYTHON,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ),
                "patterns": {
                    "function": r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*:",
                    "class": r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\([^)]*\))?\s*:",
                    "method": r"(?<=\s)def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*:",
                }
            },
            "javascript": {
                "splitter": RecursiveCharacterTextSplitter.from_language(
                    language=Language.JS,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ),
                "patterns": {
                    "function": r"(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)|(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?function|\(.*?\)\s*=>\s*{)",
                    "class": r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:extends\s+[a-zA-Z_][a-zA-Z0-9_]*\s*)?{",
                    "method": r"(?:async\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*{",
                }
            },
            "typescript": {
                "splitter": RecursiveCharacterTextSplitter.from_language(
                    language=Language.TS,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ),
                "patterns": {
                    "function": r"(?:function\s+([a-zA-Z_][a-zA-Z0-9_]*)|(?:const|let|var)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:async\s*)?function|\(.*?\)\s*=>\s*{)",
                    "class": r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:extends\s+[a-zA-Z_][a-zA-Z0-9_]*\s*)?{",
                    "method": r"(?:async\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*[:}]",
                    "interface": r"interface\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*{",
                    "type": r"type\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*=",
                }
            },
            "java": {
                "splitter": RecursiveCharacterTextSplitter.from_language(
                    language=Language.JAVA,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ),
                "patterns": {
                    "class": r"(?:public\s+|private\s+|protected\s+)?(?:abstract\s+)?class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:extends\s+[a-zA-Z_][a-zA-Z0-9_]*\s*)?(?:implements\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*\s*)?{",
                    "interface": r"(?:public\s+|private\s+|protected\s+)?interface\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:extends\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*\s*)?{",
                    "method": r"(?:public\s+|private\s+|protected\s+)?(?:static\s+)?(?:final\s+)?(?:[a-zA-Z_][a-zA-Z0-9_<>]*\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:throws\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*\s*)?{",
                }
            },
            "csharp": {
                "splitter": RecursiveCharacterTextSplitter.from_language(
                    language=Language.CSHARP,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ),
                "patterns": {
                    "class": r"(?:public\s+|private\s+|protected\s+|internal\s+)?(?:abstract\s+)?(?:sealed\s+)?class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?::\s*[a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*\s*)?{",
                    "interface": r"(?:public\s+|private\s+|protected\s+|internal\s+)?interface\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?::\s*[a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*\s*)?{",
                    "method": r"(?:public\s+|private\s+|protected\s+|internal\s+)?(?:static\s+)?(?:virtual\s+)?(?:override\s+)?(?:abstract\s+)?(?:[a-zA-Z_][a-zA-Z0-9_<>]*\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*{",
                    "property": r"(?:public\s+|private\s+|protected\s+|internal\s+)?(?:[a-zA-Z_][a-zA-Z0-9_<>]*\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*{\s*(?:get|set)\s*;?\s*(?:(?:get|set)\s*;?\s*)?}",
                }
            },
            "cpp": {
                "splitter": RecursiveCharacterTextSplitter.from_language(
                    language=Language.CPP,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ),
                "patterns": {
                    "class": r"(?:class|struct)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?::\s*(?:public|private|protected)\s+[a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*(?:public|private|protected)\s+[a-zA-Z_][a-zA-Z0-9_]*)*\s*)?{",
                    "function": r"(?:[a-zA-Z_][a-zA-Z0-9_]*\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:const)?\s*{",
                    "namespace": r"namespace\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*{",
                    "template": r"template\s*<[^>]*>\s*(?:class|struct)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
                }
            },
            "go": {
                "splitter": RecursiveCharacterTextSplitter.from_language(
                    language=Language.GO,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ),
                "patterns": {
                    "function": r"func\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:[^{]*){",
                    "method": r"func\s*\([^)]*\)\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?:[^{]*){",
                    "interface": r"type\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+interface\s*{",
                    "struct": r"type\s+([a-zA-Z_][a-zA-Z0-9_]*)\s+struct\s*{",
                }
            },
            "rust": {
                "splitter": RecursiveCharacterTextSplitter.from_language(
                    language=Language.RUST,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                ),
                "patterns": {
                    "function": r"(?:pub\s+)?fn\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<[^>]*>)?\s*\([^)]*\)\s*(?:->\s*[^{]*)?{",
                    "struct": r"(?:pub\s+)?struct\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<[^>]*>)?(?:\([^)]*\)|{)",
                    "trait": r"(?:pub\s+)?trait\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<[^>]*>)?{",
                    "impl": r"impl(?:<[^>]*>)?\s+(?:[^{]*)\s*{",
                    "enum": r"(?:pub\s+)?enum\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<[^>]*>)?{",
                }
            }
        }
        
        # Add support for more languages as needed...
        
    def _extract_code_blocks(self, text: str) -> List[Dict[str, Any]]:
        """Extract code blocks with their types and metadata"""
        settings = self.language_settings.get(self.language, self.language_settings["python"])
        patterns = settings["patterns"]
        
        blocks = []
        lines = text.split("\n")
        current_block = []
        current_type = None
        start_line = 0
        
        for i, line in enumerate(lines):
            # Check for new block start
            for block_type, pattern in patterns.items():
                if re.match(pattern, line.strip()):
                    if current_block:
                        blocks.append({
                            "text": "\n".join(current_block),
                            "metadata": SplitMetadata(
                                start_line=start_line,
                                end_line=i-1,
                                type=current_type or "code",
                                language=self.language
                            )
                        })
                    current_block = [line]
                    current_type = block_type
                    start_line = i
                    break
            else:
                current_block.append(line)
                
            # Split large blocks
            if len(current_block) >= self.max_lines_per_chunk:
                blocks.append({
                    "text": "\n".join(current_block),
                    "metadata": SplitMetadata(
                        start_line=start_line,
                        end_line=i,
                        type=current_type or "code",
                        language=self.language
                    )
                })
                current_block = []
                current_type = None
                start_line = i + 1
        
        # Add remaining block
        if current_block:
            blocks.append({
                "text": "\n".join(current_block),
                "metadata": SplitMetadata(
                    start_line=start_line,
                    end_line=len(lines)-1,
                    type=current_type or "code",
                    language=self.language
                )
            })
        
        return blocks

    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split code into chunks while preserving function and class definitions.
        
        Args:
            text: Source code text to split
            metadata: Additional metadata to include
            
        Returns:
            List of dictionaries containing text chunks and their metadata
        """
        # Extract code blocks
        blocks = self._extract_code_blocks(text)
        
        # Further split large blocks if needed
        settings = self.language_settings.get(self.language, self.language_settings["python"])
        splitter = settings["splitter"]
        
        final_chunks = []
        for block in blocks:
            if len(block["text"]) > self.chunk_size:
                sub_chunks = splitter.split_text(block["text"])
                for i, chunk in enumerate(sub_chunks):
                    final_chunks.append({
                        "text": chunk,
                        "metadata": {
                            **block["metadata"].__dict__,
                            "sub_chunk": i,
                            **(metadata or {})
                        }
                    })
            else:
                final_chunks.append({
                    "text": block["text"],
                    "metadata": {
                        **block["metadata"].__dict__,
                        **(metadata or {})
                    }
                })
        
        return final_chunks

class JupyterNotebookSplitter(BaseSplitter):
    """
    Specialized splitter for Jupyter notebooks that preserves cell structure
    and handles both code and markdown content appropriately.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_lines_per_cell: int = 100,
        include_outputs: bool = True,
        max_output_length: int = 500
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_lines_per_cell = max_lines_per_cell
        self.include_outputs = include_outputs
        self.max_output_length = max_output_length
        
        # Initialize specialized splitters for different content types
        self.code_splitter = CodeSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_lines_per_chunk=max_lines_per_cell
        )
        
        self.markdown_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def _process_cell_output(self, output: Dict[str, Any]) -> str:
        """Process and format cell output"""
        if output["output_type"] == "stream":
            return output.get("text", "")
        elif output["output_type"] == "execute_result":
            return str(output.get("data", {}).get("text/plain", ""))
        elif output["output_type"] == "display_data":
            # Handle different data formats, prioritize text
            data = output.get("data", {})
            if "text/plain" in data:
                return str(data["text/plain"])
            return str(data)
        elif output["output_type"] == "error":
            return f"Error: {output.get('ename', '')}: {output.get('evalue', '')}"
        return ""

    def _process_cell(self, cell: Dict[str, Any], cell_index: int) -> List[Dict[str, Any]]:
        """Process individual notebook cell"""
        chunks = []
        cell_type = cell["cell_type"]
        source = cell["source"]
        
        # Base metadata for the cell
        base_metadata = SplitMetadata(
            start_line=0,  # In notebooks, we use cell index instead
            end_line=0,
            type=cell_type,
            parent_type="notebook_cell"
        )

        # Process based on cell type
        if cell_type == "code":
            # Split code content
            code_chunks = self.code_splitter.split_text(source)
            for chunk in code_chunks:
                chunk["metadata"].update({
                    "cell_index": cell_index,
                    "parent_type": "notebook_cell"
                })
                chunks.append(chunk)
            
            # Process outputs if included
            if self.include_outputs and "outputs" in cell:
                output_text = "\n".join(
                    self._process_cell_output(output) 
                    for output in cell["outputs"]
                )
                if output_text.strip():
                    output_text = output_text[:self.max_output_length]
                    chunks.append({
                        "text": output_text,
                        "metadata": {
                            **base_metadata.__dict__,
                            "cell_index": cell_index,
                            "type": "output"
                        }
                    })
                    
        elif cell_type == "markdown":
            # Split markdown content
            md_chunks = self.markdown_splitter.split_text(source)
            for chunk in md_chunks:
                chunks.append({
                    "text": chunk,
                    "metadata": {
                        **base_metadata.__dict__,
                        "cell_index": cell_index,
                        "type": "markdown"
                    }
                })
                
        return chunks

    def split_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Split Jupyter notebook content while preserving cell structure.
        
        Args:
            text: Notebook content as JSON string
            metadata: Additional metadata to include
            
        Returns:
            List of dictionaries containing text chunks and their metadata
        """
        try:
            # Parse notebook content
            notebook = json.loads(text) if isinstance(text, str) else text
            
            # Process each cell
            chunks = []
            for i, cell in enumerate(notebook["cells"]):
                cell_chunks = self._process_cell(cell, i)
                for chunk in cell_chunks:
                    chunk["metadata"].update(metadata or {})
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            raise ValueError(f"Error processing notebook: {str(e)}")


