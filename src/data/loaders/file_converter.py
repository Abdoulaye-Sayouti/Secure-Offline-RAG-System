import subprocess
from pathlib import Path
import logging

class OfficeConverter:
    """
    A utility class for converting Microsoft Office and OpenDocument files to PDF format.
    
    This class provides functionality to convert various office document formats to PDF
    using LibreOffice's command-line interface. It supports both single file conversion
    and batch processing of directories.
    
    Features:
        - Single file conversion to PDF
        - Batch conversion of directories
        - Recursive directory processing
        - Custom output naming with prefix/suffix support
        - Comprehensive error handling and logging
    
    Supported Formats:
        - Microsoft Office: .docx, .doc, .ppt, .pptx
        - OpenDocument: .odt
    
    Requirements:
        - LibreOffice must be installed on the system
        - System command 'soffice' must be available
    
    Attributes:
        logger: Configured logging instance for tracking operations
        SUPPORTED_FORMATS (dict): Mapping of file extensions to format descriptions
    """
    
    # Define supported formats as a class attribute with descriptions
    SUPPORTED_FORMATS = {
        # Microsoft Office formats
        '.docx': 'Word Document',
        '.doc': 'Legacy Word Document',
        '.ppt': 'PowerPoint Presentation',
        '.pptx': 'PowerPoint Presentation',
        # OpenDocument formats
        '.odt': 'OpenDocument Text'
    }

    def __init__(self):
        """
        Initialize the OfficeConverter and verify LibreOffice installation.
        
        Raises:
            SystemError: If LibreOffice is not installed on the system
        
        Note:
            Checks for LibreOffice installation using the 'which' command
            and sets up logging configuration.
        """
        # Verify LibreOffice installation
        try:
            subprocess.run(['which', 'soffice'], check=True, capture_output=True)
        except subprocess.CalledProcessError:
            raise SystemError("LibreOffice is not installed. Please install it using: sudo apt-get install libreoffice")

        self.logger = self._setup_logger()

    def _setup_logger(self):
        """
        Configure and set up the logger for the OfficeConverter.
        
        Returns:
            logging.Logger: Configured logger instance
            
        Note:
            Sets up logging with timestamp, level, and message formatting
            at INFO level with console output.
        """
        logger = logging.getLogger('OfficeConverter')
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def convert_to_pdf(self, input_file, output_file=None, output_dir=None):
        """
        Convert a single office document to PDF format.
        
        This method handles the conversion of various office document formats to PDF,
        with support for custom output locations and naming.
        
        Args:
            input_file (str): Path to the input document
            output_file (str, optional): Desired name for the output PDF (without extension)
            output_dir (str, optional): Directory for the output PDF
            
        Returns:
            str: Path to the converted PDF file
            
        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If input file format is not supported
            RuntimeError: If conversion process fails
            
        Example:
            ```python
            converter = OfficeConverter()
            pdf_path = converter.convert_to_pdf(
                "document.docx",
                output_file="converted_doc",
                output_dir="output/pdfs"
            )
            ```
        """
        input_path = Path(input_file)
        
        # Validate input file
        if not input_path.exists():
            raise FileNotFoundError(f"Input file {input_file} does not exist")
            
        if input_path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported input format: {input_path.suffix}\n"
                f"Supported formats: {', '.join(self.SUPPORTED_FORMATS.keys())}"
            )

        # Setup output directory
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = input_path.parent

        try:
            self.logger.info(f"Converting {input_file} ({self.SUPPORTED_FORMATS[input_path.suffix.lower()]}) to PDF format...")
            
            # Set output path
            if output_file:
                output_path = output_dir / f"{output_file}.pdf"
            else:
                output_path = output_dir / f"{input_path.stem}.pdf"

            # Configure and execute conversion command
            cmd = [
                'soffice',
                '--headless',
                '--convert-to',
                'pdf',
                '--outdir',
                str(output_dir),
                str(input_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # Handle custom output name
            if output_file:
                default_output = output_dir / f"{input_path.stem}.pdf"
                if default_output.exists():
                    default_output.rename(output_path)
            
            self.logger.info(f"Successfully converted to: {output_path}")
            return str(output_path)
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Conversion failed: {e.stderr}")
            raise RuntimeError(f"Conversion failed: {e.stderr}")

    def batch_convert_to_pdf(self, input_directory, output_dir=None, recursive=False, prefix=None, suffix=None):
        """
        Convert multiple office documents to PDF format in batch.
        
        This method processes all supported documents in a directory, optionally including
        subdirectories, and converts them to PDF format with customizable output naming.
        
        Args:
            input_directory (str): Directory containing files to convert
            output_dir (str, optional): Directory for output PDFs
            recursive (bool): Whether to process subdirectories (default: False)
            prefix (str, optional): Prefix to add to output filenames
            suffix (str, optional): Suffix to add to output filenames (before .pdf)
            
        Returns:
            tuple: (
                List of (input_path, output_path) for successful conversions,
                List of (input_path, error_message) for failed conversions
            )
            
        Example:
            ```python
            converter = OfficeConverter()
            converted, errors = converter.batch_convert_to_pdf(
                "docs/",
                output_dir="pdfs/",
                recursive=True,
                prefix="converted_",
                suffix="_final"
            )
            ```
        """
        input_path = Path(input_directory)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory {input_directory} does not exist")

        converted_files = []
        errors = []

        # Determine file pattern based on recursive flag
        pattern = '**/*' if recursive else '*'
        
        # Process each supported file
        for file_path in input_path.glob(pattern):
            if file_path.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    # Build output filename with optional prefix/suffix
                    output_name = ""
                    if prefix:
                        output_name += prefix
                    output_name += file_path.stem
                    if suffix:
                        output_name += suffix
                    
                    # Convert file and track result
                    output_file = self.convert_to_pdf(
                        str(file_path),
                        output_file=output_name,
                        output_dir=output_dir
                    )
                    converted_files.append((str(file_path), output_file))
                except Exception as e:
                    errors.append((str(file_path), str(e)))

        return converted_files, errors