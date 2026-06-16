from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from policy_agent.config import PDF_FILES, RAW_DATA_DIR, CHUNK_OVERLAP, CHUNK_SIZE


def validate_pdf_exists(pdf_path: Path) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"Missing PDF file: {pdf_path}\n"
            "Please add the required policy PDFs to data/raw/ai_policy/."
        )


def load_single_pdf(pdf_path: Path, source_name: str, file_name: str) -> list[Document]:
    """
    Load one PDF file and attach clean metadata to each page.

    Args:
        pdf_path: Full path to the PDF file.
        source_name: Internal source name, e.g. 'eu_ai_act'.
        file_name: Original PDF filename, e.g. 'eu_ai_act.pdf'.

    Returns:
        A list of LangChain Document objects, one per PDF page.
    """
    validate_pdf_exists(pdf_path)

    # PyPDFLoader extracts text page by page.
    # Each page becomes a Document object.
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    # Add our own metadata.
    # This will later help with source filtering and citations.
    for page in pages:
        page.metadata["source_name"] = source_name
        page.metadata["file_name"] = file_name

    return pages


def load_pdf_documents() -> list[Document]:
    """
    Load all configured AI policy PDFs.

    Returns:
        A combined list of Document objects from all PDFs.
    """
    documents = []

    for source_name, file_name in PDF_FILES.items():
        pdf_path = RAW_DATA_DIR / file_name

        pages = load_single_pdf(
            pdf_path=pdf_path,
            source_name=source_name,
            file_name=file_name,
        )

        documents.extend(pages)

    return documents


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """
    Create the text splitter used for chunking documents.

    Returns:
        A configured RecursiveCharacterTextSplitter.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split long documents into smaller chunks.

    Args:
        documents: PDF page-level Document objects.

    Returns:
        Smaller chunk-level Document objects.
    """
    splitter = create_text_splitter()
    chunks = splitter.split_documents(documents)

    return chunks


def load_and_split_documents() -> list[Document]:
    """
    Load all PDFs and split them into chunks.

    Returns:
        Chunked Document objects ready for vector storage.
    """
    documents = load_pdf_documents()
    chunks = split_documents(documents)

    return chunks
