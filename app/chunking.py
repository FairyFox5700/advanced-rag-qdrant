"""Document loading and two-granularity chunking (child + parent) used by
the parent-document retrieval strategy (checklist item 5) and shared by the
basic ingestion pipeline (checklist item 1).
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings


@dataclass
class Document:
    doc_id: str
    source: str
    text: str


@dataclass
class ParentChunk:
    parent_id: str
    source: str
    text: str


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    source: str
    text: str


def load_documents(directory: str | None = None) -> list[Document]:
    """Load every .txt/.md file in `directory` into a Document."""
    directory = directory or settings.data_dir
    docs: list[Document] = []
    for root, _dirs, files in os.walk(directory):
        for fname in sorted(files):
            if not fname.lower().endswith((".txt", ".md")):
                continue
            path = os.path.join(root, fname)
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
            doc_id = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
            docs.append(Document(doc_id=doc_id, source=fname, text=text))
    return docs


def split_into_parents(doc: Document) -> list[ParentChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_chunk_overlap,
    )
    parents = []
    for i, chunk in enumerate(splitter.split_text(doc.text)):
        parent_id = f"{doc.doc_id}-p{i}"
        parents.append(ParentChunk(parent_id=parent_id, source=doc.source, text=chunk))
    return parents


def split_into_children(parent: ParentChunk) -> list[ChildChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    children = []
    for j, chunk in enumerate(splitter.split_text(parent.text)):
        child_id = f"{parent.parent_id}-c{j}"
        children.append(
            ChildChunk(child_id=child_id, parent_id=parent.parent_id, source=parent.source, text=chunk)
        )
    return children


def chunk_document(doc: Document) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Return (parents, children) for a document using the two-level splitter."""
    parents = split_into_parents(doc)
    children: list[ChildChunk] = []
    for parent in parents:
        children.extend(split_into_children(parent))
    return parents, children
