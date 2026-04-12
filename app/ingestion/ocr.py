"""Stage 2: OCR — extract text from image-based regions."""

import io

import pytesseract
from PIL import Image

from app.ingestion.layout_detection import DocumentRegion


async def apply_ocr(regions: list[DocumentRegion]) -> list[DocumentRegion]:
    """Run OCR on image regions; pass through text-based regions unchanged."""
    processed: list[DocumentRegion] = []

    for region in regions:
        if region.element_type == "Image" and region.metadata.get("image_bytes"):
            image = Image.open(io.BytesIO(region.metadata["image_bytes"]))
            ocr_text = pytesseract.image_to_string(image).strip()
            if ocr_text:
                processed.append(
                    DocumentRegion(
                        content=ocr_text,
                        element_type="OCRText",
                        page_number=region.page_number,
                        metadata=region.metadata,
                    )
                )
        else:
            if region.content.strip():
                processed.append(region)

    return processed
