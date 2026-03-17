import os
from PIL import Image
from pytesseract import Output
import pytesseract


def process_pdf_into_image(pdf_path: str, dpi: int = 200):
    """Render a PDF into PIL images using PyMuPDF (fitz)"""
    import fitz  # PyMuPDF
    images = []
    doc = fitz.open(pdf_path)
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            mode = "RGB" if pix.n < 4 else "RGBA"
            im = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
            if mode == "RGBA":
                im = im.convert("RGB")
            images.append(im)
    finally:
        doc.close()
    return images


def process_pdf_files_into_images_with_rotation(pdf_dir: str, image_dir: str):
    """Process PDF files within a given directory into rotated images in another directory"""
    os.makedirs(image_dir, exist_ok=True)
    for pdf_file in os.listdir(pdf_dir):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(pdf_dir, pdf_file)
            images = process_pdf_into_image(pdf_path)
            base = os.path.splitext(pdf_file)[0]
            for idx, image in enumerate(images, start=1):
                image_name = f"{base}_page_{idx}.png"
                osd_data = pytesseract.image_to_osd(image, output_type=Output.DICT)
                rotate_angle = osd_data['rotate']
                print("Detected rotation angle: ", rotate_angle)
                image = image.rotate(-rotate_angle, expand=True)
                out_path = os.path.join(image_dir, image_name)
                image.save(out_path)
