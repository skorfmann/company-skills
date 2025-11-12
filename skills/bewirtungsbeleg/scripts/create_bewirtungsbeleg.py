#!/usr/bin/env python3
"""
Script to create a German Bewirtungsbeleg (entertainment expense receipt) PDF.
"""

import argparse
import json
import os
import yaml
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from pypdf import PdfReader, PdfWriter


def load_config():
    """
    Load configuration from config.yml file.
    Falls back to generic placeholder if config file not found.

    Returns:
        dict: Configuration dictionary with 'gastgeber' key
    """
    # Try to find config file relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'config.yml')

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                if config and isinstance(config, dict):
                    return config
        except Exception as e:
            print(f"Warning: Could not load config.yml: {e}")

    # Return default if config not found or invalid
    return {
        'gastgeber': 'Your Name / Your Company'
    }


def apply_exif_orientation(img):
    """
    Apply EXIF orientation to correctly rotate image based on camera metadata.

    Args:
        img: PIL Image object

    Returns:
        PIL Image object with correct orientation
    """
    try:
        # Get EXIF data
        exif = img.getexif()
        if exif is None:
            return img

        # EXIF orientation tag is 274
        orientation = exif.get(274, 1)

        # Apply rotation based on orientation value
        if orientation == 1:
            # Normal - no rotation needed
            pass
        elif orientation == 2:
            # Mirrored horizontally
            img = img.transpose(PILImage.FLIP_LEFT_RIGHT)
        elif orientation == 3:
            # Rotated 180 degrees
            img = img.rotate(180, expand=True)
        elif orientation == 4:
            # Mirrored vertically
            img = img.transpose(PILImage.FLIP_TOP_BOTTOM)
        elif orientation == 5:
            # Mirrored horizontally then rotated 90 degrees CCW
            img = img.transpose(PILImage.FLIP_LEFT_RIGHT)
            img = img.rotate(90, expand=True)
        elif orientation == 6:
            # Rotated 90 degrees CCW
            img = img.rotate(270, expand=True)
        elif orientation == 7:
            # Mirrored horizontally then rotated 90 degrees CW
            img = img.transpose(PILImage.FLIP_LEFT_RIGHT)
            img = img.rotate(270, expand=True)
        elif orientation == 8:
            # Rotated 90 degrees CW
            img = img.rotate(90, expand=True)

        print(f"Applied EXIF orientation: {orientation}")
    except Exception as e:
        print(f"Could not apply EXIF orientation: {e}")

    return img


def image_to_pdf(image_path, output_pdf):
    """
    Convert an image file to PDF.
    Applies EXIF orientation correction and creates a PDF with the image.

    Args:
        image_path: Path to image file
        output_pdf: Path to output PDF file
    """
    from reportlab.pdfgen import canvas
    from PIL import Image as PILImage
    import tempfile
    import os

    try:
        # Open image and apply EXIF orientation
        img = PILImage.open(image_path)
        img = apply_exif_orientation(img)

        # Handle different color modes
        if img.mode == 'RGBA':
            # RGBA - composite onto white background
            background = PILImage.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3] if len(img.split()) == 4 else None)
            img = background
        elif img.mode == 'LA':
            # LA (grayscale with alpha) - composite onto white
            background = PILImage.new('L', img.size, 255)
            background.paste(img, mask=img.split()[1] if len(img.split()) == 2 else None)
            img = background.convert('RGB')
        elif img.mode not in ('RGB', 'L'):
            # Convert other modes to RGB
            img = img.convert('RGB')

        img_width, img_height = img.size

        # Create PDF with image
        c = canvas.Canvas(output_pdf, pagesize=A4)
        width, height = A4

        # Calculate scaling to fit image on A4 page
        scale = min(width / img_width, height / img_height) * 0.9  # 90% to leave margins

        new_width = img_width * scale
        new_height = img_height * scale

        # Center image on page
        x = (width - new_width) / 2
        y = (height - new_height) / 2

        # Save temp version for drawing
        temp_img = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        img.save(temp_img.name, 'PNG')
        temp_img.close()

        c.drawImage(temp_img.name, x, y, width=new_width, height=new_height)
        os.unlink(temp_img.name)

        c.save()
        print(f"PDF created successfully: {output_pdf}")

    except Exception as e:
        print(f"Error creating PDF: {e}")
        # Last resort: create a simple PDF with error message
        try:
            c = canvas.Canvas(output_pdf, pagesize=A4)
            c.drawString(100, 500, f"Error processing image: {str(e)}")
            c.save()
        except:
            pass



def create_bewirtungsbeleg(data, output_file, original_receipt_path=None, signature_path=None, config=None):
    """
    Create a Bewirtungsbeleg PDF.

    Args:
        data: Dictionary containing all required information
        output_file: Path to output PDF file
        original_receipt_path: Optional path to original receipt PDF/image to prepend
        signature_path: Optional path to signature image
        config: Optional configuration dictionary (will be loaded if not provided)
    """
    # Load config if not provided
    if config is None:
        config = load_config()
    # Create temporary file for the Bewirtungsbeleg page
    import tempfile
    temp_beleg = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_beleg_path = temp_beleg.name
    temp_beleg.close()
    
    doc = SimpleDocTemplate(temp_beleg_path, pagesize=A4,
                          topMargin=15*mm, bottomMargin=15*mm,
                          leftMargin=20*mm, rightMargin=20*mm)
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=10,
        textColor=colors.HexColor('#333333'),
        spaceAfter=3,
        spaceBefore=8,
        fontName='Helvetica-Bold'
    )
    
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor('#1a1a1a'),
        spaceAfter=3,
        fontName='Helvetica'
    )
    
    # Title
    story.append(Paragraph("Bewirtungsbeleg", title_style))
    story.append(Spacer(1, 6*mm))
    
    # Main information table
    main_data = [
        ['Datum der Bewirtung:', data.get('datum_bewirtung', '')],
        ['Ort der Bewirtung:', data.get('ort_bewirtung', '')],
        ['Gastgeber:', data.get('gastgeber', config.get('gastgeber', 'Your Name / Your Company'))],
    ]
    
    main_table = Table(main_data, colWidths=[60*mm, 100*mm])
    main_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1a1a1a')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(main_table)
    story.append(Spacer(1, 5*mm))
    
    # Guests section
    story.append(Paragraph("Bewirtete Personen:", heading_style))
    
    guests = data.get('gaeste', [])
    if guests:
        guest_data = []
        for guest in guests:
            name = guest.get('name', '')
            company = guest.get('unternehmen', '')
            if company:
                guest_data.append([f"• {name}", f"({company})"])
            else:
                guest_data.append([f"• {name}", ''])
        
        # Adjust column widths based on whether companies are present
        has_companies = any(guest.get('unternehmen', '') for guest in guests)
        if has_companies:
            guest_table = Table(guest_data, colWidths=[70*mm, 90*mm])
        else:
            guest_table = Table(guest_data, colWidths=[160*mm, 0*mm])
        
        guest_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1a1a1a')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(guest_table)
    story.append(Spacer(1, 5*mm))
    
    # Occasion section
    story.append(Paragraph("Anlass der Bewirtung:", heading_style))
    anlass = data.get('anlass', '')
    story.append(Paragraph(anlass, normal_style))
    story.append(Spacer(1, 5*mm))
    
    # Restaurant details
    story.append(Paragraph("Restaurantangaben:", heading_style))
    
    # Only include tax ID row if it's provided
    restaurant_data = [
        ['Restaurant:', data.get('restaurant_name', '')],
        ['Adresse:', data.get('restaurant_adresse', '')],
    ]
    
    # Add tax ID row only if provided
    tax_id = data.get('restaurant_steuernr', '').strip()
    if tax_id:
        restaurant_data.append(['Steuer-/USt-IdNr.:', tax_id])
    
    restaurant_table = Table(restaurant_data, colWidths=[60*mm, 100*mm])
    restaurant_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1a1a1a')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(restaurant_table)
    story.append(Spacer(1, 5*mm))
    
    # Cost breakdown
    story.append(Paragraph("Aufstellung der Kosten:", heading_style))
    
    gesamtbetrag = data.get('gesamtbetrag', 0.0)
    trinkgeld = data.get('trinkgeld', 0.0)
    
    cost_data = []
    
    # Add tip row only if provided
    if trinkgeld and trinkgeld > 0:
        rechnungsbetrag = gesamtbetrag - trinkgeld
        cost_data.append(['Rechnungsbetrag laut Beleg:', f"{rechnungsbetrag:.2f} €"])
        cost_data.append(['Trinkgeld:', f"{trinkgeld:.2f} €"])
        cost_data.append(['', ''])  # Empty row for spacing
        cost_data.append(['Gesamtbetrag:', f"{gesamtbetrag:.2f} €"])
    else:
        cost_data.append(['Gesamtbetrag laut Beleg:', f"{gesamtbetrag:.2f} €"])
    
    cost_table = Table(cost_data, colWidths=[100*mm, 60*mm])
    
    # Different styling depending on whether we have tip or not
    if trinkgeld and trinkgeld > 0:
        cost_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, 1), 'Helvetica'),
            ('FONTNAME', (1, 0), (1, 1), 'Helvetica'),
            ('FONTNAME', (0, 3), (0, 3), 'Helvetica-Bold'),
            ('FONTNAME', (1, 3), (1, 3), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1a1a1a')),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, 1), 3),
            ('LINEABOVE', (0, 3), (-1, 3), 1, colors.HexColor('#333333')),
            ('TOPPADDING', (0, 3), (-1, 3), 6),
        ]))
    else:
        cost_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1a1a1a')),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
    
    story.append(cost_table)
    story.append(Spacer(1, 8*mm))
    
    # Signature section
    ort_datum = f"{data.get('ort_bewirtung', '')}, {data.get('datum_bewirtung', '')}"
    
    story.append(Paragraph("Ort, Datum:", heading_style))
    story.append(Paragraph(ort_datum, normal_style))
    story.append(Spacer(1, 8*mm))
    
    story.append(Paragraph("Unterschrift Gastgeber:", heading_style))
    
    # Add signature image if provided
    if signature_path and os.path.exists(signature_path):
        sig_img = Image(signature_path, width=50*mm, height=15*mm)
        story.append(sig_img)
    else:
        story.append(Spacer(1, 10*mm))
    
    story.append(Spacer(1, 4*mm))
    
    # Footer note
    footer_note = "Hinweis: Die Originalrechnung des Restaurants ist als erste Seite beigefügt."
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=7,
        textColor=colors.HexColor('#666666'),
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )
    story.append(Paragraph(footer_note, footer_style))
    
    # Build the Bewirtungsbeleg PDF
    doc.build(story)
    
    # Merge original receipt with Bewirtungsbeleg
    if original_receipt_path and os.path.exists(original_receipt_path):
        try:
            merger = PdfWriter()
            
            # Check if original receipt is an image and convert to PDF if needed
            receipt_pdf_path = original_receipt_path
            temp_receipt_pdf = None
            
            if original_receipt_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.heic', '.heif')):
                # Convert image to PDF
                temp_receipt_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
                receipt_pdf_path = temp_receipt_pdf.name
                temp_receipt_pdf.close()
                
                print(f"Converting {os.path.basename(original_receipt_path)} to PDF...")
                image_to_pdf(original_receipt_path, receipt_pdf_path)
                
                # Validate that the PDF was created successfully
                if not os.path.exists(receipt_pdf_path) or os.path.getsize(receipt_pdf_path) < 100:
                    raise Exception("Failed to convert image to PDF - file too small or missing")
                
                # Try to read the PDF to validate it
                try:
                    test_reader = PdfReader(receipt_pdf_path)
                    if len(test_reader.pages) == 0:
                        raise Exception("Converted PDF has no pages")
                except Exception as pdf_error:
                    print(f"PDF validation failed: {pdf_error}")
                    raise Exception(f"Invalid PDF created from image: {pdf_error}")
            
            # Add original receipt as first page(s)
            receipt_reader = PdfReader(receipt_pdf_path)
            for page in receipt_reader.pages:
                merger.add_page(page)
            
            # Clean up temp receipt PDF if created
            if temp_receipt_pdf:
                os.unlink(receipt_pdf_path)
            
            # Add Bewirtungsbeleg as last page
            beleg_reader = PdfReader(temp_beleg_path)
            for page in beleg_reader.pages:
                merger.add_page(page)
            
            # Write merged PDF
            with open(output_file, 'wb') as f:
                merger.write(f)
            
            # Clean up temp file
            os.unlink(temp_beleg_path)
            
            print(f"Bewirtungsbeleg with original receipt created successfully: {output_file}")
        except Exception as e:
            print(f"Error merging PDFs: {e}")
            print(f"Bewirtungsbeleg created without original receipt: {output_file}")
            # If merge fails, just use the Bewirtungsbeleg
            import shutil
            shutil.copy(temp_beleg_path, output_file)
            os.unlink(temp_beleg_path)
    else:
        # No original receipt provided, just use the Bewirtungsbeleg
        import shutil
        shutil.copy(temp_beleg_path, output_file)
        os.unlink(temp_beleg_path)
        print(f"Bewirtungsbeleg created successfully: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Create a Bewirtungsbeleg PDF')
    parser.add_argument('--json', required=True, help='JSON file with receipt data')
    parser.add_argument('--output', required=True, help='Output PDF file path')
    parser.add_argument('--receipt', help='Original receipt PDF/image to prepend')
    parser.add_argument('--signature', help='Signature image to include')

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Load data from JSON
    with open(args.json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Determine signature path - use provided or default
    signature_path = args.signature
    if not signature_path:
        # Try to find signature in assets directory relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_signature = os.path.join(script_dir, '..', 'assets', 'signature.png')
        if os.path.exists(default_signature):
            signature_path = default_signature

    # Create PDF
    create_bewirtungsbeleg(
        data,
        args.output,
        original_receipt_path=args.receipt,
        signature_path=signature_path,
        config=config
    )


if __name__ == '__main__':
    main()
