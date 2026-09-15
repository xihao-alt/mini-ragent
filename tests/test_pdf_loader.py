"""Generate and validate the local NovaShop customer-support PDF."""

from __future__ import annotations

import textwrap
from pathlib import Path

from rag.loader import load_pdf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE_PDF = PROJECT_ROOT / "data" / "docs" / "novashop_support.pdf"

SECTIONS = (
    ("1. Products and Orders", "NovaShop sells electronics, home goods, apparel, and digital products. An order is confirmed only after the customer receives an order confirmation number. In-stock physical items normally enter Packing within 30 minutes. Preorder release dates are estimates and may change; customers are notified by email when a date changes."),
    ("2. Shipping and Tracking", "Standard domestic shipping takes 3-5 business days after dispatch. Express shipping takes 1-2 business days. Orders may need 1-2 business days for processing before dispatch. Tracking can take 24 hours to show movement. A package marked delivered should be checked with household members and neighbors; if still missing after 24 hours, contact NovaShop."),
    ("3. Address Changes and Cancellation", "Customers may change the delivery address or cancel an order only while its status is Confirmed. Most orders remain in that status for no more than 30 minutes. Once an order is Packing or Shipped, NovaShop cannot change the address or cancel it. For shipped orders, the customer may ask the carrier for redirection or start a return after delivery."),
    ("4. Returns and Non-returnable Items", "Eligible physical products may be returned within 30 calendar days of delivery. They must be unused, with original packaging, accessories, and proof of purchase. Personalized goods, downloadable or activated digital products, gift cards, final-sale items, and opened hygiene products such as earbuds or cosmetics are not eligible for change-of-mind returns. Defective-item rights still apply."),
    ("5. Refunds", "NovaShop issues an approved refund to the original payment method within 5-7 business days after the returned item passes inspection. Banks may take up to 10 additional business days to display the credit. Original express shipping fees are not refundable unless NovaShop sent the wrong item or the product arrived damaged. Refunds overdue by more than 10 business days after approval must be escalated to a human agent."),
    ("6. Exchanges", "Size or color exchanges are available within the 30-day return window when replacement stock exists. NovaShop does not reserve exchange inventory. If stock is unavailable, the item is refunded instead. A different product requires a return and a new order. Personalized, digital, final-sale, gift-card, and opened hygiene items cannot be exchanged for preference reasons."),
    ("7. Damaged, Incorrect, or Missing Items", "Report a damaged or incorrect item within 48 hours of delivery and provide the order number plus clear photos of the item, packaging, and shipping label. Report a missing item from a multi-item shipment within 7 calendar days. Keep all packaging until the claim is resolved. NovaShop may replace the item, ship the missing part, or issue a refund after verification."),
    ("8. Payment Failures and Duplicate Charges", "A failed payment does not create a confirmed order. A pending authorization from a failed attempt normally disappears within 3-5 business days. For an apparent duplicate charge, wait 24 hours and compare completed transactions rather than pending authorizations. Two completed charges for one order, or a duplicate charge still present after 24 hours, must be escalated with the order number and both transaction references."),
    ("9. Product Warranty", "NovaShop-branded electronics include a 12-month limited warranty; NovaShop-branded accessories include a 6-month limited warranty. The warranty covers manufacturing defects under normal use. It excludes accidental damage, liquid damage, cosmetic wear, unauthorized repair, misuse, and lost items. The customer must provide proof of purchase and the product serial number when one exists."),
    ("10. NovaPlus Membership", "Active NovaPlus members receive free standard domestic shipping on eligible items, double reward points, and early access to selected sales. Benefits apply only while membership is active and do not cover express shipping, oversized-item fees, or international orders. Reward points expire 12 months after they are earned and have no cash value."),
    ("11. Coupons and Promotions", "Only one coupon may be used per order unless a campaign explicitly allows stacking. Coupons must be entered before payment and cannot be applied after an order is placed. Expired coupons cannot be restored. Coupons do not apply to gift cards, taxes, shipping fees, or excluded brands stated in the campaign terms. A coupon has no cash value."),
    ("12. Account and Login Issues", "Customers should use Forgot Password to request a reset link, which expires after 30 minutes. After five failed login attempts, the account is locked for 30 minutes. Support may update an email address only after identity verification. Suspected account takeover, an unknown email change, or unauthorized orders require immediate escalation to a human security agent."),
    ("13. Human Support Escalation", "Escalate immediately for safety concerns, suspected fraud or account takeover, threats or legal demands, two completed charges for one order, or any single order worth more than USD 1,000. Also escalate after two failed self-service attempts, when identity verification cannot be completed, or when an approved refund is more than 10 business days overdue. Do not promise an outcome before human review."),
    ("14. Frequently Asked Questions", "Can I cancel after shipment? No; request carrier redirection or return the order after delivery. Where is my tracking update? Allow 24 hours after dispatch. Can I return a final-sale item because I changed my mind? No. When will my refund arrive? NovaShop sends it in 5-7 business days after inspection, then the bank may need up to 10 more. Can two coupons be combined? Usually no."),
)


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _page_stream(lines: list[str]) -> bytes:
    commands = ["BT", "/F1 10 Tf", "54 748 Td", "13 TL"]
    for index, line in enumerate(lines):
        if index:
            commands.append("T*")
        commands.append(f"({_escape_pdf_text(line)}) Tj")
    commands.append("ET")
    return "\n".join(commands).encode("ascii")


def create_knowledge_base_pdf(path: Path) -> None:
    """Create a readable, dependency-free, multi-page support knowledge base."""
    lines = ["NovaShop Customer Support Knowledge Base", "Version 1.0 - Fictional Training Data", ""]
    for heading, body in SECTIONS:
        lines.extend((heading, *textwrap.wrap(body, width=88), ""))
    pages = [lines[start : start + 50] for start in range(0, len(lines), 50)]
    font_id = 3 + len(pages)
    content_start = font_id + 1
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        (f"<< /Type /Pages /Kids [{' '.join(f'{3 + i} 0 R' for i in range(len(pages)))}] /Count {len(pages)} >>").encode("ascii"),
    ]
    for index in range(len(pages)):
        objects.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_start + index} 0 R >>").encode("ascii"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for page in pages:
        stream = _page_stream(page)
        objects.append(f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream")
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(pdf)); pdf.extend(f"{number} 0 obj\n".encode("ascii")); pdf.extend(obj); pdf.extend(b"\nendobj\n")
    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii")); pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    pdf.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii"))
    path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(pdf)


# Backward-compatible name used by API upload tests.
create_test_pdf = create_knowledge_base_pdf


def test_load_novashop_knowledge_base() -> None:
    create_knowledge_base_pdf(KNOWLEDGE_BASE_PDF)
    extracted = load_pdf(KNOWLEDGE_BASE_PDF)
    assert "NovaShop Customer Support Knowledge Base" in extracted
    assert len(SECTIONS) == 14
    for heading, _ in SECTIONS:
        assert heading in extracted
    assert "30 calendar days of delivery" in extracted
    assert "5-7 business days" in extracted
    assert "account takeover" in extracted


if __name__ == "__main__":
    test_load_novashop_knowledge_base()
