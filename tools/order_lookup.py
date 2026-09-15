"""Read one order belonging to the authenticated session user."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from session.store import SessionStore

from .base import Tool
from .execution_context import current_execution_context


class OrderLookupTool(Tool):
    name = "order_lookup"
    description = (
        "Look up the current customer's own order by order ID. Use before answering "
        "questions about an order's delivery, payment, cancellation, damage, or refund. "
        "Then check read_docs for the relevant NovaShop policy before giving advice."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "order_id": {
                "type": "string",
                "description": "The customer's order ID, for example NS001.",
                "minLength": 1,
            }
        },
        "required": ["order_id"],
        "additionalProperties": False,
    }

    def __init__(self, store: SessionStore) -> None:
        self.store = store

    def execute(self, *, order_id: str, **kwargs: Any) -> dict[str, Any]:
        if kwargs:
            return {"error": "order_lookup only accepts order_id"}
        if not isinstance(order_id, str) or not order_id.strip():
            return {"error": "order_id must be non-empty"}
        context = current_execution_context()
        if context is None:
            return {"error": "order_lookup requires an active session context"}
        with self.store._connect() as connection:
            row = connection.execute(
                """SELECT order_id, product, order_status, payment_status,
                          completed_charge_count, shipping_status,
                          estimated_delivery, tracking_number
                   FROM orders WHERE order_id = ? AND user_id = ?""",
                (order_id.strip().upper(), context.user_id),
            ).fetchone()
        if row is None:
            return {"error": "Order not found for the current user"}
        return dict(row)
