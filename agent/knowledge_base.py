POLICY_DOCUMENTS = [
    {
        "id": "policy_01",
        "title": "Apparel and Footwear Return Window",
        "text": (
            "Apparel and footwear items are eligible for return within 7 days "
            "of delivery when the item meets the applicable return conditions. "
            "The product should be unused and in its original condition with "
            "tags and packaging where applicable."
        ),
    },
    {
        "id": "policy_02",
        "title": "Electronics Return Window",
        "text": (
            "Eligible electronics can be returned within 7 days of delivery "
            "subject to the product-specific return policy. Serial numbers, "
            "accessories, and original packaging may need to be intact."
        ),
    },
    {
        "id": "policy_03",
        "title": "Home Products Return Window",
        "text": (
            "Eligible home products can generally be returned within 7 days "
            "of delivery when the item satisfies the applicable return "
            "conditions. The product should be unused and in acceptable "
            "original condition."
        ),
    },
    {
        "id": "policy_04",
        "title": "COD Refund Timeline",
        "text": (
            "For an eligible COD return, the refund is initiated after the "
            "returned product passes the required verification. The refund "
            "timeline can depend on the refund method selected and applicable "
            "processing requirements."
        ),
    },
    {
        "id": "policy_05",
        "title": "Prepaid Refund Timeline",
        "text": (
            "For eligible prepaid orders, refunds are processed after the "
            "return is approved and the required verification is completed. "
            "The amount is normally returned through the applicable original "
            "payment route."
        ),
    },
    {
        "id": "policy_06",
        "title": "Delivery SLA",
        "text": (
            "The estimated delivery date shown for an order is the primary "
            "delivery estimate. Delivery times can vary based on seller "
            "processing, destination, logistics conditions, and other "
            "operational factors."
        ),
    },
    {
        "id": "policy_07",
        "title": "Delayed Delivery",
        "text": (
            "If an order passes its estimated delivery date, the customer "
            "should first check the latest tracking status. A delayed shipment "
            "may receive a revised delivery estimate as logistics information "
            "is updated."
        ),
    },
    {
        "id": "policy_08",
        "title": "Reverse Pickup Eligibility",
        "text": (
            "Reverse pickup is available only for orders and locations where "
            "the applicable return service is supported. Pickup availability "
            "can depend on product category, seller policy, and serviceability."
        ),
    },
    {
        "id": "policy_09",
        "title": "Return Condition",
        "text": (
            "Returned products should normally be in the condition required "
            "by their applicable return policy. Missing accessories, damaged "
            "items, or signs of use may affect return eligibility."
        ),
    },
    {
        "id": "policy_10",
        "title": "Damaged Product",
        "text": (
            "Customers should report a product that arrives damaged as soon "
            "as possible through the applicable support or return process. "
            "The return request may require verification of the reported issue."
        ),
    },
    {
        "id": "policy_11",
        "title": "Wrong Product Received",
        "text": (
            "If the delivered product differs from the ordered product, the "
            "customer should raise the issue through the return or support "
            "flow. The request may be reviewed against the order and product "
            "details."
        ),
    },
    {
        "id": "policy_12",
        "title": "Return Pickup Process",
        "text": (
            "After an eligible return request is approved, a reverse pickup "
            "may be scheduled when the location is serviceable. The customer "
            "should keep the product ready with required accessories and "
            "packaging where applicable."
        ),
    },
    {
        "id": "policy_13",
        "title": "Order Cancellation",
        "text": (
            "Order cancellation availability depends on the current order "
            "status. Once an order has progressed to certain fulfillment "
            "stages, cancellation may no longer be available through the "
            "standard cancellation flow."
        ),
    },
    {
        "id": "policy_14",
        "title": "Refund After Return Verification",
        "text": (
            "A refund for an eligible return is processed after the returned "
            "item completes the required verification process. The final "
            "refund amount and timing can depend on the applicable order and "
            "payment conditions."
        ),
    },
]


def get_policy_documents():
    """Return the complete policy knowledge base."""
    return POLICY_DOCUMENTS