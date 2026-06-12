"""Example client for validating a bill before approval."""

from __future__ import annotations

import requests

from scripts.config import duplicate_api_url


API_URL = duplicate_api_url()


def validate_bill_before_approval(bill_data: dict) -> bool:
    response = requests.post(
        f"{API_URL}/check_bill",
        json=bill_data,
        timeout=10,
    )
    response.raise_for_status()
    result = response.json()

    print(f"\n{'=' * 60}")
    print(f"Bill ID: {result['bill_id']}")
    print(f"Status: {'FLAGGED' if result['bill_flagged'] else 'APPROVED'}")
    print(f"{'=' * 60}\n")

    for product_check in result["products"]:
        print(f"Product: {product_check['product_name']}")
        print(f"  Decision: {product_check['decision']}")
        print(f"  Best similarity: {product_check['best_similarity_score']}")

        if product_check["flagged"]:
            print("  FLAGGED")
            print(f"  Reason: {product_check['reason']}")

            if product_check["conflicting_bills"]:
                print("  Conflicting Bills:")
                for bill in product_check["conflicting_bills"]:
                    print(
                        f"    - Order: {bill['order_id']}, "
                        f"Date: {bill['supply_order_date']}"
                    )
        else:
            print("  CLEAR")
        print()

    return not result["bill_flagged"]


if __name__ == "__main__":
    new_bill = {
        "bill_id": 999,
        "transaction_id": "TXN_2024_00999",
        "order_id": "ORD_2024_00555",
        "supply_order_date": "2024-03-15",
        "fk_central_unit": 1263,
        "products": [
            "Unbranded Computer Paper, GSM 70",
            "Reynolds Blue Ink Gel Pen",
            "TVS Compatible Ribbon Cartridge",
        ],
    }

    is_approved = validate_bill_before_approval(new_bill)

    if is_approved:
        print("Bill can be approved")
    else:
        print("Bill requires manual review")
