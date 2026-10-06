"""TEMPLATE ONLY: adapt paths/schema and server guarantees before use.

No real API was contacted. The provider must enforce atomic category dedupe,
permission checks and idempotency keys on its server; HTTP retries alone do not.
"""

import httpx

from isp_support_agent.models import Customer, Service, Ticket


class HttpTicketSystem:
    def __init__(self, base_url, token, client=None):
        self.client = client or httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=5,
            trust_env=False,
        )

    def customer(self, customer_id):
        response = self.client.get(f"/customers/{customer_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return Customer.model_validate(response.json())

    def services(self, customer_id):
        return [
            Service.model_validate(row) for row in self._get(f"/customers/{customer_id}/services")
        ]

    def tickets(self, customer_id):
        return [
            Ticket.model_validate(row) for row in self._get(f"/customers/{customer_id}/tickets")
        ]

    def categories(self):
        return tuple(self._get("/categories"))

    def create_ticket(self, customer_id, category, note, key):
        return self._write(
            "/tickets", {"customer_id": customer_id, "category": category, "note": note}, key
        )

    def add_note(self, ticket_id, note, key):
        return self._write(f"/tickets/{ticket_id}/notes", {"note": note}, key)

    def resolve(self, ticket_id, key):
        return self._write(f"/tickets/{ticket_id}/resolve", {}, key)

    def _get(self, path):
        response = self.client.get(path)
        response.raise_for_status()
        return response.json()

    def _write(self, path, body, key):
        response = self.client.post(path, json=body, headers={"Idempotency-Key": key})
        response.raise_for_status()
        return Ticket.model_validate(response.json())

    def close(self):
        self.client.close()
