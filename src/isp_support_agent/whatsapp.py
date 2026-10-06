"""Cloud API sender; explicitly configured, never used by the offline demo."""

import httpx


class CloudMessageSender:
    def __init__(self, settings, client=None):
        self.url = settings.whatsapp_base_url
        self.token = settings.whatsapp_access_token
        self.client = client or httpx.Client(timeout=8, trust_env=False)

    def _post(self, body):
        response = self.client.post(
            self.url,
            json={"messaging_product": "whatsapp", **body},
            headers={"Authorization": f"Bearer {self.token}"},
        )
        response.raise_for_status()

    def read_and_typing(self, sender_id, message_id):
        self._post(
            {"status": "read", "message_id": message_id, "typing_indicator": {"type": "text"}}
        )

    def send(self, sender_id, message, key):
        self._post({"to": sender_id, "type": "text", "text": {"body": message}})

    def close(self):
        self.client.close()
