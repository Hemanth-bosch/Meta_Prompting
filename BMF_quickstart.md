# **Welcome to the Bosch Model Farm**

### The Bosch Model Farm (BMF) is the official central service for Bosch-compliant API access to third-party foundation models. It provides ready-to-use, fully managed, production-ready API endpoints for a wide range of foundation models, allowing users to leverage these models without having to manage the underlying cloud infrastructure and deal with the associated compliance requirements.

[Order in ITSP](https://service-management.bosch.tech/sp?id=sc_cat_item&sys_id=e0a942274793361c4ffebf5a436d4386&table=sc_cat_item&searchTerm=model%20farm) | [Explore Model Catalog](https://inside-docupedia.bosch.com/confluence2/x/t4FROQ) | [Read Getting Started](https://inside-docupedia.bosch.com/confluence2/x/0oFROQ)

---

# **Quick Start in 3 Steps**

## 1. Order Your Subscription

First, order your BMF subscription via the **[IT Service Portal (ITSP)](https://service-management.bosch.tech/sp?id=sc_cat_item&sys_id=4ba8219fc3982a503e43879dc001315a&referrer=popular_items)**. Once your order is processed, you will receive a Welcome Email with your subscription details.

## 2. Set Your API Key

Set the API key from your Welcome Email as an environment variable.

```bash
# Replace "YOUR_API_KEY" with the key from your email
export BMF_API_KEY="YOUR_API_KEY"
```

## 3. Make Your First API Call

Use the following `curl` command to send a test request to a model endpoint.

```bash
curl \
  -X POST "https://aoai-farm.bosch-temp.com/api/openai/deployments/gpt-5-nano-2025-08-07/chat/completions?api-version=2025-04-01-preview" \
  -H "Authorization: Bearer $BMF_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-5-nano-2025-08-07","messages":[{"role": "user","content": "Hello"}]}'
```

For more detailed instructions, please refer to **[🚀 Getting Started](https://inside-docupedia.bosch.com/confluence2/spaces/FARM/pages/961642962/%F0%9F%9A%80+Getting+Started)**.
