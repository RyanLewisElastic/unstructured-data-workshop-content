# LLM connector for Agent Builder (Instruqt / self-managed Kibana)

Replacing challenges in a cloned Elastic workshop track **does not** copy Kibana
**GenAI** configuration. The LiteLLM (or OpenAI-compatible) proxy may still run in
the sandbox (often exposed via `LLM_PROXY_URL` on `kubernetes-vm` during track
bootstrap), but **Kibana has no default AI connector** until you configure one.

Without a connector, Agent Builder can deploy tools and agents, but **chat will
not work** until a default model is selected.

## What participants should do (challenge 2)

1. Open **Kibana** (browser tab).
2. Use the global search bar and open **GenAI Settings**.
3. Under **Default AI Connector**, choose a connector that points at the lab LLM.
   - If the dropdown is empty, create a connector first (steps below).
4. **Save**.

Official reference: [Model configuration in Elastic Agent Builder](https://www.elastic.co/docs/explore-analyze/ai-features/agent-builder/models).

## Creating an OpenAI-compatible connector (LiteLLM)

1. In Kibana search, open **Connectors** (under *Alerts and insights*).
2. **Create connector** → **OpenAI** (works for OpenAI-compatible proxies).
3. Set **API URL** to your proxy’s chat base URL, typically:
   - `https://<LLM_PROXY_HOST>/v1`
   - If the sandbox only sets `LLM_PROXY_URL` to a hostname (no scheme), prepend
     `https://` and append `/v1`.
4. Set **API key** to the workshop key issued for this sandbox (Elastic’s managed
   bootstrap often fetches one during `setup-kubernetes-vm`; use the same key
   your organization documents for this track—often an Instruqt secret or the key
   printed in internal bootstrap logs—not the Elasticsearch `elastic` password).
5. Under **Additional settings**, set task type **`chat_completion`** if prompted.
6. Choose a **model** your proxy exposes (e.g. `gpt-4o`—match what LiteLLM allows).
7. Save, then return to **GenAI Settings** and select this connector as **Default AI Connector**.

## Track maintainer checklist

- [ ] After `instruqt track test`, open **GenAI Settings** and confirm a default connector exists.
- [ ] Document where the **LLM API key** comes from for your org (secret, env, or facilitator-only).
- [ ] If you automate bootstrap on `kubernetes-vm`, you can optionally add a script that calls the
  [Kibana Connectors API](https://www.elastic.co/docs/api/doc/kibana/operation/operation-post-actions-connector-id)
  to create the OpenAI connector—keep secrets out of git.

## Related env vars (Elastic managed workshops)

Sandbox logs sometimes show:

- `LLM_PROXY_URL` — proxy hostname (may omit `https://` and path).
- `LLM_MODELS` — allowed model names for the proxy.

These are hints for connector URL and model selection; they are not substitutes
for configuring **GenAI Settings** in Kibana.
