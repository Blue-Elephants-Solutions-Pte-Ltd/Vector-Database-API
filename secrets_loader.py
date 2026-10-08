"""
Optionally loads real secrets (API keys, Qdrant credentials, etc.) from
Infisical into os.environ, so every existing os.getenv(...) call across the
codebase keeps working unchanged. Must run before any other internal module
is imported.

Infisical is opt-in: if none of the INFISICAL_* connection vars below are set
in .env, this is a no-op and the app falls back to whatever API keys are
already in .env / the environment directly. If some (but not all) of them are
set, that's treated as a misconfiguration and fails fast.

  INFISICAL_CLIENT_ID
  INFISICAL_CLIENT_SECRET
  INFISICAL_PROJECT_ID
  INFISICAL_ENVIRONMENT    (environment slug configured in Infisical)
  INFISICAL_SITE_URL       (optional, defaults to Infisical Cloud)
"""

import os

from infisical_sdk import InfisicalSDKClient

DEFAULT_SITE_URL = "https://app.infisical.com"

REQUIRED_CONNECTION_VARS = (
    "INFISICAL_CLIENT_ID",
    "INFISICAL_CLIENT_SECRET",
    "INFISICAL_PROJECT_ID",
    "INFISICAL_ENVIRONMENT",
)

_loaded = False


def load_secrets():
    """Fetch secrets from Infisical and populate os.environ, if configured.

    No INFISICAL_* vars set -> no-op (use direct API keys from .env).
    Some but not all set     -> fail fast (likely a misconfiguration).
    All set                  -> fetch from Infisical. Fails fast on error.
    """
    global _loaded
    if _loaded:
        return

    configured = [var for var in REQUIRED_CONNECTION_VARS if os.getenv(var)]
    if not configured:
        print("Infisical not configured; using API keys from .env directly.")
        _loaded = True
        return

    missing = [var for var in REQUIRED_CONNECTION_VARS if not os.getenv(var)]
    if missing:
        raise RuntimeError(
            f"Infisical partially configured; missing {', '.join(missing)}. "
            "Set all INFISICAL_* vars to use Infisical, or unset them all to "
            "use direct API keys from .env instead."
        )

    client_id = os.getenv("INFISICAL_CLIENT_ID")
    client_secret = os.getenv("INFISICAL_CLIENT_SECRET")
    project_id = os.getenv("INFISICAL_PROJECT_ID")
    environment_slug = os.getenv("INFISICAL_ENVIRONMENT")
    site_url = os.getenv("INFISICAL_SITE_URL", DEFAULT_SITE_URL)

    try:
        client = InfisicalSDKClient(host=site_url)
        client.auth.universal_auth.login(client_id=client_id, client_secret=client_secret)

        secrets = client.secrets.list_secrets(
            project_id=project_id,
            environment_slug=environment_slug,
            secret_path="/",
            view_secret_value=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to fetch secrets from Infisical (project={project_id}, "
            f"env={environment_slug}): {exc}"
        ) from exc

    for secret in secrets.secrets:
        os.environ[secret.secretKey] = secret.secretValue

    _loaded = True
    print(f"Loaded {len(secrets.secrets)} secrets from Infisical (env={environment_slug})")
