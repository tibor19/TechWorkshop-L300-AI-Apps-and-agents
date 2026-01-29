from openai import AzureOpenAI, api_version
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.core.exceptions import AzureError


def get_openai_client(endpoint: str | None, deployment: str | None, api_key: str | None = None, api_version: str | None = None) -> AzureOpenAI:
    if not endpoint:
        raise ValueError("<model>_endpoint must be provided in environment variables")

    # Try Entra ID (managed identity) first
    try:
        credential = DefaultAzureCredential()
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            azure_ad_token_provider=get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default"),
            api_version=api_version
        )
        return client
    except AzureError:
        pass

    # Fallback to key
    if api_key:
        client = AzureOpenAI(
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_key=api_key,
            api_version=api_version
        )
        return client

    raise RuntimeError(
        "Failed to authenticate to Azure OpenAI using DefaultAzureCredential and no valid API key was provided"
    )

