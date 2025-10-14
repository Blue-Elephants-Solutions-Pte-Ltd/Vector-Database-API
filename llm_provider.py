from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings , ChatMistralAI
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_aws import BedrockEmbeddings, ChatBedrock

import os 
import dotenv
dotenv.load_dotenv()

def get_embeddings_model(provider, model_name, api_key):
    try:
        # Normalize provider name to avoid mismatches due to case/whitespace
        provider_normalized = (provider or "").strip().lower()
        print(f"Provider normalized: {provider_normalized}")
        print(f"Model name: {model_name}")
        print(f"API key: {api_key}")
        if provider_normalized == "openai":
            embeddings = OpenAIEmbeddings(
                model=model_name,
                api_key=api_key,    
            )
            
        elif provider_normalized == "mistral":
            embeddings = MistralAIEmbeddings(
                model=model_name,
                api_key=api_key
            )
            
        elif provider_normalized == "cohere":
            embeddings = CohereEmbeddings(
                model_name=model_name,
                api_key=api_key
            )
        
        elif provider_normalized == "aws":
            embeddings = BedrockEmbeddings(
                model_name=model_name,
                api_key=api_key
            )
        
        elif provider_normalized == "default-openai":
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            
        elif provider_normalized == "default-mistral":
            embeddings = MistralAIEmbeddings(
                model="mistral-embed",
                api_key=os.getenv("MISTRAL_API_KEY")
            )
            
        elif provider_normalized == "default-azure": # Only for Azure OpenAI-GDPR
            print("-"*100)
            print("Default Azure")
            print("-"*100)
            from langchain_openai import AzureOpenAIEmbeddings
            api_key_to_use = os.getenv("AZURE_OPENAI_API_KEY") or api_key
            api_version = "2024-02-15-preview"
            azure_endpoint = "https://openai-germany-ai-assistant.openai.azure.com/"
            # IMPORTANT: For Azure, model_name should be the deployment name passed in
            azure_deployment = model_name
            
            embeddings = AzureOpenAIEmbeddings(
                openai_api_key=api_key_to_use,
                openai_api_version=api_version,
                azure_endpoint=azure_endpoint,
                azure_deployment=azure_deployment
                )
        else:
            print("Else")
            # Only treat provider as a custom base URL if it looks like an URL
            if isinstance(provider, str) and provider.strip().lower().startswith(("http://", "https://")):
                embeddings = OpenAIEmbeddings(
                    model=model_name,
                    api_key=api_key,
                    base_url=provider
                )
            else:
                raise ValueError("Invalid provider value. Expected known provider or http(s) base URL.")
            
        return embeddings
    
    except Exception as e:
        raise Exception(f"Provider {provider} not supported. Error: {str(e)}")
    

def get_chat_model(provider, model_name, api_key):
    try:
        # Normalize provider name to avoid mismatches due to case/whitespace
        provider_normalized = (provider or "").strip().lower()
        print(f"Provider normalized: {provider_normalized}")
        print(f"Model name: {model_name}")
        print(f"API key: {api_key}")
        
        if provider_normalized == "openai":
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
            )
        
        elif provider_normalized == "mistral":
            llm = ChatMistralAI(
                model=model_name,
                api_key=api_key
            )               
        
        elif provider_normalized == "cohere":
            llm = ChatCohere(
                model=model_name,
                api_key=api_key
            )    
                        
        elif provider_normalized == "aws":
            llm = ChatBedrock(
                model=model_name,
                api_key=api_key
            )
            
        elif provider_normalized == "default-openai":
            llm = ChatOpenAI(
                model="gpt-4.1-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                default_headers={
                    "OpenAI-Data-Retention-Policy": "zero"
                }
            )
            
        elif provider_normalized == "default-mistral":
            llm = ChatMistralAI(
                model="mistral-large-latest",
                api_key=os.getenv("MISTRAL_API_KEY")
            )
            
        elif provider_normalized == "default-azure": # Only for Azure OpenAI-GDPR
            from langchain_openai import AzureChatOpenAI
            print(f"Azure Chat OpenAI")
            llm = AzureChatOpenAI(
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY") or api_key,
                openai_api_version="2024-02-15-preview",
                azure_endpoint="https://openai-germany-ai-assistant.openai.azure.com/",
                # For Azure, model_name should be the chat deployment name passed in
                azure_deployment=model_name,
                streaming=True,
                temperature=0.7,
                model_kwargs={
                    "stream_options": {"include_usage": True}
                }
            )
            
        else:
            # Only treat provider as a custom base URL if it looks like an URL
            if isinstance(provider, str) and provider.strip().lower().startswith(("http://", "https://")):
                llm = ChatOpenAI(
                    model=model_name,
                    api_key=api_key,
                    base_url=provider
                )
            else:
                raise ValueError("Invalid provider value. Expected known provider or http(s) base URL.")
        
        return llm
        
    except Exception as e:
        raise Exception(f"Provider {provider} not supported. Error: {str(e)}")

        
