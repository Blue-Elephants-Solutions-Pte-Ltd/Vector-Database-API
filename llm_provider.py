from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings , ChatMistralAI
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_aws import BedrockEmbeddings, ChatBedrock

import os 
import dotenv
dotenv.load_dotenv()

def get_embeddings_model(provider, model_name, api_key):
    try:
        if provider == "openai":
            embeddings = OpenAIEmbeddings(
                model=model_name,
                api_key=api_key,    
            )
            
        elif provider == "mistral":
            embeddings = MistralAIEmbeddings(
                model=model_name,
                api_key=api_key
            )
            
        elif provider == "cohere":
            embeddings = CohereEmbeddings(
                model_name=model_name,
                api_key=api_key
            )
        
        elif provider == "aws":
            embeddings = BedrockEmbeddings(
                model_name=model_name,
                api_key=api_key
            )
        
        elif provider == "default-openai":
            embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif provider == "default-mistral":
            embeddings = MistralAIEmbeddings(
                model="mistral-embed",
                api_key=os.getenv("MISTRAL_API_KEY")
            )
        elif provider == "default-azure": #Only for Azure OpenAI-GDPR
            print("-"*100)
            print("Default Azure")
            print("-"*100)
            from langchain_openai import AzureOpenAIEmbeddings
            api_key = os.getenv("AZURE_OPENAI_API_KEY")
            api_version = "2024-02-15-preview"
            azure_endpoint = "https://openai-germany-ai-assistant.openai.azure.com/"
            model_name = "text-embedding-3-large"
            
            embeddings = AzureOpenAIEmbeddings(
                openai_api_key = api_key,
                openai_api_version=api_version,
                azure_endpoint=azure_endpoint,
                azure_deployment=model_name
                )
        else:
            print("Else")
            embeddings = OpenAIEmbeddings(
                model=model_name,
                api_key=api_key,
                base_url=provider
            )
            
        return embeddings
    
    except Exception as e:
        raise Exception(f"Provider {provider} not supported. Error: {str(e)}")
    

def get_chat_model(provider, model_name, api_key):
    try:
        if provider == "openai":
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
            )
        
        elif provider == "mistral":
            llm = ChatMistralAI(
                model=model_name,
                api_key=api_key
            )               
        
        elif provider == "cohere":
            llm = ChatCohere(
                model=model_name,
                api_key=api_key
            )    
                        
        elif provider == "aws":
            llm = ChatBedrock(
                model=model_name,
                api_key=api_key
            )
            
        elif provider == "default-openai":
            llm = ChatOpenAI(
                model="gpt-4.1-mini",
                api_key=os.getenv("OPENAI_API_KEY"),
                default_headers={
                    "OpenAI-Data-Retention-Policy": "zero"
                }
            )
            
        elif provider == "default-mistral":
            llm = ChatMistralAI(
                model="mistral-large-latest",
                api_key=os.getenv("MISTRAL_API_KEY")
            )
            
        elif provider == "default-azure": #Only for Azure OpenAI-GDPR
            from langchain_openai import AzureChatOpenAI
            llm = AzureChatOpenAI(
                openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                openai_api_version="2024-02-15-preview",
                azure_endpoint="https://openai-germany-ai-assistant.openai.azure.com/",
                azure_deployment="gpt-4.1-mini",
                streaming=True,
                temperature=0.7,
                model_kwargs={
                    "stream_options": {"include_usage": True}
                }
            )
            
        else:
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=provider
            )
        
        return llm
        
    except Exception as e:
        raise Exception(f"Provider {provider} not supported. Error: {str(e)}")

        
