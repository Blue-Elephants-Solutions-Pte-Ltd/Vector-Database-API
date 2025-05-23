from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mistralai import MistralAIEmbeddings , ChatMistralAI
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_aws import BedrockEmbeddings, ChatBedrock

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
                model=model_name,
                api_key=api_key,
            )
        elif provider == "default-mistral":
            embeddings = MistralAIEmbeddings(
                model=model_name,
                api_key=api_key
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
        raise (f"Provider {provider} not supported. Error: {str(e)}")
    

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
                model=model_name,
                api_key=api_key,
            )
            
        elif provider == "default-mistral":
            llm = ChatMistralAI(
                model=model_name,
                api_key=api_key
            )
            
        else:
            llm = ChatOpenAI(
                model=model_name,
                api_key=api_key,
                base_url=provider
            )
        
        return llm
        
    except Exception as e:
        raise (f"Provider {provider} not supported. Error: {str(e)}")

        
