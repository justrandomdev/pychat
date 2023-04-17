import os
import qdrant_client
from typing import List
from langchain import OpenAI, ConversationChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from scraper import Scraper
import nest_asyncio
nest_asyncio.apply()


class OpenAIChat:
    def __init__(self, model_name: str, api_key, temp=0.7, conversation_buffer_token_limit=1000):
        self.__model_name = model_name
        self.__temperature = temp
        self._api_key = api_key
        self.__conversation_buffer_token_limit = conversation_buffer_token_limit

        self.__llm_factory()

    def __llm_factory(self):
        self._llm = OpenAI(temperature=self.__temperature,
                           openai_api_key=self._api_key,
                           model_name=self.__model_name)

        self.__conversation_buffer = ConversationSummaryBufferMemory(
            llm=self._llm,
            max_token_limit=self.__conversation_buffer_token_limit
        )
        self._conversation_chain = ConversationChain(
            llm=self._llm,
            memory=self.__conversation_buffer
        )

    def query(self, query) -> str:
        return self._conversation_chain.run(query)

    @property
    def temperature(self)->float:
        return self.__temperature

    @temperature.setter
    def temperature(self, value):
        self.__temperature = value
        self.__llm_factory()

    @property
    def model_name(self)->str:
        return self.__model_name

    @model_name.setter
    def model_name(self, value):
        self.__model_name = value
        self.__llm_factory()

    @property
    def conversation_buffer(self)->ConversationSummaryBufferMemory:
        return self.__conversation_buffer

    @property
    def conversation_chain(self)->ConversationChain:
        return self._conversation_chain


class OpenAISitemapWebSearch(OpenAIChat):
    def __init__(self, model_name: str, api_key: str, url: str, db_path: str, collection_name: str,
                 filter_urls: dict = List[str], temp: float = 0.7, load_docs_from_source: bool = False,
                 conversation_buffer_token_limit: int = 1000, chunk_size=1000, max_pages=0):
        super().__init__(model_name, api_key, temp, conversation_buffer_token_limit)

        if filter_urls is None:
            filter_urls = []
        self.__chunk_size = chunk_size
        self.__url = url
        self.__collection_name = collection_name
        self.__db_path = db_path
        self.__filter_urls = filter_urls
        self.__max_pages = max_pages
        self.__store = None

        self.__document_loader_factory(load_docs_from_source)

    def query(self, query) -> str:
        docs = self.__store.similarity_search(query)
        result = self._conversation_chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        if result is not None:
            return result["output_text"]
        else:
            return "No response returned"


    def __document_loader_factory(self, load_docs_from_source):
        scraper = Scraper(self.__url, max_pages=self.__max_pages)
        scraper.crawl()
        docs = scraper.pages

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.__chunk_size,
            chunk_overlap=20,
            length_function=len,
        )

        docs = text_splitter.split_documents(docs)
        embeddings = OpenAIEmbeddings(
            openai_api_key=self._api_key,
            chunk_size=self.__chunk_size
        )

        if load_docs_from_source or not os.path.exists(self.__db_path):
            self.__embed_source(docs, embeddings)
        else:
            self.__connect_store(embeddings)

        self._conversation_chain = load_qa_with_sources_chain(self._llm, chain_type="stuff")

    def __connect_store(self, embeddings: OpenAIEmbeddings):
        client = qdrant_client.QdrantClient(
            path=self.__db_path, prefer_grpc=True
        )
        self.__store = Qdrant(
            client=client, collection_name=self.__collection_name,
            embedding_function=embeddings.embed_query
        )

    def __embed_source(self, docs: List[Document], embeddings: OpenAIEmbeddings):
        self.__store = Qdrant.from_documents(
            docs, embeddings,
            path=self.__db_path,
            collection_name=self.__collection_name
        )






