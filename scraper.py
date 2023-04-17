import requests
from typing import List, Tuple
from urllib.parse import urlparse
from urllib.parse import urljoin
from bs4 import BeautifulSoup
from langchain.schema import Document


class Scraper:
    def __init__(self, start_url: str, max_pages: int = 0):
        self.__url: str = start_url
        self.__seen: List[str] = []
        self.__pages: List[Document] = []
        self.__max_pages = max_pages
        self.__hostname = urlparse(start_url).netloc

    @property
    def pages(self) -> List[Document]:
        return self.__pages

    def crawl(self):
        if not self.__valid_url(self.__url):
            raise ValueError("Invalid URL")

        self.__crawl_page(self.__url)


    def __crawl_page(self, url: str):
        if not self.__valid_url(url):
            print("Invalid url: " + url)
            return

        if url not in self.__seen and self.__hostname == urlparse(url).netloc:
            links, doc = self.__get_pages(url)
            self.__pages.append(doc)
            self.__seen.append(url)

            # check if there's a page limit
            if self.__max_pages > 0 and len(self.__pages) >= self.__max_pages:
                return

            for link in links:
                if len(self.__pages) >= self.__max_pages:
                    break

                absolute = self.__abs_url(url, link)
                self.__crawl_page(absolute)


    def __abs_url(self, base_url: str, url: str):
        return requests.compat.urljoin(base_url, url)

    def __get_pages(self, url: str) -> tuple[list[str], Document]:
        links: List[str] = []
        docs: List[Document] = []

        try:
            with requests.Session() as req:
                r = req.get(url)
                linkSoup = BeautifulSoup(r.content, 'html.parser')
                links = [item["href"] for item in linkSoup.find_all("a", href=True)]
                links = list(filter(lambda lnk: '#' not in lnk, links))

                contentSoup = BeautifulSoup(r.content, 'html.parser')
                content_list = [item.text for item in contentSoup.select("div")]
                content = ''.join(content_list)
                doc = Document(page_content=content, metadata={'source': url})

        except Exception as error:
            print(error)

        return links, doc

    def __valid_url(self, url: str):
        is_valid = False
        try:
            final_url = urlparse(urljoin(url, "/"))
            is_valid = (all([final_url.scheme, final_url.netloc, final_url.path])
                        and len(final_url.netloc.split(".")) > 1)
        except ValueError as error:
            print(error)

        return is_valid



