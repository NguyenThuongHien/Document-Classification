import scrapy
import io
import os
import time
import logging

from article import LinkSpider
from bs4 import BeautifulSoup
from scrapy import signals
from scrapy.crawler import CrawlerProcess


def formalize(str):
    str = str.replace('\n', ' ')
    str = str.replace('\r', ' ')
    str = str.replace('\t', ' ')
    str = str.replace('\t', ' ')
    str = str.replace('\t', ' ')
    str = str.replace('     ', ' ')
    str = str.replace('     ', ' ')
    str = str.replace('    ', ' ')
    str = str.replace('    ', ' ')
    str = str.replace('   ', ' ')
    str = str.replace('   ', ' ')
    str = str.replace('  ', ' ')
    str = str.replace('  ', ' ')
    return str


class NewsSpider(scrapy.Spider):

    name = "dantri"
    domain = "dantri.com.vn"
    crawled_history = "history/{}.txt".format(name)
    crawled_pages = []

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super(NewsSpider,cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed,
                                signal=signals.spider_closed)
        return spider

    def start_requests(self):

        self.load_crawled_pages()

        files = [x for x in os.listdir(self.name) if x.endswith('.txt')]
        for file in files:
            file_name = os.path.join(self.name, file)
            with open(file_name) as f:
                links = f.readlines()
                links = {x for x in links if x}
            links = [x.strip() for x in links]

            base = os.path.basename(file_name)
            directory = os.path.splitext(base)[0]

            try:
                os.mkdir(join('dantri', directory))
            except:
                pass
            for link in links:
                page = link.split('/')[-1]
                if page not in self.crawled_pages:
                    yield scrapy.Request(url=link, callback=self.parse, meta={'directory': directory})

    def parse(self, response):
        container = response.css("div#ctl00_IDContent_ctl00_divContent")[0]
        title = container.css("h1").get().strip()
        title = BeautifulSoup(title, "lxml").text.strip()

        paragraphs = container.css("div#divNewsContent p").getall()
        paragraphs = [x for x in paragraphs if '<strong>' not in x]
        paragraphs = [BeautifulSoup(p, "lxml").text.strip()
                      for p in paragraphs]
        paragraphs = [x for x in paragraphs if len(x) > 0]

        page = response.url.split('/')[-1]
        id = page.split('.')[0].split('-')[-1]
        category = response.url.split('/')[3]
        filename = '{}/{}/{}.txt'.format(self.name, category, id)

        directory_path = os.path.join(self.name, category)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        paragraphs = [formalize(p) for p in paragraphs]

        print("save: ", filename)
        with open(filename, 'w', encoding='utf-8') as f:
            for p in paragraphs:
                f.write(p + '\n')

        self.log("save: {}".format(filename),level=logging.DEBUG)

        # append history
        self.crawled_pages.append(response.url)

    def spider_closed(self, spider):
        self.log('Spider Closed')
        self.save_crawled_pages()

    def load_crawled_pages(self):
        if os.path.exists(self.crawled_history):
            with open(self.crawled_history) as f:
                pages = f.readlines()
            self.crawled_pages = [x.strip() for x in pages]
    
    def save_crawled_pages(self):
        with open(self.crawled_history,'w+') as f:
            for page in self.crawled_pages:
                f.writelines(page+'\n')
        print("save history", len(self.crawled_pages))
    
process = CrawlerProcess()
process.crawl(NewsSpider)
logging.getLogger('Scrapy').propagate = False

process.start()
    