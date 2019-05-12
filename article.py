import json
import scrapy
from  scrapy  import signals
import io


class LinkSpider(scrapy.Spider):
    
    links = dict()

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        assert cls.name, "Spider has to have a name"
        spider = super(LinkSpider, cls).from_crawler(crawler, *args, **kwargs)
        crawler.signals.connect(spider.spider_closed, signal=signals.spider_closed)
        return spider

    def update_links(self, links, extrator=None):
        if extrator == None:
            extrator = lambda x:x.split('/')[3]
        
        for link in links:
            slashes = [1 for x in link if x == '/']
            if sum(slashes) < 4:
                continue
            
            category = extrator(link)
            if category not in self.links:
                self.links[category] = set()
            self.links[category].add(link)

    def spider_closed(self):
        for category, links in self.links.items():
            path = '{}/{}.txt'.format(self.name, category)
            with io.open(path, 'a+',encoding='utf-8') as f:
                for link in links:
                    f.write(link + '\n')
