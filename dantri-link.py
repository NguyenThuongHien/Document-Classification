import scrapy
import codecs
import datetime
import os

from article import LinkSpider
from scrapy.crawler import CrawlerProcess



class DantriLinkSpider(LinkSpider):

    name = 'dantri'

    def start_requests(self, **kwagrs):
        if not os.path.exists(self.name):
            os.mkdir(self.name)

        # thay đổi thời gian lấy bài 
        date_from = datetime.date(2018,1,1)
        date_to = datetime.date(2019,1,1)

        diff = (date_to-date_from).days

        days = []

        for d in range(diff):
            today = date_from + datetime.timedelta(d)
            days.append((today.day, today.month, today.year))

        self.sub_categories = ['the-gioi/chau-a','xa-hoi','the-thao','giao-duc-khuyen-hoc','kinh-doanh','van-hoa','giai-tri','phap-luat','suc-khoe','o-to-xe-may','tinh-yeu-gioi-tinh']

        pattern = "https://dantri.com.vn/{}/{}-{}-{}.htm"


        for category in  self.sub_categories:
            for d, m, y in days:
                if d < 10:
                    day = '0'+str(d)
                else:
                    day = str(d)
                if m < 10:
                    month = '0'+str(m)
                else:
                    month = str(m)
                url = pattern.format(category, day, month, y)
                self.log(url)
                yield scrapy.Request(url=url, callback=self.parse)
    
    def parse(self,response):
        links = response.css("div#listcheckepl div a::attr(href)").getall()
        links = {x for x in links if x}
        links = {x for x in links if x[0] == '/' and x.endswith('htm')}
        links = ['https://dantri.com.vn'+ x for x in links]
        self.update_links(links)
    
process = CrawlerProcess()
process.crawl(DantriLinkSpider)
process.start()
