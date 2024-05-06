from icrawler.builtin import BingImageCrawler

classes=["body"] 
number=20

for c in classes:
    bing_crawler=BingImageCrawler(storage={'root_dir':'/Users/Adela/Desktop/data/n'})
    bing_crawler.crawl(keyword=c,filters=None,max_num=number,offset=0)
