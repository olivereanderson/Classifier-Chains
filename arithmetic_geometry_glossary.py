import requests

from parsel import Selector

url = 'https://en.wikipedia.org/wiki/Glossary_of_arithmetic_and_diophantine_geometry#S'

page = requests.get(url)

selector = Selector(text=page.text, type='html')

glossary = selector.xpath('//dfn/text()').getall()

