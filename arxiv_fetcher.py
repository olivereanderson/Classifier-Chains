# We use the requests HTTP library
import requests

# The Selector class from the Parsel library allows for extraction of data from HTML and XML using XPath selectors
from parsel import Selector

# We will use pandas to store our data in a database
import pandas as pd

# It is sometimes convenient to work with today's date which we can access from the datetime library
from datetime import date

# arXiv uses 503 Retry-After replies to implement flow control, we will use the time library to abide to these requests
import time

import numpy as np


class ArXivFetcher(object):
	"""
	Class for harvesting metadata from arXiv with the help of OAI.

	The class holds a pandas DataFrame named df that can be updated to store meta data collected from the arXiv data
	base.

	Methods:

		__init__: Constructor method

		fetch: Fetches data from arXiv and appends it to our data base.

		append_today: Fetches data from today's arXiv papers and appends the data to our data base.

		fetch_categories: Updates the categories associated to our science of interest.

		export_data_to_csv: Exports our data base to csv.

		export_categories_to_csv: Exports the category tags and associated names to csv.

		load_data_from_csv: Loads previously harvested data.

		transform_spread_labels: Spreads out the content of the categories column to a column for each category.

		return_titles_categories: Returns a pandas DataFrame of harvested titles and categories.
	"""

	def __init__(self, science):
		"""
		Constructor method.

		:param science: Name of the science to harvest arXiv metadata from. Examples include: math, cs, nlin, q-bio,
		etc. See the section named Sets at https://arxiv.org/help/oa for further explanation.

		:type science: str
		"""
		self.base_url = 'http://export.arxiv.org/oai2?verb=ListRecords'
		self.science = science
		self.meta_data = 'metadataPrefix=arXiv'
		self.df = pd.DataFrame(columns=('Titles', 'Created', 'Categories'))
		self.categories_in_science = {}
		if self.science == 'math':
			self.fetch_categories()
		self.transformed = False

	def fetch(self, start_date=None, end_date=None):
		"""
		Fetches data from arXiv in a specified time interval and appends it to our data base.

		This method is similar to the method harvest found at http://betatim.github.io/posts/analysing-the-arxiv/.

		:param start_date: We collect data from arXiv papeers starting from this date which must be supplied
		in (extended) iso format.

		:param end_date: We fetch data from arXiv papers published before this date (given in (extended) iso format).

		:type start_date: str

		:type end_date: str

		"""
		url = self.base_url + '&' + 'set=' + self.science + '&' + self.meta_data

		if start_date is not None and end_date is not None:

			print(
				'Fetching data from articles in %s published on arXiv in the period: %s to %s'
				% (self.science, start_date, end_date))

			url += '&from=' + start_date + '&until=' + end_date

		else:
			print('Fetching data from all articles in %s published on arXiv')

		page = requests.get(url)
		selector = Selector(text=page.text, type='xml')
		# The data we want to extract are xml elements registered under namespaces that we now specify
		ns = {'arXiv': 'http://arxiv.org/OAI/arXiv/', 'oai': 'http://www.openarchives.org/OAI/2.0/'}

		# The OAI-PMH framework implements flow control by means of not necessarily providing the complete list we are
		# requesting as a response. Whenever this occurs the response will contain a "resumptionToken" that can be
		# given as an argument to our next request in order to receive more of the list. The resumptionToken may have
		# an argument specifying the complete list size (i.e. how many article titles appear in the complete list
		list_size = selector.xpath('//oai:resumptionToken/@completeListSize', namespaces=ns).get()
		if list_size is not None:
			print('There is data from %s articles to be collected' % list_size)
			list_size = int(list_size)

		# We shall use a loop to obtain the data from the full list and we keep track of how many articles we have
		# collected data from along the way
		counter = 0
		while True:
			if counter != 0:
				try:
					page = requests.get(url)
					# We raise the stored HTTPError if one occurred.
					page.raise_for_status()

				except requests.HTTPError as err:
					if err.response.status_code == 503:
						# The text of err.response tells us how long to wait before retrying.
						error_selector = Selector(text=err.response.text, type='html')
						text = error_selector.xpath('//h1/text()').get()
						# text is of the form Retry after n seconds. We find n:
						timeout = int(text.split()[-2])
						print('Got 503 will retry to connect in %d seconds' % timeout)
						time.sleep(timeout)
						continue
					else:
						raise
			if counter != 0:
				selector = Selector(text=page.text, type='xml')
			# We now collect and append data (titles, date of creation and categories) to our data base
			titles = selector.xpath('//arXiv:title/text()', namespaces=ns).getall()
			created = selector.xpath('//arXiv:created/text()', namespaces=ns).getall()
			categories = selector.xpath('//arXiv:categories/text()', namespaces=ns).getall()
			temp_df = pd.DataFrame({'Titles': titles, 'Created': created, 'Categories': categories})
			self.df = self.df.append(temp_df, ignore_index=True)
			counter += len(titles)
			# If we can find a resumptionToken then there is still more data to be gathered.
			resumption_Token = selector.xpath('//oai:resumptionToken/text()', namespaces=ns).get()
			if resumption_Token is not None:
				# We (try to) read in the page again but this time with the last received resumptionToken as an argument
				url = self.base_url + '&' + 'resumptionToken=' + resumption_Token
				print('We have so far collected data from %d' % counter)
				print('There are still %d more articles to harvest data from' % (list_size - counter))

			else:
				print('All the data has been collected from the %d articles requested' % counter)
				break

	def append_today(self):
		"""
		Fetches and appends data from today's arXiv papers to our data base.

		"""
		# We find today's date using the datetime package
		today = date.today().isoformat()
		if self.transformed:
			# self.df has been transformed and will therefore not have the categories column
			temp_df = self.df.copy()
			self.df = pd.DataFrame(columns=('Titles', 'Created', 'Categories'))
			self.fetch(start_date=today, end_date=today)
			self.transform_spread_labels()
			self.df = temp_df.append(self.df, ignore_index=True)
		else:
			self.fetch(start_date=today, end_date=today)

	def fetch_categories(self):
		"""
		Updates the categories associated to our discipline (science) of interest.

		Due to how arXiv describes subject classes it takes some work to implement this method
		for all disciplines, and we have therefore so far only implemented it in the case science=math

		"""

		if self.science is not 'math':
			raise NotImplementedError('method fetch_categories has (so far) only been implemented for science=math')

		print('fetching categories')

		url = 'https://arxiv.org/archive/math'
		page = requests.get(url)
		selector = Selector(text=page.text, type='html')
		# We use Xpath to create a list of categories found on the web page.
		categories = selector.xpath('//li/b/text()').getall()
		# We remove the few items with the <li><b> tags that are not categories.
		i = 0
		while i < len(categories):
			if categories[i].startswith('math') is False:
				del categories[i]
			else:
				i += 1
		# Now we update our dictionary (self.tags).
		for entry in categories:
			separator = entry.find('-')
			category = entry[:separator - 1]
			name = entry[separator + 2:]
			self.categories_in_science[name] = category

	def export_data_to_csv(self, filename='arXiv_data.csv'):
		"""
		Export harvested data to csv.

		:param filename: The name of the csv file to be created.

		:type filename: str
		"""
		self.df.to_csv(filename, index_label=False, index=False)

	def export_categories_to_csv(self):
		"""
		Create a csv file containing the category tags and the corresponding names.
		:return: None
		"""
		temp_df = pd.DataFrame.from_dict(self.categories_in_science, orient='index')
		temp_df.to_csv('categories.csv', index_label=False)

	def load_data_from_csv(self, filename, transformed):
		"""
		Loads previously harvested data.

		warning:: This method replaces our current data base.

		:param filename: The csv file to load data from.

		:param transformed: If the data to be loaded has previously been transformed by self.transform_spread_labels
		then transformed should be set to True. In this case the method sets self.Transformed to True.


		:type filename: str

		:type transformed: bool



		"""
		self.df = pd.read_csv(filename)
		if transformed:
			self.transformed = True

	def transform_spread_labels(self):
		"""
		Spreads out the content of the categories column to a column for each category in self.science.

		self.df is transformed to a DataFrame where the categories column is replaced with one column for every
		category in self.science. Categories not belonging to self.science will no longer be visible.
		This method changes the status of self.transformed to True.

		"""
		if self.transformed:
			print('self.df has already been transformed')
			return

		# Create a temporary pd.DataFrame with all entries set to False
		temp_df = pd.DataFrame(np.zeros((self.df.shape[0], len(self.categories_in_science.values()))))
		# Rename the columns to the categories associated to our science of choice.
		temp_df.columns = self.categories_in_science.values()
		# Add all the categories as new columns filled with False entries to self.df
		self.df = pd.concat([self.df, temp_df], axis=1)

		# For each row and each category we change the value in the category column if this category is
		# contained in the Categories column of that row.
		for i, row in self.df.iterrows():
			for category, column in temp_df.iteritems():
				if category in self.df.at[i, 'Categories']:
					self.df.at[i, category] = 1

		# Finally we remove the Categories column of self.df
		del self.df['Categories']
		self.transformed = True

	def return_titles_categories(self):
		"""
		Returns a DataFrame of harvested titles and categories.

		:return: A copy of self.df with the Created column removed and no duplicate titles.

		:rtype: pandas DataFrame
		"""
		return_df = self.df.copy()
		del return_df['Created']
		return_df.drop_duplicates(subset='Titles')
		return return_df


if __name__ == '__main__':
	harvester = ArXivFetcher('math')
	harvester.fetch(start_date='2019-04-27', end_date='2019-04-26')
	harvester.export_data_to_csv()
	harvester.export_categories_to_csv()
	harvester.transform_spread_labels()

	data_set = harvester.return_titles_categories()
	print(data_set.head())
	data_set.to_csv('titles_and_categories.csv', index_label=False, index=False)






























































