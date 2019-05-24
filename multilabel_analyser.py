import pandas as pd
import numpy as np


import pickle


class MultilabelAnalyser(object):
    """
    Class for analysing a data base of multi-labeled samples.

    Throughout this class we use the words labels and categories synonymously.

    Methods:

        __init__: Constructor method.

        numbers_in_categories: Tells us how many times each label was associated with a sample.

        elements_in_several_categories: Let's us know how many samples have exactly one label, how many have two etc.

        appears_with_table: Returns a table expressing co-appearances of labels.

        related_categories_n_tuples: finds the other n-1 categories that appear most/least often in an n-tuple
        containing the given category.


    """
    def __init__(self, data, category_labels):
        self.data = data
        self.category_labels = category_labels
        self.category_data_only = data[category_labels]

    def numbers_in_categories(self):
        """
        Gives a pandas Series of the total number of samples in each category sorted in descending order.
        :return: pandas Series object
        """
        s = self.category_data_only.sum(axis=0)
        return s.sort_values(ascending=False)

    def elements_in_several_categories(self):
        """
        Returns a pandas Series where each index denotes multi-labels and the corresponding value
        is how many samples have this many labels
        """
        s = self.category_data_only.sum(axis=1)
        return s.value_counts()

    def appears_with_table(self, n=2):
        """
        A table expressing co-appearances of labels.

        More precisely the method returns a table of with rows and columns indexed by self.category labels, where the
        value at c_1,c_2 denotes how many rows (samples) (that have at least n labels) in self.data that belong to
        both c_1 and c_2.
        :param n: Number of categories required to be present in a row (for the row to be considered)
        :return: A Table expressing co-appearances of labels.
        :type n: int
        :rtype: pandas DataFrame.

        """
        # Create a table with rows and columns indexed by category_labels and all entries 0.
        temp_df = pd.DataFrame(index=list(self.category_labels), columns=list(self.category_labels)).fillna(0)
        # Only consider the rows (from the data) contained in at least n categories
        sub_df = self.category_data_only[self.category_data_only.sum(axis=1) >= n]
        for c_1, row in temp_df.iterrows():
            for c_2, column in temp_df.iteritems():
                # The following line finds the number of rows belonging to both c_1 and c_2 (among the rows contained
                # in at least n categories) and we place this number in the row corresponding to c_1 and column
                # corresponding to c_2 of temp_df
                temp_df.at[c_1, c_2] = sub_df[(sub_df[c_1] == 1) & (sub_df[c_2] == 1)].shape[0]

        return temp_df

    def related_categories_n_tuple(self, category, relative_to_size=False, n=2, most_related=True):
        """
        finds the other n-1 categories that appear most/least often in an n-tuple containing the given category.

        :param category: The given category

        :param relative_to_size: optional, default = False. If set to True we divide the frequency of the n-tuples by
        the sum of elements in each of the n -1 categories (different from the specified category) in such n-tuples.

        :param n: optional, default = 2. The size of the n-tuple.

        :param most_related: optional, default = True. If set to False we find the n-1 categories that appear
        least often in an n-tuple containing the given category. We still require that these n-1 categories appear in
        at least one n-tuple containing category.

        :return: List of length n-1 if possible otherwise we get an empty list indicating that such n-tuples do
        not exist in our data set.

        :type category: str
        :type relative_to_size: bool
        :type n: int
        :type most_related: bool
        :rtype list
        """

        # Use a mask to pick out those rows where category is present
        temp_df = self.category_data_only[self.category_data_only[category] == 1.0]
        # Use a mask to pick out the rows where n categories are present
        temp_df = temp_df[temp_df.sum(axis=1) == n]
        if temp_df.shape[0] == 0:
            # There are no n-tuples containing category
            return []

        # create a group for each unique row in temp_df
        # and make a pandas Series with the same index as temp_df
        # and columns indicates which group the index (of the row) belongs to
        x = temp_df.groupby(list(temp_df.columns)).ngroup()

        # Find which group has to most/least elements
        ascending = False
        if most_related is not True:
            ascending = True

        group_frequencies = x.value_counts(ascending=ascending)

        if relative_to_size:
            # We create a dictionary of each group and which categories occur for the rows in that group
            groups = {}
            for group in x.values:
                # we pick an arbitrary member of the group and figure out which categories the member belongs to
                i = x[x == group].index[0]
                y = temp_df.loc[i, :]
                groups[group] = y.index[y == 1.0]
                # We don't need the specified category to be in the group as it is in all groups
                bool_mask = groups[group] != category
                groups[group] = groups[group][bool_mask]

                # we also divide each value of group_frequencies by the sum of total elements in each category
                # belonging to the corresponding group
                total_elements_in_group = float(self.numbers_in_categories().loc[groups[group]].sum())
                group_frequencies.loc[group] = group_frequencies.loc[group] / total_elements_in_group
            # We have now rescaled the entries of group_frequencies and we must sort it to find the group we are
            # looking for.
            group_idx = group_frequencies.sort_values(ascending=ascending).index[0]
            return list(groups[group_idx])

        else:
            # We only need to find the categories in the group corresponding to the fist element of group_frequencies
            i = group_frequencies.index[0]
            y = temp_df.loc[i, :]
            group = y.index[y == 1.0]
            # Got all the categories we want, but also the specified category. We remove the specified category from
            # this group.
            bool_mask = group != category
            group = group[bool_mask]
            return list(group)


if __name__ == '__main__':
    data = pd.read_csv('titles_and_categories.csv')
    category_labels = data.columns[1:]
    analyser = MultilabelAnalyser(data, category_labels)
    numbers_in_categories = analyser.numbers_in_categories()
    print(numbers_in_categories)
    print(numbers_in_categories.mean())
    print(numbers_in_categories.std())

    elements_in_several_categories = analyser.elements_in_several_categories()
    print(elements_in_several_categories)
    idx = analyser.category_data_only[analyser.category_data_only.sum(axis=1) == 0].index[0]
    # print(data.loc[idx])
    # print(data.loc[idx]['Titles'])
    # print(data.shape[0])
    data = data.drop(index=idx)
    popular_categories = numbers_in_categories.index[:8]
    sub_data = data[data[popular_categories].sum(axis=1) > 0]
    sub_data = sub_data.drop(columns=[c for c in list(data.columns[1:]) if c not in list(popular_categories)])
    print(sub_data.columns)
    analyser = MultilabelAnalyser(data=sub_data, category_labels=popular_categories)
    # print(data.shape[0])
    numbers_in_categories = analyser.numbers_in_categories()
    elements_in_several_categories = analyser.elements_in_several_categories()
    print(numbers_in_categories)
    print(elements_in_several_categories)

    appears_with_table = analyser.appears_with_table()
    infile = open('pickled_categories.p', 'rb')
    categories_in_science = pickle.load(infile)
    infile.close()
    tag_look_up = {v: k for k, v in categories_in_science.items()}
    for tag in appears_with_table.index:
        name = tag_look_up[tag]
        name_with = appears_with_table.loc[tag].sort_values(ascending=False)
        print('%s is associated with %d other articles in the data set' % (name, name_with.values[0]))
        print('it most commonly shows up together with:')
        print()
        for i in range(1, 4):
            print('%s (%d times)' % (tag_look_up[name_with.index[i]], int(name_with.values[i])))
            print()
        print(' and least commonly with:')
        print()
        for i in range(1, 4):
            print('%s (%d times)' % (tag_look_up[name_with.index[-i]], int(name_with.values[-i])))
            print()

    # sub_data.to_csv('titles_and_most_popular_categories.csv', index_label=False, index=False)



