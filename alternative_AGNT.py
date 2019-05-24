# Extract the rows that were labeled with either math.AG (Algebraic geometry) or math.NT (Number theory)
import pandas as pd

df = pd.read_csv('AGNT.csv')

AGNT = df[(df['math.AG'].values == 1) | (df['math.NT'].values == 1)]
# Remove all columns except 'Titles', 'math.AG', and 'math.NT'
AGNT = AGNT.drop(columns=[c for c in list(AGNT.columns) if c not in ['Titles', 'math.AG', 'math.NT']])

# Find the rows of AGNT where math.AG (Algebraic geometry) is labeled.
AG = AGNT[(AGNT['math.AG'].values == 1)]
# Find the rows of AGNT where math.NT (Number theory) is labeled.
NT = AGNT[(AGNT['math.NT'].values == 1)]

# The rows where both math.AG and math.NT are labeled.
AG_and_NT = AGNT[(AGNT['math.AG'].values == 1) & (AGNT['math.NT'].values == 1)]
# The rows where math.AG is labeled but not math.NT
AG_not_NT = AGNT[(AGNT['math.NT'].values == 0)]
# The rows where math.NT is labeled but not math.AG
NT_not_AG = AGNT[(AGNT['math.AG'].values == 0)]

# Let us see how many items each of these DataFrames contain:

total = int(AGNT.shape[0])

num_AG = int(AG.shape[0])

num_NT = int(NT.shape[0])

num_AG_and_NT = int(AG_and_NT.shape[0])

num_AG_not_NT = int(AG_not_NT.shape[0])

num_NT_not_AG = int(NT_not_AG.shape[0])
print('Number of articles tagged either AG or NT (or both): %d' % total)
print('number of articles tagged AG: %d' % num_AG)
print('number of articles tagged NT: %d' % num_NT)
print('number of articles tagged AG and NT: %d' % num_AG_and_NT)
print('number of articles tagged AG but not NT: %d' % num_AG_not_NT)
print('number of articles tagged NT but not AG: %d' % num_NT_not_AG)

# We compute the corresponding probabilities:
pr_AG = num_AG/total
pr_NT = num_NT/total
pr_AG_and_NT = num_AG_and_NT/total
pr_AG_not_NT = num_AG_not_NT/total
pr_NT_not_AG = num_NT_not_AG/total
pr_AG_given_NT = num_AG_and_NT/num_NT
pr_not_AG_given_NT = 1 - pr_AG_given_NT
pr_NT_given_AG = num_AG_and_NT/num_AG
pr_not_NT_given_AG = 1 - pr_NT_given_AG
print('probability of math.AG = %f' % pr_AG)
print('probability of math.NT = %f' % pr_NT)
print('product of these probabilities is: %f' % (pr_AG * pr_NT))
print('probability of math.AG and math.NT is: %f' % pr_AG_and_NT)

print('probability of AG but not NT: %f' % pr_AG_not_NT)
print('probability of NT but not AG: %f' % pr_NT_not_AG)
print('probability of AG given NT: %f' % pr_AG_given_NT)
print('probability of NT given AG: %f' % pr_NT_given_AG)
print('probability of not NT given AG: %f' % pr_not_NT_given_AG)
print('probability of not AG given NT: %f' % pr_not_AG_given_NT)