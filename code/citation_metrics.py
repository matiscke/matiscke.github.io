import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

min_year = 2018
max_year = 2023

m = pd.read_csv('metrics-indices.csv', delimiter=', ')
m = m[m['Year'].between(min_year, max_year)]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(m['Year'], m['h Index'], label='h Index')
# ax.plot(m['Year'], m['read10 Index'], label='read10 Index')

ax.set_xlabel('Year')
ax.legend()

plt.show()
