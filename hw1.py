
import pandas as pd
import json
import requests
from pandas.io.json import json_normalize 


URL_2017 = ""
URL_2018 = "https://data.cityofchicago.org/resource/3i3m-jwuy.json"
TOK = "JC6TefVpi0uqK3mS0rolqVQVw" 
CENSUS_KEY = '01b84bdd6af611973590f4db54816866c3d50e70'


def access_data():

	r = requests.get(URL_2018), headers={"X-App-Token": TOK})
	data = r.json()
	df = pd.DataFrame.from_dict(json_normalize(data), orient='columns')

	return df



# Problem 2

df = pd.read_csv('ps1/data/Crimes_-_2018.csv')
r = requests.get("https://api.census.gov/data/2017/acs/acs1/profile?get=group(DP02)&for=us:1&key="CENSUS_KEY)
data = r.json()
df = pd.DataFrame.from_dict(json_normalize(data), orient='columns')

codes = {
	
}

from uszipcode import SearchEngine
search = SearchEngine(simple_zipcode=True)

df(lambda x: search) # to create column 
# create zip code column, drop where there aren't lat/long, drop duplicate blocks
# when creating api call url, for = zip code tabulation area then join the two dataframes on zip code

short_df = df.iloc[0:1000,:]
short_df.dropna(subset=['Location'], inplace=True)
short_df['zipcode'] = short_df.apply(lambda x: search.by_coordinates(x['Latitude'], x['Longitude'])[0].zipcode, axis=1) 
short_df.drop_duplicates(subset=['Block'],inplace=True) 


response = requests.get('https://api.census.gov/data/2017/acs/acs5?get=B15003_017E,B01003_001E,B17001_002E,B10050_003E,B02009_001E&for=zip code tabulation area:*')
census_df = pd.DataFrame(data[1:], columns=data[0])
data = response.json() 

codes = {
	'B15003_017E': 'educ',
	'B01003_001E': 'pop',
	'B17001_002E': 'total_below_poverty',
	'B10050_003E': 'total_with_grandparents',
	'B02009_001E': 'total_black',
	'zip code tabulation area': 'zipcode'
}

census_df.columns = [codes[col] for col in census_df.columns] 
merged_df = pd.merge(short_df, census_df, on='zipcode', how='inner')


## FILTERS 
### Battery
bat_df = merged_df[merged_df['Primary Type']=='BATTERY']
bat_df = bat_df.loc[:,['educ','pop','total_below_poverty','total_with_grandparents']]

for col in bat_df.columns: 
        bat_df[col] = pd.to_numeric(bat_df[col])

# number of homicides over percentage of population with degree

