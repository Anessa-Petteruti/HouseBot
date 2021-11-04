import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd

def scrape(URL):

    # grab html content
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, "html.parser")

    # get table headers
    header_text = []
    headers = soup.find_all("th", class_="ant-table-cell")
    for x in range(len(headers)): 
        header_text.append(headers[x].text)

    # get all entries of table
    data = []
    for tr in soup.find_all('tr'):
        tds = tr.find_all('td')
        for x in tds: 
            data.append([tds[0].text, tds[1].text, tds[2].text, tds[3].text, tds[4].text, tds[5].text])
            print(tds[1].text)

    return header_text, data

def make_df(header_text, data):

    df = pd.DataFrame(data, columns=header_text)
    df = df.drop_duplicates()
    # drop empty row after headers
    df = df.drop(labels=0, axis=0)
    # drop rows containing astericked entries
    df = df[~df["Object Type"].str.contains('\*')]
    # reset index after removing duplicates
    df.reset_index(drop=True, inplace=True)
    df.to_csv('ithor.csv', sep=',', quoting=csv.QUOTE_NONNUMERIC)
    print(df) 

def main():
    URL = "https://ai2thor.allenai.org/ithor/documentation/objects/object-types"
    header_text, data = scrape(URL)
    make_df(header_text, data)

if __name__ == "__main__":
    main()

