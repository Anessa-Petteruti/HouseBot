import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd

# <- OPENING URL ------------------------------------------>

URL = "https://ai2thor.allenai.org/ithor/documentation/objects/object-types"
page = requests.get(URL)
# print(page.text)

soup = BeautifulSoup(page.content, "html.parser")
# print(soup)
# print(soup.prettify())

# <- RETRIEVING DATA ------------------------------------------>

# EXPLORATION
# # this are the table headings
# table_headings = soup.find('table')
# print("table_headings: ", table_headings.prettify())
# # this is the actual table content
# table_content = soup.find('tbody')
# print("table_content: ", table_content.prettify())

# TABLE HEADERS
header_text = []
# headers = soup.find_all("span", {"class": "ant-table-filter-column-title"}
headers = soup.find_all("th", class_="ant-table-cell")
for x in range(len(headers)): 
    # print(str.strip(headers[x].text))
    header_text.append(headers[x].text)
# print(header_text)

# OBJECT TYPES
# object_type = []
# obj = soup.find_all('h3') 
# for x in range(len(obj)): 
#     # print(obj[x].contents[0])
#     object_type.append(str.strip(obj[x].text))
# print(object_type)

# ALL ENTRIES OF TABLE
data = []
for tr in soup.find_all('tr'):
    # print("ROW:", row.prettify())
    tds = tr.find_all('td')
    for x in tds: 
        data.append([tds[0].text, tds[1].text, tds[2].text, tds[3].text, tds[4].text, tds[5].text])
# print(data)

# <- APPEND DATA TO DF ------------------------------------------>

df = pd.DataFrame(data, columns=header_text)
df = df.drop_duplicates()
# reset index after removing duplicates
df.reset_index(drop=True, inplace=True)
# drop empty row after headers
df = df.drop(labels=0, axis=0)
# drop rows containing astericked entries
df = df[~df["Object Type"].str.contains('\*')]
df.to_csv('ithor.csv', sep=',', quoting=csv.QUOTE_NONNUMERIC)
print(df) 

# <- WHAT DOES THE HTML LOOK LIKE FOR A GIVEN ENTRY ------------------------------------------>

# </td></tr><tr class="ant-table-row ant-table-row-level-0" data-row-key="WineBottle"><td class="ant-table-cell ant-table-cell-fix-left ant-table-cell-fix-left-last" style="position:sticky;left:0"><div class="css-bjn8wh"><div class="css-1wrbovg" id="WineBottle"></div></div><a href="/ithor/documentation/objects/object-types/#WineBottle"><span class="css-sysblq"><h3 class="css-1sn4r2g">WineBottle</h3></span></a></td><td class="ant-table-cell">Kitchens (10/30)</td><td class="ant-table-cell">Pickupable, Fillable, Breakable</td><td class="ant-table-cell">Temperature, Mass, SalientMaterials</td><td class="ant-table-cell">Fridge, Dresser, Desk, Cabinet, DiningTable, TVStand, CoffeeTable, SideTable, CounterTop, Shelf, GarbageCan</td><td class="ant-table-cell">Will automatically break if subjected to enough force. If empty, will automatically fill with water if placed under a running water source.</td></tr></tbody></table></div></div></div></div></div></div></div><div class="css-1a07e8w"><div><span id="definitions"></span><a href="/ithor/documentation/objects/object-types/#definitions"><div class="css-1hrg4jl"><span aria-label="📖, book, open_book" class="emoji-mart-emoji"><span style="width:30px;height:30px;display:inline-block;background-image:url(https://unpkg.com/emoji-datasource-apple@5.0.1/img/apple/sheets-256/64.png);background-size:5700% 5700%;background-position:46.42857142857143% 78.57142857142857%"></span></span></div><h1 class="css-1ra23ls e1ev2ax80">Definitions</h1>

# <tr class="ant-table-row ant-table-row-level-0" data-row-key="WineBottle">
#                  <td class="ant-table-cell ant-table-cell-fix-left ant-table-cell-fix-left-last" style="position:sticky;left:0">
#                   <div class="css-bjn8wh">
#                    <div class="css-1wrbovg" id="WineBottle">
#                    </div>
#                   </div>
#                   <a href="/ithor/documentation/objects/object-types/#WineBottle">
#                    <span class="css-sysblq">
#                     <h3 class="css-1sn4r2g">
#                      WineBottle
#                     </h3>
#                    </span>
#                   </a>
#                  </td>
#                  <td class="ant-table-cell">
#                   Kitchens (10/30)
#                  </td>
#                  <td class="ant-table-cell">
#                   Pickupable, Fillable, Breakable
#                  </td>
#                  <td class="ant-table-cell">
#                   Temperature, Mass, SalientMaterials
#                  </td>
#                  <td class="ant-table-cell">
#                   Fridge, Dresser, Desk, Cabinet, DiningTable, TVStand, CoffeeTable, SideTable, CounterTop, Shelf, GarbageCan
#                  </td>
#                  <td class="ant-table-cell">
#                   Will automatically break if subjected to enough force. If empty, will automatically fill with water if placed under a running water source.
#                  </td>
#                 </tr>

