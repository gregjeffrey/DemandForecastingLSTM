#Get demand data from ISO-NE
import csv
import requests

start_date = '20150401'
end_date = '20180401'
CSV_URL = 'https://www.iso-ne.com/transform/csv/hourlysystemdemand?start=%s&end=%s' % (start_date,end_date)

with requests.Session() as s:
    download = s.get(CSV_URL)
    decoded_content = download.content.decode('utf-8')
    f = open('ISO-NE_hourly_data.csv','w')
    f.write(decoded_content)
    f.close()
    

    


