import requests
import os
import sys
import urllib
import os
# import requests-ftp

BASE_URL = "https://www.aoml.noaa.gov/hrd/data_sub/dropsonde.html"
def get_html(url):
    r = requests.get(BASE_URL)
    content = str(r.content).split('\\n')
    for c in content:
        if("http://www.aoml.noaa.gov/hrd/Storm_pages/" in c):
            url = c[c.index("http://"): c.index(".html")+5]
            hurrName = c[c.index("Storm_pages/") + 12: c.index("/sonde.html")]
            print(hurrName)
            os.system("mkdir " + hurrName)
            hurricane_temp = str(requests.get(url).content).split("\\n")
            for d in hurricane_temp:
                if("AVP.tar.gz" in d):
                    avpUrl = d[d.index("ftp://"): d.index(".gz")+3]
                    dirCmd = "cd " + hurrName + ";"
                    os.system(dirCmd + " curl -O " + avpUrl + ";cd ..")




# def find_links():
#
print(get_html(BASE_URL))
