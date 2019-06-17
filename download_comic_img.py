# -*- coding: utf-8 -*-
"""
Created on Fri May 31 18:41:21 2019

@author: Wei-Hsiang, Shen
"""

import requests
import shutil
from bs4 import BeautifulSoup

def Get_HTML_in_URL(url):
    # Fake a header
    headers = {'user-agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36', 'referer': 'https://www.webtoons.com'}
    r = requests.get(url, headers=headers)

    if r.status_code != requests.codes.ok:
        raise RuntimeError("[ERROR] Unable to access the server. (Is the website correct or the website is down?")

    return r.text

def Save_Comic_Img(html_text, episode_num):
    soup = BeautifulSoup(html_text, 'html.parser') # Load html raw text into soup
    # Get Patent ID in title
    title = soup.title.string
    print(title)

    div_of_img = soup.find('div', {"class": "viewer_img"})
    img_tags = div_of_img.findAll('img')

    for index, img_div in enumerate(img_tags):
        if index >= (len(img_tags)-1): # exclude the last image
            break
        img_url = img_div['data-url']
        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/52.0.2743.116 Safari/537.36', 'referer': 'https://www.webtoons.com'}
        response = requests.get(img_url, headers=headers, stream=True)
        with open('./comic_img/comic{}_{}.png'.format(episode_num, index), 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)

    return

if __name__ == '__main__':
    for i_episode in range(1,284): # episode 1 to 283
        url = 'https://www.webtoons.com/en/romance/yumi-cell/ep-0-prologue/viewer?title_no=478&episode_no={}'.format(i_episode)
        html_text = Get_HTML_in_URL(url)
        img_url_list = Save_Comic_Img(html_text, i_episode)
