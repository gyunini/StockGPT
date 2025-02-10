import multiprocessing as mp
mp.set_start_method("fork", force=True)  # necessary for parmap

import glob
import io
import logging
from pathlib import Path
import random
import time

from bs4 import BeautifulSoup as bs4
import pandas as pd
import parmap
import re
import requests
import tqdm


def parse_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df) == 0: return pd.DataFrame()
        else:
            return df.assign(ticker=file_path.split('/')[-1].split('.')[0])
    except:
        return pd.DataFrame()


def load_universe():
    """
    Scrapes the Stooq website to retrieve a list of US stock symbols and saves them to a text file.

    The function iterates through 84 pages of Stooq's US stock listings, extracts stock symbols
    that end in ".US", and writes them to a file. A delay is introduced between requests to avoid
    being blocked.

    Raises
    ------
    requests.HTTPError
        If the HTTP request to Stooq fails.
    AssertionError
        If any page (except the last) contains a number of symbols different from 100.
    
    Notes
    -----
    - The number of pages (84) is hardcoded and might change if Stooq updates its pagination.
    - The function uses a random sleep interval (0.5 to 2 seconds) between requests to mimic
      human-like behavior and reduce the risk of getting blocked.
    - Destination path assumes calling from home directory
    """
    stooq_us_url = "https://stooq.com/t/?i=518&v=0&l={page_num}"
    us_universe = []

    # TODO: hard coding the number of pages in stooq -> maybe detect max page later
    for page_num in tqdm.tqdm(range(1, 85)):
        page_url = stooq_us_url.format(page_num=page_num)
        response = requests.get(page_url)
        response.raise_for_status()
        page = response.text

        stock_name_pattern = re.compile(r"^q/\?s=.*$")

        soup = bs4(page, 'html.parser')
        name_links = soup.find_all('a', href=stock_name_pattern)
        stock_names = [name_link.get_text(strip=True) for name_link in name_links]
        valid_names = [name for name in stock_names if name.split('.')[-1] == 'US']

        if page_num != 84:
            assert len(valid_names) == 100

        us_universe += valid_names

        # api rate limit aware
        time.sleep(random.uniform(0.5,2))

    dest_path = Path('data/stooq/universe/universe.txt') 
    dest_path.write_text('\n'.join(us_universe))


def load_stooq(logger: logging.Logger):
    loaded_tickers = [pth.split('/')[-1].split('.')[0] for pth in glob.glob('/Users/sjkdan/desk/StockGPT/data/stooq/*/*.csv')]
    stooq_tickers = [ticker.split('.')[0] for ticker in open('data/stooq/universe/universe.txt').readlines()]
    tickers = [ticker for ticker in stooq_tickers if ticker not in loaded_tickers]

    download_url = "https://stooq.com/q/d/l/?s={stooq_ticker}&i=d"
    for ticker in tqdm.tqdm(tickers):
        stooq_ticker = ticker + '.US'
        url = download_url.format(stooq_ticker=stooq_ticker.lower())
        response = requests.get(url)
        response.raise_for_status()

        resp_text = response.content.decode('utf-8')
        if resp_text == 'No data':
            logger.info("No data for %s", ticker)
        else:
            df = pd.read_csv(io.StringIO(resp_text))
            if len(df) > 0:
                df.to_csv(f'data/stooq/us/{ticker}.csv', index=False)
                logger.info("Downloaded %s", ticker)
            else:
                logger.info("Possibly exceeded api limit for the day")
                logger.info(resp_text)
                break

        time.sleep(random.uniform(0.5, 2))


if __name__=='__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO) 
    logger.addHandler(console_handler)

    load_stooq(logger)