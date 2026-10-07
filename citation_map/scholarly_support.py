# Copyright (c) 2024 Chen Liu
# All rights reserved.
import random
import time
from bs4 import BeautifulSoup
from typing import List
from scholarly import MaxTriesExceededException
from selenium import webdriver

NO_AUTHOR_FOUND_STR = 'No_author_found'

# Elements that Google uses to show a CAPTCHA instead of (or on top of) the search results.
CAPTCHA_ELEMENT_IDS = ['gs_captcha_ccl', 'recaptcha', 'captcha-form']

# Observation: the Nominatim package is very bad at getting the geolocation of companies (geolocation of universities are fine).
# Temporary solution: hard code the geolocations of the companies.
# NOTE: The headquarter represents the whole company which usually has many offices accross the world.
KNOWN_AFFILIATION_DICT = {
    'amazon': ('King County', 'Seattle', 'Washington', 'USA', 47.622721, -122.337176),
    'meta': ('Menlo Park', 'San Mateo', 'California', 'USA', 37.4851, -122.1483),
    'microsoft': ('King County', 'Redmond', 'Washington', 'USA', 47.645695, -122.131803),
    'ibm': ('Westchester', 'Armonk', 'New York', 'USA', 41.108252, -73.719887),
    'google': ('Santa Clara', 'Mountain View', 'California', 'USA', 37.421473, -122.080679),
    'morgan stanley': ('New York', 'New York', 'New York', 'USA', 40.760251, -73.98518),
    'siemens healthineers': ('Forchheim', 'Forchheim', 'Bavaria', 'Germany', 49.702088, 11.055870),
    'oracle': ('Travis', 'Austin', 'Texas', 'USA', 30.242913, -97.721641)
}

global_driver = None

def get_driver():
    global global_driver
    if global_driver is None:
        global_driver = webdriver.Chrome()
        print("[INFO] Browser opened. You can solve CAPTCHAs (if prompted) in the browser window.")
        print("[INFO] KEEP THE POP-UP BROWSER OPEN until the CitationMap program is complete.")
    return global_driver

def wait_for_captcha(driver):
    '''
    Wait for user to solve CAPTCHA if present.
    '''
    page_source = driver.page_source
    if 'CAPTCHA' in page_source or 'not a robot' in page_source:
        print("\n" + "="*60)
        print("CAPTCHA DETECTED! Please solve it in the browser.")
        print("Press Enter here after you've solved it...")
        print("="*60)
        input()  # Wait for user to press Enter
        time.sleep(1)
    return

def get_html_per_citation_page(soup) -> List[str]:
    '''
    Utility to query each page containing results for
    cited work.
    Parameters
    --------
    soup: Beautiful Soup object pointing to current page.
    '''
    citing_authors_and_citing_papers = []

    for result in soup.find_all('div', class_='gs_ri'):
        title_tag = result.find('h3', class_='gs_rt')
        if title_tag:
            paper_parsed = False
            author_links = result.find_all('a', href=True)
            title_text = title_tag.get_text()
            title = title_text.replace('[HTML]', '').replace('[PDF]', '')
            for link in author_links:
                if 'user=' in link['href']:
                    author_id = link['href'].split('user=')[1].split('&')[0]
                    citing_authors_and_citing_papers.append((author_id, title))
                    paper_parsed = True
            if not paper_parsed:
                print("[WARNING!] Could not find author links for ", title)
                citing_authors_and_citing_papers.append((NO_AUTHOR_FOUND_STR, title))
        else:
            continue
    return citing_authors_and_citing_papers


def is_results_page(soup) -> bool:
    '''
    Check if the page lists Google Scholar search results, rather than a CAPTCHA or an error page.
    '''
    if soup.find(id=CAPTCHA_ELEMENT_IDS) is not None:
        return False
    return soup.find('div', class_='gs_ri') is not None or soup.find(id='gs_res_ccl_mid') is not None

def load_results_page(driver, url: str, blocked_wait_minutes: float = 30, max_blocked_waits: int = 8):
    '''
    Load a page of Google Scholar search results and return its soup.
    If there is a CAPTCHA, wait for the user to solve it in the browser.
    If we are blocked, wait `blocked_wait_minutes` and load the same page again, giving up after `max_blocked_waits` waits.
    We never return a page that is not a results page, so that no citing paper is silently missed.
    '''
    num_blocked_waits = 0
    while True:
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        if is_results_page(soup):
            return soup

        if soup.find(id=CAPTCHA_ELEMENT_IDS) is not None or 'CAPTCHA' in driver.page_source or 'not a robot' in driver.page_source:
            print("\n" + "="*60)
            print("CAPTCHA DETECTED! Please solve it in the browser.")
            print("Press Enter here after you've solved it...")
            print("="*60)
            input()  # Wait for user to press Enter
            continue  # Load the page again, in case solving the CAPTCHA did not bring us back to it.

        if num_blocked_waits >= max_blocked_waits:
            raise MaxTriesExceededException(
                'Still blocked by Google Scholar after waiting %d times when loading %s. '
                'Progress is saved. Run again later (or from another network) to resume.' % (num_blocked_waits, url))
        num_blocked_waits += 1
        print('\n[WARNING!] Blocked by Google Scholar when loading %s. '
              'Waiting %g minutes before loading it again (wait %d/%d). '
              'You can also stop now (Ctrl+C) and rerun later to resume.' % (url, blocked_wait_minutes, num_blocked_waits, max_blocked_waits))
        time.sleep(blocked_wait_minutes * 60)

def get_citing_author_ids_and_citing_papers(paper_url: str,
                                            blocked_wait_minutes: float = 30,
                                            max_blocked_waits: int = 8) -> List[str]:
    '''
    Find the (Google Scholar IDs of authors, titles of papers) who cite a given paper on Google Scholar.

    Parameters
    --------
    paper_url: URL of the paper BEING cited.
    blocked_wait_minutes, max_blocked_waits: what to do when Google Scholar blocks us, see `load_results_page`.
    '''
    citing_authors_and_citing_papers = []

    driver = get_driver()
    url = paper_url
    page_number = 1
    while url is not None:
        time.sleep(random.uniform(1, 5))  # Random delay to reduce risk of being blocked.
        soup = load_results_page(driver, url, blocked_wait_minutes=blocked_wait_minutes, max_blocked_waits=max_blocked_waits)

        # Loop through the citation results and find citing authors and papers.
        citing_authors_and_citing_papers += get_html_per_citation_page(soup)

        # Find the link to the next page. Look for it on every page rather than only on the first one,
        # which may not link to all pages.
        page_number += 1
        url = None
        for navigation in soup.find_all('a', class_='gs_nma'):
            if navigation.text.strip() == str(page_number):
                url = 'https://scholar.google.com' + navigation['href']
                break

    return citing_authors_and_citing_papers

def get_organization_name(organization_id: str) -> str:
    '''
    Get the official name of the organization defined by the unique Google Scholar organization ID.
    '''

    driver = get_driver()
    time.sleep(random.uniform(1, 5))  # Random delay to reduce risk of being blocked.

    url = f'https://scholar.google.com/citations?view_op=view_org&org={organization_id}&hl=en'

    time.sleep(random.uniform(1, 5))  # Random delay to reduce risk of being blocked.

    driver.get(url)
    wait_for_captcha(driver)

    soup = BeautifulSoup(driver.page_source, 'html.parser')
    tag = soup.find('h2', {'class': 'gsc_authors_header'})
    if not tag:
        raise Exception(f'When getting organization name, failed to parse {url}.')
    return tag.text.replace('Learn more', '').strip()
