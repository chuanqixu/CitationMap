# Copyright (c) 2024 Chen Liu
# All rights reserved.
import folium
import itertools
import pandas as pd
import os
import pickle
import pycountry
import re
import random
import time

from contextlib import nullcontext
from functools import partial
from geopy.geocoders import Nominatim
from multiprocessing import Pool
from scholarly import scholarly, ProxyGenerator, MaxTriesExceededException
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Optional

from .scholarly_support import get_citing_author_ids_and_citing_papers, get_organization_name, NO_AUTHOR_FOUND_STR, KNOWN_AFFILIATION_DICT


def find_all_citing_authors(scholar_id: str, num_processes: int = 16) -> List[Tuple[str]]:
    '''
    Step 1. Find all publications of the given Google Scholar ID.
    Step 2. Find all citing authors.
    '''
    # Find Google Scholar Profile using Scholar ID.
    author = scholarly.search_author_id(scholar_id)
    author = scholarly.fill(author, sections=['publications'])
    publications = author['publications']
    print('Author profile found, with %d publications.\n' % len(publications))

    # Fetch metadata for all publications.
    if isinstance(num_processes, int) and num_processes > 1:
        with Pool(processes=num_processes) as pool:
            all_publications = list(tqdm(pool.imap(__fill_publication_metadata, publications),
                                         desc='Filling metadata for your %d publications' % len(publications),
                                         total=len(publications)))
    else:
        all_publications = []
        for pub in tqdm(publications,
                        desc='Filling metadata for your %d publications' % len(publications),
                        total=len(publications)):
            all_publications.append(__fill_publication_metadata(pub))

    # Convert all publications to Google Scholar publication IDs and paper titles.
    # This is fast and no parallel processing is needed.
    all_publication_info = []
    for pub in all_publications:
        if 'cites_id' in pub:
            for cites_id in pub['cites_id']:
                pub_title = pub['bib']['title']
                all_publication_info.append((cites_id, pub_title))

    # Find all citing authors from all publications.
    # To best solve CAPTCHA problems, we won't perform parallel processing here.
    all_citing_author_paper_info_nested = []
    for pub in tqdm(all_publication_info,
                    desc='Finding citing authors and papers on your %d publications' % len(all_publication_info),
                    total=len(all_publication_info)):
        all_citing_author_paper_info_nested.append(__citing_authors_and_papers_from_publication(pub))
    all_citing_author_paper_tuple_list = list(itertools.chain(*all_citing_author_paper_info_nested))
    return all_citing_author_paper_tuple_list

def find_all_citing_affiliations(all_citing_author_paper_tuple_list: List[Tuple[str]],
                                 num_processes: int = 16,
                                 affiliation_conservative: bool = False,
                                 progress_cache_path: Optional[str] = None,
                                 blocked_wait_minutes: float = 30,
                                 max_blocked_waits: int = 8) -> Tuple[List[Tuple[str]], int]:
    '''
    Step 3. Find all citing affiliations.

    Each citing author is looked up only once, and every result is saved to `progress_cache_path`
    as soon as it arrives, so an interrupted run resumes where it stopped.
    When Google Scholar blocks us, we wait `blocked_wait_minutes` and retry the same author, giving up after
    `max_blocked_waits` waits in a row without any progress.
    A citing author is only skipped if their lookup fails again after waiting while Google Scholar still
    returns an author we already found, i.e. their own profile is unavailable.

    Returns the (author, citing paper, cited paper, affiliation) tuples, and the number of citing authors
    that were skipped in this run (they are not cached, so the next run retries them).
    '''
    # The same author often cites several of your papers, but only needs to be looked up once.
    unique_author_ids = list(dict.fromkeys(author_id for author_id, _, _ in all_citing_author_paper_tuple_list
                                           if author_id != NO_AUTHOR_FOUND_STR))

    # Citing author ID -> (author name, affiliation), or None if the author has no usable affiliation.
    # It may also hold authors from an earlier list of citing authors, so we never count it directly.
    affiliation_by_author_id = {}
    def num_done_authors():
        return len([author_id for author_id in unique_author_ids if author_id in affiliation_by_author_id])

    if progress_cache_path is not None and os.path.exists(progress_cache_path):
        affiliation_by_author_id = load_cache(progress_cache_path)
        print('Resuming from %s: %d/%d citing authors already looked up.\n' % (
            progress_cache_path, num_done_authors(), len(unique_author_ids)))

    failed_author_ids = set()  # Authors whose lookup failed before, so we already waited for them once.
    unavailable_author_ids = set()  # Authors whose own profiles fail while Google Scholar works.
    num_blocked_waits = 0
    while True:
        # Keep the original order, so that after waiting we retry the author we were blocked on.
        pending_author_ids = [author_id for author_id in unique_author_ids
                              if author_id not in affiliation_by_author_id and author_id not in unavailable_author_ids]
        if len(pending_author_ids) == 0:
            break

        num_done_before = num_done_authors()
        blocked = __lookup_citing_authors(pending_author_ids, affiliation_by_author_id,
                                          failed_author_ids, unavailable_author_ids,
                                          num_total_authors=len(unique_author_ids),
                                          affiliation_conservative=affiliation_conservative,
                                          num_processes=num_processes,
                                          progress_cache_path=progress_cache_path)
        if not blocked:
            continue

        if num_done_authors() > num_done_before:
            num_blocked_waits = 0
        if num_blocked_waits >= max_blocked_waits:
            raise MaxTriesExceededException(
                'Still blocked by Google Scholar after waiting %d times. %d/%d citing authors are done and saved. '
                'Run again later (or from another network) to resume.' % (
                    num_blocked_waits, num_done_authors(), len(unique_author_ids)))
        num_blocked_waits += 1
        print('\nBlocked by Google Scholar. %d/%d citing authors are done and saved. '
              'Waiting %g minutes before retrying (wait %d/%d). '
              'You can also stop now (Ctrl+C) and rerun later to resume.' % (
                  num_done_authors(), len(unique_author_ids), blocked_wait_minutes, num_blocked_waits, max_blocked_waits))
        time.sleep(blocked_wait_minutes * 60)

    num_skipped_authors = len(unique_author_ids) - num_done_authors()

    author_paper_affiliation_tuple_list = []
    for citing_author_id, citing_paper_title, cited_paper_title in all_citing_author_paper_tuple_list:
        if citing_author_id == NO_AUTHOR_FOUND_STR:
            author_paper_affiliation_tuple_list.append((NO_AUTHOR_FOUND_STR, citing_paper_title, cited_paper_title, NO_AUTHOR_FOUND_STR))
        elif affiliation_by_author_id.get(citing_author_id) is not None:
            author_name, affiliation = affiliation_by_author_id[citing_author_id]
            author_paper_affiliation_tuple_list.append((author_name, citing_paper_title, cited_paper_title, affiliation))
    return author_paper_affiliation_tuple_list, num_skipped_authors

def clean_affiliation_names(author_paper_affiliation_tuple_list: List[Tuple[str]]) -> List[Tuple[str]]:
    '''
    Optional Step. Clean up the names of affiliations from the authors' affiliation tab on their Google Scholar profiles.
    NOTE: This logic is very naive. Please send an issue or pull request if you have any idea how to improve it.
    Currently we will not consider any paid service or tools that pose extra burden on the users, such as GPT API.
    '''
    cleaned_author_paper_affiliation_tuple_list = []
    for author_name, citing_paper_title, cited_paper_title, affiliation_string in author_paper_affiliation_tuple_list:
        if author_name == NO_AUTHOR_FOUND_STR:
            cleaned_author_paper_affiliation_tuple_list.append((NO_AUTHOR_FOUND_STR, citing_paper_title, cited_paper_title, NO_AUTHOR_FOUND_STR))
        else:
            # Use a regular expression to split the string by ';' or 'and'.
            substring_list = [part.strip() for part in re.split(r'[;]|\band\b', affiliation_string)]
            # Further split the substrings by ',' if the latter component is not a country.
            substring_list = __country_aware_comma_split(substring_list)

            for substring in substring_list:
                # Use a regular expression to remove anything before 'at', or '@'.
                cleaned_affiliation = re.sub(r'.*?\bat\b|.*?@', '', substring, flags=re.IGNORECASE).strip()
                # Use a regular expression to filter out strings that represent
                # a person's identity rather than affiliation.
                is_common_identity_string = re.search(
                    re.compile(
                        r'\b(director|manager|chair|engineer|programmer|scientist|professor|lecturer|phd|ph\.d|postdoc|doctor|student|department of)\b',
                        re.IGNORECASE),
                    cleaned_affiliation)
                if not is_common_identity_string:
                    cleaned_author_paper_affiliation_tuple_list.append((author_name, citing_paper_title, cited_paper_title, cleaned_affiliation))
    return cleaned_author_paper_affiliation_tuple_list

def fill_known_affiliations(affiliation_name: str) -> Optional[str]:
    '''
    If the affiliation is known, return its geolocation.
    If not, return None.
    The reason we have this function is taht geolocator may return hilarious results,
    such as putting the company Amazon in Amazon rain forest.
    NOTE: This is a temporal fix. Can be replaced by smarter natural language parsers.
    '''
    for key in KNOWN_AFFILIATION_DICT:
        if key in affiliation_name.lower():
            return KNOWN_AFFILIATION_DICT[key]
    return None

def affiliation_invalid(affiliation_name: str) -> bool:
    '''
    Check if the affiliation is invalid.
    Typical invalid affiliation contains non-affiliation words, such as 'computer science'.
    Invalid affiliations will waste time in geolocator.geocode(affiliation_name).
    NOTE: This is a temporal fix. Can be replaced by smarter natural language parsers.
    '''
    invalid_affiliation_set = {
        NO_AUTHOR_FOUND_STR.lower(),
        'computer', 'computer science', 'electrical', 'engineering', 'researcher',
        'scholar', 'inc.', 'school', 'department', 'student', 'candidate', 'professor', 'faculty', 'associate'
    }
    for key in invalid_affiliation_set:
        if key in affiliation_name.lower():
            return True
    return False

def affiliation_text_to_geocode(author_paper_affiliation_tuple_list: List[Tuple[str]], max_attempts: int = 3) -> List[Tuple[str]]:
    '''
    Step 4: Convert affiliations in plain text to Geocode.
    '''
    coordinates_and_info = []
    # NOTE: According to the Nominatim Usage Policy (https://operations.osmfoundation.org/policies/nominatim/),
    # we are explicitly asked not to submit bulk requests on multiple threads.
    # Therefore, we will keep it to a loop instead of multiprocessing.
    geolocator = Nominatim(user_agent='citation_mapper')

    # Find unique affiliations and record their corresponding entries.
    affiliation_map = {}
    for entry_idx, (_, _, _, affiliation_name) in enumerate(author_paper_affiliation_tuple_list):
        if affiliation_name not in affiliation_map.keys():
            affiliation_map[affiliation_name] = [entry_idx]
        else:
            affiliation_map[affiliation_name].append(entry_idx)

    num_total_affiliations = len(affiliation_map)
    num_located_affiliations = 0
    for affiliation_name in tqdm(affiliation_map,
                                 desc='Finding geographic coordinates from %d unique citing affiliations in %d entries' % (
                                     len(affiliation_map), len(author_paper_affiliation_tuple_list)),
                                 total=len(affiliation_map)):
        if affiliation_invalid(affiliation_name):
            # If an affiliation is invalid, we will not run geolocator on it.
            # However, we still record it in the csv, so that the user can choose to manually correct it.
            corresponding_entries = affiliation_map[affiliation_name]
            for entry_idx in corresponding_entries:
                author_name, citing_paper_title, cited_paper_title, affiliation_name = author_paper_affiliation_tuple_list[entry_idx]
                coordinates_and_info.append((author_name, citing_paper_title, cited_paper_title, affiliation_name,
                                            '', '', '', '', '', ''))
        else:
            # Directly enter information if the affiliation is known.
            geo_location = fill_known_affiliations(affiliation_name)
            if geo_location is not None:
                county, city, state, country, latitude, longitude = geo_location
                corresponding_entries = affiliation_map[affiliation_name]
                for entry_idx in corresponding_entries:
                    author_name, citing_paper_title, cited_paper_title, affiliation_name = author_paper_affiliation_tuple_list[entry_idx]
                    coordinates_and_info.append((author_name, citing_paper_title, cited_paper_title, affiliation_name,
                                                latitude, longitude, county, city, state, country))
                # This location is successfully recorded.
                num_located_affiliations += 1
            else:
                for _ in range(max_attempts):
                    try:
                        geo_location = geolocator.geocode(affiliation_name)
                        if geo_location is not None:
                            # Get the full location metadata that includes county, city, state, country, etc.
                            location_metadata = geolocator.reverse(str(geo_location.latitude) + ',' + str(geo_location.longitude), language='en')
                            address = location_metadata.raw['address']
                            county, city, state, country = None, None, None, None
                            if 'county' in address:
                                county = address['county']
                            if 'city' in address:
                                city = address['city']
                            if 'state' in address:
                                state = address['state']
                            if 'country' in address:
                                country = address['country']

                            corresponding_entries = affiliation_map[affiliation_name]
                            for entry_idx in corresponding_entries:
                                author_name, citing_paper_title, cited_paper_title, affiliation_name = author_paper_affiliation_tuple_list[entry_idx]
                                coordinates_and_info.append((author_name, citing_paper_title, cited_paper_title, affiliation_name,
                                                            geo_location.latitude, geo_location.longitude,
                                                            county, city, state, country))
                            # This location is successfully recorded.
                            num_located_affiliations += 1
                            break
                    except:
                        continue
    print('\nConverted %d/%d affiliations to Geocodes.' % (num_located_affiliations, num_total_affiliations))
    coordinates_and_info = [item for item in coordinates_and_info if item is not None]  # Filter out empty entries.
    return coordinates_and_info

def export_dict_to_csv(coordinates_and_info: List[Tuple[str]], csv_output_path: str) -> None:
    '''
    Step 5.1: Export csv file recording citation information.
    '''

    citation_df = pd.DataFrame(coordinates_and_info,
                               columns=['citing author name', 'citing paper title', 'cited paper title',
                                        'affiliation', 'latitude', 'longitude',
                                        'county', 'city', 'state', 'country'])

    citation_df.to_csv(csv_output_path)
    return

def read_csv_to_dict(csv_path: str) -> None:
    '''
    Step 5.1: Read csv file recording citation information.
    Only relevant if `read_from_csv` is True.
    '''

    citation_df = pd.read_csv(csv_path, index_col=0)
    coordinates_and_info = list(citation_df.itertuples(index=False, name=None))
    return coordinates_and_info

def create_map(coordinates_and_info: List[Tuple[str]], pin_colorful: bool = True):
    '''
    Step 5.2: Create the Citation World Map.

    For authors under the same affiliations, they will be displayed in the same pin.
    '''
    citation_map = folium.Map(location=[20, 0], zoom_start=2)

    # Find unique affiliations and record their corresponding entries.
    affiliation_map = {}
    for entry_idx, (_, _, _, affiliation_name, _, _, _, _, _, _) in enumerate(coordinates_and_info):
        if affiliation_name == NO_AUTHOR_FOUND_STR:
            continue
        elif affiliation_name not in affiliation_map.keys():
            affiliation_map[affiliation_name] = [entry_idx]
        else:
            affiliation_map[affiliation_name].append(entry_idx)

    if pin_colorful:
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
                  'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
                  'darkpurple', 'pink', 'lightblue', 'lightgreen',
                  'gray', 'black', 'lightgray']
        for affiliation_name in affiliation_map:
            color = random.choice(colors)
            corresponding_entries = affiliation_map[affiliation_name]
            author_name_list = []
            location_valid = True
            for entry_idx in corresponding_entries:
                author_name, _, _, _, lat, lon, _, _, _, _  = coordinates_and_info[entry_idx]
                if pd.isna(lat) or pd.isna(lon) or lat == '' or lon == '':
                    location_valid = False
                author_name_list.append(author_name)
            if location_valid:
                folium.Marker([lat, lon], popup='%s (%s)' % (affiliation_name, ' & '.join(author_name_list)),
                            icon=folium.Icon(color=color)).add_to(citation_map)
    else:
        for affiliation_name in affiliation_map:
            corresponding_entries = affiliation_map[affiliation_name]
            author_name_list = []
            location_valid = True
            for entry_idx in corresponding_entries:
                author_name, _, _, _, lat, lon, _, _, _, _  = coordinates_and_info[entry_idx]
                if pd.isna(lat) or pd.isna(lon) or lat == '' or lon == '':
                    location_valid = False
                author_name_list.append(author_name)
            if location_valid:
                folium.Marker([lat, lon], popup='%s (%s)' % (affiliation_name, ' & '.join(author_name_list))).add_to(citation_map)
    return citation_map

def count_citation_stats(coordinates_and_info: List[Tuple[str]]) -> List[int]:
    '''
    Count the number of citing authors, affiliations and countries.
    '''
    unique_author_list, unique_affiliation_list, unique_country_list = set(), set(), set()
    for (author_name, _, _, affiliation_name, _, _, _, _, _, country) in coordinates_and_info:
        if affiliation_name == NO_AUTHOR_FOUND_STR:
            continue
        unique_author_list.add(author_name)
        unique_affiliation_list.add(affiliation_name)
        unique_country_list.add(country)
        num_authors, num_affiliations, num_countries = \
            len(unique_author_list), len(unique_affiliation_list), len(unique_country_list)
    return num_authors, num_affiliations, num_countries

def __fill_publication_metadata(pub):
    time.sleep(random.uniform(1, 5))  # Random delay to reduce risk of being blocked.
    return scholarly.fill(pub)

def __citing_authors_and_papers_from_publication(cites_id_and_cited_paper: Tuple[str, str]):
    cites_id, cited_paper_title = cites_id_and_cited_paper
    citing_paper_search_url = 'https://scholar.google.com/scholar?hl=en&cites=' + cites_id
    citing_authors_and_citing_papers = get_citing_author_ids_and_citing_papers(citing_paper_search_url)
    citing_author_paper_info = []
    for citing_author_id, citing_paper_title in citing_authors_and_citing_papers:
        citing_author_paper_info.append((citing_author_id, citing_paper_title, cited_paper_title))
    return citing_author_paper_info

def __lookup_citing_authors(author_ids: List[str],
                            affiliation_by_author_id: Dict[str, Optional[Tuple[str, str]]],
                            failed_author_ids: set,
                            unavailable_author_ids: set,
                            num_total_authors: int,
                            affiliation_conservative: bool,
                            num_processes: int,
                            progress_cache_path: Optional[str]) -> bool:
    '''
    Look up the affiliations of `author_ids`, recording each result in `affiliation_by_author_id`
    and saving it to `progress_cache_path` right away.
    `author_ids` are what remains of `num_total_authors` citing authors, so the progress bar continues from there.
    Returns True as soon as Google Scholar blocks us, so that the caller can wait and retry.
    '''
    lookup = partial(__affiliation_from_author_id, affiliation_conservative=affiliation_conservative)
    use_pool = isinstance(num_processes, int) and num_processes > 1
    # Leaving the `with` block terminates the pool, which is what we want when blocked.
    with (Pool(processes=num_processes) if use_pool else nullcontext()) as pool:
        results = pool.imap_unordered(lookup, author_ids) if use_pool else map(lookup, author_ids)
        for author_id, affiliation, error in tqdm(results,
                                                  desc='Finding citing affiliations from %d citing authors' % num_total_authors,
                                                  total=num_total_authors,
                                                  initial=num_total_authors - len(author_ids)):
            if error is not None:
                tqdm.write('[Warning!] Failed to look up citing author %s: %s' % (author_id, error))
                if author_id not in failed_author_ids:
                    # Most likely we are blocked. Wait, then retry this same author.
                    failed_author_ids.add(author_id)
                    return True

                # This author failed again after we waited. Either we are still blocked, or this profile itself
                # is unavailable. Looking up an author we already found (or another pending one) tells us which.
                probe_author_id = next(reversed(affiliation_by_author_id), None) or \
                    next((other_id for other_id in author_ids if other_id != author_id), None)
                if probe_author_id is None or not __google_scholar_returns_author(probe_author_id):
                    return True

                # Google Scholar works. Retry once more in case the block was lifted right before the probe.
                _, affiliation, error = lookup(author_id)
                if error is not None:
                    tqdm.write('[Warning!] Google Scholar works, but not for citing author %s. '
                               'Skipping this author in this run.' % author_id)
                    unavailable_author_ids.add(author_id)
                    continue

            affiliation_by_author_id[author_id] = affiliation
            if progress_cache_path is not None:
                save_cache(affiliation_by_author_id, progress_cache_path)
    return False

def __google_scholar_returns_author(author_id: str) -> bool:
    time.sleep(random.uniform(1, 5))  # Random delay to reduce risk of being blocked.
    try:
        scholarly.search_author_id(author_id)
        return True
    except Exception:
        return False

def __affiliation_from_author_id(citing_author_id: str, affiliation_conservative: bool):
    '''
    Returns (citing_author_id, affiliation, error).
    `affiliation` is (author name, affiliation), or None if the author has no usable affiliation.
    `error` is None on success, or a description of why Google Scholar did not give us the author.

    Conservative: only use Google Scholar verified organization.
    This will have higher precision and lower recall.
    Aggressive: use the self-reported affiliation string from the Google Scholar affiliation panel.
    This will have lower precision and higher recall.
    '''
    time.sleep(random.uniform(1, 5))  # Random delay to reduce risk of being blocked.
    try:
        citing_author = scholarly.search_author_id(citing_author_id)
    except Exception as e:
        return (citing_author_id, None, '%s: %s' % (type(e).__name__, e))

    if affiliation_conservative:
        if 'organization' in citing_author:
            try:
                author_organization = get_organization_name(citing_author['organization'])
                return (citing_author_id, (citing_author['name'], author_organization), None)
            except Exception as e:
                print('[Warning!]', e)
        return (citing_author_id, None, None)

    if 'affiliation' in citing_author:
        return (citing_author_id, (citing_author['name'], citing_author['affiliation']), None)
    return (citing_author_id, None, None)

def __country_aware_comma_split(string_list: List[str]) -> List[str]:
    comma_split_list = []

    for part in string_list:
        # Split the strings by comma.
        # NOTE: The non-English comma is entered intentionally.
        sub_parts = [sub_part.strip() for sub_part in re.split(r'[,，]', part)]
        sub_parts_iter = iter(sub_parts)

        # Merge the split strings if the latter component is a country name.
        for sub_part in sub_parts_iter:
            if __iscountry(sub_part):
                continue  # Skip country names if they appear as the first sub_part.
            next_part = next(sub_parts_iter, None)
            if __iscountry(next_part):
                comma_split_list.append(f"{sub_part}, {next_part}")
            else:
                comma_split_list.append(sub_part)
                if next_part:
                    comma_split_list.append(next_part)
    return comma_split_list

def __iscountry(string: str) -> bool:
    try:
        pycountry.countries.lookup(string)
        return True
    except LookupError:
        return False

def __print_author_and_affiliation(author_paper_affiliation_tuple_list: List[Tuple[str]]) -> None:
    __author_affiliation_tuple_list = []
    for author_name, _, _, affiliation_name in sorted(author_paper_affiliation_tuple_list):
        if author_name == NO_AUTHOR_FOUND_STR:
            continue
        __author_affiliation_tuple_list.append((author_name, affiliation_name))

    # Take unique tuples.
    __author_affiliation_tuple_list = list(set(__author_affiliation_tuple_list))
    for author_name, affiliation_name in sorted(__author_affiliation_tuple_list):
        print('Author: %s. Affiliation: %s.' % (author_name, affiliation_name))
    print('')
    return


def save_cache(data: Any, fpath: str) -> None:
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    # Write to a temporary file first, so that an interruption never leaves a half-written cache.
    tmp_fpath = fpath + '.tmp'
    with open(tmp_fpath, "wb") as fd:
        pickle.dump(data, fd)
    os.replace(tmp_fpath, fpath)

def load_cache(fpath: str) -> Any:
    with open(fpath, "rb") as fd:
        return pickle.load(fd)

def generate_citation_map(scholar_id: str,
                          output_path: str = 'citation_map.html',
                          csv_output_path: str = 'citation_info.csv',
                          parse_csv: bool = False,
                          cache_folder: str = 'cache',
                          affiliation_conservative: bool = False,
                          num_processes: int = 16,
                          use_proxy: bool = False,
                          pin_colorful: bool = True,
                          print_citing_affiliations: bool = True,
                          blocked_wait_minutes: float = 30,
                          max_blocked_waits: int = 8):
    '''
    Google Scholar Citation World Map.

    Parameters
    ----
    scholar_id: str
        Your Google Scholar ID.
    output_path: str
        (default is 'citation_map.html')
        The path to the output HTML file.
    csv_output_path: str
        (default is 'citation_info.csv')
        The path to the output csv file.
    parse_csv: bool
        (default is False)
        If True, will directly jump to Step 5.2, using the information loaded from the csv.
    cache_folder: str
        (default is 'cache')
        The folder to save intermediate results, after finding (author, paper) but before finding the affiliations.
        This is because the user might want to try the aggressive vs. conservative approach.
        Set to None if you do not want caching.
    affiliation_conservative: bool
        (default is False)
        If true, we will use a more conservative approach to identify affiliations.
        If false, we will use a more aggressive approach to identify affiliations.
    num_processes: int
        (default is 16)
        Number of processes for parallel processing.
    use_proxy: bool
        (default is False)
        If true, we will use a scholarly proxy.
        It is necessary for some environments to avoid blocks, but it usually makes things slower.
    pin_colorful: bool
        (default is True)
        If true, the location pins will have a variety of colors.
        Otherwise, it will only have one color.
    print_citing_affiliations: bool
        (default is True)
        If true, print the list of citing affiliations (affiliations of citing authors).
    blocked_wait_minutes: float
        (default is 30)
        When Google Scholar blocks us while finding affiliations, wait this many minutes before retrying.
        Citing authors already looked up are saved in `cache_folder`, so you can also stop and rerun later.
    max_blocked_waits: int
        (default is 8)
        Give up after waiting this many times in a row without any progress.
    '''

    if not parse_csv:

        if use_proxy:
            pg = ProxyGenerator()
            pg.FreeProxies()
            scholarly.use_proxy(pg)
            print('Using proxy.')

        if cache_folder is not None:
            cache_path = os.path.join(cache_folder, scholar_id, 'all_citing_author_paper_tuple_list.pkl')
        else:
            cache_path = None

        if cache_path is None or not os.path.exists(cache_path):
            print('No cache found for this author. Finding citing authors from scratch.\n')

            # NOTE: Step 1. Find all publications of the given Google Scholar ID.
            #       Step 2. Find all citing authors.
            all_citing_author_paper_tuple_list = find_all_citing_authors(scholar_id=scholar_id,
                                                                         num_processes=num_processes)
            print('A total of %d citing authors recorded.\n' % len(all_citing_author_paper_tuple_list))
            if cache_path is not None and len(all_citing_author_paper_tuple_list) > 0:
                save_cache(all_citing_author_paper_tuple_list, cache_path)
            print('Saved to cache: %s.\n' % cache_path)

        else:
            print('Cache found. Loading author paper information from cache.\n')
            all_citing_author_paper_tuple_list = load_cache(cache_path)
            print('Loaded from cache: %s.\n' % cache_path)
            print('A total of %d citing authors loaded.\n' % len(all_citing_author_paper_tuple_list))

        if cache_folder is not None:
            cache_path = os.path.join(cache_folder, scholar_id, 'author_paper_affiliation_tuple_list.pkl')
            progress_cache_path = os.path.join(cache_folder, scholar_id, 'citing_author_affiliations_%s.pkl' % (
                'conservative' if affiliation_conservative else 'aggressive'))
        else:
            cache_path = None
            progress_cache_path = None

        if cache_path is None or not os.path.exists(cache_path):
            print('No cache found for this author. Finding citing affiliations from scratch.\n')

            # NOTE: Step 2. Find all citing affiliations.
            print('Identifying affiliations using the %s approach.' % ('conservative' if affiliation_conservative else 'aggressive'))
            author_paper_affiliation_tuple_list, num_skipped_authors = find_all_citing_affiliations(
                all_citing_author_paper_tuple_list,
                num_processes=num_processes,
                affiliation_conservative=affiliation_conservative,
                progress_cache_path=progress_cache_path,
                blocked_wait_minutes=blocked_wait_minutes,
                max_blocked_waits=max_blocked_waits)
            print('\nA total of %d citing affiliations recorded.\n' % len(author_paper_affiliation_tuple_list))
            if num_skipped_authors > 0:
                print('[Warning!] %d citing authors could not be looked up and are missing from this map. '
                      'Run again later to retry them.\n' % num_skipped_authors)
            # Take unique tuples.
            author_paper_affiliation_tuple_list = list(set(author_paper_affiliation_tuple_list))

            # NOTE: Step 3. Clean the affiliation strings (optional, only used if taking the aggressive approach).
            if print_citing_affiliations:
                if affiliation_conservative:
                    print('Taking the conservative approach. Will not need to clean the affiliation names.')
                    print('List of all citing authors and affiliations:\n')
                else:
                    print('Taking the aggressive approach. Cleaning the affiliation names.')
                    print('List of all citing authors and affiliations before cleaning:\n')
                __print_author_and_affiliation(author_paper_affiliation_tuple_list)
            if not affiliation_conservative:
                cleaned_author_paper_affiliation_tuple_list = clean_affiliation_names(author_paper_affiliation_tuple_list)
                if print_citing_affiliations:
                    print('List of all citing authors and affiliations after cleaning:\n')
                    __print_author_and_affiliation(cleaned_author_paper_affiliation_tuple_list)
                # Use the merged set to maximize coverage.
                author_paper_affiliation_tuple_list += cleaned_author_paper_affiliation_tuple_list
                # Take unique tuples.
                author_paper_affiliation_tuple_list = list(set(author_paper_affiliation_tuple_list))

            # Only cache complete results, so that the next run retries the skipped citing authors.
            if cache_path is not None and len(author_paper_affiliation_tuple_list) > 0 and num_skipped_authors == 0:
                save_cache(author_paper_affiliation_tuple_list, cache_path)
                print('Saved to cache: %s.\n' % cache_path)

        else:
            print('Cache found. Loading author paper and affiliation information from cache.\n')
            author_paper_affiliation_tuple_list = load_cache(cache_path)
            print('List of all citing authors and affiliations loaded:\n')
            __print_author_and_affiliation(author_paper_affiliation_tuple_list)

        # NOTE: Step 4. Convert affiliations in plain text to Geocode.
        coordinates_and_info = affiliation_text_to_geocode(author_paper_affiliation_tuple_list)
        # Take unique tuples.
        coordinates_and_info = sorted(list(set(coordinates_and_info)))

        # NOTE: Step 5.1. Export csv file recording citation information.
        export_dict_to_csv(coordinates_and_info, csv_output_path)
        print('\nCitation information exported to %s.' % csv_output_path)

    else:
        print('\nDirectly parsing the csv. Skipping all previous steps.')
        assert os.path.isfile(csv_output_path), '`csv_output_path` is not a file.'
        coordinates_and_info = read_csv_to_dict(csv_output_path)
        print('\nCitation information loaded from %s.' % csv_output_path)

    # NOTE: Step 5.2. Create the citation world map.
    citation_map = create_map(coordinates_and_info, pin_colorful=pin_colorful)
    citation_map.save(output_path)
    print('\nHTML map created and saved at %s.\n' % output_path)

    num_authors, num_affiliations, num_countries = count_citation_stats(coordinates_and_info)
    print('\nYou have been cited by %s researchers from %s affiliations and %s countries.\n' % (
        num_authors, num_affiliations, num_countries))
    return


if __name__ == '__main__':
    # Replace this with your Google Scholar ID.
    scholar_id = '3rDjnykAAAAJ'
    generate_citation_map(scholar_id,
                          output_path='citation_map.html',
                          csv_output_path='citation_info.csv',
                          parse_csv=False,
                          cache_folder='cache',
                          affiliation_conservative=True,
                          num_processes=16,
                          use_proxy=False,
                          pin_colorful=True,
                          print_citing_affiliations=True)
