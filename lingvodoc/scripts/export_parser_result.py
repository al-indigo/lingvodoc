
# Standard library imports.

import json
import logging
import pdb
import pprint
import re
import requests

import xml.dom as dom
import xml.etree.ElementTree as ElementTree

import bs4
from pdb import set_trace as A

# Setting up logging.
log = logging.getLogger(__name__)

format_flag = True

graphql_url = 'http://lingvodoc.ispras.ru/api/graphql'
perspective_url = 'http://lingvodoc.ispras.ru/dictionary/3648/5/perspective/3648/8/view'

entity_query_str = ('''
  query Perspective {
    perspective(id: [3648, 8]) {
      id
      created_at
      lexical_entries(mode: "published") {
        id
        created_at
        entities(mode: "published") {
          id
          created_at
          field_id
          is_subject_for_parsing
          content
        }
      }
    }
  }
''')

parser_result_query_str = ('''
  query Parser_Results {{
    parser_results(entity_id: {}) {{
      id
      created_at
      parser_id
      content
    }}
  }}
''')

# Exporting raw data.

def main_raw():

    result = (

        requests.post(
            graphql_url,
            json = {'query': entity_query_str}))

    print(
        'entity_query: {}'.format(
            result.status_code))

    if result.status_code != 200:
        raise NotImplementedError

    result_json = result.json()

    if 'data' not in result_json:
        raise NotImplementedError

    entity_data = result_json['data']

    # Getting parsing results.

    for lexical_entry in entity_data['perspective']['lexical_entries']:
        for entity in lexical_entry['entities']:

            if not entity['is_subject_for_parsing']:
                continue

            query_str = (
                parser_result_query_str.format(entity['id']))

            result = (

                requests.post(
                    graphql_url,
                    json = {'query': query_str}))

            if result.status_code != 200:
                raise NotImplementedError

            result_json = result.json()

            if 'data' not in result_json:
                raise NotImplementedError

            entity['parser_results'] = (
                result_json['data']['parser_results'])

            print(
                'parser_result_query {}: {}'.format(
                    entity['id'],
                    result.status_code))

    # Saving.

    with open('corpus-raw.json', 'w') as corpus_file:

        json.dump(
            entity_data,
            corpus_file,
            ensure_ascii = False,
            indent = 2)

def process_item_list(item_list):

    source_item_list = item_list
    item_list = []

    for item in source_item_list:

        if (isinstance(item, str) and
            item_list and
            isinstance(item_list[-1], str)):

            item_list[-1] += item

        else:

            item_list.append(item)

    if (item_list and
        isinstance(item_list[0], str)):

        item_list[0] = item_list[0].strip()

        if not item_list[0]:
            item_list.pop(0)

    if (item_list and
        isinstance(item_list[-1], str)):

        item_list[-1] = item_list[-1].strip()

        if not item_list[-1]:
            item_list.pop(-1)

    return item_list

def process_paragraph_et(
    paragraph_xml,
    debug_flag):

    item_list = []

    check_list = []
    check_flag = False

    def f(element):
        """
        Recursive element processing.
        """

        if debug_flag:

            element_str = (

                ElementTree.tostring(
                    element, encoding = 'unicode'))

            if element.tail:
                element_str = element_str[ : -len(element.tail) ]

            log.debug(
                '\n' + 
                dom.minidom
                    .parseString(element_str)
                    .toprettyxml(indent = '  '))

        nonlocal check_flag

        # Parsing result.

        if (element.tag == 'span' or
            element.tag.endswith('}span') or
            element.tag.endswith(':span')):

            class_list = (
                element.attrib['class'].split())

            text_list = []

            approved_list = []
            other_list = []

            if element.text:
                text_list.append(element.text)

            # Looking through variants.

            for sub_element in element:
                
                if (sub_element.tag != 'span' and
                    not sub_element.tag.endswith('}span') and
                    not element.tag.endswith(':span')):

                    text_list.append(
                        ''.join(sub_element.itertext()))

                    continue

                sub_class_list = (
                    sub_element.attrib['class'].split())

                if ('result' not in sub_class_list and
                    sub_element.attrib.keys() != {'class', 'id'}):
                    
                    log.debug(
                        f'\nsub_class_list: {sub_class_list}'
                        f'\nsub_element.attrib.keys(): {sub_element.attrib.keys()}')

                    raise NotImplementedError

                sub_element_json = (
                    json.loads(''.join(sub_element.itertext())))

                ((approved_list if 'approved' in sub_class_list else other_list)
                    .append(sub_element_json))

                if sub_element.tail:
                    text_list.append(sub_element.tail)

            # Saving gathered info.

            result_text = ''.join(text_list)

            if ('verified' not in class_list or
                len(approved_list) != 1):

                check_flag = True
                check_list.append([result_text])

            else:

                check_list.append(result_text)

            item_list.append(
                [result_text, approved_list, other_list])

            if element.tail:
                item_list.append(element.tail)

            return

        # Getting text, looking at subelements.

        if element.text:
            item_list.append(element.text)

        for sub_element in element:
            f(sub_element)

        if element.tail:
            item_list.append(element.tail)

    # Processing paragraph HTML.

    f(paragraph_xml)

    # Post-processing by joining strings and removing unnecessary whitespace.

    item_list = (
        process_item_list(item_list))

    return item_list, check_list, check_flag

def process_paragraph_bs(
    paragraph_html,
    debug_flag):

    item_list = []

    check_list = []
    check_flag = False

    def f(element):
        """
        Recursive element processing.
        """

        if debug_flag:
            log.debug('\n' + element.prettify())

        nonlocal check_flag

        # Parsing result.

        if element.name == 'span':

            class_list = element.attrs.get('class')

            text_list = []

            approved_list = []
            other_list = []

            # Looking through variants.

            for content in element.children:

                if isinstance(content, bs4.NavigableString):

                    text_list.append(str(content))
                    continue

                elif not isinstance(content, bs4.Tag):

                    continue
                
                if (content.name != 'span'):

                    text_list.append(content.get_text())
                    continue

                sub_class_list = content.attrs.get('class')

                if (sub_class_list is not None and
                        'result' not in sub_class_list and
                        content.attrs.keys() != {'class', 'id'}):
                    
                    log.debug(
                        f'sub_class_list: {sub_class_list}'
                        f'\ncontent.attrs.keys(): {content.attrs.keys()}')

                    raise NotImplementedError

                try:

                    sub_element_text = (
                        content.get_text())

                    sub_element_json = (
                        json.loads(sub_element_text))

                    ((approved_list if 'approved' in sub_class_list else other_list)
                        .append(sub_element_json))

                except:

                    log.warning(
                        f'\nsub_element_text:\n{repr(sub_element_text)}')

            # Saving gathered info.

            result_text = ''.join(text_list)

            if (class_list is not None and
                    'verified' not in class_list or
                    len(approved_list) != 1):

                check_flag = True
                check_list.append([result_text])

            else:

                check_list.append(result_text)

            item_list.append(
                [result_text, approved_list, other_list])

            if element.tail:
                item_list.append(element.tail)

            return

        # Getting text, looking at subelements.

        for content in element.children:

            if isinstance(content, bs4.NavigableString):
                item_list.append(str(content))

            elif isinstance(content, bs4.Tag):
                f(content)

    # Processing paragraph HTML.

    f(paragraph_html)

    # Post-processing by joining strings and removing unnecessary whitespace.

    item_list = (
        process_item_list(item_list))

    return item_list, check_list, check_flag

def iterate_element_tree(
    content,
    debug_flag):

    root = ElementTree.fromstring(content)

    if debug_flag:

        with open(
            '__content__.xml', 'w') as content_file:

            content_file.write(content)

    for paragraph_index, paragraph_xml in (
        enumerate(root.iter('{http://www.w3.org/1999/xhtml}p'))):

        if debug_flag:

            paragraph_str = (

                dom.minidom
                    .parseString(ElementTree.tostring(paragraph_xml))
                    .toprettyxml(indent = '  '))

            with open(
                '__paragraph__.xml', 'w') as paragraph_file:

                paragraph_file.write(paragraph_str)

        yield (
            paragraph_index,
            process_paragraph_et(
                paragraph_xml, debug_flag))

def iterate_beautiful_soup(
    content,
    debug_flag):

    soup = bs4.BeautifulSoup(content, 'html.parser')

    if debug_flag:

        with open(
            '__content__.html', 'w') as content_file:

            content_file.write(
                soup.prettify())

    for paragraph_index, paragraph_html in (
        enumerate(soup.find_all('p'))):

        if debug_flag:

            with open(
                '__paragraph__.html', 'w') as paragraph_file:

                paragraph_file.write(
                    paragraph_html.prettify())

        yield (
            paragraph_index,
            process_paragraph_bs(
                paragraph_html, debug_flag))

def process_content(
    content,
    debug_flag,
    format_flag,
    iterate_f,
    only_count_flag = False):

    paragraph_list = []

    token_count = 0
    approved_count = 0

    # Processing each paragraph.

    for (
        paragraph_index,
        (paragraph_item_list, check_list, check_flag)) in (

        iterate_f(
            content, debug_flag)):

        if not paragraph_item_list:
            continue

        if debug_flag:

            log.debug(
                '\ncheck_list:\n' + str(check_list))

        paragraph_text = (

            ''.join(
                item if isinstance(item, str) else item[0]
                for item in paragraph_item_list))

        # Requested info.

        if format_flag:

            token_list = []
            variant_list = []

            for item in paragraph_item_list:

                # Text part which is not a parsed token.

                if isinstance(item, str):

                    if not only_count_flag:

                        token_str = re.sub(r'\s+', ' ', item.strip())

                        if token_str:

                            token_list.append({'token': token_str})
                            variant_list.append({'token': token_str})

                # Parsed token without variants.

                elif not item[1] and not item[2]:

                    if not only_count_flag:

                        token_str = re.sub(r'\s+', ' ', item[0].strip())

                        if token_str:

                            token_list.append({'token': token_str})
                            variant_list.append({'token': token_str})

                # Parsed token with variants.

                elif only_count_flag:

                    token_count += 1

                    if item[1]:
                        approved_count += 1

                else:

                    token_str = item[0]

                    approved_list = [
                        dict(variant_dict, token = token_str)
                        for variant_dict in item[1]]

                    other_list = [
                        dict(variant_dict, token = token_str)
                        for variant_dict in item[2]]

                    token_list.append(
                        approved_list[0] if approved_list else other_list[0])

                    variant_list.append(
                        approved_list + other_list)

                    token_count += 1

                    if approved_list:
                        approved_count += 1

            if not only_count_flag:

                paragraph_dict = {
                    'index': paragraph_index,
                    'text': paragraph_text,
                    'tokens': token_list,
                    'variants': variant_list}

        # All info.

        else:

            approved_list = [
                item[:2]
                for item in paragraph_item_list
                if not isinstance(item, str)]

            variant_list = [
                item
                for item in paragraph_item_list
                if not isinstance(item, str)]

            paragraph_dict = {
                'index': paragraph_index,
                'text': paragraph_text,
                'source': paragraph_item_list,
                'tokens': approved_list,
                'variants': variant_list}

        if debug_flag:

            log.debug(
                '\nparagraph_dict:\n' +
                pprint.pformat(paragraph_dict, width = 192))

        if not only_count_flag:

            paragraph_list.append(paragraph_dict)

    return paragraph_list, token_count, approved_count

def process_parser_result(
    content,
    debug_flag,
    format_flag):

    try:

        return (

            process_content(
                content,
                debug_flag,
                format_flag,
                iterate_element_tree))

    except:

        pass

    return (

        process_content(
            content,
            debug_flag,
            format_flag,
            iterate_beautiful_soup))

def main_processed():
    """
    Exports processed data.
    """

    result = (

        requests.post(
            graphql_url,
            json = {'query': entity_query_str}))

    print(
        'entity_query: {}'.format(
            result.status_code))

    if result.status_code != 200:
        raise NotImplementedError

    result_json = result.json()

    if 'data' not in result_json:
        raise NotImplementedError

    entity_data = result_json['data']

    # Getting all parsing results.

    parser_result_list = []
    token_count = 0

    for lexical_entry in entity_data['perspective']['lexical_entries']:

        comment_str = None

        for entity in lexical_entry['entities']:

            if entity['field_id'] == [674, 5]:

                comment_str = entity['content']
                continue

            elif not entity['is_subject_for_parsing']:

                continue

            # Getting parsing result data.

            query_str = (
                parser_result_query_str.format(entity['id']))

            result = (

                requests.post(
                    graphql_url,
                    json = {'query': query_str}))

            print(
                'parser_result_query {}: {}'.format(
                    entity['id'],
                    result.status_code))

            if result.status_code != 200:
                raise NotImplementedError

            result_json = result.json()

            if 'data' not in result_json:
                raise NotImplementedError

            print(f'comment_str: {repr(comment_str)}')

            # Processing parsing results for this source.

            for parser_result_index, parser_result in (
                enumerate(result_json['data']['parser_results'])):

                with open(f'content_{parser_result_index:02d}.html', 'w') as content_file:
                    content_file.write(parser_result['content'])

                paragraph_list, parser_result_token_count, _ = (

                    process_parser_result(
                        parser_result['content'],
                        debug_flag = True,
                        format_flag = format_flag))

                # Ok, got another parser result.

                parser_result_dict = {
                    'index': parser_result_index,
                    'id': parser_result['id'],
                    'title': comment_str,
                    'paragraphs': paragraph_list}

                parser_result_list.append(
                    parser_result_dict)

                token_count += parser_result_token_count

    # Saving.

    with open('corpus.json', 'w') as corpus_file:

        json.dump(
            parser_result_list,
            corpus_file,
            ensure_ascii = False,
            indent = 2)

    print(f'token_count: {token_count}')

# If we are being run as a script.
#
# python -m lingvodoc.scripts.export_parser_result

if __name__ == '__main__':

    main_processed()

