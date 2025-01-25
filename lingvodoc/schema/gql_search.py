import collections
import datetime
import gzip
import hashlib
import io
import itertools
from itertools import chain
import logging
import os
import os.path
import pickle
import pprint
import re
import traceback
import uuid

# Just in case, don't have that on Windows.

try:
    import resource
except:
    pass

import shutil
import time

import graphene
import pathvalidate
import xlsxwriter

from lingvodoc.schema.gql_holders import (
    LingvodocObjectType,
    CreatedAt,
    IdHolder,
    MarkedForDeletion,
    AdditionalMetadata,
    Name,
    About,
    del_object,
    acl_check_by_id,
    ResponseError,
    LingvodocID,
    ObjectVal,
    get_published_translation_gist_id_cte_query,
)

import lingvodoc.models as models
from lingvodoc.models import (
    Organization as dbOrganization,
    Client,
    Dictionary as dbDictionary,
    DictionaryPerspective as dbPerspective,
    Language as dbLanguage,
    LexicalEntry as dbLexicalEntry,
    Entity as dbEntity,
    PublishingEntity as dbPublishingEntity,
    User as dbUser,
    BaseGroup as dbBaseGroup,
    Group as dbGroup,
    DBSession,
    Field as dbField,
    TranslationGist as dbTranslationGist,
    TranslationAtom as dbTranslationAtom,
    ENGLISH_LOCALE,
    SLBigInteger,
)

from lingvodoc.schema.gql_entity import Entity
from lingvodoc.schema.gql_dictionary import Dictionary

from lingvodoc.schema.gql_dictionaryperspective import (
    DictionaryPerspective as Perspective,
    entries_with_entities
)

from lingvodoc.schema.gql_lexicalentry import LexicalEntry
from lingvodoc.scripts.save_dictionary import Save_Context

from lingvodoc.utils.search import (
    recursive_sort,
    translation_gist_search
)

import lingvodoc.utils as utils

from lingvodoc.utils import (
    ids_to_id_query,
    ids_to_id_cte,
    ids_to_id_cte_query,
)

from lingvodoc.views.v2.utils import (
    storage_file,
    as_storage_file
)

import sqlalchemy
from sqlalchemy import (
    and_,
    Boolean,
    cast,
    exists,
    func,
    literal,
    not_,
    or_,
    tuple_,
    union
)

import sqlalchemy.dialects.postgresql as postgresql
from sqlalchemy.orm.util import aliased
import sqlalchemy.sql.expression as expression

from poioapi.eaf_search import eaf_search


# Setting up logging.
log = logging.getLogger(__name__)


def graphene_obj(dbobj, cur_cls):
    obj = cur_cls(id=(dbobj.client_id, dbobj.object_id))
    obj.dbObject = dbobj
    return obj

def has_wildcard(elem):
    wild_cards = {'@', '?'}
    for wild_card in wild_cards:
        if wild_card in elem:
            return True
    return False

def add_as_strict(elem):
    query = elem.replace('"', '')
    return query

def add_as_like(elem):
    query = '%' + elem.replace('"', '').replace('@', '%').replace('?', '_') + '%'
    return query

def boolean_search(test_string,  exclude_char = '-'):
    regex = '(-?"[^\n\r]+?"|[^\s\n\r]+)'
    and_blocks_strings = test_string.split(" | ")
    and_blocks_strings = [x.lower() for x in and_blocks_strings]
    #print(and_blocks_strings)
    and_blocks = []
    for block_string in and_blocks_strings:
        and_blocks.append(re.findall(regex, block_string))
    #print(and_blocks)

    and_blocks_queries = []
    #excludes = []
    for and_block in and_blocks:
        and_blocks_queries.append([])
        and_block_queries = and_blocks_queries[len(and_blocks_queries) - 1]
        for element in and_block:
            if element.startswith(exclude_char):
                element = element[1:]
                and_block_queries.append({"matching_type": 'exclude',
                                          "search_string": element})
                #excludes.append(element[1:])
                continue
            if '"' in element:
                if has_wildcard(element):
                    query = add_as_like(element)
                    and_block_queries.append({"matching_type": 'substring',
                                          "search_string": query})
                else:
                    query = add_as_strict(element)
                    and_block_queries.append({"matching_type": 'full_string',
                                          "search_string": query})
            else:
                query = add_as_like(element)
                and_block_queries.append({"matching_type": 'substring',
                                          "search_string": query})
    return and_blocks_queries


def save_xlsx_data(
    xlsx_context,
    dictionary_list,
    perspective_list,
    lexical_entry_list):
    """
    Exports search results to as Xlsx data.
    """

    # Ordering perspectives in the same way as dictionaries later, by creation time descending.

    perspective_list = sorted(
        perspective_list,
        key = lambda perspective: perspective.created_at,
        reverse = True)

    perspective_dict = collections.defaultdict(list)
    lexical_entry_dict = collections.defaultdict(list)

    for perspective in perspective_list:

        parent = perspective.parent
        perspective_dict[(parent.client_id, parent.object_id)].append(perspective)

    for lexical_entry in lexical_entry_list:

        parent = lexical_entry.parent
        lexical_entry_dict[(parent.client_id, parent.object_id)].append(lexical_entry)

    # Dictionary and language info.

    dictionary_dict = collections.defaultdict(list)

    language_id_set = set()
    language_list = []

    def f(language):
        """
        Recursively gathers containing languages.
        """

        language_id = (
            language.client_id, language.object_id)

        if language_id in language_id_set:
            return

        language_list.append(language)
        language_id_set.add(language_id)

        parent = language.parent

        if parent is not None:
            f(language.parent)

    # Standard dictionary ordering, creation time descending, see resolve_dictionaries(), order_by() clause.

    dictionary_list = sorted(
        dictionary_list,
        key = lambda dictionary: dictionary.created_at,
        reverse = True)

    for dictionary in dictionary_list:

        parent = dictionary.parent
        dictionary_dict[(parent.client_id, parent.object_id)].append(dictionary)

        f(parent)

    # Standard language ordering, see resolve_language_tree().

    language_list.sort(
        key = lambda language: (
            language.parent_client_id or -1,
            language.parent_object_id or -1,
            language.additional_metadata['younger_siblings']))

    visited = set()
    stack = set()
    result = list()
    recursive_sort(language_list, visited, stack, result)

    language_dict = collections.defaultdict(list)
    id_to_language_dict = {}

    for language in language_list:

        parent_id = (
            language.parent_client_id, language.parent_object_id)

        language_dict[parent_id].append(language)

        language_id = (
            language.client_id, language.object_id)

        id_to_language_dict[language_id] = language

    # And now determining standard main language for each language and grouping dictionaries by main
    # language.

    standard_dict = {}

    group_dict = collections.defaultdict(list)
    group_list = []

    standard_language_id_query = (

        DBSession

            .query(
                dbLanguage.client_id, 
                dbLanguage.object_id)

            .filter(
                dbLanguage.marked_for_deletion == False,
                cast(dbLanguage.additional_metadata['toc_mark'], Boolean)))

    standard_language_id_set = (
        set(tuple(id) for id in standard_language_id_query.all()))

    def f(
        language_id,
        standard_id = None):

        if language_id in standard_language_id_set:

            standard_id = language_id
            group_list.append(standard_id)

        standard_dict[language_id] = standard_id

        # First sublanguages, and only then own dictionaries.

        for language in language_dict[language_id]:

            f(
                (language.client_id, language.object_id),
                standard_id)

        group_dict[standard_id].extend(dictionary_dict[language_id])

    f((None, None))

    if group_dict[None]:
        group_list.append(None)

    # Showing what we've got.

    log.debug(
        '\nlen(dictionary_list): {0}'
        '\nlen(language_list): {1}'
        '\nlen(language_dict): {2}'
        '\nlanguage_dict:\n{3}'
        '\nlen(standard_dict): {4}'
        '\nstandard_dict:\n{5}'
        '\nlen(group_dict): {6}'
        '\ngroup_dict:\n{7}'
        '\nlen(group_list): {8}'
        '\ngroup_list:\n{9}'.format(
            len(dictionary_list),
            len(language_list),
            len(language_dict),
            pprint.pformat(language_dict, width = 192),
            len(standard_dict),
            pprint.pformat(standard_dict, width = 192),
            len(group_dict),
            pprint.pformat(group_dict, width = 192),
            len(group_list),
            pprint.pformat(group_list, width = 192)))

    # Saving data of all found lexical entries.

    for language_id in group_list:

        if language_id is None:

            language_name = '(ungrouped)'

        else:

            language = id_to_language_dict[language_id]
            language_name = language.get_translation(xlsx_context.locale_id)

        dictionary_list = group_dict[language_id]

        log.debug(
            '\n{0}: {1} dictionaries'.format(
                repr(language_name),
                len(dictionary_list)))

        if not dictionary_list:
            continue

        xlsx_context.ready_worksheet(language_name)

        # Another dictionary and its perspectives.

        for dictionary in dictionary_list:

            for perspective in perspective_dict[
                (dictionary.client_id, dictionary.object_id)]:

                xlsx_context.ready_perspective(
                    perspective,
                    dictionary,
                    worksheet_flag = False,
                    list_flag = True)

                for lexical_entry in lexical_entry_dict[
                    (perspective.client_id, perspective.object_id)]:

                    xlsx_context.save_lexical_entry(
                        lexical_entry, published = True, accepted = True)()


def get_text_field_cte(session):
    """
    Texts fields, used in search mechanisms to restrict search without field specification.
    """

    return (

        session

            .query(
                dbField.client_id,
                dbField.object_id)

            .filter(
                dbField.marked_for_deletion == False,
                dbTranslationAtom.parent_client_id == dbField.data_type_translation_gist_client_id,
                dbTranslationAtom.parent_object_id == dbField.data_type_translation_gist_object_id,
                dbTranslationAtom.locale_id == ENGLISH_LOCALE,
                dbTranslationAtom.content == 'Text')

            .cte())


def get_markup_field_cte(session):
    """
    Markup fields, used in search mechanisms to restrict search without field specification.
    """

    return (

        session

            .query(
                dbField.client_id,
                dbField.object_id)

            .filter(
                dbField.marked_for_deletion == False,
                dbTranslationAtom.parent_client_id == dbField.data_type_translation_gist_client_id,
                dbTranslationAtom.parent_object_id == dbField.data_type_translation_gist_object_id,
                dbTranslationAtom.locale_id == ENGLISH_LOCALE,
                dbTranslationAtom.content == 'Markup')

            .cte())


def get_perspectives_dictionaries(res_lexical_entries):
    """
    Gets perspectives and dictionaries of the search result from its lexical entries.
    """

    perspective_ids = (

        set(
            lexical_entry.dbObject.parent_id
            for lexical_entry in res_lexical_entries))

    if len(perspective_ids) > 2:
        perspective_ids = ids_to_id_query(perspective_ids)

    perspective_query = (

        DBSession

            .query(
                dbPerspective)

            .filter(

                tuple_(
                    dbPerspective.client_id,
                    dbPerspective.object_id)

                    .in_(
                        perspective_ids)))

    log.debug(
        '\n perspective_query:\n ' +
        str(perspective_query.statement.compile(compile_kwargs = {"literal_binds": True})))

    perspective_list = (
        perspective_query.all())

    res_perspectives = [
        graphene_obj(perspective, Perspective)
        for perspective in perspective_list]

    dictionary_ids = (

        set(
            perspective.dbObject.parent_id
            for perspective in res_perspectives))

    if len(dictionary_ids) > 2:
        dictionary_ids = ids_to_id_query(dictionary_ids)

    dictionary_query = (

        DBSession

            .query(
                dbDictionary)

            .filter(

                tuple_(
                    dbDictionary.client_id,
                    dbDictionary.object_id)

                    .in_(
                        dictionary_ids)))

    log.debug(
        '\n dictionary_query:\n ' +
        str(dictionary_query.statement.compile(compile_kwargs = {"literal_binds": True})))

    dictionary_list = (
        dictionary_query.all())

    res_dictionaries = [
        graphene_obj(dictionary, Dictionary)
        for dictionary in dictionary_list]

    return res_perspectives, res_dictionaries


def search_mechanism(
    dictionaries,
    category,
    search_strings,
    publish,
    accept,
    adopted,
    etymology,
    diacritics,
    category_field_cte_query,
    load_entities = True,
    __debug_flag__ = False):

    """
    1) published dictionaries
    2) field filter
    3) metadata filter
    4)
    """
    # 1) old filter

    if __debug_flag__:

        # If we are in debug mode, we try to load already computed search data to reduce debugging time.

        dictionary_id_list = sorted(dictionaries.all())
        field_id_list = sorted(category_field_cte_query.all())

        search_list = [

            [sorted(search_string.items())
                for search_string in search_block]

            for search_block in search_strings]

        search_digest = hashlib.md5(

            repr([
                dictionary_id_list,
                category,
                search_list,
                publish,
                accept,
                adopted,
                etymology,
                diacritics,
                field_id_list])

                .encode('utf-8')).hexdigest()

        search_data_file_name = (
            '__search_data_{0}__.gz'.format(search_digest))

        # Checking if we have saved data.

        if os.path.exists(search_data_file_name):

            with gzip.open(search_data_file_name, 'rb') as search_data_file:

                (lexical_entry_id_list,
                    perspective_id_list,
                    dictionary_id_list) = pickle.load(search_data_file)

            # Loading search data.

            perspective_list = (

                DBSession.query(dbPerspective)
                
                    .filter(
                        tuple_(dbPerspective.client_id, dbPerspective.object_id).in_(
                            perspective_id_list))
                        
                    .all())

            dictionary_list = (

                DBSession.query(dbDictionary)
                
                    .filter(
                        tuple_(dbDictionary.client_id, dbDictionary.object_id).in_(
                            dictionary_id_list))
                        
                    .all())

            # Compiling search results.

            result_lexical_entries, _ = (

                entries_with_entities(
                    lexical_entry_id_list,
                    accept = True,
                    delete = False,
                    mode = None,
                    publish = True,
                    check_perspective = False))

            res_perspectives = [
                graphene_obj(dbpersp, Perspective) for dbpersp in perspective_list]

            res_dictionaries = [
                graphene_obj(dbdict, Dictionary) for dbdict in dictionary_list]

            return result_lexical_entries, res_perspectives, res_dictionaries

    # Base set of lexical entries we'd be searching in.

    lexes = (

        DBSession

            .query(
                dbLexicalEntry.client_id,
                dbLexicalEntry.object_id)

            .filter(
                dbLexicalEntry.marked_for_deletion == False,
                dbLexicalEntry.parent_client_id == dbPerspective.client_id,
                dbLexicalEntry.parent_object_id == dbPerspective.object_id,
                dbPerspective.marked_for_deletion == False,

                tuple_(
                    dbPerspective.parent_client_id,
                    dbPerspective.parent_object_id)

                    .in_(
                        DBSession.query(dictionaries.cte()))))

    lexes_cte = lexes.cte()

    # Base set of entities we'd be searching in, from filtering of everything by fields and entity content
    # conditions.

    all_entity_content_filter = list()
    all_block_field_set = set()
    fields_flag = True

    if diacritics == 'ignore':

        xform_func = func.diacritic_xform
        xform_bag_func = func.diacritic_xform_bag

    else:

        xform_func = func.lower
        xform_bag_func = func.lower_bag

    # Entity content conditions.

    for search_block in search_strings:

        if category != 1:

            all_block_field_set.update(
                tuple(sb.get('field_id')) for sb in search_block if sb.get('field_id'))

        for search_string in search_block:

            search_value = (
                search_string['search_string'])

            if not search_string.get('field_id'):

                fields_flag = False

            matching_type = (
                search_string.get('matching_type'))

            # Corpora.

            if category == 1:

                if matching_type == 'substring':

                    curr_bs_search_blocks = (
                        boolean_search(search_value))

                    for ss in chain.from_iterable(curr_bs_search_blocks):

                        xform_ss = xform_func(ss['search_string'])

                        if ss.get('matching_type') == 'substring':

                            all_entity_content_filter.append(
                                xform_func(dbEntity.additional_metadata['bag_of_words'].astext)
                                    .like(xform_ss))

                        elif ss.get('matching_type') == 'full_string':

                            all_entity_content_filter.append(
                                xform_bag_func(dbEntity.additional_metadata['bag_of_words'])
                                    .op('@>')(func.to_jsonb(xform_ss)))

                        elif ss.get('matching_type') == 'regexp':

                            all_entity_content_filter.append(
                                xform_func(dbEntity.additional_metadata['bag_of_words'].astext)
                                    .op('~*')(xform_ss))

                elif matching_type == 'full_string':

                    all_entity_content_filter.append(
                        xform_bag_func(dbEntity.additional_metadata['bag_of_words'])
                            .op('@>')(func.to_jsonb(xform_func(search_value))))

                elif matching_type == 'regexp':

                    all_entity_content_filter.append(
                        xform_func(dbEntity.additional_metadata['bag_of_words'].astext)
                            .op('~*')(xform_func(search_value)))

            # Can have category other than 0 or 1, e.g. 2 for morphological dictionaries.

            else:

                if matching_type == 'substring':

                    curr_bs_search_blocks = (
                        boolean_search(search_value))

                    for ss in chain.from_iterable(curr_bs_search_blocks):

                        xform_ss = xform_func(ss['search_string'])

                        if ss.get('matching_type') == 'substring':

                            all_entity_content_filter.append(
                                xform_func(dbEntity.content)
                                    .like(xform_ss))

                        elif ss.get('matching_type') == 'full_string':

                            all_entity_content_filter.append(
                                xform_func(dbEntity.content) ==
                                    xform_ss)

                        elif ss.get('matching_type') == 'regexp':

                            all_entity_content_filter.append(
                                xform_func(dbEntity.content)
                                    .op('~*')(xform_ss))

                elif matching_type == 'full_string':

                    all_entity_content_filter.append(
                        xform_func(dbEntity.content) ==
                            xform_func(search_value))

                elif matching_type == 'regexp':

                    all_entity_content_filter.append(
                        xform_func(dbEntity.content)
                            .op('~*')(xform_func(search_value)))

    # Field conditions.

    if fields_flag and category != 1:

        all_block_field_cte = ids_to_id_cte(all_block_field_set)

        all_block_field_query = (

            DBSession

                .query(all_block_field_cte)

                .filter(
                    tuple_(
                        all_block_field_cte.c.client_id,
                        all_block_field_cte.c.object_id)

                        .in_(category_field_cte_query)))

        all_entity_content_filter = (

            and_(
                or_(*all_entity_content_filter),

                tuple_(dbEntity.field_client_id, dbEntity.field_object_id)
                    .in_(all_block_field_query)))

    else:

        all_entity_content_filter = (

            and_(
                or_(*all_entity_content_filter),

                tuple_(dbEntity.field_client_id, dbEntity.field_object_id)
                    .in_(category_field_cte_query)))

    # filter unused entitities

    select_query = [
        dbEntity.field_client_id,
        dbEntity.field_object_id]

    if category == 1:

        select_query.append(
            dbEntity.additional_metadata)

    else:

        select_query.append(dbEntity.content)

    published_filter = []

    if accept is not None:

        published_filter.append(
            dbPublishingEntity.accepted == accept)

    if publish is not None:

        published_filter.append(
            dbPublishingEntity.published == publish)

    if published_filter:

        published_filter.append(
            dbPublishingEntity.id == dbEntity.id)

    all_entities_cte = (

        DBSession

            .query(
                lexes_cte.c.client_id.label('entry_client_id'),
                lexes_cte.c.object_id.label('entry_object_id'),
                *select_query)

            .filter(
                dbEntity.parent_client_id == lexes_cte.c.client_id,
                dbEntity.parent_object_id == lexes_cte.c.object_id,
                dbEntity.marked_for_deletion == False,
                *published_filter,
                all_entity_content_filter)

            .cte())

    # old mechanism + cte

    full_or_block = list()

    for search_block in search_strings:

        and_lexes_sum = list()

        for search_string in search_block:

            inner_and = list()

            cur_dbEntity = all_entities_cte.c

            if search_string.get('field_id'):

                inner_and.extend((
                    cur_dbEntity.field_client_id == search_string['field_id'][0],
                    cur_dbEntity.field_object_id == search_string['field_id'][1]))

            else:

                inner_and.append(

                    tuple_(
                        cur_dbEntity.field_client_id,
                        cur_dbEntity.field_object_id)

                        .in_(
                            category_field_cte_query))

            matching_type = (
                search_string.get('matching_type'))

            if not matching_type in ('full_string', 'substring', 'regexp', 'exclude'):
                raise ResponseError(message = 'wrong matching_type')

            search_value = (
                search_string['search_string'])

            # Corpora.

            if category == 1:

                if matching_type == 'full_string':

                    inner_and.append(
                        xform_bag_func(cur_dbEntity.additional_metadata['bag_of_words'])
                            .op('@>')(func.to_jsonb(xform_func(search_value))))

                elif matching_type == 'substring':

                    curr_bs_search_blocks = (
                        boolean_search(search_value))

                    for ss in chain.from_iterable(curr_bs_search_blocks):

                        xform_ss = xform_func(ss['search_string'])

                        if ss.get('matching_type') == 'substring':

                            inner_and.append(
                                xform_func(cur_dbEntity.additional_metadata['bag_of_words'].astext)
                                    .like(xform_ss))

                        elif ss.get('matching_type') == 'full_string':

                            inner_and.append(
                                xform_bag_func(cur_dbEntity.additional_metadata['bag_of_words'])
                                    .op('@>')(func.to_jsonb(xform_ss)))

                        elif ss.get('matching_type') == 'regexp':

                            inner_and.append(
                                xform_func(cur_dbEntity.additional_metadata['bag_of_words'].astext)
                                    .op('~*')(xform_ss))

                elif matching_type == 'regexp':

                    inner_and.append(
                        xform_func(cur_dbEntity.additional_metadata['bag_of_words'].astext)
                            .op('~*')(xform_func(search_value)))

            # Can have category other than 0 or 1, e.g. 2 for morphological dictionaries.

            else:

                if matching_type == 'full_string':

                    inner_and.append(
                        xform_func(cur_dbEntity.content) ==
                            xform_func(search_value))

                elif matching_type == 'substring':

                    curr_bs_search_blocks = (
                        boolean_search(search_value))

                    bs_or_block_list = list()

                    for bs_or_block in curr_bs_search_blocks:

                        bs_and = list()

                        for ss in bs_or_block:

                            xform_ss = xform_func(ss['search_string'])

                            if ss.get('matching_type') == 'substring':

                                bs_and.append(
                                    xform_func(cur_dbEntity.content)
                                        .like(xform_ss))

                            elif ss.get('matching_type') == 'full_string':

                                bs_and.append(
                                    xform_func(cur_dbEntity.content) ==
                                        xform_ss)

                            elif ss.get('matching_type') == 'regexp':

                                bs_and.append(
                                    xform_func(cur_dbEntity.content)
                                        .op('~*')(xform_ss))

                            elif ss.get('matching_type') == 'exclude':

                                bs_and.append(
                                    xform_func(cur_dbEntity.content) !=
                                        xform_ss)

                        bs_or_block_list.append(and_(*bs_and))

                    inner_and.append(or_(*bs_or_block_list))

                elif matching_type == 'regexp':

                    inner_and.append(
                        xform_func(cur_dbEntity.content)
                            .op('~*')(xform_func(search_value)))

            and_lexes_query = (

                DBSession
                
                    .query(
                        all_entities_cte.c.entry_client_id.label('entry_client_id'),
                        all_entities_cte.c.entry_object_id.label('entry_object_id'))

                    .filter(
                        and_(*inner_and).self_group())

                    .group_by(
                        all_entities_cte.c.entry_client_id,
                        all_entities_cte.c.entry_object_id))

            and_lexes_sum.append(and_lexes_query)

        if and_lexes_sum:

            and_lexes_sum_query = (

                and_lexes_sum[0].intersect(
                    *and_lexes_sum[1:]))

            full_or_block.append(and_lexes_sum_query)
        
    # Finalizing search for lexical entries.

    if full_or_block:

        entry_id_query = (
            full_or_block[0])

        if len(full_or_block) > 1:

            entry_id_query = (

                entry_id_query.union(
                    *full_or_block[1:]))
        
    # Just in case check for an empty search, should not be happening.

    else:

        entry_id_query = (

            DBSession
            
                .query(
                    all_entities_cte.c.entry_client_id.label('entry_client_id'),
                    all_entities_cte.c.entry_object_id.label('entry_object_id'))

                .group_by(
                    all_entities_cte.c.entry_client_id,
                    all_entities_cte.c.entry_object_id))

    # Additional final non-specific conditions.

    if (adopted is not None or
        etymology is not None):

        entry_id_cte = (
            entry_id_query.cte())

        condition_list = []

        # Filtering for entries which have or do not have content marked as adopted.

        if adopted is not None:

            dbEntityAdopted = aliased(dbEntity)

            adopted_query = (

                DBSession

                    .query(
                        entry_id_cte.c.entry_client_id,
                        entry_id_cte.c.entry_object_id)

                    .filter(
                        dbEntityAdopted.parent_client_id == entry_id_cte.c.entry_client_id,
                        dbEntityAdopted.parent_object_id == entry_id_cte.c.entry_object_id,
                        dbEntityAdopted.marked_for_deletion == False,
                        func.lower(dbEntityAdopted.content).contains('заим.'))

                    .group_by(
                        entry_id_cte.c.entry_client_id,
                        entry_id_cte.c.entry_object_id))

            if adopted:

                condition_list.append(

                    tuple_(
                        entry_id_cte.c.entry_client_id,
                        entry_id_cte.c.entry_object_id)

                        .in_(
                            DBSession.query(adopted_query.cte())))

            else:

                condition_list.append(

                    tuple_(
                        entry_id_cte.c.entry_client_id,
                        entry_id_cte.c.entry_object_id)

                        .notin_(
                            DBSession.query(adopted_query.cte())))

        # Filtering for entries based on whether they contain etymology or not.

        if etymology is not None:

            dbEntityEtymology = aliased(dbEntity)
            dbFieldEtymology = aliased(dbField)

            etymology_gist = translation_gist_search('Grouping Tag')

            etymology_query = (

                DBSession

                    .query(
                        entry_id_cte.c.entry_client_id,
                        entry_id_cte.c.entry_object_id)

                    .filter(
                        dbEntityEtymology.parent_client_id == entry_id_cte.c.entry_client_id,
                        dbEntityEtymology.parent_object_id == entry_id_cte.c.entry_object_id,
                        dbEntityEtymology.marked_for_deletion == False,
                        dbFieldEtymology.id == dbEntityEtymology.field_id,
                        dbFieldEtymology.marked_for_deletion == False,
                        dbFieldEtymology.data_type_translation_gist_id == etymology_gist.id)

                    .group_by(
                        entry_id_cte.c.entry_client_id,
                        entry_id_cte.c.entry_object_id))

            if etymology:

                condition_list.append(

                    tuple_(
                        entry_id_cte.c.entry_client_id,
                        entry_id_cte.c.entry_object_id)

                        .in_(
                            DBSession.query(etymology_query.cte())))

            else:

                condition_list.append(

                    tuple_(
                        entry_id_cte.c.entry_client_id,
                        entry_id_cte.c.entry_object_id)

                        .notin_(
                            DBSession.query(etymology_query.cte())))

        entry_id_query = (

            DBSession

                .query(
                    entry_id_cte.c.entry_client_id.label('entry_client_id'),
                    entry_id_cte.c.entry_object_id.label('entry_object_id'))

                .filter(
                    *condition_list))

    # Searching for and getting lexical entries.

    query_list = (
        [dbLexicalEntry.client_id, dbLexicalEntry.object_id] if load_entities else
        [dbLexicalEntry])

    entry_query = (

        DBSession

            .query(
                *query_list)

            .filter(
                tuple_(
                    dbLexicalEntry.client_id,
                    dbLexicalEntry.object_id)
                    .in_(entry_id_query)))

    # Showing overall entry query.

    log.info(
        '\n entry_query:\n ' +
        str(entry_query.statement.compile(compile_kwargs = {"literal_binds": True})))

    if load_entities:

        res_lexical_entries, _ = (

            # Don't need to check for perspective deletion, we explicitly look only in undeleted dictionaries
            # and undeleted perspectives.

            entries_with_entities(
                entry_query,
                accept = True,
                delete = False,
                mode = None,
                publish = True,
                check_perspective = False))

    else:

        res_lexical_entries = [
            graphene_obj(lexical_entry, LexicalEntry)
            for lexical_entry in entry_query]

    res_perspectives, res_dictionaries = (
        get_perspectives_dictionaries(res_lexical_entries))

    # Saving search results data, if required.

    if __debug_flag__:

        with gzip.open(
            search_data_file_name, 'wb') as search_data_file:

            search_data = (

                sorted(
                    lexical_entry.dbObject.id
                    for lexical_entry in res_lexical_entries),

                sorted(
                    perspective.dbObject.id
                    for perspective in res_perspectives),

                sorted(
                    dictionary.dbObject.id
                    for dictionary in res_dictionaries))

            pickle.dump(
                search_data, search_data_file)

    return res_lexical_entries, res_perspectives, res_dictionaries


def search_mechanism_simple(
    dictionaries,
    category,
    search_strings,
    publish,
    accept,
    adopted,
    etymology,
    diacritics,
    category_field_cte_query):

    # Empty list of AND conditions means no results.

    if not search_strings:

        return [], [], []

    dictionaries = (

        dictionaries.filter(
            dbDictionary.category == category))

    lexes = (

        DBSession

            .query(
                dbLexicalEntry)

            .filter(
                dbLexicalEntry.marked_for_deletion == False,
                dbLexicalEntry.parent_client_id == dbPerspective.client_id,
                dbLexicalEntry.parent_object_id == dbPerspective.object_id,
                dbPerspective.marked_for_deletion == False,

                tuple_(
                    dbPerspective.parent_client_id,
                    dbPerspective.parent_object_id)

                    .in_(
                        DBSession.query(dictionaries.cte()))))

    if adopted is not None or etymology is not None:
        lexes.join(dbLexicalEntry.entity)
        if adopted is False:
            lexes = lexes.filter(~func.lower(dbEntity.content).contains('заим.%'))
        elif adopted:
            lexes = lexes.filter(func.lower(dbEntity.content).contains('заим.%'))
        if etymology is not None:
            gist = translation_gist_search('Grouping Tag')
            fields = DBSession.query(dbField.client_id, dbField.object_id).filter(
                tuple_(dbField.data_type_translation_gist_client_id,
                       dbField.data_type_translation_gist_object_id) == (gist.client_id, gist.object_id))
            if etymology:
                lexes = lexes.filter(not_(tuple_(dbEntity.field_client_id, dbEntity.field_object_id).in_(fields)))
            else:
                lexes = lexes.filter(tuple_(dbEntity.field_client_id, dbEntity.field_object_id).in_(fields))

    if diacritics == 'ignore':

        xform_func = func.diacritic_xform
        xform_bag_func = func.diacritic_xform_bag

    else:

        xform_func = func.lower
        xform_bag_func = func.lower_bag

    publish_or_accept = (
        publish is not None and
        accept is not None)

    and_block = list()

    for search_block in search_strings:

        cur_dbEntity = aliased(dbEntity)

        and_block.extend((
            cur_dbEntity.marked_for_deletion == False,
            cur_dbEntity.parent_client_id == dbLexicalEntry.client_id,
            cur_dbEntity.parent_object_id == dbLexicalEntry.object_id))

        if publish_or_accept:

            cur_dbPublishingEntity = aliased(dbPublishingEntity)

            and_block.extend((
                cur_dbPublishingEntity.client_id == cur_dbEntity.client_id,
                cur_dbPublishingEntity.object_id == cur_dbEntity.object_id))

            if publish is not None:
                and_block.append(cur_dbPublishingEntity.published == publish)

            if accept is not None:
                and_block.append(cur_dbPublishingEntity.accepted == accept)

        or_block = list()

        for search_string in search_block:

            inner_and = list()

            if search_string.get('field_id'):

                inner_and.extend((
                    cur_dbEntity.field_client_id == search_string['field_id'][0],
                    cur_dbEntity.field_object_id == search_string['field_id'][1]))

            else:

                inner_and.append(

                    tuple_(
                        cur_dbEntity.field_client_id,
                        cur_dbEntity.field_object_id)

                        .in_(
                            category_field_cte_query))

            matching_type = (
                search_string.get('matching_type'))

            if not matching_type in ('full_string', 'substring', 'regexp'):
                raise ResponseError(message = 'wrong matching_type')

            search_value = (
                search_string['search_string'])

            # Corpora.

            if category == 1:

                if matching_type == 'full_string':

                    inner_and.append(
                        xform_bag_func(cur_dbEntity.additional_metadata['bag_of_words'])
                            .op('@>')(func.to_jsonb(xform_func(search_value))))

                elif matching_type == 'substring':

                    inner_and.append(
                        xform_func(cur_dbEntity.additional_metadata['bag_of_words'].astext)
                            .like(xform_func('%' + search_value + '%')))

                elif matching_type == 'regexp':

                    inner_and.append(
                        xform_func(cur_dbEntity.additional_metadata['bag_of_words'].astext)
                            .op('~*')(search_value))

            # Can have category other than 0 or 1, e.g. 2 for morphological dictionaries.

            else:

                if matching_type == 'full_string':

                    inner_and.append(
                        xform_func(cur_dbEntity.content) ==
                            xform_func(search_value))

                elif matching_type == 'substring':

                    inner_and.append(
                        xform_func(cur_dbEntity.content)
                            .like(xform_func('%' + search_value + '%')))

                elif matching_type == 'regexp':

                    inner_and.append(
                        xform_func(cur_dbEntity.content)
                            .op('~*')(xform_func(search_value)))

            or_block.append(and_(*inner_and))

        and_block.append(or_(*or_block))

    entry_query = (

        lexes.filter(
            and_(*and_block)))

    # Showing overall entry query.

    log.info(
        '\n entry_query:\n ' +
        str(entry_query.statement.compile(compile_kwargs = {"literal_binds": True})))

    resolved_search = entry_query.all()

    if not resolved_search:

        return [], [], []

    res_lexical_entries = [
        graphene_obj(lexical_entry, LexicalEntry)
        for lexical_entry in resolved_search]

    res_perspectives, res_dictionaries = (
        get_perspectives_dictionaries(res_lexical_entries))

    return res_lexical_entries, res_perspectives, res_dictionaries


def dictionaries_with_audio_id_cte():

    return (

        DBSession

            .query(
                dbPerspective.parent_client_id,
                dbPerspective.parent_object_id)

            .filter(
                dbPerspective.marked_for_deletion == False,
                dbPerspective.client_id == dbLexicalEntry.parent_client_id,
                dbPerspective.object_id == dbLexicalEntry.parent_object_id,
                dbLexicalEntry.marked_for_deletion == False,
                dbLexicalEntry.client_id == dbEntity.parent_client_id,
                dbLexicalEntry.object_id == dbEntity.parent_object_id,
                dbEntity.marked_for_deletion == False,
                dbPublishingEntity.client_id == dbEntity.client_id,
                dbPublishingEntity.object_id == dbEntity.object_id,
                dbPublishingEntity.accepted == True,
                dbField.client_id == dbEntity.field_client_id,
                dbField.object_id == dbEntity.field_object_id,
                dbField.marked_for_deletion == False,
                dbTranslationAtom.parent_client_id == dbField.data_type_translation_gist_client_id,
                dbTranslationAtom.parent_object_id == dbField.data_type_translation_gist_object_id,
                dbTranslationAtom.locale_id == 2,
                dbTranslationAtom.content == 'Sound')

            .distinct()

            .cte())


def save_xlsx(info, xlsx_context, xlsx_filename):

    xlsx_context.workbook.close()

    storage = info.context.request.registry.settings['storage']
    time_str = '{0:.6f}'.format(time.time())

    storage_dir = (

        os.path.join(
            storage['path'],
            'map_search',
            time_str))

    os.makedirs(storage_dir, exist_ok = True)

    xlsx_path = os.path.join(
        storage_dir, xlsx_filename)

    with open(xlsx_path, 'wb') as xlsx_file:

        xlsx_context.stream.seek(0)
        shutil.copyfileobj(xlsx_context.stream, xlsx_file)

    return ''.join([
        storage['prefix'],
        storage['static_route'],
        'map_search', '/',
        time_str, '/',
        xlsx_filename])


regexp_check_set = {'', '.*', '.+', '.', '..*', '.*.', '..+', '.+.', '..', '...', '....'}


class AdvancedSearch(LingvodocObjectType):
    entities = graphene.List(Entity)
    lexical_entries = graphene.List(LexicalEntry)
    perspectives = graphene.List(Perspective)
    dictionaries = graphene.List(Dictionary)
    xlsx_url = graphene.String()

    @classmethod
    def constructor(
        cls,
        info,
        languages,
        dicts_to_filter,
        tag_list,
        category,
        adopted,
        etymology,
        diacritics,
        search_strings,
        publish,
        accept,
        search_metadata,
        xlsx_export = False,
        cognates_flag = True,
        load_entities = True,
        __debug_flag__ = False):

        try:

            log.info(
                f'\n advanced_search'
                f'\n languages: {languages}'
                f'\n dicts_to_filter: {dicts_to_filter}'
                f'\n tag_list: {tag_list}'
                f'\n category: {category}'
                f'\n adopted: {adopted}'
                f'\n etymology: {etymology}'
                f'\n diacritics: {diacritics}'
                f'\n search_strings: {search_strings}'
                f'\n publish: {publish}'
                f'\n accept: {accept}'
                f'\n search_metadata: {search_metadata}'
                f'\n xlsx_export: {xlsx_export}'
                f'\n cognates_flag: {cognates_flag}'
                f'\n load_entities: {load_entities}'
                f'\n __debug_flag__: {__debug_flag__}')

            # Checking for too permissive search conditions.

            for block in search_strings:

                if not block:

                    return ResponseError('Empty AND blocks in OR mode are not allowed.')

                for condition in block:

                    matching_type = condition.get('matching_type')

                    if matching_type == 'substring':

                        if condition['search_string'] == '':
                            return ResponseError('Empty search substrings are not allowed.')

                    elif matching_type == 'regexp':

                        if condition['search_string'] in regexp_check_set:
                            return ResponseError('Too broad search regular expressions are not allowed.')

            # Preparing the search.

            dictionaries = (

                DBSession

                    .query(
                        dbDictionary.client_id,
                        dbDictionary.object_id)

                    .filter_by(
                        marked_for_deletion = False))

            d_filter = []

            if dicts_to_filter:

                dictionary_ids = dicts_to_filter

                if len(dictionary_ids) > 2:
                    dictionary_ids = ids_to_id_query(dictionary_ids)

                d_filter.append(

                    tuple_(
                        dbDictionary.client_id,
                        dbDictionary.object_id)

                        .in_(
                            dictionary_ids))

            if languages:

                if dicts_to_filter:

                    dictionaries = (
                        dictionaries.join(dbLanguage))

                    language_ids = languages

                    if len(language_ids) > 2:
                        language_ids = ids_to_id_query(language_ids)

                    d_filter.append(

                        and_(

                            tuple_(
                                dbLanguage.client_id,
                                dbLanguage.object_id)

                                .in_(
                                    language_ids),

                            dbLanguage.marked_for_deletion == False))

            dictionaries = (
                dictionaries.filter(or_(*d_filter)))

            if publish:

                published_query = (
                    get_published_translation_gist_id_cte_query())

                dbPerspectiveP = aliased(dbPerspective)

                dictionaries = (

                    dictionaries

                        .filter(

                            tuple_(
                                dbDictionary.state_translation_gist_client_id,
                                dbDictionary.state_translation_gist_object_id)

                                .in_(published_query))

                        .join(dbPerspectiveP)

                        .filter(
                            dbPerspectiveP.marked_for_deletion == False,

                            tuple_(
                                dbPerspectiveP.state_translation_gist_client_id,
                                dbPerspectiveP.state_translation_gist_object_id)

                                .in_(published_query)))

            if tag_list:

                dictionaries = (

                    dictionaries.filter(
                        dbDictionary.additional_metadata['tag_list'].contains(tag_list)))

            if search_metadata:

                has_audio = (
                    search_metadata.get('hasAudio'))

                if has_audio is not None:

                    # Using CTE and exists predicate because otherwise, as testing shows, in particular when
                    # looking through dictionaries _without_ audio we have not optimal query plans and
                    # unnecessary long execution times.

                    audio_dict_id_cte = (
                        dictionaries_with_audio_id_cte())

                    exists_query = (

                        DBSession

                            .query(literal(1))

                            .filter(
                                audio_dict_id_cte.c.parent_client_id == dbDictionary.client_id,
                                audio_dict_id_cte.c.parent_object_id == dbDictionary.object_id)

                            .exists())

                    dictionaries = (

                        dictionaries.filter(
                            exists_query if has_audio else not_(exists_query)))

                # Going to check if we need to stop because of any non-existing metadata tags.

                meta_condition_list = []
                meta_key_value_list = []

                for meta_key in ['authors', 'humanSettlement', 'years']:

                    meta_value_list = search_metadata.get(meta_key)

                    if not meta_value_list:
                        continue

                    meta_condition_list.append(
                        dbDictionary.additional_metadata[meta_key].op('?|')(
                            postgresql.array(meta_value_list)))

                    meta_key_value_list.append(
                        (meta_key, meta_value_list))

                kind = search_metadata.get('kind')

                if kind:

                    meta_condition_list.append(
                        dbDictionary.additional_metadata['kind'].astext == kind)

                # If we don't have any dictionaries for any of the metadata conditions, we return no
                # results.

                for dictionary_condition in meta_condition_list:

                    result = (

                        DBSession

                            .query(

                                DBSession

                                    .query(
                                        literal(1))

                                    .filter(
                                        dbDictionary.marked_for_deletion == False,
                                        dictionary_condition)

                                    .exists())

                            .scalar())

                    if not result:

                        return (

                            cls(
                                entities = [],
                                lexical_entries = [],
                                perspectives = [],
                                dictionaries = []))

                # Processing metadata tags conditions. If it's just another single condition, we add it on.

                if len(meta_condition_list) == 1:

                    dictionaries = (

                        dictionaries.filter(
                            meta_condition_list[0]))

                # For multiple conditions we'll have to use a base CTE.

                elif len(meta_condition_list) > 1:

                    dictionary_cte = (

                        dictionaries
                            .add_columns(dbDictionary.additional_metadata)
                            .cte())

                    meta_condition_list = []

                    for meta_key, meta_value_list in meta_key_value_list:

                        meta_condition_list.append(
                            dictionary_cte.c.additional_metadata[meta_key].op('?|')(
                                postgresql.array(meta_value_list)))

                    if kind:

                        meta_condition_list.append(
                            dictionary_cte.c.additional_metadata['kind'].astext == kind)

                    query_list = [

                        DBSession

                            .query(
                                dictionary_cte.c.client_id,
                                dictionary_cte.c.object_id)

                            .filter(
                                condition)

                        for condition in meta_condition_list]

                    dictionaries = (
                        query_list[0].union(*query_list[1:]))

            # If we don't actually have a search query, it means that we should return just the dictionaries
            # and perspectives, which were possibly filtered through additional conditions earlier.

            if not search_strings:

                log.info(
                    '\n dictionary_query:\n ' +
                    str(dictionaries.statement.compile(compile_kwargs = {'literal_binds': True})))

                # Temporary table to reuse our possibly costly dictionary id query both for dictionaries and
                # for pespectives.

                dictionary_table_name = (

                    'dictionary_' +
                     str(uuid.uuid4()).replace('-', '_'))

                dictionary_id_table = (

                    sqlalchemy.Table(
                        dictionary_table_name,
                        models.Base.metadata,
                        sqlalchemy.Column('client_id', SLBigInteger),
                        sqlalchemy.Column('object_id', SLBigInteger),
                        prefixes = ['temporary'],
                        postgresql_on_commit = 'drop'))

                dictionary_id_table.create(
                    DBSession.connection())

                DBSession.execute(

                    dictionary_id_table

                        .insert()

                        .from_select(
                            (dbDictionary.client_id, dbDictionary.object_id),
                            dictionaries))

                # Getting and returning dictionaries and perspectives.

                dictionary_list = (

                    DBSession

                        .query(
                            dbDictionary)

                        .filter(
                            dbDictionary.client_id == dictionary_id_table.c.client_id,
                            dbDictionary.object_id == dictionary_id_table.c.object_id)

                        .all())

                gql_dictionary_list = [
                    graphene_obj(dictionary, Dictionary)
                    for dictionary in dictionary_list]

                perspective_list = (

                   DBSession

                        .query(
                            dbPerspective)

                        .filter(
                            dbPerspective.parent_client_id == dictionary_id_table.c.client_id,
                            dbPerspective.parent_object_id == dictionary_id_table.c.object_id,
                            dbPerspective.marked_for_deletion == False)

                        .all())

                gql_perspective_list = [
                    graphene_obj(perspective, Perspective)
                    for perspective in perspective_list]

                return (

                    cls(
                        entities = [],
                        lexical_entries = [],
                        perspectives = gql_perspective_list,
                        dictionaries = gql_dictionary_list))

            res_lexical_entries = list()
            res_perspectives = list()
            res_dictionaries = list()

            # Setting up export to an XLSX file, if required.

            xlsx_context = (

                None if not xlsx_export else

                Save_Context(
                    info.context.get('locale_id'),
                    DBSession,
                    cognates_flag = cognates_flag,
                    __debug_flag__ = __debug_flag__))

            # Normal dictionaries and morphological dictionaries, which have category 2.

            if category != 1:

                res_lexical_entries, res_perspectives, res_dictionaries = (

                    search_mechanism(
                        dictionaries = dictionaries,
                        category = 0,
                        search_strings = search_strings,
                        publish = publish,
                        accept = accept,
                        adopted = adopted,
                        etymology = etymology,
                        diacritics = diacritics,
                        category_field_cte_query = DBSession.query(get_text_field_cte(DBSession)),
                        load_entities = load_entities,
                        __debug_flag__ = __debug_flag__))

            # Corpora.

            if category != 0:

                tmp_lexical_entries, tmp_perspectives, tmp_dictionaries = (

                    search_mechanism(
                        dictionaries = dictionaries,
                        category = 1,
                        search_strings = search_strings,
                        publish = publish,
                        accept = accept,
                        adopted = adopted,
                        etymology = etymology,
                        diacritics = diacritics,
                        category_field_cte_query = DBSession.query(get_markup_field_cte(DBSession)),
                        load_entities = load_entities,
                        __debug_flag__ = __debug_flag__))

                res_lexical_entries += tmp_lexical_entries
                res_perspectives += tmp_perspectives
                res_dictionaries += tmp_dictionaries

            # Exporting search results as an XLSX data, if required.

            if xlsx_context is not None:

                if __debug_flag__:

                    start_time = time.time()

                save_xlsx_data(
                    xlsx_context,
                    [dictionary.dbObject for dictionary in res_dictionaries],
                    [perspective.dbObject for perspective in res_perspectives],
                    [lexical_entry.dbObject for lexical_entry in res_lexical_entries])

                if __debug_flag__:

                    elapsed_time = time.time() - start_time
                    resident_memory = utils.get_resident_memory()

                    log.debug(
                        '\nelapsed_time, resident_memory: {0:.3f}s, {1:.3f}m'.format(
                        elapsed_time,
                        resident_memory / 1048576.0))

            # Saving XLSX-exported search results, if required.

            xlsx_url = None
            
            if xlsx_export:

                query_str = (

                    '_'.join([
                        search_string["search_string"]
                        for search_block in search_strings
                        for search_string in search_block]))

                xlsx_filename = (

                    pathvalidate.sanitize_filename(
                        'Search_' + query_str)[:64] + '.xlsx')

                xlsx_url = (

                    save_xlsx(
                        info, xlsx_context, xlsx_filename))

                # Saving resulting Excel workbook for debug purposes, if required.

                if __debug_flag__:

                    xlsx_context.stream.seek(0)

                    with open(xlsx_filename, 'wb') as xlsx_file:
                        shutil.copyfileobj(xlsx_context.stream, xlsx_file)

            return (

                cls(
                    entities = [],
                    lexical_entries = res_lexical_entries,
                    perspectives = res_perspectives,
                    dictionaries = res_dictionaries,
                    xlsx_url = xlsx_url))

        except Exception as exception:

            traceback_string = (
                    
                ''.join(
                    traceback.format_exception(
                        exception, exception, exception.__traceback__))[:-1])

            log.warning('advanced_search: exception')
            log.warning(traceback_string)

            return (
                ResponseError(
                    'Exception:\n' + traceback_string))


class AdvancedSearchSimple(LingvodocObjectType):
    entities = graphene.List(Entity)
    lexical_entries = graphene.List(LexicalEntry)
    perspectives = graphene.List(Perspective)
    dictionaries = graphene.List(Dictionary)
    xlsx_url = graphene.String()

    @classmethod
    def constructor(
        cls,
        info,
        languages,
        dicts_to_filter,
        tag_list,
        category,
        adopted,
        etymology,
        diacritics,
        search_strings,
        publish,
        accept,
        xlsx_export = False,
        cognates_flag = True,
        __debug_flag__ = False):

        try:

            log.info(
                f'\n advanced_search_simple'
                f'\n languages: {languages}'
                f'\n dicts_to_filter: {dicts_to_filter}'
                f'\n tag_list: {tag_list}'
                f'\n category: {category}'
                f'\n adopted: {adopted}'
                f'\n etymology: {etymology}'
                f'\n diacritics: {diacritics}'
                f'\n search_strings: {search_strings}'
                f'\n publish: {publish}'
                f'\n accept: {accept}'
                f'\n xlsx_export: {xlsx_export}'
                f'\n cognates_flag: {cognates_flag}'
                f'\n __debug_flag__: {__debug_flag__}')

            # Checking for too permissive search conditions.

            for block in search_strings:

                if not block:

                    return ResponseError('Empty OR blocks in AND mode are not allowed.')

                for condition in block:

                    matching_type = condition.get('matching_type')

                    if matching_type == 'substring':

                        if condition['search_string'] == '':
                            return ResponseError('Empty search substrings are not allowed.')

                    elif matching_type == 'regexp':

                        if condition['search_string'] in regexp_check_set:
                            return ResponseError('Too broad search regular expressions are not allowed.')

            # Preparing the search.

            dictionaries = (

                DBSession

                    .query(
                        dbDictionary.client_id,
                        dbDictionary.object_id)

                    .filter_by(
                        marked_for_deletion = False))

            d_filter = []

            if dicts_to_filter:

                dictionary_ids = dicts_to_filter

                if len(dictionary_ids) > 2:
                    dictionary_ids = ids_to_id_query(dictionary_ids)

                d_filter.append(

                    tuple_(
                        dbDictionary.client_id,
                        dbDictionary.object_id)

                        .in_(
                            dictionary_ids))

            if languages:

                if dicts_to_filter:

                    dictionaries = (
                        dictionaries.join(dbLanguage))

                    language_ids = languages

                    if len(language_ids) > 2:
                        language_ids = ids_to_id_query(language_ids)

                    d_filter.append(

                        and_(

                            tuple_(
                                dbLanguage.client_id,
                                dbLanguage.object_id)

                                .in_(
                                    language_ids),

                            dbLanguage.marked_for_deletion == False))

            dictionaries = (
                dictionaries.filter(or_(*d_filter)))

            if publish:

                published_query = (
                    get_published_translation_gist_id_cte_query())

                dbPerspectiveP = aliased(dbPerspective)

                dictionaries = (

                    dictionaries

                        .filter(

                            tuple_(
                                dbDictionary.state_translation_gist_client_id,
                                dbDictionary.state_translation_gist_object_id)

                                .in_(published_query))

                        .join(dbPerspectiveP)

                        .filter(
                            dbPerspectiveP.marked_for_deletion == False,

                            tuple_(
                                dbPerspectiveP.state_translation_gist_client_id,
                                dbPerspectiveP.state_translation_gist_object_id)

                                .in_(published_query)))

            if tag_list:

                dictionaries = (

                    dictionaries.filter(
                        dbDictionary.additional_metadata["tag_list"].contains(tag_list)))

            res_lexical_entries = list()
            res_perspectives = list()
            res_dictionaries = list()

            # Setting up export to an XLSX file, if required.

            xlsx_context = (

                None if not xlsx_export else

                Save_Context(
                    info.context.get('locale_id'),
                    DBSession,
                    cognates_flag = cognates_flag,
                    __debug_flag__ = __debug_flag__))

            # Normal dictionaries.

            if category != 1:

                res_lexical_entries, res_perspectives, res_dictionaries = (

                    search_mechanism_simple(
                        dictionaries = dictionaries,
                        category = 0,
                        search_strings = search_strings,
                        publish = publish,
                        accept = accept,
                        adopted = adopted,
                        etymology = etymology,
                        diacritics = diacritics,
                        category_field_cte_query = DBSession.query(get_text_field_cte(DBSession))))

            # Corpora.

            if category != 0:

                tmp_lexical_entries, tmp_perspectives, tmp_dictionaries = (

                    search_mechanism_simple(
                        dictionaries = dictionaries,
                        category = 1,
                        search_strings = search_strings,
                        publish = publish,
                        accept = accept,
                        adopted = adopted,
                        etymology = etymology,
                        diacritics = diacritics,
                        category_field_cte_query = DBSession.query(get_markup_field_cte(DBSession))))

                res_lexical_entries += tmp_lexical_entries
                res_perspectives += tmp_perspectives
                res_dictionaries += tmp_dictionaries

            # Exporting search results as an XLSX data, if required.

            if xlsx_context is not None:

                save_xlsx_data(
                    xlsx_context,
                    [dictionary.dbObject for dictionary in res_dictionaries],
                    [perspective.dbObject for perspective in res_perspectives],
                    [lexical_entry.dbObject for lexical_entry in res_lexical_entries])

            # Saving XLSX-exported search results, if required.

            xlsx_url = None
            
            if xlsx_export:

                query_str = '_'.join([
                    search_string["search_string"]
                    for search_block in search_strings
                    for search_string in search_block])

                xlsx_filename = ('Search_' + query_str)[:64] + '.xlsx'

                xlsx_url = save_xlsx(
                    info, xlsx_context, xlsx_filename)

                # Saving resulting Excel workbook for debug purposes, if required.

                if __debug_flag__:

                    xlsx_context.stream.seek(0)

                    with open(xlsx_filename, 'wb') as xlsx_file:
                        shutil.copyfileobj(xlsx_context.stream, xlsx_file)

            return (

                cls(
                    entities = [],
                    lexical_entries = res_lexical_entries,
                    perspectives = res_perspectives,
                    dictionaries = res_dictionaries,
                    xlsx_url = xlsx_url))

        except Exception as exception:

            traceback_string = (
                    
                ''.join(
                    traceback.format_exception(
                        exception, exception, exception.__traceback__))[:-1])

            log.warning('advanced_search_simple: exception')
            log.warning(traceback_string)

            return (
                ResponseError(
                    'Exception:\n' + traceback_string))


class EafSearch(LingvodocObjectType):

    result_list = graphene.List(ObjectVal)
    xlsx_url = graphene.String()

    @classmethod
    def constructor(
        cls,
        info,
        perspective_id = None,
        search_query = None,
        __debug_flag__ = False):

        log.debug(
            '\neaf_search'
            '\n  perspective_id: {0}'
            '\n  search_query: {1}'
            '\n  __debug_flag__: {2}'.format(
                perspective_id,
                search_query,
                __debug_flag__))

        # Getting query string to use for XLSX file name.

        def f(query_dict):

            value = query_dict.get('value')

            if isinstance(value, list):

                for generator in map(f, value):
                    yield from generator

            elif isinstance(value, str):

                yield value

        query_str = '_'.join(f(search_query))

        xlsx_filename = (
            ('Search_' + query_str)[:64] + '.xlsx')

        # Constructing path and URL to the resulting XLSX file.

        storage = info.context.request.registry.settings['storage']
        time_str = '{0:.6f}'.format(time.time())

        storage_dir = (

            os.path.join(
                storage['path'],
                'eaf_search',
                time_str))

        os.makedirs(storage_dir, exist_ok = True)

        xlsx_path = os.path.join(
            storage_dir, xlsx_filename)

        xlsx_url = ''.join([
            storage['prefix'],
            storage['static_route'],
            'eaf_search', '/',
            time_str, '/',
            xlsx_filename])

        log.debug(
            '\neaf_search'
            '\n  xlsx_path: {0}'
            '\n  xlsx_url: {1}'.format(
                xlsx_path,
                xlsx_url))

        # Preparing to get EAF data.

        field_query = (DBSession
                
            .query(
                dbField.client_id, dbField.object_id)

            .filter(
                dbField.marked_for_deletion == False,
                dbField.data_type_translation_gist_client_id == dbTranslationGist.client_id,
                dbField.data_type_translation_gist_object_id == dbTranslationGist.object_id,
                dbTranslationGist.marked_for_deletion == False,
                dbTranslationAtom.parent_client_id == dbTranslationGist.client_id,
                dbTranslationAtom.parent_object_id == dbTranslationGist.object_id,
                dbTranslationAtom.marked_for_deletion == False,
                dbTranslationAtom.locale_id == 2,
                dbTranslationAtom.content == 'Markup'))

        eaf_query = (DBSession
                
            .query(
                dbEntity.client_id,
                dbEntity.object_id,
                dbEntity.content)

            .filter(
                tuple_(dbEntity.field_client_id, dbEntity.field_object_id)
                    .in_(field_query.subquery()),
                dbEntity.marked_for_deletion == False,
                dbEntity.parent_client_id == dbLexicalEntry.client_id,
                dbEntity.parent_object_id == dbLexicalEntry.object_id,
                dbEntity.content.op('~*')('.*\\.eaf.*'),
                dbLexicalEntry.marked_for_deletion == False,
                dbLexicalEntry.moved_to == None))

        if perspective_id is not None:

            eaf_query = eaf_query.filter(
                dbLexicalEntry.parent_client_id == perspective_id[0],
                dbLexicalEntry.parent_object_id == perspective_id[1])

        else:

            eaf_query = eaf_query.filter(
                dbLexicalEntry.parent_client_id == dbPerspective.client_id,
                dbLexicalEntry.parent_object_id == dbPerspective.object_id,
                dbPerspective.marked_for_deletion == False)

        eaf_count = eaf_query.count()

        # Processing EAF corpora.

        result_list = []

        open(xlsx_path, 'w').close()

        storage_f = (
            as_storage_file if __debug_flag__ else storage_file)

        for index, (entity_client_id, entity_object_id, eaf_url) in (
            enumerate(eaf_query.yield_per(256))):

            log.debug(
                '\neaf_search {0}/{1}: entity {2}/{3} {4}'.format(
                    index + 1,
                    eaf_count,
                    entity_client_id,
                    entity_object_id,
                    repr(eaf_url)))

            sheet_name = (
                'entity_{0}_{1}'.format(
                    entity_client_id, entity_object_id))

            with storage_f(storage, eaf_url) as eaf_file:

                eaf_search(
                    eaf_file,
                    xlsx_path,
                    search_query,
                    sheet_name,
                    result_list)

        # If we do not have any results, we try to ensure that no unnecessary empty XLSX files are left.

        if not result_list:
            
            try:
                os.remove(xlsx_path)

            except:
                pass

            xlsx_url = None

        # Saving debug copy, if required.

        elif __debug_flag__:
            shutil.copy(xlsx_path, '__eaf_search__.xlsx')

        log.debug(
            '\neaf_search'
            '\nlen(result_list): {0}'
            '\nresult_list: {1}'.format(
                len(result_list),
                pprint.pformat(result_list, width = 192)))

        return cls(
            result_list = result_list,
            xlsx_url = xlsx_url)

