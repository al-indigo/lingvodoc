
# Standard library imports.

import base64
import collections
import copy
import gzip
import hashlib
import io
import logging
import math
import os.path
import pickle
import pprint
import random
import re
import shutil
import string
import tempfile
import time
import traceback
import uuid

# Library imports.

import graphene
import graphene.types

# So that matplotlib does not require display stuff, in particular, tkinter. See e.g. https://
# stackoverflow.com/questions/4931376/generating-matplotlib-graphs-without-a-running-x-server.
import matplotlib
matplotlib.use('Agg', warn = False)

from matplotlib import pyplot

import minio
import numpy
import pathvalidate
import psycopg2.errors
import pylab

from pyramid.httpexceptions import HTTPOk

from pyramid.request import Request

import scipy.optimize
import scipy.sparse.csgraph

import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics
import sklearn.mixture

import sqlalchemy

from sqlalchemy import (
    and_,
    cast,
    create_engine,
    func,
    or_,
    tuple_,
    union,
    desc)

import sqlalchemy.dialects.postgresql as postgresql

from sqlalchemy.orm import (
    aliased,
    joinedload)

from sqlalchemy.orm.attributes import flag_modified

import sqlalchemy.types

from zope.sqlalchemy import mark_changed
import sqlite3

import transaction
import xlsxwriter

# Project imports.

import lingvodoc.acl as acl

import lingvodoc.cache.caching as caching

from lingvodoc.cache.caching import (
    initialize_cache,
    TaskStatus)

import lingvodoc.models as models

from lingvodoc.models import (
    BaseGroup as dbBaseGroup,
    Client,
    DBSession,
    Dictionary as dbDictionary,
    DictionaryPerspective as dbPerspective,
    DictionaryPerspectiveToField as dbColumn,
    Email as dbEmail,
    ENGLISH_LOCALE,
    Entity as dbEntity,
    Field as dbField,
    Grant as dbGrant,
    Group as dbGroup,
    Language as dbLanguage,
    LexicalEntry as dbLexicalEntry,
    Locale as dbLocale,
    Organization as dbOrganization,
    Parser as dbParser,
    ParserResult as dbParserResult,
    PublishingEntity as dbPublishingEntity,
    TranslationAtom as dbTranslationAtom,
    TranslationGist as dbTranslationGist,
    UnstructuredData as dbUnstructuredData,
    User as dbUser,
    user_to_group_association,
    UserBlobs as dbUserBlobs,
    UserRequest as dbUserRequest,
    ValencyAnnotationData as dbValencyAnnotationData,
    ValencyEafData as dbValencyEafData,
    ValencyInstanceData as dbValencyInstanceData,
    ValencyMergeData as dbValencyMergeData,
    ValencyParserData as dbValencyParserData,
    ValencySentenceData as dbValencySentenceData,
    ValencySourceData as dbValencySourceData,
    AdverbAnnotationData as dbAdverbAnnotationData,
    AdverbInstanceData as dbAdverbInstanceData)

from lingvodoc.queue.celery import celery

from lingvodoc.schema.gql_basegroup import (
    AddUserToBasegroup,
    BaseGroup,
    CreateBasegroup)

from lingvodoc.schema.gql_cognate import (
    CognateAnalysis,
    ComplexDistance,
    MorphCognateAnalysis,
    PhonemicAnalysis,
    SwadeshAnalysis,
    XlsxBulkDisconnect)

from lingvodoc.schema.gql_column import (
    Column,
    CreateColumn,
    DeleteColumn,
    UpdateColumn)

from lingvodoc.schema.gql_convert_dictionary import (
    ConvertDictionary,
    ConvertFiveTiers)

from lingvodoc.schema.gql_copy_field import (
    CopySingleField,
    CopySoundMarkupFields)

from lingvodoc.schema.gql_dictionary import (
    AddDictionaryRoles,
    CreateDictionary,
    DeleteDictionary,
    DeleteDictionaryRoles,
    Dictionary,
    UndeleteDictionary,
    UpdateDictionary,
    UpdateDictionaryAtom,
    UpdateDictionaryStatus)

from lingvodoc.schema.gql_dictionaryperspective import (
    AddPerspectiveRoles,
    CreateDictionaryPerspective,
    DeleteDictionaryPerspective,
    DeletePerspectiveRoles,
    DictionaryPerspective as Perspective,
    UndeleteDictionaryPerspective,
    UpdateDictionaryPerspective,
    UpdatePerspectiveAtom,
    UpdatePerspectiveStatus,
    entries_with_entities)

from lingvodoc.schema.gql_entity import (
    ApproveAllForUser,
    BulkCreateEntity,
    BulkDeleteEntity,
    BulkUpdateEntityContent,
    CreateEntity,
    DeleteEntity,
    Entity,
    UpdateEntity,
    UpdateEntityContent)

from lingvodoc.schema.gql_field import (
    CreateField,
    Field)

from lingvodoc.schema.gql_grant import (
    CreateGrant,
    Grant,
    UpdateGrant)

from lingvodoc.schema.gql_group import Group

from lingvodoc.schema.gql_holders import (
    AdditionalMetadata,
    client_id_check,
    CreatedAt,
    del_object,
    get_published_translation_gist_id_cte_query,
    gql_none_value,
    LingvodocID,
    ObjectVal,
    PermissionException,
    ResponseError,
    UnstructuredData,
    Upload)

from lingvodoc.schema.gql_language import (
    CreateLanguage,
    DeleteLanguage,
    Language,
    Language_Resolver,
    language_resolver_args,
    LanguageTree,
    MoveLanguage,
    UpdateLanguage,
    UpdateLanguageAtom)

from lingvodoc.schema.gql_lexicalentry import (
    BulkCreateLexicalEntry,
    BulkDeleteLexicalEntry,
    BulkUndeleteLexicalEntry,
    ConnectLexicalEntries,
    CreateLexicalEntry,
    DeleteGroupingTags,
    DeleteLexicalEntry,
    LexicalEntry)

from lingvodoc.schema.gql_merge import MergeBulk

from lingvodoc.schema.gql_organization import (
    CreateOrganization,
    DeleteOrganization,
    Organization,
    UpdateOrganization)

from lingvodoc.schema.gql_parser import Parser

from lingvodoc.schema.gql_parserresult import (
    CreateValencyData,
    DeleteParserResult,
    ExecuteParser,
    ParserResult,
    SaveValencyData,
    SetValencyAnnotation,
    UpdateParserResult,
    Valency,
    ValencyVerbCases)

from lingvodoc.schema.gql_search import (
    AdvancedSearch,
    AdvancedSearchSimple,
    EafSearch)

from lingvodoc.schema.gql_sync import (
    DownloadDictionaries,
    DownloadDictionary,
    Synchronize)

from lingvodoc.schema.gql_tasks import (
    DeleteTask,
    Task)

from lingvodoc.schema.gql_translationatom import (
    CreateTranslationAtom,
    DeleteTranslationAtom,
    TranslationAtom,
    UpdateTranslationAtom)

from lingvodoc.schema.gql_translationgist import (
    CreateTranslationGist,
    DeleteTranslationGist,
    TranslationGist,
    UndeleteTranslationGist)

from lingvodoc.schema.gql_user import (
    ActivateDeactivateUser,
    CreateUser,
    UpdateUser,
    User)

from lingvodoc.schema.gql_userblobs import (
    CreateUserBlob,
    DeleteUserBlob,
    UserBlobs)

from lingvodoc.schema.gql_userrequest import (
    AcceptUserRequest,
    AddDictionaryToGrant,
    AddDictionaryToOrganization,
    AdministrateOrg,
    CreateGrantPermission,
    ParticipateOrg,
    UserRequest)

from lingvodoc.scripts import elan_parser

import lingvodoc.scripts.adverb as adverb
import lingvodoc.scripts.valency as valency
import lingvodoc.scripts.docx_import as docx_import
import lingvodoc.scripts.docx_to_xlsx as docx_to_xlsx
import lingvodoc.scripts.list_cognates as list_cognates

from lingvodoc.scripts.save_dictionary import save_dictionary as sync_save_dictionary

import lingvodoc.utils as utils

from lingvodoc.utils import (
    plain_text_converter,
    render_statement,
    starling_converter)

from lingvodoc.utils.creation import (
    create_entity,
    get_result_json,
    edit_role)

from lingvodoc.utils.elan_functions import tgt_to_eaf
from lingvodoc.utils.merge import merge_suggestions

from lingvodoc.utils.phonology import (
    gql_phonology as utils_phonology,
    gql_phonology_skip_list as utils_phonology_skip_list,
    gql_phonology_tier_list as utils_phonology_tier_list,
    gql_phonology_link_perspective_data,
    gql_sound_and_markup)

from lingvodoc.utils.proxy import try_proxy

from lingvodoc.utils.search import (
    get_id_to_field_dict,
    translation_gist_search)

import lingvodoc.version

from lingvodoc.views.v2.phonology import (
    get_vowel_class,
    Phonology_Parameters,
    process_sound_markup,
    PickleCache)

from lingvodoc.views.v2.save_dictionary.core import async_save_dictionary

from lingvodoc.views.v2.utils import (
    anonymous_userid,
    storage_file,
    view_field_from_object)

from pdb import set_trace as A

# Setting up logging.
log = logging.getLogger(__name__)


class TierList(graphene.ObjectType):
    tier_count = graphene.Field(ObjectVal)
    total_count = graphene.Int()


class SkipList(graphene.ObjectType):
    markup_count = graphene.Int()
    neighbour_list = graphene.Field(ObjectVal)
    skip_list = graphene.Field(ObjectVal)
    total_neighbour_count = graphene.Int()
    total_skip_count = graphene.Int()


class Link_Perspective_Data(graphene.ObjectType):
    field_data_list = graphene.List(ObjectVal)
    perspective_id_list = graphene.List(LingvodocID)


class LexicalEntriesAndEntities(graphene.ObjectType):
    entities = graphene.List(Entity)
    lexical_entries = graphene.List(LexicalEntry)


class Permissions(graphene.ObjectType):
    edit = graphene.List(Perspective)
    view = graphene.List(Perspective)
    publish = graphene.List(Perspective)
    limited = graphene.List(Perspective)


class StarlingField(graphene.InputObjectType):
    starling_name = graphene.String(required=True)
    starling_type = graphene.Int(required=True)
    field_id = LingvodocID(required=True)
    fake_id = graphene.String()
    link_fake_id = LingvodocID()  # graphene.String()


class DialeqtInfo(graphene.ObjectType):
    dictionary_name = graphene.String()
    dialeqt_id = graphene.String()


class MergeSuggestions(graphene.ObjectType):
    match_result = graphene.List(ObjectVal)
    user_has_permissions = graphene.Boolean()


def get_dict_attributes(sqconn):
    dict_trav = sqconn.cursor()
    dict_trav.execute("""SELECT
                        dict_name,
                        dict_identificator,
                        dict_description
                        FROM
                        dict_attributes
                        WHERE
                        id = 1;""")
    req = dict()
    for dictionary in dict_trav:
        req['dictionary_name'] = dictionary[0]
        req['dialeqt_id'] = dictionary[1]
    return req


class Query(graphene.ObjectType):
    client = graphene.String()
    dictionaries = graphene.List(Dictionary, published=graphene.Boolean(),
                                 mode=graphene.Int(),
                                 category=graphene.Int(),
                                 proxy=graphene.Boolean())
    dictionary = graphene.Field(Dictionary, id=LingvodocID())
    perspectives = graphene.List(Perspective,
        published=graphene.Boolean(),
        with_phonology_data=graphene.Boolean(),
        with_valency_data=graphene.Boolean())
    perspective = graphene.Field(Perspective, id=LingvodocID())
    entity = graphene.Field(Entity, id=LingvodocID())
    language = graphene.Field(Language, id=LingvodocID())

    languages = (

        graphene.List(
            Language,
            id_list = graphene.List(LingvodocID),
            only_in_toc = graphene.Boolean(),
            only_with_dictionaries_recursive = graphene.Boolean(),
            dictionary_category = graphene.Int(),
            dictionary_published = graphene.Boolean(),
            in_tree_order = graphene.Boolean()))

    user = graphene.Field(User, id=graphene.Int())
    users = graphene.List(User, search=graphene.String())
    field = graphene.Field(Field, id=LingvodocID())
    translationgist = graphene.Field(TranslationGist, id=LingvodocID())
    userblob = graphene.Field(UserBlobs, id=LingvodocID())
    translationatom = graphene.Field(TranslationAtom, id=LingvodocID())
    organization = graphene.Field(Organization, id=LingvodocID())

    organizations = (

        graphene.List(
            Organization,
            has_participant = graphene.Boolean(),
            participant_deleted = graphene.Boolean(),
            participant_category = graphene.Int(),
            participant_published = graphene.Boolean()))

    lexicalentry = graphene.Field(LexicalEntry, id=LingvodocID())
    basic_search = graphene.Field(LexicalEntriesAndEntities, searchstring=graphene.String(),
                                  can_add_tags=graphene.Boolean(),
                                  perspective_id=LingvodocID(), field_id=LingvodocID(),
                                  search_in_published=graphene.Boolean())
    advanced_lexicalentries = graphene.List(LexicalEntry, searchstrings=graphene.List(ObjectVal),
                                            perspectives=LingvodocID(),
                                            adopted=graphene.Boolean(),
                                            adopted_type=LingvodocID(),
                                            with_entimology=graphene.Boolean())
    translationgists = graphene.List(TranslationGist, gists_type=graphene.String())

    translation_search = (
        graphene.List(
            TranslationGist,
            searchstring = graphene.String(),
            search_case_insensitive = graphene.Boolean(),
            search_regular_expression = graphene.Boolean(),
            translation_type = graphene.String(),
            deleted = graphene.Boolean(),
            order_by_type = graphene.Boolean(),
            no_result_error_flag = graphene.Boolean()))

    translation_service_search = graphene.Field(TranslationGist, searchstring=graphene.String())
    advanced_translation_search = graphene.List(TranslationGist, searchstrings=graphene.List(graphene.String))
    optimized_translation_search = graphene.List(graphene.String, searchstrings=graphene.List(graphene.String))
    all_locales = graphene.List(ObjectVal)
    user_blobs = graphene.List(UserBlobs, data_type=graphene.String(), is_global=graphene.Boolean())
    userrequest = graphene.Field(UserRequest, id=graphene.Int())
    userrequests = graphene.List(UserRequest)
    all_basegroups = graphene.List(BaseGroup)
    all_data_types = graphene.List(TranslationGist)
    all_fields = graphene.List(Field, common=graphene.Boolean(), parallel=graphene.Boolean())
    common_fields = graphene.List(Field)
    all_statuses = graphene.List(TranslationGist)
    template_fields = graphene.List(Field, mode=graphene.String())
    template_modes = graphene.List(graphene.String)
    grant = graphene.Field(Grant, id=graphene.Int())

    grants = (

        graphene.List(
            Grant,
            has_participant = graphene.Boolean(),
            participant_deleted = graphene.Boolean(),
            participant_category = graphene.Int(),
            participant_published = graphene.Boolean()))

    column = graphene.Field(Column, id=LingvodocID())

    phonology_tier_list = graphene.Field(TierList, perspective_id=LingvodocID(required=True))
    phonology_skip_list = graphene.Field(SkipList, perspective_id=LingvodocID(required=True))

    phonology_link_perspective_data = graphene.Field(
        Link_Perspective_Data,
        perspective_id = LingvodocID(required = True),
        field_id_list = graphene.List(LingvodocID, required = True))

    connected_words = graphene.Field(LexicalEntriesAndEntities, id=LingvodocID(required=True),
                                     field_id=LingvodocID(required=True), mode=graphene.String())

    advanced_search = (

        graphene.Field(
            AdvancedSearch,
            languages = graphene.List(LingvodocID),
            dicts_to_filter = graphene.List(LingvodocID),
            tag_list = graphene.List(graphene.String),
            category = graphene.Int(),
            adopted = graphene.Boolean(),
            etymology = graphene.Boolean(),
            diacritics = graphene.String(),
            search_strings = graphene.List(graphene.List(ObjectVal), required = True),
            mode = graphene.String(),
            search_metadata = ObjectVal(),
            simple = graphene.Boolean(),
            xlsx_export = graphene.Boolean(),
            cognates_flag = graphene.Boolean(),
            load_entities = graphene.Boolean(),
            debug_flag = graphene.Boolean()))

    advanced_search_simple = (

        graphene.Field(
            AdvancedSearchSimple,
            languages = graphene.List(LingvodocID),
            dicts_to_filter = graphene.List(LingvodocID),
            tag_list = graphene.List(graphene.String),
            category = graphene.Int(),
            adopted = graphene.Boolean(),
            etymology = graphene.Boolean(),
            diacritics = graphene.String(),
            search_strings = graphene.List(graphene.List(ObjectVal), required = True),
            mode = graphene.String()))

    convert_markup = graphene.Field(
        graphene.String, id=LingvodocID(required=True))

    permission_lists = graphene.Field(Permissions, proxy=graphene.Boolean(required=True))
    tasks = graphene.List(Task)
    is_authenticated = graphene.Boolean()
    dictionary_dialeqt_get_info = graphene.Field(DialeqtInfo, blob_id=LingvodocID(required=True))

    convert_five_tiers_validate = (

        graphene.Field(
            graphene.List(ObjectVal),
            markup_id_list = graphene.List(LingvodocID, required=True)))

    merge_suggestions = graphene.Field(MergeSuggestions, perspective_id=LingvodocID(required=True),
                                       algorithm=graphene.String(required=True),
                                       entity_type_primary=graphene.String(),
                                       entity_type_secondary=graphene.String(),
                                       levenshtein=graphene.Int(),
                                       threshold=graphene.Float(),
                                       field_selection_list=graphene.List(ObjectVal), )
    select_tags_metadata = ObjectVal()
    get_user_groups = graphene.Field(
        graphene.List(graphene.String), id=graphene.Int(required=False))
    perspectives_fields_intersection = graphene.Field(
        graphene.List(Field), perspectives=graphene.List(LingvodocID),)

    eaf_search = (
        graphene.Field(EafSearch,
            perspective_id = LingvodocID(),
            search_query = graphene.Argument(ObjectVal),
            debug_flag = graphene.Boolean()))

    version = graphene.String()
    version_uniparser = ObjectVal()

    client_list = (
        graphene.Field(
            graphene.List(graphene.List(graphene.Int)),
            client_id_list = graphene.List(graphene.Int, required = True)))
    parser_results = graphene.Field((graphene.List(ParserResult)),
                                    entity_id = LingvodocID(), parser_id=LingvodocID())
    parser_result = graphene.Field(ParserResult, id=LingvodocID(), exact_fmt=graphene.String())
    parsers = graphene.Field(graphene.List(Parser))

    unstructured_data = (

        graphene.Field(
            UnstructuredData,
            id = graphene.String(required = True)))

    valency_data = (

        graphene.Field(
            ObjectVal,
            perspective_id = LingvodocID(required = True),
            offset = graphene.Int(),
            limit = graphene.Int(),
            verb_prefix = graphene.String(),
            case_flag = graphene.Boolean(),
            accept_value = graphene.Boolean(),
            sort_order_list = graphene.List(graphene.String),
            debug_flag = graphene.Boolean()))

    adverb_data = (

        graphene.Field(
            ObjectVal,
            perspective_id = LingvodocID(required = True),
            offset = graphene.Int(),
            limit = graphene.Int(),
            specificity_flag = graphene.Boolean(),
            adverb_prefix = graphene.String(),
            case_flag = graphene.Boolean(),
            accept_value = graphene.Boolean(),
            sort_order_list = graphene.List(graphene.String),
            debug_flag = graphene.Boolean()))

    language_toc = graphene.List(Language)

    language_tree = (

        graphene.Field(
            LanguageTree,
            dictionary_category = graphene.Int(),
            dictionary_published = graphene.Boolean(),
            language_id = LingvodocID(),
            by_grants = graphene.Boolean(),
            grant_id = graphene.Int(),
            by_organizations = graphene.Boolean(),
            organization_id = graphene.Int(),
            debug_flag = graphene.Boolean()))

    fill_logs = graphene.String(worker = graphene.Int())

    def resolve_fill_logs(self, info, worker=1):
        # Check if the current user is administrator
        client_id = info.context.client_id
        user_id = DBSession.query(Client.user_id).filter_by(id=client_id).scalar()
        if user_id != 1:
            return ResponseError("Only administrator can fill in logs using request.")

        period = 300
        step = 0.25
        times = int(period // step)
        for _ in range(times):
            log.debug(f"Lingvodoc({worker}) " * 500)
            time.sleep(step)
        return "Done"

    def resolve_language_tree(
        self,
        info,
        dictionary_category = None,
        dictionary_published = None,
        language_id = None,
        by_grants = False,
        grant_id = None,
        by_organizations = False,
        organization_id = None,
        debug_flag = False):

        try:

            language_field_asts = []

            for field in info.field_asts:

                if field.name.value != 'language_tree':
                    continue

                language_field_asts = (
                    field.selection_set.selections)

            resolver = (

                Language_Resolver(

                    info,
                    language_field_asts,

                    language_resolver_args(
                        only_with_dictionaries_recursive = True,
                        dictionary_category = dictionary_category,
                        dictionary_published = dictionary_published,
                        language_id = language_id,
                        by_grants = by_grants,
                        grant_id = grant_id,
                        by_organizations = by_organizations,
                        organization_id = organization_id,
                        in_tree_order = True),

                    debug_flag = debug_flag))

            gql_language_list = (
                resolver.run())

            from_to_dict = (
                resolver.from_to_dict)

            filtered_out_id_set = (
                resolver.filtered_out_id_set)

            tree_object = None

            # Constructing per grant / per organization language trees.

            if resolver.grant_or_organization:

                count_dict = (
                    resolver.aggregate_count_dict)

                count_dict[(None, None)] = {}

                recursive_count_dict = {}

                def f(language_id):

                    group_count_dict = (
                        recursive_count_dict.get(language_id))

                    if group_count_dict is not None:
                        return group_count_dict

                    group_count_dict = (

                        collections.Counter(
                            count_dict[language_id]))

                    for to_id in from_to_dict[language_id]:

                        group_count_dict += f(to_id)

                    recursive_count_dict[language_id] = group_count_dict

                    return group_count_dict

                # Getting properly ordered list of grants / organizations with dictionaries.

                id_str_set = (

                    f((None, None)).keys())

                id_empty_flag = (

                    '' in id_str_set)

                id_list = [

                    int(id_str)
                    for id_str in id_str_set
                    if id_str]

                if by_grants:

                    id_row_list = (

                        DBSession

                            .query(
                                cast(dbGrant.id, models.UnicodeText))

                            .filter(
                                dbGrant.id.in_(
                                    utils.values_query(
                                        id_list, models.SLBigInteger)))

                            .order_by(
                                dbGrant.grant_number,
                                dbGrant.id)

                            .all())

                else:

                    id_row_list = (

                        DBSession

                            .query(
                                cast(dbOrganization.id, models.UnicodeText))

                            .filter(
                                dbOrganization.id.in_(
                                    utils.values_query(
                                        id_list, models.SLBigInteger)))

                            .order_by(
                                dbOrganization.id)

                            .all())

                id_str_list = [

                    id_row[0]
                    for id_row in id_row_list]

                if id_empty_flag:

                    id_str_list.append('')

                if debug_flag:

                    log.debug(
                        f'\n id_list:\n{id_list}')

                tree_object_list = []

                def g(
                    language_id,
                    group_id_str,
                    dictionary_count = 0):

                    node_list = [

                        language_id if language_id != (None, None) else
                        int(group_id_str) if group_id_str else
                        None]

                    item_list = []

                    for to_id in from_to_dict[language_id]:

                        count = (
                            f(to_id).get(group_id_str, 0))

                        item, count = (
                            g(to_id, group_id_str, count))

                        if count > 0:

                            item_list.append(item)
                            dictionary_count += count

                    if item_list:

                        node_list.append(item_list)

                    return (
                        node_list, dictionary_count)

                for id_str in id_str_list:

                    tree_object, _ = (
                        g((None, None), id_str))

                    tree_object_list.append(
                        tree_object)

                tree_object = [
                    None, tree_object_list]

            # Constructing standard language tree, if we actually have language data.

            elif gql_language_list:

                root_id = (

                    tuple(language_id) if language_id else
                    (None, None))

                def f(language_id):

                    node_list = [

                        None if language_id == (None, None) else
                        language_id]

                    item_list = [

                        f(to_id)
                        for to_id in from_to_dict[language_id]
                        if to_id not in filtered_out_id_set]

                    if item_list:

                        node_list.append(item_list)

                    return node_list
                        
                tree_object = f(root_id)

            if debug_flag:

                log.debug(

                    '\n tree_object:\n' +

                    pprint.pformat(
                        tree_object, width = 144))

            return (

                LanguageTree(
                    tree = tree_object,
                    languages = gql_language_list))

        except Exception as exception:

            traceback_string = (

                ''.join(
                    traceback.format_exception(
                        exception, exception, exception.__traceback__))[:-1])

            log.warning('language_tree: exception')
            log.warning(traceback_string)

            return (
                ResponseError(
                    'Exception:\n' + traceback_string))

    def resolve_valency_data(
        self,
        info,
        perspective_id,
        offset = 0,
        limit = 25,
        verb_prefix = None,
        case_flag = False,
        accept_value = None,
        sort_order_list = None,
        debug_flag = False,
        **args):

        log.debug(
            f'\nperspective_id: {perspective_id}'
            f'\noffset: {offset}'
            f'\nlimit: {limit}'
            f'\nverb_prefix: {repr(verb_prefix)}'
            f'\ncase_flag: {case_flag}'
            f'\naccept_value: {accept_value}'
            f'\nsort_order_list: {sort_order_list}'
            f'\ndebug_flag: {debug_flag}')

        if sort_order_list is None:
            sort_order_list = ['verb', 'case', 'accept']

        verb_flag = verb_prefix is not None
        accept_flag = accept_value is not None

        # We'll need source deletion info and acceptance info to filter out not accepted instances from
        # deleted sources.

        source_id_cte = (

            DBSession

                .query(
                    dbValencySourceData.id)

                .filter(
                    dbValencySourceData.perspective_client_id == perspective_id[0],
                    dbValencySourceData.perspective_object_id == perspective_id[1])

                .cte())

        source_parser_query = (

            DBSession

                .query(

                    source_id_cte.c.id
                        .label('id'),

                    (dbParserResult.marked_for_deletion |
                        dbEntity.marked_for_deletion |
                        dbLexicalEntry.marked_for_deletion)

                        .label('marked_for_deletion'))

                .filter(
                    source_id_cte.c.id == dbValencyParserData.id,
                    dbParserResult.client_id == dbValencyParserData.parser_result_client_id,
                    dbParserResult.object_id == dbValencyParserData.parser_result_object_id,
                    dbEntity.client_id == dbParserResult.entity_client_id,
                    dbEntity.object_id == dbParserResult.entity_object_id,
                    dbLexicalEntry.client_id == dbEntity.parent_client_id,
                    dbLexicalEntry.object_id == dbEntity.parent_object_id))

        source_eaf_query = (

            DBSession

                .query(

                    source_id_cte.c.id
                        .label('id'),

                    (dbEntity.marked_for_deletion |
                        dbLexicalEntry.marked_for_deletion)

                        .label('marked_for_deletion'))

                .filter(
                    source_id_cte.c.id == dbValencyEafData.id,
                    dbEntity.client_id == dbValencyEafData.entity_client_id,
                    dbEntity.object_id == dbValencyEafData.entity_object_id,
                    dbLexicalEntry.client_id == dbEntity.parent_client_id,
                    dbLexicalEntry.object_id == dbEntity.parent_object_id))

        source_subquery = (

            source_parser_query
                .union(source_eaf_query)
                .subquery())

        accept_subquery = (

            DBSession

                .query(
                    dbValencyAnnotationData.instance_id,

                    func.bool_or(dbValencyAnnotationData.accepted)
                        .label('accept_value'))

                .group_by(
                    dbValencyAnnotationData.instance_id)

                .subquery())

        instance_query = (

            DBSession

                .query(
                    dbValencyInstanceData)

                .filter(
                    dbValencySentenceData.source_id == source_subquery.c.id,
                    dbValencyInstanceData.sentence_id == dbValencySentenceData.id)

                .outerjoin(
                    accept_subquery,
                    dbValencyInstanceData.id == accept_subquery.c.instance_id)

                .filter(
                    or_(
                        ~source_subquery.c.marked_for_deletion,
                        func.coalesce(
                            accept_subquery.c.accept_value, False))))

        # We'll need that for getting verb merge groupings.

        merge_filter_list = [
            dbValencyMergeData.perspective_client_id == perspective_id[0],
            dbValencyMergeData.perspective_object_id == perspective_id[1]]

        if verb_prefix:

            verb_prefix_filter_str = (
                verb_prefix.replace('%', '\\%') + '%')

            merge_filter_list.append(

                dbValencyMergeData.verb_lex.ilike(
                    verb_prefix_filter_str))

        lex_id_cte = (

            DBSession

                .query(
                    dbValencyMergeData.verb_lex,
                    dbValencyMergeData.merge_id)

                .filter(
                    *merge_filter_list)

                .cte())

        id_lex_list_cte = (

            DBSession

                .query(
                    dbValencyMergeData.merge_id,

                    func.array_agg(
                        postgresql.aggregate_order_by(
                            dbValencyMergeData.verb_lex,
                            dbValencyMergeData.verb_lex))

                        .label('verb_lex_list'))

                .filter(
                    dbValencyMergeData.perspective_client_id == perspective_id[0],
                    dbValencyMergeData.perspective_object_id == perspective_id[1],

                    dbValencyMergeData.merge_id.in_(
                        DBSession.query(lex_id_cte.c.merge_id)))

                .group_by(
                    dbValencyMergeData.merge_id)

                .cte())

        # Filtering by verb prefix, if required.

        if verb_prefix:

            verb_lex_subquery = (

                DBSession

                    .query(
                        dbValencyMergeData.verb_lex)

                    .filter(
                        dbValencyMergeData.perspective_client_id == perspective_id[0],
                        dbValencyMergeData.perspective_object_id == perspective_id[1],

                        dbValencyMergeData.merge_id.in_(
                            DBSession.query(lex_id_cte.c.merge_id)))

                    .subquery())

            instance_query = (

                instance_query.filter(

                    or_(
                        dbValencyInstanceData.verb_lex.ilike(
                            verb_prefix_filter_str),

                        dbValencyInstanceData.verb_lex.in_(
                            verb_lex_subquery))))

        instance_count = (
            instance_query.count())

        if debug_flag:

            log.debug(
                '\ninstance_query:\n' +
                str(instance_query.statement.compile(compile_kwargs = {'literal_binds': True})))

            log.debug(
                f'\ninstance_count: {instance_count}')

        # Getting ready to sort, if required.

        order_by_list = []

        for sort_type in sort_order_list:

            if sort_type == 'verb':

                if verb_flag:

                    lex_list_subquery = (

                        DBSession

                            .query(
                                dbValencyMergeData.verb_lex,
                                id_lex_list_cte.c.verb_lex_list)

                            .filter(
                                dbValencyMergeData.perspective_client_id == perspective_id[0],
                                dbValencyMergeData.perspective_object_id == perspective_id[1],
                                dbValencyMergeData.merge_id == id_lex_list_cte.c.merge_id)

                            .subquery())

                    instance_query = (

                        instance_query

                            .outerjoin(
                                lex_list_subquery,
                                dbValencyInstanceData.verb_lex == lex_list_subquery.c.verb_lex))

                    order_by_list.extend((

                        func.coalesce(
                            lex_list_subquery.c.verb_lex_list,
                            postgresql.array(
                                [dbValencyInstanceData.verb_lex])),

                        dbValencyInstanceData.verb_lex))

            elif sort_type == 'case':

                if case_flag:

                    # Getting case ordering mapping as a temporary table.

                    case_table_name = (

                        'case_table_' +
                        str(uuid.uuid4()).replace('-', '_'))

                    case_value_str = (

                        ', '.join(
                            f'(\'{case_str}\', {index})'
                            for index, case_str in (
                                enumerate(valency.cases))))

                    DBSession.execute(f'''

                        create temporary table

                        {case_table_name} (
                          case_str TEXT PRIMARY KEY,
                          order_value INT NOT NULL)

                        on commit drop;

                        insert into {case_table_name}
                        values {case_value_str};

                        ''')

                    class tmpCaseOrder(models.Base):

                        __tablename__ = case_table_name

                        case_str = (
                            sqlalchemy.Column(sqlalchemy.types.UnicodeText, primary_key = True))

                        order_value = (
                            sqlalchemy.Column(sqlalchemy.types.Integer, nullable = False))

                    # Ordering by cases.

                    instance_query = (

                        instance_query.outerjoin(
                            tmpCaseOrder,
                            dbValencyInstanceData.case_str == tmpCaseOrder.case_str))

                    order_by_list.append(
                        tmpCaseOrder.order_value)

            elif sort_type == 'accept':

                if accept_flag:

                    order_by_list.append(
                        func.coalesce(accept_subquery.c.accept_value, False) != accept_value)

        order_by_list.append(
            dbValencyInstanceData.id)

        # Getting annotation instances and related info.

        instance_query = (

            instance_query
                .order_by(*order_by_list)
                .offset(offset)
                .limit(limit))

        instance_list = instance_query.all()

        if debug_flag:

            log.debug(
                f'\ninstance_query ({len(instance_list)}):\n' +
                str(instance_query.statement.compile(compile_kwargs = {'literal_binds': True})))

        instance_id_set = (
            set(instance.id for instance in instance_list))

        instance_verb_lex_set = (
            set(instance.verb_lex for instance in instance_list))

        sentence_id_set = (
            set(instance.sentence_id for instance in instance_list))

        log.debug(
            '\ninstance_id_set: {}'
            '\nsentence_id_set: {}'.format(
                instance_id_set,
                sentence_id_set))

        sentence_list = []

        if sentence_id_set:

            sentence_list = (

                DBSession

                    .query(
                        dbValencySentenceData)

                    .filter(
                        dbValencySentenceData.id.in_(

                            utils.values_query(
                                sentence_id_set, models.SLBigInteger)))

                    .all())

        annotation_list = []

        if instance_id_set:

            annotation_list = (

                DBSession

                    .query(
                        dbValencyAnnotationData.instance_id,

                        func.jsonb_agg(
                            func.jsonb_build_array(
                                dbValencyAnnotationData.user_id,
                                dbValencyAnnotationData.accepted)))

                    .filter(
                        dbValencyAnnotationData.instance_id.in_(

                            utils.values_query(
                                instance_id_set, models.SLBigInteger)))

                    .group_by(
                        dbValencyAnnotationData.instance_id)

                    .all())

        user_id_set = (

            set(user_id
                for _, user_annotation_list in annotation_list
                for user_id, _ in user_annotation_list))

        user_list = []

        if user_id_set:

            user_list = (

                DBSession

                    .query(
                        dbUser.id, dbUser.name)

                    .filter(
                        dbUser.id.in_(

                            utils.values_query(
                                user_id_set, models.SLBigInteger)))

                    .all())

        instance_list = [

            {'id': instance.id,
                'sentence_id': instance.sentence_id,
                'index': instance.index,
                'verb_lex': instance.verb_lex,
                'case_str': instance.case_str}

            for instance in instance_list]

        sentence_list = [
            dict(sentence.data, id = sentence.id)
            for sentence in sentence_list]

        merge_list = []

        if instance_verb_lex_set:

            merge_id_subquery = (

                DBSession

                    .query(
                        lex_id_cte.c.merge_id)

                    .filter(
                        lex_id_cte.c.verb_lex.in_(

                            utils.values_query(
                                instance_verb_lex_set, models.String)))

                    .subquery())

            merge_list = (

                DBSession

                    .query(
                        id_lex_list_cte.c.verb_lex_list)

                    .filter(
                        id_lex_list_cte.c.merge_id.in_(
                            merge_id_subquery))

                    .all())

            merge_list = [
                item[0] for item in merge_list]

        log.debug(

            '\ninstance_list ({}):\n{}'
            '\nmerge_list ({}):\n{}'
            '\nsentence_list ({}):\n{}'
            '\nannotation_list ({}):\n{}'
            '\nuser_list ({}):\n{}'.format(

                len(instance_list),
                pprint.pformat(instance_list, width = 192),
                len(merge_list),
                pprint.pformat(merge_list, width = 192),
                len(sentence_list),
                pprint.pformat(sentence_list, width = 192),
                len(annotation_list),
                pprint.pformat(annotation_list, width = 192),
                len(user_list),
                pprint.pformat(user_list, width = 192)))

        result_dict = {
            'instance_count': instance_count,
            'instance_list': instance_list,
            'merge_list': merge_list,
            'sentence_list': sentence_list,
            'annotation_list': annotation_list,
            'user_list': user_list}

        # Getting all verbs, without filtering, if required.

        if verb_flag:

            if verb_prefix:

                verb_query = (

                    DBSession

                        .query(
                            dbValencyInstanceData.verb_lex,
                            dbValencyInstanceData.verb_lex.ilike(
                                verb_prefix_filter_str)))
            else:

                verb_query = (

                    DBSession

                        .query(
                            dbValencyInstanceData.verb_lex))

            verb_list = (

                verb_query

                    .filter(
                        dbValencySourceData.perspective_client_id == perspective_id[0],
                        dbValencySourceData.perspective_object_id == perspective_id[1],
                        dbValencySentenceData.source_id == dbValencySourceData.id,
                        dbValencyInstanceData.sentence_id == dbValencySentenceData.id)

                    .distinct()

                    .order_by(dbValencyInstanceData.verb_lex)

                    .all())

            if verb_prefix:

                result_dict['verb_list'] = (
                    [list(row) for row in verb_list])

            else:

                result_dict['verb_list'] = (
                    [[row[0], True] for row in verb_list])

        return result_dict

    def resolve_unstructured_data(self, info, id):
        return UnstructuredData(id = id)

    def resolve_client_list(
        self,
        info,
        client_id_list):

        client_list = (

            DBSession

                .query(
                    Client.id,
                    Client.user_id)

                .filter(
                    Client.id.in_(set(client_id_list)))

                .distinct())

        return client_list

    def resolve_version(self, info):
        return lingvodoc.version.__version__

    def resolve_version_uniparser(self, info):
        return lingvodoc.version.uniparser_version_dict

    def resolve_eaf_search(
        self,
        info,
        perspective_id = None,
        search_query = None,
        debug_flag = False):

        return EafSearch.constructor(
            info, perspective_id, search_query, debug_flag)

    def resolve_perspectives_fields_intersection(self, info, perspectives=None):
        """
        query qwe($perspective_list: [LingvodocID] ){
        perspective_fields_intersection(perspectives: $perspective_list){
            id
            data_type
            translation
        }
        }


        {
            "perspective_list": [
                [
                    504,
                    5
                ],
                [
                    743,
                    127939
                ]
            ]
        }
        """

        field_ids_cte = DBSession.query(
            dbColumn.field_client_id, dbColumn.field_object_id).filter(
            tuple_(dbColumn.parent_client_id, dbColumn.parent_object_id).in_(perspectives)).distinct().cte()
        field_objects = DBSession.query(dbField).join(field_ids_cte).filter(dbField.client_id==field_ids_cte.c.field_client_id,
                                                                            dbField.object_id==field_ids_cte.c.field_object_id,
                                                                            ).distinct()


        gql_fields = list()
        for db_field in field_objects:
            if db_field.data_type == "Text":
                gql_field = Field()
                gql_field.dbObject = db_field
                gql_fields.append(gql_field)

        return gql_fields

    def resolve_get_user_groups(self, info, id=None):
        if id:
            user = DBSession.query(dbUser).filter(dbUser.id == id).first()
        else:
            client_id = info.context.client_id
            user = Client.get_user_by_client_id(client_id)
        if not user:
            return None
        all_basegroups = DBSession.query(dbBaseGroup).all()
        user_groups = list()
        for base in all_basegroups:
            group = DBSession.query(dbGroup).filter(dbGroup.base_group_id==base.id).first()
            if group:
                if user in group.users:
                    user_groups.append(base.name)
        return user_groups

    def resolve_select_tags_metadata(self, info):

        def get_sorted_metadata_keys(metadata_name):

            all_values = DBSession.query(dbDictionary.additional_metadata[metadata_name]) \
                .filter(dbDictionary.additional_metadata[metadata_name] != None,
                        dbDictionary.marked_for_deletion==False)

            value_set = set()

            for value, in all_values:

                value_set.update(
                    (value_str.strip() for value_str in re.split(r'\s*,\s*', value))
                        if isinstance(value, str) else
                        value)

            return sorted(value_set)

        menu_json_data = {}
        authors_list = get_sorted_metadata_keys("authors")
        menu_json_data["hasAudio"] = [False, True]
        menu_json_data["authors"] = authors_list
        menu_json_data["humanSettlement"] = get_sorted_metadata_keys("humanSettlement")
        menu_json_data["years"] = get_sorted_metadata_keys("years")
        menu_json_data["kind"] = ["Expedition", "Archive"]
        menu_json_data["nativeSpeakersCount"] = ["Vulnerable",
                                                 "Definitely endangered",
                                                 "Critically endangered",
                                                 "Extinct",
                                                 "Severely endangered",
                                                 "Safe"]
        return menu_json_data



    def resolve_merge_suggestions(self, info, perspective_id, algorithm, threshold=0.1,
                                  entity_type_primary='Transcription',
                                  entity_type_secondary='Translation',
                                  levenshtein=1,
                                  field_selection_list=None
                                  ):
        request = info.context.request
        locale_id = info.context.locale_id

        old_field_selection_list = copy.deepcopy(field_selection_list)

        for field_selection in old_field_selection_list:
            field_selection['client_id'] = field_selection['field_id'][0]
            field_selection['object_id'] = field_selection['field_id'][1]

        result = merge_suggestions(request=request,
                                   perspective_client_id=perspective_id[0],
                                   perspective_object_id=perspective_id[1],
                                   algorithm=algorithm,
                                   entity_type_primary=entity_type_primary,
                                   entity_type_secondary=entity_type_secondary,
                                   threshold=threshold, levenshtein=levenshtein,
                                   field_selection_list=old_field_selection_list,
                                   locale_id=locale_id)

        return MergeSuggestions(user_has_permissions=result['user_has_permissions'],
                                match_result=result['match_result'])

    def resolve_convert_five_tiers_validate(self, info, markup_id_list):

        result_list = []

        for index, markup_id in enumerate(markup_id_list):

            if not markup_id:
                result_list.append(False)
                continue

            client_id, object_id = markup_id
            entity = DBSession.query(dbEntity).filter_by(client_id=client_id, object_id=object_id).first()

            if not entity:
                return ResponseError(f'No entity {client_id} / {object_id}.')

            try:

                storage = (
                    info.context.request.registry.settings['storage'])

                with storage_file(
                    storage, entity.content) as content_stream:

                    content = content_stream.read()

            except:
                return ResponseError(f'Cannot access file \'{entity.content}\'.')

            fd, filename = tempfile.mkstemp()
            with open(filename, 'wb') as temp:
                markup = tgt_to_eaf(content, entity.additional_metadata)
                temp.write(markup.encode("utf-8"))
                temp.flush()

                elan_check = elan_parser.ElanCheck(filename)
                elan_check.parse()
                is_valid = elan_check.check()

                if index == 0 and is_valid:
                    elan_reader = elan_parser.Elan(filename)
                    elan_reader.parse()
                    result_list.append(
                        elan_reader.preview())
                else:
                    result_list.append(is_valid)

            os.close(fd)
            os.remove(filename)

        return result_list

    def resolve_dictionary_dialeqt_get_info(self, info, blob_id):  # TODO: test
        blob_client_id, blob_object_id = blob_id
        blob = DBSession.query(dbUserBlobs).filter_by(client_id=blob_client_id, object_id=blob_object_id).first()
        if blob:
            filename = blob.real_storage_path
            sqconn = sqlite3.connect(filename)
            try:
                dict_attributes = get_dict_attributes(sqconn)
            except:
                raise ResponseError(message="database disk image is malformed")
            return DialeqtInfo(dictionary_name=dict_attributes['dictionary_name'], dialeqt_id=dict_attributes["dialeqt_id"])
        raise ResponseError(message="No such blob in the system")

    def resolve_tasks(self, info):
        request = info.context.request
        client_id = info.context.client_id
        if not client_id:
            tasks_dicts = TaskStatus.get_user_tasks(anonymous_userid(request), clear_out=True)
            tasks = [Task(**task_dict) for task_dict in tasks_dicts]
            return tasks
        user = Client.get_user_by_client_id(client_id)
        tasks_dicts = TaskStatus.get_user_tasks(user.id, clear_out=True)
        tasks = [Task(**task_dict) for task_dict in tasks_dicts]
        return tasks

    def resolve_permission_lists(self, info, proxy):
        request = info.context.request
        if proxy:
            try_proxy(request)
        client_id = info.context.client_id

        subreq = Request.blank('/translation_service_search')
        subreq.method = 'POST'
        subreq.headers = request.headers
        headers = dict()
        if request.headers.get('Cookie'):
            headers = {'Cookie': request.headers['Cookie']}
        subreq.headers = headers
        subreq.json = {'searchstring': 'Published'}
        resp = request.invoke_subrequest(subreq)

        if 'error' not in resp.json:
            published_gist_object_id, published_gist_client_id = resp.json['object_id'], resp.json['client_id']
        else:
            raise KeyError("Something wrong with the base", resp.json['error'])

        subreq = Request.blank('/translation_service_search')
        subreq.method = 'POST'
        subreq.headers = request.headers
        headers = dict()
        if request.headers.get('Cookie'):
            headers = {'Cookie': request.headers['Cookie']}
        subreq.headers = headers
        subreq.json = {'searchstring': 'Limited access'}  # todo: fix
        resp = request.invoke_subrequest(subreq)

        if 'error' not in resp.json:
            limited_gist_object_id, limited_gist_client_id = resp.json['object_id'], resp.json['client_id']
        else:
            raise KeyError("Something wrong with the base", resp.json['error'])


        dblimited = DBSession.query(dbPerspective).filter(
            and_(dbPerspective.state_translation_gist_client_id == limited_gist_client_id,
                 dbPerspective.state_translation_gist_object_id == limited_gist_object_id)
        )

        # limited_perms = [("limited", True), ("read", False), ("write", False), ("publish", False)]
        limited = list()
        for dbperspective in dblimited.all():
            perspective = Perspective(id=[dbperspective.client_id, dbperspective.object_id])
            perspective.dbObject = dbperspective
            perspective.list_name='limited'
            limited.append(perspective)
            # fulfill_permissions_on_perspectives(intermediate, pers, limited_perms)


        dbpublished = DBSession.query(dbPerspective).filter(
            and_(dbPerspective.state_translation_gist_client_id == published_gist_client_id,
                 dbPerspective.state_translation_gist_object_id == published_gist_object_id)
        )
        existing = list()
        view = list()
        for dbperspective in dbpublished.all():
            perspective = Perspective(id=[dbperspective.client_id, dbperspective.object_id])
            perspective.dbObject = dbperspective
            perspective.list_name='view'
            view.append(perspective)
            existing.append([dbperspective.client_id, dbperspective.object_id])

        if not client_id:
            return Permissions(limited=limited, view=view, edit=list(), publish=list())

        user = DBSession.query(Client).filter(client_id == Client.id).first()
        if not user:
            return None
        user_id = user.user_id
        editor_basegroup = DBSession.query(dbBaseGroup).filter(
            and_(dbBaseGroup.subject == "lexical_entries_and_entities", dbBaseGroup.action == "create")).first()
        editable_perspectives = DBSession.query(dbPerspective).join(dbGroup, and_(
            dbPerspective.client_id == dbGroup.subject_client_id,
            dbPerspective.object_id == dbGroup.subject_object_id)).join(dbGroup.users).filter(
            and_(dbUser.id == user_id,
                 dbGroup.base_group_id == editor_basegroup.id,
                 dbPerspective.marked_for_deletion == False)).all()
        edit = list()
        for dbperspective in editable_perspectives:
            perspective = Perspective(id=[dbperspective.client_id, dbperspective.object_id])
            perspective.dbObject = dbperspective
            perspective.list_name='edit'
            edit.append(perspective)

        reader_basegroup = DBSession.query(dbBaseGroup).filter(
            and_(dbBaseGroup.subject == "approve_entities", dbBaseGroup.action == "view")).first()
        readable_perspectives = DBSession.query(dbPerspective).join(dbGroup, and_(
            dbPerspective.client_id == dbGroup.subject_client_id,
            dbPerspective.object_id == dbGroup.subject_object_id)).join(dbGroup.users).filter(
            and_(dbUser.id == user_id, dbGroup.base_group_id == reader_basegroup.id)).all()

        view = list()
        for dbperspective in readable_perspectives:
            if [dbperspective.client_id, dbperspective.object_id] not in existing:
                perspective = Perspective(id=[dbperspective.client_id, dbperspective.object_id])
                perspective.dbObject = dbperspective
                perspective.list_name='view'
                view.append(perspective)

        publisher_basegroup = DBSession.query(dbBaseGroup).filter(
            and_(dbBaseGroup.subject == "approve_entities", dbBaseGroup.action == "create")).first()

        approvable_perspectives = DBSession.query(dbPerspective).join(dbGroup, and_(
            dbPerspective.client_id == dbGroup.subject_client_id,
            dbPerspective.object_id == dbGroup.subject_object_id)).join(dbGroup.users).filter(
            and_(dbUser.id == user_id, dbGroup.base_group_id == publisher_basegroup.id)).all()
        publish = list()
        for dbperspective in approvable_perspectives:
            perspective = Perspective(id=[dbperspective.client_id, dbperspective.object_id])
            perspective.dbObject = dbperspective
            perspective.list_name='publish'
            publish.append(perspective)
        return Permissions(limited=limited, view=view, edit=edit, publish=publish)


    def resolve_advanced_search(
        self,
        info,
        search_strings,
        languages = None,
        dicts_to_filter = None,
        tag_list = None,
        category = None,
        adopted = None,
        etymology = None,
        diacritics = None,
        search_metadata = None,
        mode = 'published',
        simple = True,
        xlsx_export = False,
        cognates_flag = True,
        load_entities = True,
        debug_flag = False):

        if mode == 'all':
            publish = None
            accept = True
        elif mode == 'published':
            publish = True
            accept = True
        elif mode == 'not_accepted':
            publish = None
            accept = False
        elif mode == 'deleted':
            publish = None
            accept = None
        elif mode == 'all_with_deleted':
            publish = None
            accept = None
        else:
            return ResponseError('mode: <all|published|not_accepted>')

        if simple:

            return (

                AdvancedSearchSimple().constructor(
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
                    xlsx_export,
                    cognates_flag,
                    debug_flag))

        return (

            AdvancedSearch().constructor(
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
                xlsx_export,
                cognates_flag,
                load_entities,
                debug_flag))

    def resolve_advanced_search_simple(
        self,
        info,
        search_strings,
        languages = None,
        dicts_to_filter = None,
        tag_list = None,
        category = None,
        adopted = None,
        etymology = None,
        diacritics = None,
        mode = 'published'):

        if mode == 'all':
            publish = None
            accept = True
        elif mode == 'published':
            publish = True
            accept = True
        elif mode == 'not_accepted':
            publish = None
            accept = False
        elif mode == 'deleted':
            publish = None
            accept = None
        elif mode == 'all_with_deleted':
            publish = None
            accept = None
        else:
            return ResponseError(message="mode: <all|published|not_accepted>")

        return (

            AdvancedSearchSimple().constructor(
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
                accept))

    def resolve_template_modes(self, info):
        return ['corpora']

    def resolve_template_fields(self, info, mode=None):
        response = list()
        request = info.context.request
        if mode == 'corpora':
            data_type_query = DBSession.query(dbField) \
                .join(dbTranslationGist,
                      and_(dbField.translation_gist_object_id == dbTranslationGist.object_id,
                           dbField.translation_gist_client_id == dbTranslationGist.client_id)) \
                .join(dbTranslationGist.translationatom)
            sound_field = data_type_query.filter(dbTranslationAtom.locale_id == ENGLISH_LOCALE,
                                                 dbTranslationAtom.content == 'Sound').one()  # todo: a way to find this fields if wwe cannot use one
            markup_field = data_type_query.filter(dbTranslationAtom.locale_id == ENGLISH_LOCALE,
                                                  dbTranslationAtom.content == 'Markup').one()
            comment_field = data_type_query.filter(dbTranslationAtom.locale_id == ENGLISH_LOCALE,
                                                    dbTranslationAtom.content == 'Comment').one()
            sound_field =view_field_from_object(request=request, field=sound_field)
            markup_field = view_field_from_object(request=request, field=markup_field)
            comment_field = view_field_from_object(request=request, field=comment_field)
            fake_id_1 = '6f355d7a-e68d-44ab-9cf6-36f78e8f1b34'  # chosen by fair dice roll
            fake_id_2 = '51fbe0b6-2cea-4d40-a994-f6bb6f501d48'  # guaranteed to be random
            f = Field(id=[sound_field["client_id"], sound_field["object_id"]], fake_id = fake_id_1)
            f2 = Field(id=[markup_field["client_id"], markup_field["object_id"]], fake_id = fake_id_2, self_fake_id = fake_id_1)
            f.dbObject = DBSession.query(dbField).filter_by(client_id=sound_field["client_id"], object_id=sound_field["object_id"]).first()
            f2.dbObject = DBSession.query(dbField).filter_by(client_id=markup_field["client_id"], object_id=markup_field["object_id"]).first()
            response.append(f)
            response.append(f2)

            f3 = Field(id=[comment_field["client_id"], comment_field["object_id"]])
            f3.dbObject = DBSession.query(dbField).filter_by(client_id=comment_field["client_id"]).first()
            response.append(f3)
            # response[0]['contains'] = [view_field_from_object(request=request, field=markup_field)]
            # response.append(view_field_from_object(request=request, field=markup_field))
            # response.append(view_field_from_object(request=request, field=comment_field))
            #
            # return response
            #
            # response.append(TranslationGist(id=[sound_field.translation_gist_client_id, sound_field.data_type_translation_gist_object_id]))
            # response.append(TranslationGist(id=[markup_field.data_type_translation_gist_client_id, markup_field.data_type_translation_gist_object_id]))
            # response.append(TranslationGist(id=[comment_field.data_type_translation_gist_client_id, comment_field.data_type_translation_gist_object_id]))
            return response
        else:
            raise ResponseError(message='no such mode')

    def resolve_all_statuses(self, info):
        gql_statuses = list()
        for status in ['WiP', 'Published', 'Limited access', 'Hidden']:
            db_tr_gist = translation_gist_search(status)
            gql_tr_gist = TranslationGist(id=[db_tr_gist.client_id, db_tr_gist.object_id ])
            gql_tr_gist.dbObject = db_tr_gist
            gql_statuses.append(gql_tr_gist)
        return gql_statuses

    def resolve_all_fields(self, info, common=False, parallel=False):
        fields = DBSession.query(dbField).filter_by(marked_for_deletion=False).all()
        if common:
            field_to_psersp_dict = collections.defaultdict(list)
            p_to_field = DBSession.query(dbColumn.parent_client_id,
                                         dbColumn.parent_object_id,
                                         dbColumn.field_client_id,
                                         dbColumn.field_object_id).filter(
                dbColumn.marked_for_deletion == False
            ).all()

            for perspective_client_id, perspective_object_id, field_client_id, field_object_id in p_to_field:
                field_to_psersp_dict[(field_client_id, field_object_id)].append((perspective_client_id,
                                                                                 perspective_object_id))
        gql_fields = list()
        for db_field in fields:
            if common:
                if len(field_to_psersp_dict[(db_field.client_id, db_field.object_id)]) <= 3:
                    continue
                if db_field.data_type != "Text":
                    continue
            if parallel and not (db_field.additional_metadata and db_field.additional_metadata.get('parallel')):
                continue
            gql_field = Field(id=[db_field.client_id, db_field.object_id])
            gql_field.dbObject = db_field
            gql_fields.append(gql_field)

        return gql_fields




    def resolve_all_data_types(self, info):
        response = list()
        for data_type in ['Text', 'Image', 'Sound', 'Markup', 'Link', 'Grouping Tag']:
            db_tr_gist = translation_gist_search(data_type)
            gql_tr_gist = TranslationGist(id=[db_tr_gist.client_id, db_tr_gist.object_id ])
            gql_tr_gist.dbObject = db_tr_gist
            response.append(gql_tr_gist)
        return response

    def resolve_dictionaries(self, info, published=None, mode=None, category=None, proxy=False):
        """
        example:

        query DictionaryList {
            dictionaries(published: true) {
                id
                translation
                parent_id
                translation_gist_id
                state_translation_gist_id
                category
                domain
            }
        }
        """
        request = info.context.request
        if proxy:
            try_proxy(request)

        client_id = info.context.client_id
        client = DBSession.query(Client).filter_by(id=client_id).first()

        dbdicts = (

            DBSession
                .query(dbDictionary)
                .filter_by(marked_for_deletion = False))

        published_cte_query = (
            get_published_translation_gist_id_cte_query())

        if published:

            dbdicts = (

                dbdicts

                    .filter(
                        tuple_(
                            dbDictionary.state_translation_gist_client_id,
                            dbDictionary.state_translation_gist_object_id)

                            .in_(published_cte_query))

                    .join(dbPerspective)

                    .filter(
                        dbPerspective.marked_for_deletion == False,

                        tuple_(
                            dbPerspective.state_translation_gist_client_id,
                            dbPerspective.state_translation_gist_object_id)

                            .in_(published_cte_query))

                    .group_by(dbDictionary))

        if category is not None:
            dbdicts = dbdicts.filter(dbDictionary.category == category)
        dbdicts = dbdicts.order_by(dbDictionary.created_at.desc())
        if mode is not None and client:
            user = DBSession.query(dbUser).filter_by(id=client.user_id).first()

            if not mode:
                # my dictionaries

                client_query = (

                    DBSession
                        .query(Client.id)
                        .filter(Client.user_id == user.id)
                        .subquery()) # user,id?

                dbdicts = dbdicts.filter(dbDictionary.client_id.in_(client_query))

            else:
                # available dictionaries

                dictstemp_set = set()
                group_tuples = []
                isadmin = False
                for group in user.groups: # todo: LOOK AT ME this is really bad. rewrite me from group point of view
                    subject_id = (group.subject_client_id, group.subject_object_id)
                    if group.parent.dictionary_default:
                        if group.subject_override:
                            isadmin = True
                            break
                        dictstemp_set.add(subject_id)
                    if group.parent.perspective_default:
                        if group.subject_override:
                            isadmin = True
                            break
                    group_tuples.append(subject_id)

                if not isadmin:

                    for i in range(0, len(group_tuples), 1000):

                        dictstemp_set.update(

                            DBSession
                                .query(dbDictionary.client_id, dbDictionary.object_id)
                                .join(dbPerspective)
                                .filter(
                                    tuple_(
                                        dbPerspective.client_id,
                                        dbPerspective.object_id)
                                        .in_(group_tuples[i : i + 1000]))
                                .all())

                    dbdicts = [o for o in dbdicts if (o.client_id, o.object_id) in dictstemp_set]

        dictionaries_list = list()
        for dbdict in dbdicts:
            gql_dict = Dictionary(id=[dbdict.client_id, dbdict.object_id])
            gql_dict.dbObject = dbdict
            dictionaries_list.append(gql_dict)
        return dictionaries_list

    def resolve_dictionary(self, info, id):
        return Dictionary(id=id)

    def resolve_perspectives(
        self,
        info,
        published = None,
        with_phonology_data = None,
        with_valency_data = None):
        """
        example:

        query LanguagesList {
            perspectives(published: true) {
                id
                translation
                parent_id
                translation_gist_id
                state_translation_gist_id
                import_source
                import_hash
            }
        }
        """

        perspective_query = (

            DBSession

                .query(
                    dbPerspective)

                .filter(
                    dbPerspective.marked_for_deletion == False))

        if published:

            db_published_gist = translation_gist_search('Published')
            state_translation_gist_client_id = db_published_gist.client_id
            state_translation_gist_object_id = db_published_gist.object_id
            db_la_gist = translation_gist_search('Limited access')
            limited_client_id, limited_object_id = db_la_gist.client_id, db_la_gist.object_id

            perspective_query = perspective_query.filter(
                or_(and_(dbPerspective.state_translation_gist_object_id == state_translation_gist_object_id,
                         dbPerspective.state_translation_gist_client_id == state_translation_gist_client_id),
                    and_(dbPerspective.state_translation_gist_object_id == limited_object_id,
                         dbPerspective.state_translation_gist_client_id == limited_client_id)))

        # If required, filtering out pespectives without phonology data.
        #
        # Experiments have shown that filtering is faster through id in select group by than through exists
        # subquery. For previous exists-based filterting see the file's history.

        if with_phonology_data is not None:

            dbMarkup = aliased(dbEntity, name = 'Markup')
            dbSound = aliased(dbEntity, name = 'Sound')

            dbPublishingMarkup = aliased(dbPublishingEntity, name = 'PublishingMarkup')
            dbPublishingSound = aliased(dbPublishingEntity, name = 'PublishingSound')

            phonology_query = (

                DBSession

                    .query(
                        dbLexicalEntry.parent_client_id,
                        dbLexicalEntry.parent_object_id)

                    .filter(
                        dbLexicalEntry.marked_for_deletion == False,
                        dbMarkup.parent_client_id == dbLexicalEntry.client_id,
                        dbMarkup.parent_object_id == dbLexicalEntry.object_id,
                        dbMarkup.marked_for_deletion == False,
                        dbMarkup.additional_metadata.contains({'data_type': 'praat markup'}),
                        dbPublishingMarkup.client_id == dbMarkup.client_id,
                        dbPublishingMarkup.object_id == dbMarkup.object_id,
                        dbPublishingMarkup.published == True,
                        dbPublishingMarkup.accepted == True,
                        dbSound.client_id == dbMarkup.self_client_id,
                        dbSound.object_id == dbMarkup.self_object_id,
                        dbSound.marked_for_deletion == False,
                        dbPublishingSound.client_id == dbSound.client_id,
                        dbPublishingSound.object_id == dbSound.object_id,
                        dbPublishingSound.published == True,
                        dbPublishingSound.accepted == True)

                    .group_by(
                        dbLexicalEntry.parent_client_id,
                        dbLexicalEntry.parent_object_id))

            id_tuple = (

                tuple_(
                    dbPerspective.client_id,
                    dbPerspective.object_id))

            perspective_query = (

                perspective_query.filter(

                    (id_tuple.in_ if with_phonology_data else
                        id_tuple.notin_)(

                        DBSession.query(
                            phonology_query.cte()))))

        # If required, filtering out perspectives without valency data.
        #
        # NOTE: We explicitly need a union, if we try to use an or condition, due to something or other in
        # PostgreSQL's planner query execution time jumps from 2 to 180 seconds, like what?

        if with_valency_data is not None:

            parser_result_query = (

                DBSession

                    .query(
                        dbLexicalEntry.parent_client_id,
                        dbLexicalEntry.parent_object_id)

                    .filter(
                        dbLexicalEntry.marked_for_deletion == False,
                        dbEntity.parent_client_id == dbLexicalEntry.client_id,
                        dbEntity.parent_object_id == dbLexicalEntry.object_id,
                        dbEntity.marked_for_deletion == False,
                        dbEntity.content.op('~*')('.*\.(doc|docx|odt)'),
                        dbPublishingEntity.client_id == dbEntity.client_id,
                        dbPublishingEntity.object_id == dbEntity.object_id,
                        dbPublishingEntity.published == True,
                        dbPublishingEntity.accepted == True,
                        dbParserResult.entity_client_id == dbEntity.client_id,
                        dbParserResult.entity_object_id == dbEntity.object_id,
                        dbParserResult.marked_for_deletion == False)

                    .group_by(
                        dbLexicalEntry.parent_client_id,
                        dbLexicalEntry.parent_object_id))

            eaf_corpus_query = (

                DBSession

                    .query(
                        dbLexicalEntry.parent_client_id,
                        dbLexicalEntry.parent_object_id)

                    .filter(
                        dbLexicalEntry.marked_for_deletion == False,
                        dbEntity.parent_client_id == dbLexicalEntry.client_id,
                        dbEntity.parent_object_id == dbLexicalEntry.object_id,
                        dbEntity.marked_for_deletion == False,
                        dbEntity.content.ilike('%.eaf'),
                        dbEntity.additional_metadata.contains({'data_type': 'elan markup'}),
                        dbPublishingEntity.client_id == dbEntity.client_id,
                        dbPublishingEntity.object_id == dbEntity.object_id,
                        dbPublishingEntity.published == True,
                        dbPublishingEntity.accepted == True)

                    .group_by(
                        dbLexicalEntry.parent_client_id,
                        dbLexicalEntry.parent_object_id))

            union_list = [
                DBSession.query(parser_result_query.cte()),
                DBSession.query(eaf_corpus_query.cte()),
            ]

            id_tuple = (

                tuple_(
                    dbPerspective.client_id,
                    dbPerspective.object_id))

            perspective_query = (

                perspective_query.filter(

                    (id_tuple.in_ if with_valency_data else
                        id_tuple.notin_)(

                        union(*union_list))))

        log.debug(
            '\nperspective_query:\n' +
            render_statement(perspective_query.statement))

        perspectives_list = []

        for db_persp in perspective_query.all():

            gql_persp = Perspective(id=[db_persp.client_id, db_persp.object_id])
            gql_persp.dbObject = db_persp
            perspectives_list.append(gql_persp)

        return perspectives_list


    def resolve_perspective(self, info, id):
        return Perspective(id=id)

    def resolve_language(self, info, id):
        return Language(id=id)

    def resolve_languages(
        self,
        info,
        id_list = None,
        only_in_toc = False,
        only_with_dictionaries_recursive = False,
        dictionary_category = None,
        dictionary_published = None,
        in_tree_order = False):
        """
        example:

        query LanguagesList {
            languages {
                id
                translation
                parent_id
                translation_gist_id
            }
        }
        """

        try:

            __debug_flag__ = False

            resolver = (

                Language_Resolver(

                    info,
                    info.field_asts,

                    language_resolver_args(
                        id_list = id_list,
                        only_in_toc = only_in_toc,
                        only_with_dictionaries_recursive = only_with_dictionaries_recursive,
                        dictionary_category = dictionary_category,
                        dictionary_published = dictionary_published,
                        in_tree_order = in_tree_order),

                    debug_flag = __debug_flag__))

            return (
                resolver.run())

        except Exception as exception:

            traceback_string = (

                ''.join(
                    traceback.format_exception(
                        exception, exception, exception.__traceback__))[:-1])

            log.warning('languages: exception')
            log.warning(traceback_string)

            return (
                ResponseError(
                    'Exception:\n' + traceback_string))

    def resolve_entity(self, info, id):
        return Entity(id=id)

    def resolve_user(self, info, id=None):
        if id is None:
            client_id = info.context.client_id
            client = DBSession.query(Client).filter_by(id=client_id).first()
            if not client:
                return None
            id = client.user_id
        return User(id=id)

    def resolve_is_authenticated(self, info):
        client_id = info.context.client_id
        client = DBSession.query(Client).filter_by(id=client_id).first()
        if not client:
            return False
        return True

    def resolve_users(self, info, search=None):
        """
        example:

        query UsersList {
            users(search: "modis") {
                login
                name
                intl_name
            }
        }
        """
        users = DBSession.query(dbUser).join(dbUser.email)
        if search:
            name = search + '%'
            users = users.filter(or_(
                dbUser.name.startswith(name),
                dbUser.login.startswith(name),
                dbUser.intl_name.startswith(name),
                dbEmail.email.startswith(name)
            ))
        users_list = list()
        for db_user in users:
            gql_user = User(name=db_user.name)
            gql_user.dbObject = db_user
            users_list.append(gql_user)

        return users_list


    # def resolve_datetime(self, args, context, info):
    #     id = args.get('id')
    #     return DateTime(id=id)

    def resolve_basegroup(self, info, id):
        return BaseGroup(id=id)

    def resolve_client(self, info):
        context = info.context
        return context.get('client')

    def resolve_column(self, info, id):
        return Column(id=id)

    def resolve_group(self, info, id):
        return Group(id=id)

    def resolve_organization(self, info, id):
        return Organization(id=id)

    def resolve_organizations(
        self,
        info,
        has_participant = None,
        participant_deleted = None,
        participant_published = None,
        participant_category = None):

        __debug_flag__ = False

        # Analyzing query.

        object_flag = False

        selection_dict = {

            'additional_metadata': (
                dbOrganization.additional_metadata,),

            'created_at': (
                dbOrganization.created_at,),

            'id': (
                dbOrganization.id,),

            'marked_for_deletion': (
                dbOrganization.marked_for_deletion,),

            'about_translations': (
                dbOrganization.about_translation_gist_client_id,
                dbOrganization.about_translation_gist_object_id),

            'translations': (
                dbOrganization.translation_gist_client_id,
                dbOrganization.translation_gist_object_id)}

        selection_set = set()
        selection_list = []

        for field in info.field_asts:

            if field.name.value != 'organizations':
                continue

            for subfield in field.selection_set.selections:

                name_str = subfield.name.value

                if name_str in selection_dict:

                    selection_set.add(name_str)
                    selection_list.extend(selection_dict[name_str])

                elif name_str != '__typename':

                    object_flag = True

        if object_flag:

            selection_list = [dbOrganization]

        elif 'id' not in selection_set:

            # For standard organization ordering.

            selection_set.add('id')
            selection_list.extend(selection_dict['id'])

        # If we are going to query both the usual and about translations, we'll have to use separate joins
        # to a CTE-based queries to avoid joining to actually a cross product of two translations.

        translations_flag = 'translations' in selection_set
        about_translations_flag = 'about_translations' in selection_set

        cte_flag = (
            translations_flag and
            about_translations_flag)

        organization_query = None
        organization_cte = None

        # No participant filtering, getting every organization.

        if has_participant is None:

            organization_query = (

                DBSession

                    .query(
                        *selection_list)

                    .filter_by(
                        marked_for_deletion = False))

            organization_c = dbOrganization

        # Simple participant count filter.

        elif (
            participant_deleted is None and
            participant_category is None and
            participant_published is None):

            participant_count = (

                func.jsonb_array_length(
                    dbOrganization.additional_metadata['participant']))

            organization_query = (

                DBSession

                    .query(
                        *selection_list)

                    .filter(
                        dbOrganization.marked_for_deletion == False,

                        participant_count > 0 if has_participant else
                        participant_count <= 0))

            organization_c = dbOrganization

        # Additional conditions on participants, we'll have to check them through a join.
        #
        # We have to use raw SQL due to SQLAlchemy being bad with PostgreSQL's jsonb_to_recordset.

        else:

            dictionary_condition_list = []

            if participant_deleted is not None:

                dictionary_condition_list.append(
                    '\n and D.marked_for_deletion = true' if participant_deleted else
                    '\n and D.marked_for_deletion = false')

            if participant_category is not None:

                dictionary_condition_list.append(
                    f'\n and D.category = {participant_category}')

            if participant_published is not None:

                if not participant_published:
                    raise NotImplementedError

                dictionary_condition_list.append('''

                    and (
                      D.state_translation_gist_client_id,
                      D.state_translation_gist_object_id) in (

                      select
                        T.client_id,
                        T.object_id

                      from
                        translationgist T,
                        translationatom A

                      where
                        T.marked_for_deletion = false and
                        T.type = 'Service' and
                        A.parent_client_id = T.client_id and
                        A.parent_object_id = T.object_id and
                        A.locale_id = 2 and
                        A.marked_for_deletion = false and (
                          A.content = 'Published' or
                          A.content = 'Limited access'))

                    ''')

            dictionary_condition_str = (
                ''.join(dictionary_condition_list))

            if object_flag:

                selection_str = 'O.*'

            else:

                selection_str = (

                    ', '.join(
                        f'O.{selection.name}'
                        for selection in selection_list))

            sql_text = (

                sqlalchemy.text(f'''

                    select
                      {selection_str}

                    from
                      organization O

                    cross join
                      jsonb_to_recordset(O.additional_metadata -> 'participant')
                        P (client_id bigint, object_id bigint)

                    join
                      dictionary D

                    on
                      D.client_id = P.client_id and
                      D.object_id = P.object_id {dictionary_condition_str}

                    where
                      O.marked_for_deletion = false

                    group by
                      O.id

                    '''))

            if object_flag:

                sql_text = (

                    aliased(
                        dbOrganization,

                        sql_text

                            .columns(
                                **{column.name: column.type
                                    for column in dbOrganization.__table__.c})

                            .alias(),

                        adapt_on_names = True))

                organization_c = (
                    sql_text)

            else:

                sql_text = (

                    sql_text

                        .columns(
                            *selection_list)

                        .alias())

                organization_c = (
                    sql_text.c)

            organization_query = (
                DBSession.query(sql_text))

            if not has_participant:
                raise NotImplementedError

        # Establishing a CTE if we'll need it.

        if cte_flag:

            organization_cte = organization_query.cte()
            organization_c = organization_cte.c

            if object_flag:

                organization_cte = (

                    aliased(
                        dbOrganization,
                        organization_cte,
                        adapt_on_names = True))

                organization_c = organization_cte

            organization_query = (
                DBSession.query(organization_cte))

        # Getting translations through a join, if required.

        if translations_flag:

            if organization_cte is not None:

                translation_query = (

                    DBSession

                        .query(
                            organization_c.id)

                        .outerjoin(
                            dbTranslationAtom,

                            and_(
                                dbTranslationAtom.parent_client_id == organization_c.translation_gist_client_id,
                                dbTranslationAtom.parent_object_id == organization_c.translation_gist_object_id,
                                dbTranslationAtom.marked_for_deletion == False))

                        .add_columns(

                            func.jsonb_object_agg(
                                dbTranslationAtom.locale_id,
                                dbTranslationAtom.content)

                                .filter(dbTranslationAtom.locale_id != None)
                                .label('translations'))

                        .group_by(
                            organization_c.id)

                        .subquery())

                organization_query = (

                    organization_query

                        .join(
                            translation_query,
                            translation_query.c.id == organization_c.id)

                        .add_columns(
                            translation_query.c.translations))

            else:

                organization_query = (

                    organization_query

                        .outerjoin(
                            dbTranslationAtom,

                            and_(
                                dbTranslationAtom.parent_client_id == organization_c.translation_gist_client_id,
                                dbTranslationAtom.parent_object_id == organization_c.translation_gist_object_id,
                                dbTranslationAtom.marked_for_deletion == False))

                        .add_columns(

                            func.jsonb_object_agg(
                                dbTranslationAtom.locale_id,
                                dbTranslationAtom.content)

                                .filter(dbTranslationAtom.locale_id != None)
                                .label('translations'))

                        .group_by(
                            *([organization_c] if object_flag else organization_c)))

        # Getting about translations through a join, if required.

        if about_translations_flag:

            if organization_cte is not None:

                about_translation_query = (

                    DBSession

                        .query(
                            organization_c.id)

                        .outerjoin(
                            dbTranslationAtom,

                            and_(
                                dbTranslationAtom.parent_client_id == organization_c.about_translation_gist_client_id,
                                dbTranslationAtom.parent_object_id == organization_c.about_translation_gist_object_id,
                                dbTranslationAtom.marked_for_deletion == False))

                        .add_columns(

                            func.jsonb_object_agg(
                                dbTranslationAtom.locale_id,
                                dbTranslationAtom.content)

                                .filter(dbTranslationAtom.locale_id != None)
                                .label('about_translations'))

                        .group_by(
                            organization_c.id)

                        .subquery())

                organization_query = (

                    organization_query

                        .join(
                            about_translation_query,
                            about_translation_query.c.id == organization_c.id)

                        .add_columns(
                            about_translation_query.c.about_translations))

            else:

                organization_query = (

                    organization_query

                        .outerjoin(
                            dbTranslationAtom,

                            and_(
                                dbTranslationAtom.parent_client_id == organization_c.about_translation_gist_client_id,
                                dbTranslationAtom.parent_object_id == organization_c.about_translation_gist_object_id,
                                dbTranslationAtom.marked_for_deletion == False))

                        .add_columns(

                            func.jsonb_object_agg(
                                dbTranslationAtom.locale_id,
                                dbTranslationAtom.content)

                                .filter(dbTranslationAtom.locale_id != None)
                                .label('about_translations'))

                        .group_by(
                            *([organization_c] if object_flag else organization_c)))

        # Getting organization info, preparing it for GraphQL.

        organization_query = (

            organization_query
                .order_by(organization_c.id))

        result_list = organization_query.all()

        if __debug_flag__:

            log.debug(
                '\n organization_query:\n ' +
                render_statement(organization_query.statement))

        gql_organization_list = []

        if object_flag:

            # We are getting full ORM dbOrganization objects.

            attribute_set = selection_set.copy()

            attribute_set.discard('translations')
            attribute_set.discard('about_translations')

            for result in result_list:

                organization = (
                    result[0] if translations_flag or about_translations_flag else
                    result)

                gql_organization = (
                    Organization(id = organization.id))

                gql_organization.dbObject = organization

                if translations_flag:

                    translations = (
                        result.translations)

                    gql_organization.translations = (
                        translations if translations is not None else gql_none_value)

                if about_translations_flag:

                    translations = (
                        result.about_translations)

                    gql_organization.about_translations = (
                        translations if translations is not None else gql_none_value)

                for attribute in attribute_set:

                    value = getattr(organization, attribute)

                    if attribute == 'additional_metadata':

                        value = AdditionalMetadata.from_object(value)

                    elif attribute == 'created_at':

                        value = CreatedAt.from_timestamp(value)

                    setattr(
                        gql_organization,
                        attribute,
                        value)

                gql_organization_list.append(gql_organization)

        else:

            # We are getting attribute values as they are.

            for result in result_list:

                gql_organization = (
                    Organization(id = result.id))

                for selection in selection_set:

                    value = getattr(result, selection)

                    if selection == 'additional_metadata':

                        value = AdditionalMetadata.from_object(value)

                    elif selection == 'created_at':

                        value = CreatedAt.from_timestamp(value)

                    setattr(
                        gql_organization,
                        selection,
                        value if value is not None else gql_none_value)

                gql_organization_list.append(gql_organization)

        return gql_organization_list

    # def resolve_passhash(self, args, context, info):
    #     id = args.get('id')
    #     return Passhash(id=id)

    # def resolve_objecttoc(self, args, context, info):
    #     id = args.get('id')
    #     return ObjectTOC(id=id)


    def resolve_translationatom(self, info, id):
        return TranslationAtom(id=id)

    def resolve_translationgist(self, info, id):
        return TranslationGist(id=id)

    def resolve_translationgists(self, info, gists_type=None):
        """
        example:
        query GistsList {
            translationgists {
                id
                type
            }
        }
        """

        gists_query = DBSession.query(dbTranslationGist).filter_by(marked_for_deletion=False)
        if gists_type:
            gists_query = gists_query.filter(dbTranslationGist.type == gists_type)
        else:
            gists_query = gists_query.order_by(dbTranslationGist.type)
        gists = gists_query.all()
        gists_list = list()
        for db_gist in gists:
            gql_gist = TranslationGist(id=[db_gist.client_id, db_gist.object_id])
            gql_gist.dbObject = db_gist
            gists_list.append(gql_gist)

        return gists_list

    def resolve_translation_search(
        self,
        info,
        searchstring = None,
        search_case_insensitive = False,
        search_regular_expression = False,
        translation_type = None,
        deleted = None,
        order_by_type = False,
        no_result_error_flag = True):
        """
        query TranslationsList {
            translation_search(searchstring: "словарь") {
                id
                type
                translationatoms {
                     id
                     content
                }
            }
        }
        """

        # Analyzing query.

        atoms_flag = False
        atoms_deleted = None

        def f(argument):

            try:

                return argument.value.value

            except AttributeError:

                return (
                    info.variable_values.get(
                        argument.value.name.value, None))

        for field in info.field_asts:

            if field.name.value != 'translation_search':
                continue

            for subfield in field.selection_set.selections:

                if subfield.name.value != 'translationatoms':
                    continue

                atoms_flag = True

                for argument in subfield.arguments:

                    if argument.name.value != 'deleted':

                        atoms_flag = False
                        break

                    atoms_deleted = f(argument)

        # Getting ready to get gists.

        gist_query = (

            DBSession.query(
                dbTranslationGist))

        if deleted is not None:

            gist_query = (

                gist_query.filter(
                    dbTranslationGist.marked_for_deletion == deleted))

        if searchstring:

            if search_regular_expression:

                search_filter = (

                    dbTranslationAtom.content.op('~*') if search_case_insensitive else
                    dbTranslationAtom.content.op('~'))(

                        searchstring)

            else:

                search_filter = (

                    dbTranslationAtom.content.ilike if search_case_insensitive else
                    dbTranslationAtom.content.like)(

                        '%' + searchstring + '%')

            gist_id_query = (

                DBSession

                    .query(
                        dbTranslationAtom.parent_client_id,
                        dbTranslationAtom.parent_object_id)

                    .filter(search_filter))

            gist_query = (

                gist_query.filter(

                    tuple_(
                        dbTranslationGist.client_id,
                        dbTranslationGist.object_id)

                        .in_(gist_id_query)))

        if translation_type:

            gist_query = (

                gist_query
                    .filter(dbTranslationGist.type == translation_type))

        # If we need to get atoms, we'll use the gist query as subquery before we add ordering to it.

        if atoms_flag:

            gist_subquery = (
                gist_query.subquery())

            atom_query = (

                DBSession

                    .query(
                        dbTranslationAtom)

                    .filter(

                        tuple_(
                            dbTranslationAtom.parent_client_id,
                            dbTranslationAtom.parent_object_id)

                            .in_(
                                DBSession.query(
                                    gist_subquery.c.client_id,
                                    gist_subquery.c.object_id))))

            if atoms_deleted is not None:

                atom_query = (

                    atom_query.filter(
                        dbTranslationAtom.marked_for_deletion == atoms_deleted))

        if order_by_type and not translation_type:

            gist_query = (

                gist_query
                    .order_by(dbTranslationGist.type))

        try:

            gist_list = gist_query.all()

        except sqlalchemy.exc.DataError as data_error:

            if isinstance(data_error.orig, psycopg2.errors.InvalidRegularExpression):
                return ResponseError('InvalidRegularExpression')

            raise

        if gist_list or not no_result_error_flag:

            gql_gist_list = []

            if atoms_flag:
                gql_gist_dict = {}

            for gist in gist_list:

                id = gist.id

                gql_gist = TranslationGist(id = id)
                gql_gist.dbObject = gist

                gql_gist_list.append(gql_gist)

                if atoms_flag:

                    gql_gist.translationatoms = []
                    gql_gist_dict[id] = gql_gist

            # Getting atoms info if required.

            if atoms_flag:

                for atom in atom_query.all():

                    gql_atom = TranslationAtom(id = atom.id)
                    gql_atom.dbObject = atom

                    gql_gist_dict[atom.parent_id].translationatoms.append(gql_atom)

            return gql_gist_list

        raise ResponseError(message="Error: no result")

    def resolve_translation_service_search(self, info, searchstring):
        """
        query TranslationsList {
            translation_service_search(searchstring: "Converting 80%") {
                id
                type
                translationatoms {
                     id
                     content
                }
            }
        }
        """
        db_translationgist = translation_gist_search(searchstring)
        if not db_translationgist:
            raise ResponseError(message="Error: no result")
        # translationatoms_list = list()
        # for db_translationatom in db_translationgist.translationatom:
        #     translationatom_object = TranslationAtom(id=[db_translationatom.client_id, db_translationatom.object_id])
        #     translationatom_object.dbObject = db_translationatom
        #     translationatoms_list.append(translationatom_object)
        gql_translationgist = TranslationGist(id=[db_translationgist.client_id, db_translationgist.object_id])
        gql_translationgist.dbObject = db_translationgist
        return gql_translationgist


    def resolve_advanced_translation_search(self, info, searchstrings):
        """
        query TranslationsList {
            advanced_translation_search(searchstrings: ["Converting 80%", "Available dictionaries"]) {
                id
                type
                translationatoms {
                     id
                     content
                }
            }
        }
        """

        debug_flag = False

        if not searchstrings:
            raise ResponseError(message = "Error: no search strings")

        search_table_name = (

            'search_table_' +
            str(uuid.uuid4()).replace('-', '_'))

        DBSession.execute(f'''

            create temporary table

            {search_table_name} (
              index INT PRIMARY KEY,
              search_string TEXT NOT NULL)

            on commit drop;

            ''')

        class tmpSearchString(models.Base):

            __tablename__ = search_table_name

            index = (
                sqlalchemy.Column(sqlalchemy.types.Integer, primary_key = True))

            search_string = (
                sqlalchemy.Column(sqlalchemy.types.UnicodeText, nullable = False))

        insert_list = [
            {'index': index, 'search_string': search_string}
            for index, search_string in enumerate(searchstrings)]

        insert_query = (

            tmpSearchString.__table__
                .insert()
                .values(insert_list))

        if debug_flag:

            log.debug(
                '\ninsert_query:\n' +
                str(insert_query.compile(compile_kwargs = {'literal_binds': True})))

        DBSession.execute(insert_query)

        gist_query = (

            DBSession

                .query(
                    tmpSearchString.index,
                    dbTranslationGist)

                .filter(
                    dbTranslationGist.type == 'Service',
                    dbTranslationGist.marked_for_deletion == False,
                    dbTranslationAtom.parent_client_id == dbTranslationGist.client_id,
                    dbTranslationAtom.parent_object_id == dbTranslationGist.object_id,
                    dbTranslationAtom.locale_id == ENGLISH_LOCALE,
                    dbTranslationAtom.marked_for_deletion == False,
                    dbTranslationAtom.content == tmpSearchString.search_string)

                .distinct(
                    tmpSearchString.index)

                .order_by(
                    tmpSearchString.index,
                    dbTranslationGist.client_id,
                    dbTranslationGist.object_id))

        if debug_flag:

            log.debug(
                '\ngist_query:\n' +
                str(gist_query.statement.compile(compile_kwargs = {'literal_binds': True})))

        gist_list = gist_query.all()

        result_list = [None] * len(searchstrings)

        for index, gist in gist_list:

            gql_gist = (

                TranslationGist(
                    id = [gist.client_id, gist.object_id],
                    translation = searchstrings[index]))

            gql_gist.dbObject = gist

            result_list[index] = gql_gist

        return result_list

    def resolve_optimized_translation_search(self, info, searchstrings):

        if not searchstrings:
            raise ResponseError(message = "Error: no search strings")

        search_table_name = (

            'search_table_' +
            str(uuid.uuid4()).replace('-', '_'))

        DBSession.execute(f'''

            create temporary table

            {search_table_name} (
              index INT PRIMARY KEY,
              search_string TEXT NOT NULL)

            on commit drop;

            ''')

        insert_sql_str_list = [

            f'insert into {search_table_name} values ',
            '(0, \'{}\')'.format(searchstrings[0].replace('\'', '\'\''))]

        for index, search_string in enumerate(searchstrings[1:], 1):

            insert_sql_str_list.append(
                ', ({}, \'{}\')'.format(index, search_string.replace('\'', '\'\'')))

        insert_sql_str_list.append(';')

        DBSession.execute(
            ''.join(insert_sql_str_list))

        locale_id = (
            info.context.locale_id)

        row_list = (

            DBSession

                .execute(f'''

                    select
                    distinct on (S.index)
                    S.index,
                    A2.content

                    from
                    {search_table_name} S

                    left outer join
                    translationatom A1
                    on
                    A1.content = S.search_string

                    left outer join
                    translationgist G
                    on
                    A1.parent_client_id = G.client_id and
                    A1.parent_object_id = G.object_id

                    left outer join
                    translationatom A2
                    on
                    A2.parent_client_id = G.client_id and
                    A2.parent_object_id = G.object_id

                    where
                    G.type = 'Service' and
                    G.marked_for_deletion = false and
                    A1.locale_id = 2 and
                    A1.marked_for_deletion = false and
                    A2.locale_id = {locale_id} and
                    A2.marked_for_deletion = false

                    order by
                    S.index, G.client_id, G.object_id, A2.client_id, A2.object_id;

                    ''')

                .fetchall())

        result_list = [None] * len(searchstrings)

        for index, result in row_list:
            result_list[index] = result

        return result_list

    def resolve_userblob(self, info, id):
        return UserBlobs(id=id)

    def resolve_field(self, info, id):
        return Field(id=id)

    def resolve_lexicalentry(self, info, id):
        return LexicalEntry(id=id)

    def resolve_all_locales(self, info):
        response = list()
        locales = DBSession.query(dbLocale).all()
        for locale in locales:
            locale_json = dict()
            locale_json['shortcut'] = locale.shortcut
            locale_json['intl_name'] = locale.intl_name
            locale_json['created_at'] = locale.created_at
            locale_json['id'] = locale.id
            response.append(locale_json)
        return response


    def resolve_basic_search(self, info, searchstring, search_in_published, field_id=None, perspective_id=None, can_add_tags=None): #basic_search() function
        """
        query EntriesList {
            basic_search(searchstring: "следить", search_in_published: true) {
                id
                entities {
                     id
                     content
                }
            }
        }

        """

        if (not searchstring or
            len(searchstring) < 1):

            raise ResponseError(message="Bad string")

        field = None
        if field_id:
            field_client_id, field_object_id = field_id[0], field_id[1]
            field = DBSession.query(dbField).filter_by(client_id=field_client_id, object_id=field_object_id).first()

        client_id = info.context.client_id

        group = DBSession.query(dbGroup).filter(dbGroup.subject_override == True).join(dbBaseGroup) \
            .filter(dbBaseGroup.subject == 'lexical_entries_and_entities', dbBaseGroup.action == 'view') \
            .join(dbUser, dbGroup.users).join(Client) \
            .filter(Client.id == client_id).first()

        # See get_hidden() in models.py.

        hidden_id = (

            DBSession

                .query(
                    dbTranslationGist.client_id,
                    dbTranslationGist.object_id)

                .join(dbTranslationAtom)

                .filter(
                    dbTranslationGist.type == 'Service',
                    dbTranslationAtom.content == 'Hidden',
                    dbTranslationAtom.locale_id == ENGLISH_LOCALE)

                .first())

        # NOTE: due to no-op nature of publishing entity checks (see note below), removing joins
        # with PublishingEntity.

        results_cursor = (

            DBSession

                .query(
                    dbLexicalEntry.client_id,
                    dbLexicalEntry.object_id)

                .join(dbEntity)
                .join(dbPerspective)
                .join(dbDictionary)

                .filter(
                    dbEntity.content.like('%' + searchstring + '%'),
                    dbEntity.marked_for_deletion == False,
                    dbLexicalEntry.marked_for_deletion == False,
                    dbPerspective.marked_for_deletion == False,
                    dbDictionary.marked_for_deletion == False,
                    or_(dbDictionary.state_translation_gist_client_id != hidden_id[0],
                        dbDictionary.state_translation_gist_object_id != hidden_id[1]),
                    or_(dbPerspective.state_translation_gist_client_id != hidden_id[0],
                        dbPerspective.state_translation_gist_object_id != hidden_id[1])))

        if perspective_id:

            results_cursor = (

                results_cursor.filter(
                    dbPerspective.client_id == perspective_id[0],
                    dbPerspective.object_id == perspective_id[1]))

        # NOTE:
        #
        # Well, intention is clear, but this does not work and is actually a no-op.
        #
        # To work, should be:
        #
        #   results_cursor = results_cursor.filter(...)
        #
        # So, broken or at least not working as intended for almost 2.5 years.
        #
        # But this GraphQL api is currently used in only a single place, searching words for grouping, so
        # apparently all right, no problem.
        #
        # Commenting out for optimization.

#       if search_in_published is not None:
#           results_cursor.filter(dbPublishingEntity.published == search_in_published)

#       results_cursor.filter(dbPublishingEntity.accepted == True)

        if not group:

            # We do not have a single group giving us all necessary permissions.
            #
            # So, we look in either published perspectives or perspectives we have nesessary permissions
            # for.

            db_published_gist = translation_gist_search('Published')

            published_id = (
                db_published_gist.client_id,
                db_published_gist.object_id)

            group_query = (

                DBSession

                    .query(dbBaseGroup.action)
                    .join(dbGroup)

                    .filter(
                        dbGroup.subject_client_id == dbPerspective.client_id,
                        dbGroup.subject_object_id == dbPerspective.object_id,
                        dbGroup.id == user_to_group_association.c.group_id,
                        user_to_group_association.c.user_id == Client.user_id,
                        Client.id == client_id,
                        dbBaseGroup.subject == 'lexical_entries_and_entities'))

            # Do we need both view and create permissions?

            if can_add_tags:

                group_query = (

                    group_query

                        .filter(or_(
                            dbBaseGroup.action == 'create',
                            dbBaseGroup.action == 'view'))

                        .group_by(dbBaseGroup.action)
                        .subquery())

                group_count_query = (

                    DBSession
                        .query(func.count(group_query.c.action))
                        .as_scalar())

                group_condition = (
                    group_count_query == 2)

            # Only view persmissions.

            else:

                group_query = (
                    group_query.filter(dbBaseGroup.action == 'view'))

                group_condition = group_query.exists()

            results_cursor = (

                results_cursor.filter(

                    or_(

                        and_(
                            dbPerspective.state_translation_gist_client_id == published_id[0],
                            dbPerspective.state_translation_gist_object_id == published_id[1]),

                        group_condition)))

        if field:
            results_cursor = results_cursor.join(dbPerspective.dictionaryperspectivetofield).filter(
                dbColumn.field == field)

        entry_list, _ = (

            entries_with_entities(
                results_cursor.distinct(),
                publish = None if not search_in_published else True,
                accept = True,
                check_perspective = False))

        entity_list = []

        for entry in entry_list:
            if entry.gql_Entities:
                entity_list.extend(entry.gql_Entities)

        return (
            LexicalEntriesAndEntities(
                entities = entity_list,
                lexical_entries = entry_list))

    def resolve_advanced_lexicalentries(self, info, searchstrings, perspectives=None, adopted=None,
                                        adopted_type=None, with_etimology=None): #advanced_search() function

        """
        query EntriesList {
            advanced_lexicalentries(searchstrings: [{searchstring: "смотреть следить"}]) {
                id
                entities {
                     id
                     content
                }
            }
        }

        """

        if not perspectives:
            db_published_gist = translation_gist_search('Published')
            state_translation_gist_client_id = db_published_gist.client_id
            state_translation_gist_object_id = db_published_gist.object_id
            db_la_gist = translation_gist_search('Limited access')
            limited_client_id, limited_object_id = db_la_gist.client_id, db_la_gist.object_id

            perspectives = [(persp.client_id, persp.object_id) for persp in DBSession.query(dbPerspective).filter(
                dbPerspective.marked_for_deletion == False,
                or_(and_(dbPerspective.state_translation_gist_client_id == state_translation_gist_client_id,
                         dbPerspective.state_translation_gist_object_id == state_translation_gist_object_id),
                    and_(dbPerspective.state_translation_gist_client_id == limited_client_id,
                         dbPerspective.state_translation_gist_object_id == limited_object_id))).all()]

        def make_query(searchstring, perspectives):
            results_cursor = DBSession.query(dbLexicalEntry).join(dbEntity.parent) \
                .join(dbEntity.field).join(dbTranslationAtom,
                                         and_(dbField.translation_gist_client_id == dbTranslationAtom.parent_client_id,
                                              dbField.translation_gist_object_id == dbTranslationAtom.parent_object_id,
                                              dbField.marked_for_deletion == False)) \
                .distinct(dbEntity.parent_client_id, dbEntity.parent_object_id)
            if perspectives:
                results_cursor = results_cursor.filter(
                    tuple_(dbLexicalEntry.parent_client_id, dbLexicalEntry.parent_object_id).in_(perspectives))
            if not searchstring["searchstring"]:
                raise ResponseError(message="Error: bad argument 'searchstring'.")
            search_parts = searchstring["searchstring"].split()
            search_expression = dbEntity.content.like('%' + search_parts[0] + '%')
            to_do_or = searchstring.get('search_by_or', True)

            for part in search_parts[1:]:
                search_expression = or_(search_expression, dbEntity.content.like('%' + part + '%'))
            if 'entity_type' in searchstring and searchstring['entity_type']:
                search_expression = and_(search_expression, dbField.client_id == searchstring['entity_type'][0],
                                         dbField.object_id == searchstring['entity_type'][1])

            results_cursor = results_cursor.filter(search_expression)
            return results_cursor, to_do_or

        if not searchstrings[0]:
            raise ResponseError(message="Error: bad argument 'searchstrings'")

        results_cursor, to_do_or = make_query(searchstrings[0], perspectives)

        pre_results = set(results_cursor.all())
        if adopted:
            results_cursor = DBSession.query(dbLexicalEntry).join(dbEntity.parent).filter(
                dbEntity.content.like('%заим.%'))
            if adopted_type:
                results_cursor = results_cursor.join(dbEntity.field) \
                    .join(dbTranslationAtom,
                          and_(dbField.translation_gist_client_id == dbTranslationAtom.parent_client_id,
                               dbField.translation_gist_object_id == dbTranslationAtom.parent_object_id,
                               dbField.marked_for_deletion == False)) \
                    .filter(dbTranslationAtom.content == adopted_type,
                            dbTranslationAtom.locale_id == ENGLISH_LOCALE)
            pre_results = pre_results & set(results_cursor.all())
        if with_etimology:
            results_cursor = DBSession.query(dbLexicalEntry).join(dbEntity.parent).join(dbEntity.field) \
                .join(dbTranslationAtom,
                      and_(dbField.data_type_translation_gist_client_id == dbTranslationAtom.parent_client_id,
                           dbField.data_type_translation_gist_object_id == dbTranslationAtom.parent_object_id,
                           dbField.marked_for_deletion == False)) \
                .filter(dbTranslationAtom.content == 'Grouping Tag',
                        dbTranslationAtom.locale_id == ENGLISH_LOCALE)

        pre_results = pre_results & set(results_cursor.all())

        for search_string in searchstrings[1:]:
            results_cursor, to_do_or_new = make_query(search_string, perspectives)
            if to_do_or:
                pre_results = pre_results | set(results_cursor.all())
            else:
                pre_results = pre_results & set(results_cursor.all())
            to_do_or = to_do_or_new

        lexes_composite_list = [(lex.created_at,
                                 lex.client_id, lex.object_id, lex.parent_client_id, lex.parent_object_id,
                                 lex.marked_for_deletion, lex.additional_metadata,
                                 lex.additional_metadata.get('came_from')
                                 if lex.additional_metadata and 'came_from' in lex.additional_metadata else None)
                                for lex in pre_results]

        lexical_entries = (

            dbLexicalEntry.track_multiple(
                lexes_composite_list,
                info.context.locale_id,
                publish = True,
                accept = True))

        lexical_entries_list = list()
        for entry in lexical_entries:
            entities = []
            for ent in entry['contains']:
                del ent["contains"]
                del ent["level"]
                del ent["accepted"]
                del ent["published"]
                if "link_client_id" in ent and "link_object_id" in ent:
                    ent["link_id"] = (ent["link_client_id"], ent["link_object_id"])
                else:
                    ent["link_id"] = None
                ent["field_id"] = (ent["field_client_id"], ent["field_object_id"])
                if "self_client_id" in ent and "self_object_id" in ent:
                    ent["self_id"] = (ent["self_client_id"], ent["self_object_id"])
                else:
                    ent["self_id"] = None
                if "content" not in ent:
                    ent["content"] = None
                if "additional_metadata" in ent:
                    ent["additional_metadata_string"] = ent["additional_metadata"]
                    del ent["additional_metadata"]
                if 'entity_type' in ent:
                    del ent['entity_type']

                gr_entity_object = Entity(id=[ent['client_id'],
                                              ent['object_id']],
                                          # link_id = (ent["link_client_id"], ent["link_object_id"]),
                                          parent_id=(ent["parent_client_id"], ent["parent_object_id"]),
                                          **ent  # all other args from sub_result
                                          )
                entities.append(gr_entity_object)

            del entry["published"]
            del entry["contains"]
            del entry["level"]
            gr_lexicalentry_object = LexicalEntry(id=[entry['client_id'],
                                                      entry['object_id']],
                                                  entities=entities, **entry)

            lexical_entries_list.append(gr_lexicalentry_object)
        return lexical_entries_list

    # @client_id_check()
    def resolve_user_blobs(self, info, data_type=None, is_global=None):
        allowed_global_types = ["sociolinguistics", "pdf"]
        client_id = info.context.client_id
        client = DBSession.query(Client).filter_by(id=client_id).first()
        user_blobs = list()
        if not data_type and is_global:
            raise ResponseError("Error: cannot list globally without data_type")
        if data_type and is_global:
            if data_type in allowed_global_types:
                user_blobs = DBSession.query(dbUserBlobs).filter_by(marked_for_deletion=False, data_type=data_type).all()
            else:
                raise ResponseError(message="Error: you can not list that data type globally.")
        elif not client:
            raise ResponseError('not authenticated')
        if data_type:
            if not is_global:
                user_blobs = DBSession.query(dbUserBlobs).filter_by(marked_for_deletion=False, user_id=client.user_id, data_type=data_type).all()
        else:
            user_blobs = DBSession.query(dbUserBlobs).filter_by(marked_for_deletion=False, user_id=client.user_id).all()
        user_blobs_list = list()
        for db_blob in user_blobs:
            gql_blob = UserBlobs(id=[db_blob.client_id, db_blob.object_id])
            gql_blob.dbObject = db_blob
            user_blobs_list.append(gql_blob)
        return user_blobs_list

    def resolve_userrequest(self, info, id):
        """
        query myQuery {
          userrequest(id: 6) {
                id
           }
        }
        """
        return UserRequest(id=id)

    #@client_id_check()
    def resolve_userrequests(self, info):
        """
        query myQuery {
          userrequests {
                id
                sender_id
                type
           }
        }
        """
        client_id = info.context.client_id

        client = DBSession.query(Client).filter_by(id=client_id).first()
        user = DBSession.query(dbUser).filter_by(id=client.user_id).first()
        if not user:
            raise ResponseError(message="This client id is orphaned. Try to logout and then login once more.")

        userrequests = DBSession.query(dbUserRequest).filter(dbUserRequest.recipient_id == user.id).order_by(
            dbUserRequest.created_at).all()
        userrequests_list = list()
        for db_userrequest in userrequests:
            gql_userrequest = UserRequest(id=db_userrequest.id)
            userrequests_list.append(gql_userrequest)
        return userrequests_list

    def resolve_all_basegroups(self, info):  # tested
        basegroups = list()
        for basegroup_object in DBSession.query(dbBaseGroup).all():
            basegroup = BaseGroup(id=basegroup_object.id)
            basegroup.dbObject = basegroup_object
            basegroups.append(basegroup)
        return basegroups

    def resolve_grant(self, info, id):
        return Grant(id=id)

    def resolve_grants(
        self,
        info,
        has_participant = None,
        participant_deleted = None,
        participant_published = None,
        participant_category = None):
        """
        query myQuery {
          grants {
                id
           }
        }
        """

        __debug_flag__ = False

        # Analyzing query.

        object_flag = False

        selection_dict = {

            'additional_metadata': (
                dbGrant.additional_metadata,),

            'begin': (
                dbGrant.begin,),

            'created_at': (
                dbGrant.created_at,),

            'end': (
                dbGrant.end,),

            'grant_number': (
                dbGrant.grant_number,),

            'grant_url': (
                dbGrant.grant_url,),

            'id': (
                dbGrant.id,),

            'issuer_translations': (
                dbGrant.issuer_translation_gist_client_id,
                dbGrant.issuer_translation_gist_object_id),

            'issuer_url': (
                dbGrant.issuer_url,),

            'owners': (
                dbGrant.owners,),

            'translations': (
                dbGrant.translation_gist_client_id,
                dbGrant.translation_gist_object_id,)}

        selection_set = set()
        selection_list = []

        owners_flag = False
        email_flag = False

        for field in info.field_asts:

            if field.name.value != 'grants':
                continue

            for subfield in field.selection_set.selections:

                name_str = subfield.name.value

                if name_str in selection_dict:

                    selection_set.add(name_str)
                    selection_list.extend(selection_dict[name_str])

                    if name_str == 'owners':

                        owners_flag = True

                        for owners_field in subfield.selection_set.selections:

                            if owners_field.name.value == 'email':

                                email_flag = True
                                break

                elif name_str != '__typename':

                    object_flag = True

        if object_flag:

            selection_list = [dbGrant]

        elif 'grant_number' not in selection_set:

            # For standard grant ordering.

            selection_set.add('grant_number')
            selection_list.extend(selection_dict['grant_number'])

        # If we are going to query both the usual and issuer translations, we'll have to use separate joins
        # to a CTE-based queries to avoid joining to actually a cross product of two translations.

        translations_flag = (
            'translations' in selection_set)

        issuer_translations_flag = (
            'issuer_translations' in selection_set)

        cte_flag = (
            translations_flag and
            issuer_translations_flag)

        grant_query = None
        grant_cte = None

        # No participant filtering, getting every grant.

        if has_participant is None:

            grant_query = (
                DBSession.query(*selection_list))

            grant_c = dbGrant

        # Simple participant count filter.

        elif (
            participant_deleted is None and
            participant_category is None and
            participant_published is None):

            participant_count = (

                func.jsonb_array_length(
                    dbGrant.additional_metadata['participant']))

            grant_query = (

                DBSession

                    .query(
                        *selection_list)

                    .filter(
                        participant_count > 0 if has_participant else
                        participant_count <= 0))

            grant_c = dbGrant

        # Additional conditions on participants, we'll have to check them through a join.
        #
        # We have to use raw SQL due to SQLAlchemy being bad with PostgreSQL's jsonb_to_recordset.

        else:

            dictionary_condition_list = []

            if participant_deleted is not None:

                dictionary_condition_list.append(
                    '\n and D.marked_for_deletion = true' if participant_deleted else
                    '\n and D.marked_for_deletion = false')

            if participant_category is not None:

                dictionary_condition_list.append(
                    f'\n and D.category = {participant_category}')

            if participant_published is not None:

                if not participant_published:
                    raise NotImplementedError

                dictionary_condition_list.append('''

                    and (
                      D.state_translation_gist_client_id,
                      D.state_translation_gist_object_id) in (

                      select
                        T.client_id,
                        T.object_id

                      from
                        translationgist T,
                        translationatom A

                      where
                        T.marked_for_deletion = false and
                        T.type = 'Service' and
                        A.parent_client_id = T.client_id and
                        A.parent_object_id = T.object_id and
                        A.locale_id = 2 and
                        A.marked_for_deletion = false and (
                          A.content = 'Published' or
                          A.content = 'Limited access'))

                    ''')

            dictionary_condition_str = (
                ''.join(dictionary_condition_list))

            if object_flag:

                selection_str = 'G.*'

            else:

                selection_str = (

                    ', '.join(
                        f'G.{selection.name}'
                        for selection in selection_list))

            sql_text = (

                sqlalchemy.text(f'''

                    select
                      {selection_str}

                    from
                      public.grant G

                    cross join
                      jsonb_to_recordset(G.additional_metadata -> 'participant')
                        P (client_id bigint, object_id bigint)

                    join
                      dictionary D

                    on
                      D.client_id = P.client_id and
                      D.object_id = P.object_id {dictionary_condition_str}

                    group by
                      G.id

                    '''))

            if object_flag:

                sql_text = (

                    aliased(
                        dbGrant,

                        sql_text

                            .columns(
                                **{column.name: column.type
                                    for column in dbGrant.__table__.c})

                            .alias(),

                        adapt_on_names = True))

                grant_c = (
                    sql_text)

            else:

                sql_text = (

                    sql_text

                        .columns(
                            *selection_list)

                        .alias())

                grant_c = (
                    sql_text.c)

            grant_query = (
                DBSession.query(sql_text))

            if not has_participant:
                raise NotImplementedError

        # Establishing a CTE if we'll need it.

        if cte_flag:

            grant_cte = grant_query.cte()
            grant_c = grant_cte.c

            if object_flag:

                grant_cte = (

                    aliased(
                        dbGrant,
                        grant_cte,
                        adapt_on_names = True))

                grant_c = grant_cte

            grant_query = (
                DBSession.query(grant_cte))

        # Getting translations through a join, if required.

        if translations_flag:

            if grant_cte is not None:

                translation_query = (

                    DBSession

                        .query(
                            grant_c.id)

                        .outerjoin(
                            dbTranslationAtom,

                            and_(
                                dbTranslationAtom.parent_client_id == grant_c.translation_gist_client_id,
                                dbTranslationAtom.parent_object_id == grant_c.translation_gist_object_id,
                                dbTranslationAtom.marked_for_deletion == False))

                        .add_columns(

                            func.jsonb_object_agg(
                                dbTranslationAtom.locale_id,
                                dbTranslationAtom.content)

                                .filter(dbTranslationAtom.locale_id != None)
                                .label('translations'))

                        .group_by(
                            grant_c.id)

                        .subquery())

                grant_query = (

                    grant_query

                        .join(
                            translation_query,
                            translation_query.c.id == grant_c.id)

                        .add_columns(
                            translation_query.c.translations))

            else:

                grant_query = (

                    grant_query

                        .outerjoin(
                            dbTranslationAtom,

                            and_(
                                dbTranslationAtom.parent_client_id == grant_c.translation_gist_client_id,
                                dbTranslationAtom.parent_object_id == grant_c.translation_gist_object_id,
                                dbTranslationAtom.marked_for_deletion == False))

                        .add_columns(

                            func.jsonb_object_agg(
                                dbTranslationAtom.locale_id,
                                dbTranslationAtom.content)

                                .filter(dbTranslationAtom.locale_id != None)
                                .label('translations'))

                        .group_by(
                            *([grant_c] if object_flag else grant_c)))

        # Getting issuer translations through a join, if required.

        if issuer_translations_flag:

            if grant_cte is not None:

                issuer_translation_query = (

                    DBSession

                        .query(
                            grant_c.id)

                        .outerjoin(
                            dbTranslationAtom,

                            and_(
                                dbTranslationAtom.parent_client_id == grant_c.issuer_translation_gist_client_id,
                                dbTranslationAtom.parent_object_id == grant_c.issuer_translation_gist_object_id,
                                dbTranslationAtom.marked_for_deletion == False))

                        .add_columns(

                            func.jsonb_object_agg(
                                dbTranslationAtom.locale_id,
                                dbTranslationAtom.content)

                                .filter(dbTranslationAtom.locale_id != None)
                                .label('issuer_translations'))

                        .group_by(
                            grant_c.id)

                        .subquery())

                grant_query = (

                    grant_query

                        .join(
                            issuer_translation_query,
                            issuer_translation_query.c.id == grant_c.id)

                        .add_columns(
                            issuer_translation_query.c.issuer_translations))

            else:

                grant_query = (

                    grant_query

                        .outerjoin(
                            dbTranslationAtom,

                            and_(
                                dbTranslationAtom.parent_client_id == grant_c.issuer_translation_gist_client_id,
                                dbTranslationAtom.parent_object_id == grant_c.issuer_translation_gist_object_id,
                                dbTranslationAtom.marked_for_deletion == False))

                        .add_columns(

                            func.jsonb_object_agg(
                                dbTranslationAtom.locale_id,
                                dbTranslationAtom.content)

                                .filter(dbTranslationAtom.locale_id != None)
                                .label('issuer_translations'))

                        .group_by(
                            *([grant_c] if object_flag else grant_c)))

        # Getting grant info, preparing it for GraphQL.

        grant_query = (

            grant_query
                .order_by(grant_c.grant_number))

        result_list = grant_query.all()

        if __debug_flag__:

            log.debug(
                '\n grant_query:\n ' +
                render_statement(grant_query.statement))

        gql_grant_list = []

        if object_flag:

            # We are getting full ORM dbGrant objects.

            attribute_set = selection_set.copy()

            attribute_set.discard('translations')
            attribute_set.discard('issuer_translations')

            for result in result_list:

                grant = (
                    result[0] if translations_flag or issuer_translations_flag else
                    result)

                gql_grant = (
                    Grant(id = grant.id))

                gql_grant.dbObject = grant

                if translations_flag:

                    translations = (
                        result.translations)

                    gql_grant.translations = (
                        translations if translations is not None else gql_none_value)

                if issuer_translations_flag:

                    translations = (
                        result.issuer_translations)

                    gql_grant.issuer_translations = (
                        translations if translations is not None else gql_none_value)

                for attribute in attribute_set:

                    value = getattr(grant, attribute)

                    if attribute == 'additional_metadata':

                        value = AdditionalMetadata.from_object(value)

                    elif attribute == 'begin' or attribute == 'end':

                        value = Grant.from_date(value)

                    elif attribute == 'created_at':

                        value = CreatedAt.from_timestamp(value)

                    setattr(
                        gql_grant,
                        attribute,
                        value)

                gql_grant_list.append(gql_grant)

        else:

            # We are getting attribute values as they are.

            for result in result_list:

                gql_grant = (
                    Grant(id = result.id))

                for selection in selection_set:

                    value = getattr(result, selection)

                    if selection == 'additional_metadata':

                        value = AdditionalMetadata.from_object(value)

                    elif selection == 'begin' or selection == 'end':

                        value = Grant.from_date(value)

                    elif selection == 'created_at':

                        value = CreatedAt.from_timestamp(value)

                    setattr(
                        gql_grant,
                        selection,
                        value if value is not None else gql_none_value)

                gql_grant_list.append(gql_grant)

        # Loading owners if required.

        if (gql_grant_list and
            owners_flag):

            owner_id_set = set()

            owner_id_set.update(

                *(gql_grant.owners
                    for gql_grant in gql_grant_list))

            user_query = (

                DBSession
                    .query(dbUser)

                    .filter(

                        dbUser.id.in_(
                            utils.values_query(
                                owner_id_set, models.SLBigInteger))))

            if email_flag:

                user_query = (

                    user_query.options(
                        joinedload(dbUser.email)))

            gql_user_dict = {}

            for user in user_query:

                user_id = user.id

                gql_user = User(id = user_id)
                gql_user.dbObject = user

                gql_user_dict[user_id] = gql_user

            for gql_grant in gql_grant_list:

                gql_grant.owners = [

                    gql_user_dict[owner_id]
                    for owner_id in gql_grant.owners]

        return gql_grant_list

    def resolve_phonology_tier_list(self, info, perspective_id):
        """
        query MyQuery {
          phonology_tier_list(perspective_id: [330, 4]) {
            tier_count
            total_count
          }
        }
        """
        storage = info.context.request.registry.settings['storage']

        answer = utils_phonology_tier_list(storage, *perspective_id)
        return TierList(**answer)

    def resolve_phonology_skip_list(self, info, perspective_id):
        """
        query MyQuery {
          phonology_skip_list(perspective_id: [1251, 14]) {
            markup_count
            neighbour_list
            skip_list
            total_neighbour_count
            total_skip_count
          }
        }
        """
        storage = info.context.request.registry.settings['storage']

        answer = utils_phonology_skip_list(storage, *perspective_id)
        return SkipList(**answer)

    def resolve_phonology_link_perspective_data(self, info, perspective_id, field_id_list):
        """
        query MyQuery {
          phonology_link_perspective_data(
            perspective_id: [657, 4],
            field_id_list: [[1, 213]])
          {
            perspective_id_list
          }
        }
        """

        answer = gql_phonology_link_perspective_data(perspective_id, field_id_list)
        return Link_Perspective_Data(**answer)

    def resolve_connected_words(self, info, id, field_id, mode=None):

        client_id = id[0]
        object_id = id[1]
        field_client_id = field_id[0]
        field_object_id = field_id[1]
        if mode == 'all':
            publish = None
            accept = True
        elif mode == 'published':
            publish = True
            accept = True
        elif mode == 'not_accepted':
            publish = None
            accept = False

        # NOTE: modes 'deleted' and 'all_with_deleted' are currently not implemented.

        elif mode == 'deleted':
            publish = None
            accept = None
        elif mode == 'all_with_deleted':
            publish = None
            accept = None

        else:
            raise ResponseError(message="mode: <all|published|not_accepted>")

        # Getting lexical entry group info.

        marked_for_deletion = (

            DBSession
                .query(dbLexicalEntry.marked_for_deletion)

                .filter_by(
                    client_id = client_id,
                    object_id = object_id)

                .scalar())

        if (marked_for_deletion is None or
            marked_for_deletion):

            raise ResponseError(message = 'No such lexical entry in the system')

        if publish is None and accept is None:

            sql_str = '''

                select * from linked_group_no_publishing(
                    :field_client_id,
                    :field_object_id,
                    :client_id,
                    :object_id)

                '''

        else:

            sql_str = '''

                select * from linked_group(
                    :field_client_id,
                    :field_object_id,
                    :client_id,
                    :object_id,
                    :publish,
                    :accept)

                '''

        entry_query = (

            DBSession

                .query(
                    dbLexicalEntry.client_id,
                    dbLexicalEntry.object_id)

                .filter(

                    tuple_(
                        dbLexicalEntry.client_id,
                        dbLexicalEntry.object_id)

                    .in_(sqlalchemy.text(sql_str)))

                .params({
                    'field_client_id': field_client_id,
                    'field_object_id': field_object_id,
                    'client_id': client_id,
                    'object_id': object_id,
                    'publish': publish,
                    'accept': accept}))

        entry_list, _ = (

            entries_with_entities(
                entry_query,
                publish = publish,
                accept = accept,
                check_perspective = False))

        entity_list = []

        for entry in entry_list:
            if entry.gql_Entities:
                entity_list.extend(entry.gql_Entities)

        return (
            LexicalEntriesAndEntities(
                entities = entity_list,
                lexical_entries = entry_list))

    def resolve_convert_markup(self, info, id):

        # TODO: permission check
        """
        query myQuery {
            convert_markup(id: [742, 5494] )
        }
        """
        # client_id = info.context.client_id
        # client = DBSession.query(Client).filter_by(id=client_id).first()
        # user = DBSession.query(dbUser).filter_by(id=client.user_id).first()
        client_id, object_id = id
        entity = DBSession.query(dbEntity).filter_by(client_id=client_id, object_id=object_id).first()

        if not entity:
            return ResponseError(f'No entity {client_id} / {object_id}.')

        try:

            storage = (
                info.context.request.registry.settings['storage'])

            with storage_file(
                storage, entity.content) as content_stream:

                content = content_stream.read()

        except:
            return ResponseError(f'Cannot access file \'{entity.content}\'.')

        return tgt_to_eaf(content, entity.additional_metadata)

    def resolve_parser_results(self, info, entity_id):
        entity_client_id, entity_object_id = entity_id
        results = DBSession.query(dbParserResult).filter_by(entity_client_id=entity_client_id,
                                                            entity_object_id=entity_object_id,
							    marked_for_deletion=False
                                                            ).all()
        return_list = list()
        for result in results:
            new_parser_result = ParserResult(id = [result.client_id, result.object_id])
            new_parser_result.dbObject = result
            return_list.append(new_parser_result)
        return return_list

    def resolve_parser_result(self, info, id, exact_fmt='html'):
        client_id, object_id = id
        result_orig = DBSession.query(dbParserResult).filter_by(client_id=client_id,
                                                                object_id=object_id,
                                                                ).first()
        if not result_orig or result_orig.marked_for_deletion:
            return None

        if exact_fmt == 'json':
            result = copy.copy(result_orig)
            result.content = get_result_json(result.content)
        else:
            result = result_orig

        parser_result = ParserResult(id=[result.client_id, result.object_id])
        parser_result.dbObject = result
        return parser_result

    def resolve_parsers(self, info):
        parsers = DBSession.query(dbParser).all()
        return_list = list()
        for parser in parsers:
            element = Parser(id=[parser.client_id, parser.object_id])
            element.dbObject = parser
            return_list.append(element)
        return return_list

    def resolve_adverb_data(
        self,
        info,
        perspective_id,
        offset = 0,
        limit = 25,
        specificity_flag = True,
        adverb_prefix = None,
        case_flag = False,
        accept_value = None,
        sort_order_list = None,
        debug_flag = False,
        **args):

        log.debug(
            f'\nperspective_id: {perspective_id}'
            f'\noffset: {offset}'
            f'\nlimit: {limit}'
            f'\nspecificity_flag: {specificity_flag}'
            f'\nadverb_prefix: {adverb_prefix}'
            f'\ncase_flag: {case_flag}'
            f'\naccept_value: {accept_value}'
            f'\nsort_order_list: {sort_order_list}'
            f'\ndebug_flag: {debug_flag}')

        if sort_order_list is None:
            sort_order_list = ['specificity', 'adverb', 'case', 'accept']

        specificity_flag = True

        adverb_flag = adverb_prefix is not None
        accept_flag = accept_value is not None

        instance_query = (

            DBSession

                .query(
                    dbAdverbInstanceData)

                .filter(
                    dbValencySourceData.perspective_client_id == perspective_id[0],
                    dbValencySourceData.perspective_object_id == perspective_id[1],
                    dbValencySentenceData.source_id == dbValencySourceData.id,
                    dbAdverbInstanceData.sentence_id == dbValencySentenceData.id))

        # If we are going to sort by specificity, we will need a base instance CTE to generate specificity
        # metrics.

        instance_c = (
            dbAdverbInstanceData)

        if specificity_flag:

            instance_cte = (
                instance_query.cte())

            instance_c = (
                instance_cte.c)

            instance_query = (
                DBSession.query(instance_cte))

        # Filtering by adverb prefix if required.

        if adverb_prefix:
            adverb_prefix_filter_str = (
                adverb_prefix.replace('%', '\\%') + '%')

            instance_query = (

                instance_query.filter(
                    instance_c.adverb_lex.ilike(
                        adverb_prefix_filter_str)))

        instance_count = (
            instance_query.count())

        # Getting ready to sort, if required.

        order_by_list = []

        for sort_type in sort_order_list:

            if sort_type == 'specificity':

                if specificity_flag:

                    count_cte = (
                        DBSession

                            .query(
                                instance_c.adverb_lex,
                                func.unnest(
                                    func.string_to_array(
                                        instance_c.case_str, ',')).label('mark_str'),
                                func.count('*').label('c_i'))

                            .group_by(
                                instance_c.adverb_lex,
                                'mark_str')

                           .cte())

                    specificity_subquery = (
                        DBSession

                            .query(
                                count_cte.c.adverb_lex,
                                func.count('*').label('c_unique'),

                                (func.log(
                                    2, func.sum(count_cte.c.c_i))

                                    - func.sum(
                                        count_cte.c.c_i * func.log(2, count_cte.c.c_i))
                                        / func.sum(count_cte.c.c_i))

                                    .label('H'))

                            .group_by(
                                count_cte.c.adverb_lex)

                            .subquery())

                    instance_query = (

                        instance_query.outerjoin(
                            specificity_subquery,
                            instance_c.adverb_lex == specificity_subquery.c.adverb_lex))

                    order_by_list.extend((
                        specificity_subquery.c.c_unique,
                        specificity_subquery.c.H))

            elif sort_type == 'adverb':

                if adverb_flag:

                    order_by_list.append(
                        instance_c.adverb_lex)

            elif sort_type == 'case':

                if case_flag:

                    order_by_list.append(
                        func.string_to_array(instance_c.case_str, ','))

            elif sort_type == 'accept':

                if accept_flag:

                    accept_subquery = (

                        DBSession

                            .query(
                                dbAdverbAnnotationData.instance_id,

                                func.bool_or(dbAdverbAnnotationData.accepted)
                                    .label('accept_value'))

                            .group_by(
                                dbAdverbAnnotationData.instance_id)

                            .subquery())

                    instance_query = (

                        instance_query.outerjoin(
                            accept_subquery,
                            instance_c.id == accept_subquery.c.instance_id))

                    order_by_list.append(
                        func.coalesce(accept_subquery.c.accept_value, False) != accept_value)

        order_by_list.append(
            instance_c.id)

        # Getting annotation instances and related info.

        instance_query = (

            instance_query
                .order_by(*order_by_list)
                .offset(offset)
                .limit(limit))

        instance_list = instance_query.all()

        if debug_flag:

            log.debug(
                '\n' +
                str(instance_query.statement.compile(compile_kwargs = {'literal_binds': True})))

        instance_id_set = (
            set(instance.id for instance in instance_list))

        sentence_id_set = (
            set(instance.sentence_id for instance in instance_list))

        log.debug(
            '\ninstance_id_set: {}'
            '\nsentence_id_set: {}'.format(
                instance_id_set,
                sentence_id_set))

        sentence_list = []

        if sentence_id_set:

            sentence_list = (

                DBSession

                    .query(
                        dbValencySentenceData)

                    .filter(
                        dbValencySentenceData.id.in_(sentence_id_set))

                    .all())

        annotation_list = []

        if instance_id_set:

            annotation_list = (

                DBSession

                    .query(
                        dbAdverbAnnotationData.instance_id,

                        func.jsonb_agg(
                            func.jsonb_build_array(
                                dbAdverbAnnotationData.user_id,
                                dbAdverbAnnotationData.accepted)))

                        .filter(
                        dbAdverbAnnotationData.instance_id.in_(instance_id_set))

                    .group_by(
                        dbAdverbAnnotationData.instance_id)

                    .all())

        user_id_set = (

            set(user_id
                for _, user_annotation_list in annotation_list
                for user_id, _ in user_annotation_list))

        user_list = []

        if user_id_set:

            user_list = (

                DBSession

                    .query(
                        dbUser.id, dbUser.name)

                    .filter(
                        dbUser.id.in_(user_id_set))

                    .all())

        instance_list = [

            {'id': instance.id,
                'sentence_id': instance.sentence_id,
                'index': instance.index,
                'adverb_lex': instance.adverb_lex,
                'case_str': instance.case_str}

            for instance in instance_list]

        sentence_list = [
            dict(sentence.data, id = sentence.id)
            for sentence in sentence_list]

        log.debug(
            '\ninstance_list:\n{}'
            '\nsentence_list:\n{}'
            '\nannotation_list:\n{}'
            '\nuser_list:\n{}'.format(
                pprint.pformat(instance_list, width = 192),
                pprint.pformat(sentence_list, width = 192),
                pprint.pformat(annotation_list, width = 192),
                pprint.pformat(user_list, width = 192)))

        result_dict = {
            'instance_count': instance_count,
            'instance_list': instance_list,
            'sentence_list': sentence_list,
            'annotation_list': annotation_list,
            'user_list': user_list}

        # Getting all adverbs, without filtering, if required.

        if adverb_flag:

            if adverb_prefix:

                adverb_query = (

                    DBSession

                        .query(
                            dbAdverbInstanceData.adverb_lex,
                            dbAdverbInstanceData.adverb_lex.ilike(
                                adverb_prefix_filter_str)))

            else:

                adverb_query = (

                    DBSession

                        .query(
                            dbAdverbInstanceData.adverb_lex))

            adverb_list = (

                adverb_query

                    .filter(
                        dbValencySourceData.perspective_client_id == perspective_id[0],
                        dbValencySourceData.perspective_object_id == perspective_id[1],
                        dbValencySentenceData.source_id == dbValencySourceData.id,
                        dbAdverbInstanceData.sentence_id == dbValencySentenceData.id)

                    .distinct()

                    .order_by(dbAdverbInstanceData.adverb_lex)

                    .all())

            if adverb_prefix:

                result_dict['adverb_list'] = (
                    [list(row) for row in adverb_list])

            else:

                result_dict['adverb_list'] = (
                    [[row[0], True] for row in adverb_list])

        return result_dict

class PerspectivesAndFields(graphene.InputObjectType):
    perspective_id = LingvodocID()
    field_id = LingvodocID()

class StarlingEtymologyObject(graphene.InputObjectType):
    starling_perspective_id = LingvodocID()
    perspectives_and_fields = graphene.List(PerspectivesAndFields)

class StarlingEtymology(graphene.Mutation):

    class Arguments:
        etymology_field_id = LingvodocID()
        complex_list = graphene.List(StarlingEtymologyObject)

    triumph = graphene.Boolean()

    @staticmethod
    @client_id_check()
    def mutate(root, info, **args):

        client_id = info.context.client_id

        client = DBSession.query(Client).filter_by(id=client_id).first()
        user = DBSession.query(dbUser).filter_by(id=client.user_id).first()
        if not user:
            raise ResponseError(message="This client id is orphaned. Try to logout and then login once more.")
        if user.id != 1:
            raise ResponseError(message="not admin")

        etymology_field_id = args['etymology_field_id']
        etymology_field = DBSession.query(dbField).filter_by(client_id=etymology_field_id[0],
                                                             object_id=etymology_field_id[1],
                                                             marked_for_deletion=False).first()
        if not etymology_field:
            raise ResponseError(message='no such field')
        timestamp = (
            time.asctime(time.gmtime()) + ''.join(
                random.SystemRandom().choice(string.ascii_uppercase + string.digits) for c in range(10)))
        for complex_element in args['complex_list']:
            starling_ids = complex_element['starling_perspective_id']
            starling_perspective = DBSession.query(dbPerspective).filter_by(client_id=starling_ids[0],
                                                                                      object_id=starling_ids[1],
                                                                                      marked_for_deletion=False).first()
            if not starling_perspective:
                raise ResponseError(message='no such starling perspective')
            for persp_and_field in complex_element['perspectives_and_fields']:
                persp_ids = persp_and_field['perspective_id']
                cur_persp = DBSession.query(dbPerspective).filter_by(client_id=persp_ids[0],
                                                                               object_id=persp_ids[1],
                                                                               marked_for_deletion=False).first()
                field_id = persp_and_field['field_id']
                field = DBSession.query(dbField).filter_by(client_id=field_id[0],
                                                           object_id=field_id[1],
                                                           marked_for_deletion=False).first()
                if not cur_persp:
                    raise ResponseError(message='no such perspective')

                StarlingEntity = aliased(dbEntity)
                StarlingLexicalEntry = aliased(dbLexicalEntry)
                result = DBSession.query(dbEntity, dbLexicalEntry, StarlingEntity, StarlingLexicalEntry).filter(
                    dbEntity.parent_client_id == dbLexicalEntry.client_id,
                    dbEntity.parent_object_id == dbLexicalEntry.object_id,
                    StarlingEntity.parent_client_id == StarlingLexicalEntry.client_id,
                    StarlingEntity.parent_object_id == StarlingLexicalEntry.object_id,
                    dbLexicalEntry.parent_client_id == cur_persp.client_id,
                    dbLexicalEntry.parent_object_id == cur_persp.object_id,
                    StarlingLexicalEntry.parent_client_id == starling_perspective.client_id,
                    StarlingLexicalEntry.parent_object_id == starling_perspective.object_id,
                    dbEntity.content == StarlingEntity.content,
                    dbEntity.field == field).all()
                for sub_res in result:
                    first_lex = sub_res[1]
                    second_lex = sub_res[3]
                    tag = sub_res[0].content + timestamp
                    first_tag = dbEntity(client_id = client.id, parent=first_lex, content=tag, field=etymology_field)
                    first_tag.publishingentity.accepted = True
                    first_tag.publishingentity.published = True
                    # DBSession.add(first_tag)
                    second_tag = dbEntity(client_id = client.id, parent=second_lex, content=tag, field=etymology_field)
                    second_tag.publishingentity.accepted = True
                    second_tag.publishingentity.published = True
                    # DBSession.add(second_tag)
                    caching.CACHE.set(objects = [first_tag, second_tag], DBSession=DBSession)

        return StarlingEtymology(triumph=True)


class Phonology(graphene.Mutation):

    class Arguments:
        perspective_id=LingvodocID(required=True)
        limit=graphene.Int()
        limit_exception=graphene.Int()
        limit_no_vowel=graphene.Int()
        limit_result=graphene.Int()
        no_cache=graphene.Boolean()
        interval_only=graphene.Boolean()
        group_by_description=graphene.Boolean(required=True)
        maybe_translation_field=LingvodocID()
        only_first_translation=graphene.Boolean(required=True)
        vowel_selection=graphene.Boolean(required=True)
        maybe_tier_list=graphene.List(graphene.String)
        keep_list=graphene.List(graphene.Int)
        join_list=graphene.List(graphene.Int)
        chart_threshold=graphene.Int()
        generate_csv=graphene.Boolean()
        link_field_list=graphene.List(LingvodocID)
        link_perspective_list=graphene.List(graphene.List(LingvodocID))
        use_fast_track=graphene.Boolean()
        synchronous=graphene.Boolean()
        debug_flag=graphene.Boolean()

    triumph = graphene.Boolean()

    @staticmethod
    # @client_id_check()

    def mutate(self, info, **args):
        """
        query MyQuery {
          phonology(
            perspective_id: [126, 5],
            group_by_description: false,
            only_first_translation: false,
            vowel_selection: false,
            maybe_tier_list: [],
            maybe_translation_field: [66, 19])
        }
        """

        parameters = (
            Phonology_Parameters.from_graphql(args))

        parameters.__debug_flag__ = (
            args.get('debug_flag', False))

        locale_id = info.context.locale_id
        request = info.context.get('request')

        utils_phonology(request, locale_id, parameters)

        return Phonology(triumph=True)


@celery.task
def async_phonological_statistical_distance(
    id_list,
    vowel_selection,
    chart_threshold,
    locale_id,
    storage,
    task_key,
    cache_kwargs,
    sqlalchemy_url,
    __debug_flag__):
    """
    Sets up and launches cognate analysis in asynchronous mode.
    """

    # NOTE: copied from phonology.
    #
    # As far as we know, this is a no-op with current settings, we use it to enable logging inside celery
    # tasks, because somehow this does it, and otherwise we couldn't set it up.

    logging.debug('async_phonological_statistical_distance')

    # Ok, and now we go on with task execution.

    engine = create_engine(sqlalchemy_url)
    DBSession.configure(bind = engine)
    initialize_cache(cache_kwargs)
    task_status = TaskStatus.get_from_cache(task_key)

    with transaction.manager:

        try:
            PhonologicalStatisticalDistance.perform_phonological_statistical_distance(
                id_list,
                vowel_selection,
                chart_threshold,
                locale_id,
                storage,
                task_status,
                __debug_flag__)

        # Some unknown external exception.

        except Exception as exception:

            traceback_string = ''.join(traceback.format_exception(
                exception, exception, exception.__traceback__))[:-1]

            log.warning('phonological_statistical_distance: exception')
            log.warning(traceback_string)

            if task_status is not None:

                task_status.set(1, 100,
                    'Finished (ERROR), exception:\n' + traceback_string)


class PhonologicalStatisticalDistance(graphene.Mutation):

    class Arguments:

        id_list = graphene.List(LingvodocID, required = True)
        vowel_selection = graphene.Boolean(required = True)
        chart_threshold = graphene.Int()

        synchronous_flag = graphene.Boolean()
        debug_flag = graphene.Boolean()

    triumph = graphene.Boolean()

    def perform_phonological_statistical_distance(
        id_list,
        vowel_selection,
        chart_threshold,
        locale_id,
        storage,
        task_status,
        __debug_flag__):
        """
        Performs phonological statistical distance computation in either synchronous or asynchronous mode.
        """

        Markup = aliased(dbEntity, name = 'Markup')
        Sound = aliased(dbEntity, name = 'Sound')

        PublishingMarkup = aliased(dbPublishingEntity, name = 'PublishingMarkup')
        PublishingSound = aliased(dbPublishingEntity, name = 'PublishingSound')

        info_list = []
        total_count = 0

        # Getting preliminary perspective info.

        for perspective_id in id_list:

            perspective = DBSession.query(dbPerspective).filter_by(
                client_id = perspective_id[0], object_id = perspective_id[1]).first()

            dictionary = perspective.parent

            name = '{0} - {1}'.format(
                dictionary.get_translation(locale_id),
                perspective.get_translation(locale_id))

            query = DBSession.query(
                Markup, Sound).filter(
                    dbLexicalEntry.parent_client_id == perspective_id[0],
                    dbLexicalEntry.parent_object_id == perspective_id[1],
                    dbLexicalEntry.marked_for_deletion == False,
                    Markup.parent_client_id == dbLexicalEntry.client_id,
                    Markup.parent_object_id == dbLexicalEntry.object_id,
                    Markup.marked_for_deletion == False,
                    Markup.additional_metadata.contains({'data_type': 'praat markup'}),
                    PublishingMarkup.client_id == Markup.client_id,
                    PublishingMarkup.object_id == Markup.object_id,
                    PublishingMarkup.published == True,
                    PublishingMarkup.accepted == True,
                    Sound.client_id == Markup.self_client_id,
                    Sound.object_id == Markup.self_object_id,
                    Sound.marked_for_deletion == False,
                    PublishingSound.client_id == Sound.client_id,
                    PublishingSound.object_id == Sound.object_id,
                    PublishingSound.published == True,
                    PublishingSound.accepted == True)

            count = query.count()

            info_list.append((perspective, dictionary, name, query, count))
            total_count += count

        # Showing what we've got.

        log.debug(
            'phonological_statistical_distance ({0} perspectives): {1} sound/markup pairs\n{2}'.format(
            len(id_list),
            total_count,
            '\n'.join('{0}: {1} sound/markup pairs'.format(name, count)
                for p, d, name, q, count in info_list)))

        # And now we are going to gather formant distribution data.

        formant_data_list = []
        row_count = 0
        perspective_cache_dict = {}

        x_min, x_max = None, None
        y_min, y_max = None, None

        no_index_list = []
        yes_index_list = []

        for perspective_index, (perspective, dictionary, name, query, count) in enumerate(info_list):

            formant_list = []
            vowel_counter = collections.Counter()

            '''
            Cache initializing
            '''
            perspective_id = (perspective.client_id, perspective.object_id)
            if not perspective_cache_dict.get(perspective_id):
                perspective_cache_dict[perspective_id] = PickleCache(storage, perspective_id)

            for row_index, row in enumerate(query.yield_per(100)):

                # Showing what sound/markup pair we are to process.

                sound_id = (row.Sound.client_id, row.Sound.object_id)
                sound_url = row.Sound.content

                markup_id = (row.Markup.client_id, row.Markup.object_id)
                markup_url = row.Markup.content

                log_str = (
                    '\nphonological_statistical_distance ({0}, {1}): '
                    'sound entity {2}/{3}, markup entity {4}/{5}'.format(
                        perspective_index, row_index,
                        sound_id[0], sound_id[1],
                        markup_id[0], markup_id[1]))

                # Getting phonology data of another sound/markup pair.

                textgrid_result_list = (

                    process_sound_markup(
                        log_str,
                        sound_id,
                        sound_url,
                        markup_id,
                        markup_url,
                        storage,
                        perspective_cache_dict[perspective_id],
                        vowel_selection,
                        __debug_flag__))

                row_count += 1

                if task_status is not None:

                    task_status.set(1,
                        int(math.floor(95.0 * row_count / total_count)),
                        'Gathering formant distribution data')

                # Extracting F1/F2 formant data.

                if textgrid_result_list is None:
                    continue

                for tier_number, tier_name, tier_result_list in textgrid_result_list:

                    if tier_result_list == 'no_vowel' or tier_result_list == 'no_vowel_selected':
                        continue

                    for tier_result in tier_result_list:

                        if vowel_selection:

                            # Either only for longest interval and interval with highest intensity, or...

                            f_list_a = tuple(map(float, tier_result.max_length_f_list[:2]))
                            f_list_b = tuple(map(float, tier_result.max_intensity_f_list[:2]))

                            text_a_list = tier_result.max_length_str.split()
                            text_b_list = tier_result.max_intensity_str.split()

                            vowel_a = get_vowel_class(
                                tier_result.max_length_source_index, tier_result.source_interval_list)

                            vowel_b = get_vowel_class(
                                tier_result.max_intensity_source_index, tier_result.source_interval_list)

                            formant_list.append(f_list_a)
                            vowel_counter[vowel_a] += 1

                            if text_b_list[2] != text_a_list[2]:

                                formant_list.append(f_list_b)
                                vowel_counter[vowel_b] += 1

                        else:

                            # ...for all intervals.

                            for (interval_str, interval_r_length, j_local, s_local, p_mean, i_list, f_list,
                                sign_longest, sign_highest, source_index) in tier_result.interval_data_list:

                                formant_list.append(tuple(map(float, f_list[:2])))

                                vowel_counter[get_vowel_class(
                                    source_index, tier_result.source_interval_list)] += 1

            # If we do not have enough formant data, we skip modelling of this perspective.

            if len(formant_list) < 2:

                formant_data_list.append((vowel_counter, formant_list, None))
                no_index_list.append(perspective_index)

                log.debug(
                    '\nphonological_statistical_distance:'
                    '\nperspective {0} ({1}, {2}/{3}):'
                    '\n{4} vowels, {5} formant vectors'
                    '\nno model, not enough data'.format(
                    perspective_index,
                    name,
                    perspective.client_id,
                    perspective.object_id,
                    len(vowel_counter),
                    len(formant_list)))

                continue

            yes_index_list.append(perspective_index)

            # And now we are going to model the distribution as a gaussian mixture.

            formant_array = numpy.array(formant_list)
            n_components = min(len(vowel_counter) * 2, len(formant_list))

            mixture_kwargs = {
                'n_components': n_components,
                'weight_concentration_prior_type': 'dirichlet_distribution',
                'weight_concentration_prior': 1.0,
                'reg_covar': 1e-9,
                'max_iter': 1024}

            cache_key = 'psd.' + (
                base64.b85encode(hashlib.md5(pickle.dumps(
                    (formant_array, mixture_kwargs)))
                        .digest()).decode('ascii'))

            cache_result = caching.CACHE.get(cache_key)

            # If we have cached model, we use it, otherwise we build it anew.

            if cache_result:

                mixture, component_count = cache_result

            else:

                mixture = sklearn.mixture.BayesianGaussianMixture(
                    verbose = 2 if __debug_flag__ else 0,
                    **mixture_kwargs)

                mixture.fit(formant_array)

                min_weight = min(mixture.weights_)

                component_where = mixture.weights_ > min_weight
                component_count = component_where.sum()

                # Removing any non-significant components, if required.

                if component_count < n_components - 2:

                    for name in [
                        'weights_',
                        'weight_concentration_',
                        'means_',
                        'covariances_',
                        'precisions_cholesky_',
                        'degrees_of_freedom_',
                        'mean_precision_']:

                        setattr(mixture, name,
                            getattr(mixture, name)[component_where])

                    mixture.weights_ /= sum(mixture.weights_)

                # Caching newly built model.

                caching.CACHE.set(cache_key,
                    (mixture, component_count))

            # Updating distribution data.

            #if perspective_index <= 0:
            if any((val is None for val in (x_min, x_max, y_min, y_max))):

                x_min, x_max = formant_array[:,0].min(), formant_array[:,0].max()
                y_min, y_max = formant_array[:,1].min(), formant_array[:,1].max()

            else:

                x_min = min(x_min, formant_array[:,0].min())
                x_max = max(x_max, formant_array[:,0].max())

                y_min = min(y_min, formant_array[:,1].min())
                y_max = max(y_max, formant_array[:,1].max())

            formant_data_list.append((vowel_counter, formant_array, mixture))

            # Another distribution of F1/F2 formant vectors we've got.

            log.debug(
                '\nphonological_statistical_distance:'
                '\nperspective {0} ({1}, {2}/{3}):'
                '\n{4} vowels, {5} formant vectors'
                '\n{6}'
                '\n{7}model with {8} / {9} significant components'.format(
                perspective_index,
                name,
                perspective.client_id,
                perspective.object_id,
                len(vowel_counter),
                len(formant_list),
                pprint.pformat(vowel_counter, width = 144),
                'CACHEd ' if cache_result else '',
                component_count if component_count <= n_components - 2 else n_components,
                n_components))

            if task_status is not None:

                task_status.set(1,
                    int(math.floor(95.0 + 2.0 * (perspective_index + 1) / len(info_list))),
                    'Creating formant distribution models')

        '''
        Writing cache to file if there is any change
        '''
        for cache in perspective_cache_dict.values():
            cache.flush()

        # Preparing for compilation of modelling results.

        workbook_stream = io.BytesIO()

        workbook = xlsxwriter.Workbook(
            workbook_stream, {'in_memory': True})

        worksheet_distance = (
            workbook.add_worksheet(
                utils.sanitize_worksheet_name('Distance')))

        worksheet_list = []

        # Getting integrable density grids for each distribution model.

        x_min = x_min or 500
        x_max = x_max or 1000

        y_min = y_min or 500
        y_max = y_max or 1000

        x_low = max(0, x_min - (x_max - x_min) * 0.25)
        x_high = x_max + (x_max - x_min) * 0.25

        y_low = max(0, y_min - (y_max - y_min) * 0.25)
        y_high = y_max + (y_max - y_min) * 0.25

        n_step = 1024

        x, y = pylab.meshgrid(
            numpy.linspace(x_low, x_high, n_step),
            numpy.linspace(y_low, y_high, n_step))

        # Showing grid parameters.

        log.debug(
            '\nphonological_statistical_distance: '
            '\nmin/max: {0:.3f}, {1:.3f}, {2:.3f}, {3:.3f}'
            '\nlow/high: {4:.3f}, {5:.3f}, {6:.3f}, {7:.3f}'
            '\n{8} steps'.format(
            x_min, x_max, y_min, y_max,
            x_low, x_high, y_low, y_high,
            n_step))

        grid_data_list = []

        for perspective_index, (
            (perspective, dictionary, name, query, count),
            (vowel_counter, formant_array, mixture)) in (

            enumerate(zip(info_list, formant_data_list))):

            # Checking if we have no data for this perspective.

            if mixture is None:

                grid_data_list.append(None)
                continue

            z = None

            worksheet_list.append(
                workbook.add_worksheet(
                    utils.sanitize_worksheet_name(
                        'Figure {0}'.format(len(worksheet_list) + 1))))

            # If we are in debug mode, we try to load saved grid data we might have.

            if __debug_flag__:

                z_digest = (
                    hashlib.md5(pickle.dumps(
                        (formant_array, mixture_kwargs, x_low, x_high, y_low, y_high, n_step)))
                            .hexdigest())

                z_file_name = (
                    '__grid_data_{0}_{1}_{2}__.gz'.format(
                        perspective.client_id, perspective.object_id, z_digest))

                if os.path.exists(z_file_name):

                    with gzip.open(z_file_name, 'rb') as z_file:
                        z, elapsed_time, integral, check = pickle.load(z_file)

            # Computing density grid data.

            if z is None:

                x.shape = (n_step ** 2,)
                y.shape = (n_step ** 2,)

                start_time = time.time()

                z = numpy.exp(mixture.score_samples(numpy.stack((x, y), 1)))
                z.shape = (n_step, n_step)

                elapsed_time = time.time() - start_time

                x.shape = (n_step, n_step)
                y.shape = (n_step, n_step)

                # Integrating its interpolation via Simpson's rule, normalizing if required.

                integral = scipy.integrate.simps(scipy.integrate.simps(z, x), y[:,0])
                z /= integral

                check = scipy.integrate.simps(scipy.integrate.simps(z, x), y[:,0])

                if __debug_flag__:

                    with gzip.open(z_file_name, 'wb') as z_file:
                        pickle.dump((z, elapsed_time, integral, check), z_file)

            # And we have another density grid data.

            log.debug(
                '\nphonological_statistical_distance:'
                '\nperspective {0} ({1}, {2}/{3}):'
                '\n{4} components, {5} ** 2 ({6}) samples in {7:.3f}s'
                '\nintegral of interpolation {8:.6f}, {9:.6f} after normalization'.format(
                perspective_index, name, perspective.client_id, perspective.object_id,
                len(mixture.weights_), n_step, n_step ** 2, elapsed_time,
                integral, check))

            grid_data_list.append(z)

            # Preparing figures with modelling results.

            figure = pyplot.figure(figsize = (16, 10))

            axes = figure.add_subplot(111)
            axes.set_title(name, fontsize = 14, family = 'Gentium')

            axes.invert_xaxis()
            axes.invert_yaxis()

            # F1/F2 formant data points.

            axes.scatter(
                formant_array[:,1],
                formant_array[:,0],
                marker = 'o', color = 'black', s = 4)

            x_lim = axes.get_xlim()
            y_lim = axes.get_ylim()

            x_lim = (min(x_lim[0], y_high), max(x_lim[1], y_low))
            y_lim = (min(y_lim[0], x_high), max(y_lim[1], x_low))

            # Probability density heat map.

            axes.imshow(z.T[::-1,:],
                aspect = 'auto',
                extent = (y_low, y_high, x_low, x_high),
                cmap = 'YlOrRd',
                alpha = 0.5)

            axes.set_xlim(x_lim)
            axes.set_ylim(y_lim)

            # Plotting standard deviation ellipses, see
            # https://scikit-learn.org/stable/auto_examples/mixture/plot_concentration_prior.html.

            for n in range(mixture.means_.shape[0]):

                eig_vals, eig_vecs = numpy.linalg.eigh(mixture.covariances_[n])
                unit_eig_vec = eig_vecs[0] / numpy.linalg.norm(eig_vecs[0])

                angle = numpy.arctan2(unit_eig_vec[1], unit_eig_vec[0])
                angle = 180 * angle / numpy.pi

                eig_vals = 2 * numpy.sqrt(2) * numpy.sqrt(eig_vals)

                ellipse = (
                    matplotlib.patches.Ellipse(
                        mixture.means_[n][::-1],
                        eig_vals[1],
                        eig_vals[0],
                        180 - angle,
                        edgecolor = 'black'))

                ellipse.set_alpha(max(mixture.weights_[n], 0.0625))
                ellipse.set_facecolor('#E0E0E0')

                axes.add_artist(ellipse)

            # Saving figure as a PNG stream.

            pyplot.setp(axes.texts, family = 'Gentium')

            pyplot.xticks(fontsize = 14, family = 'Gentium')
            pyplot.yticks(fontsize = 14, family = 'Gentium')

            figure_stream = io.BytesIO()

            figure.savefig(
                figure_stream,
                bbox_inches = 'tight',
                pad_inches = 0.25,
                format = 'png')

            # Adding figure to the workbook.

            figure_file_name = (
                'figure phonology distribution 2d {0} {1}.png'.format(
                perspective.client_id,
                perspective.object_id))

            worksheet_list[-1].insert_image(
                'A1', figure_file_name, {'image_data': figure_stream})

            if __debug_flag__:

                figure_stream.seek(0)

                with open(figure_file_name, 'wb') as figure_file:
                    shutil.copyfileobj(figure_stream, figure_file)

            # And now 3d with the density plot.

            figure_3d = pyplot.figure()
            figure_3d.set_size_inches(16, 10)

            axes_3d = figure_3d.add_subplot(111, projection = '3d')
            axes_3d.set_title(name, fontsize = 14, family = 'Gentium')

            axes_3d.autoscale(tight = True)
            axes_3d.autoscale_view(tight = True)

            axes_3d.invert_xaxis()
            axes_3d.view_init(elev = 40, azim = -165)

            axes_3d.scatter(
                formant_array[:,0],
                formant_array[:,1],
                numpy.zeros(formant_array.shape[0]),
                color = 'black', s = 4, depthshade = False)

            # Plotting density.

            x_where = numpy.logical_and(x[0,:] >= y_lim[1], x[0,:] <= y_lim[0])
            y_where = numpy.logical_and(y[:,0] >= x_lim[1], y[:,0] <= x_lim[0])

            axes_3d.plot_surface(
                x[:,x_where][y_where,:],
                y[:,x_where][y_where,:],
                z[:,x_where][y_where,:],
                rstride = 8, cstride = 8,
                color = '#56B4E9', alpha = 0.5)

            axes_3d.set_xlim(y_lim)
            axes_3d.set_ylim((x_lim[1], x_lim[0]))

            grid_sum = z.sum() * (x_high - x_low) * (y_high - y_low) / n_step ** 2

            log.debug('\ndensity grid sum: {0:.6f}'.format(grid_sum))

            # Saving 3d modelling figure as a PNG stream.

            pyplot.setp(axes_3d.texts, family = 'Gentium')

            pyplot.xticks(fontsize = 14, family = 'Gentium')
            pyplot.yticks(fontsize = 14, family = 'Gentium')

            for label in axes_3d.get_zticklabels():

                label.set_fontsize(14)
                label.set_family('Gentium')

            figure_3d_stream = io.BytesIO()

            figure_3d.savefig(
                figure_3d_stream,
                bbox_inches = 'tight',
                pad_inches = 0.25,
                format = 'png')

            # Adding figure to the workbook.

            figure_3d_file_name = (
                'figure phonology distribution 3d {0} {1}.png'.format(
                perspective.client_id,
                perspective.object_id))

            worksheet_list[-1].insert_image(
                'A44', figure_3d_file_name, {'image_data': figure_3d_stream})

            if __debug_flag__:

                figure_3d_stream.seek(0)

                with open(figure_3d_file_name, 'wb') as figure_3d_file:
                    shutil.copyfileobj(figure_3d_stream, figure_3d_file)

            # Updating task status, if required.

            if task_status is not None:

                task_status.set(1,
                    int(math.floor(97.0 + 2.0 * (perspective_index + 1) / len(info_list))),
                    'Computing formant distribution density data')

        # And now we should compute pairwise total variation statistical distances between computed
        # approximations of formant distributions.

        d_ij = numpy.zeros((len(yes_index_list), len(yes_index_list)))

        for i in range(len(yes_index_list) - 1):
            for j in range(i + 1, len(yes_index_list)):

                index_i = yes_index_list[i]
                index_j = yes_index_list[j]

                perspective_i, name_i = info_list[index_i][0], info_list[index_i][2]
                perspective_j, name_j = info_list[index_j][0], info_list[index_j][2]

                z_i = grid_data_list[index_i]
                z_j = grid_data_list[index_j]

                delta_abs = numpy.abs(z_i - z_j)

                d_value = (0.5 *
                    scipy.integrate.simps(
                        scipy.integrate.simps(delta_abs, x), y[:,0]))

                # Saving and showing another statistical distance.

                d_ij[i,j] = d_value
                d_ij[j,i] = d_value

                log.debug(
                    '\nphonological_statistical_distance:'
                    '\nperspective {0} ({1}, {2}/{3}),'
                    '\nperspective {4} ({5}, {6}/{7}):'
                    '\ntotal variation distance {8:.6f}'.format(
                    index_i, name_i, perspective_i.client_id, perspective_i.object_id,
                    index_j, name_j, perspective_j.client_id, perspective_j.object_id,
                    d_value))

        # Adding distance data to the workbook.

        name_list = [info_list[i][2] for i in yes_index_list]

        worksheet_distance.set_column(0, 0, 64)

        worksheet_distance.write_row('B1', name_list)
        worksheet_distance.write_column('A2', name_list)

        for i in range(len(yes_index_list)):

            worksheet_distance.write_row(
                'B{0}'.format(i + 2),
                [round(value, 4) for value in d_ij[i,:]])

        if no_index_list:

            worksheet_distance.write_column(
                'A{0}'.format(len(yes_index_list) + 3),
                ['Insufficient formant data:'] +
                    [info_list[i][2] for i in no_index_list])

        workbook.close()

        xlsx_file_name = pathvalidate.sanitize_filename(
            'Phonological statistical distance ({0} perspectives).xlsx'.format(len(info_list)))

        if __debug_flag__:

            workbook_stream.seek(0)

            with open(xlsx_file_name, 'wb') as xlsx_file:
                shutil.copyfileobj(workbook_stream, xlsx_file)

        # Storing XLSX file with distribution comparison results.

        cur_time = time.time()

        storage_dir = os.path.join(storage['path'], 'psd', str(cur_time))

        xlsx_path = os.path.join(storage_dir, xlsx_file_name)
        os.makedirs(os.path.dirname(xlsx_path), exist_ok = True)

        workbook_stream.seek(0)

        with open(xlsx_path, 'wb') as xlsx_file:
            shutil.copyfileobj(workbook_stream, xlsx_file)

        xlsx_url = ''.join([
            storage['prefix'], storage['static_route'],
            'psd', '/', str(cur_time), '/', xlsx_file_name])

        # Finalizing task status, if required.

        if task_status is not None:
            task_status.set(1, 100, 'Finished', result_link = xlsx_url)

        return PhonologicalStatisticalDistance(triumph = True)

    @staticmethod
    def mutate(self, info, **args):
        """
        query MyQuery {
          phonological_statistical_distance(
            id_list: [[656, 3]],
            vowel_selection: false,
            chart_threshold: 1)
        }
        """

        id_list = args['id_list']
        vowel_selection = args['vowel_selection']
        chart_threshold = args['chart_threshold']

        synchronous_flag = args.get('synchronous_flag', False)

        __debug_flag__ = args.get('debug_flag', False)

        try:

            # Showing our arguments.

            log.debug(
                'phonological_statistical_distance ({0} perspectives):\n'
                'id_list: {1}\n'
                'vowel_selection: {2}\n'
                'chart_threshold: {3}\n'
                'synchronous_flag: {4}\n'
                'debug_flag: {5}\n'.format(
                len(id_list),
                id_list,
                vowel_selection,
                chart_threshold,
                synchronous_flag,
                __debug_flag__))

            locale_id = info.context.locale_id

            request = info.context.get('request')
            storage = request.registry.settings['storage']

            # Simple synchronous phonological statistical distance computation.

            if synchronous_flag:

                return PhonologicalStatisticalDistance.perform_phonological_statistical_distance(
                    id_list,
                    vowel_selection,
                    chart_threshold,
                    locale_id,
                    storage,
                    None,
                    __debug_flag__)

            # Asynchronous phonological statistical distance computation with task status setup.

            client_id = info.context.client_id

            user_id = (
                Client.get_user_by_client_id(client_id).id
                    if client_id else anonymous_userid(request))

            if __debug_flag__ and user_id != 1:

                return (

                    ResponseError(
                        message = 'Only administrator can use debug mode.'))

            task_status = TaskStatus(
                user_id,
                'Phonological statistical distance computation',
                '{0} perspectives'.format(len(id_list)),
                1)

            # Launching asynchronous phonological statistical distance computation.

            request.response.status = HTTPOk.code

            async_phonological_statistical_distance.delay(
                id_list,
                vowel_selection,
                chart_threshold,
                locale_id,
                storage,
                task_status.key,
                request.registry.settings['cache_kwargs'],
                request.registry.settings['sqlalchemy.url'],
                __debug_flag__)

            return PhonologicalStatisticalDistance(triumph = True)

        # Some unknown external exception.

        except Exception as exception:

            traceback_string = ''.join(traceback.format_exception(
                exception, exception, exception.__traceback__))[:-1]

            log.warning('phonological_statistical_distance: exception')
            log.warning(traceback_string)

            return ResponseError(message =
                'Exception:\n' + traceback_string)


class SoundAndMarkup(graphene.Mutation):

    class Arguments:
        perspective_id = LingvodocID(required = True)
        published_mode = graphene.String(required = True)

    triumph = graphene.Boolean()

    def mutate(self, info, **args):
        """
        query MyQuery {
          sound_and_markup(
            perspective_id: [657, 4])
        }
        """

        locale_id = info.context.locale_id
        request = info.context.get('request')

        gql_sound_and_markup(request, locale_id, args['perspective_id'], args['published_mode'])

        return SoundAndMarkup(triumph = True)


def save_dictionary(
    dict_id,
    dictionary_obj,
    request,
    user_id,
    locale_id,
    publish,
    sound_flag = False,
    markup_flag = False,
    f_type = 'xlsx',
    synchronous = False,
    debug_flag = False):

    my_args = dict()
    my_args["client_id"] = dict_id[0]
    my_args["object_id"] = dict_id[1]
    my_args["locale_id"] = locale_id
    my_args["storage"] = request.registry.settings["storage"]
    my_args['sqlalchemy_url'] = request.registry.settings["sqlalchemy.url"]
    try:
        gist = DBSession.query(dbTranslationGist). \
            filter_by(client_id=dictionary_obj.translation_gist_client_id,
                      object_id=dictionary_obj.translation_gist_object_id).first()
        dict_name = gist.get_translation(locale_id)

        if not synchronous:

            task_name_str = 'Saving dictionary'

            modifier_list = [
                'published only' if publish else 'with unpublished']

            if sound_flag:
                modifier_list.append('with sound')

            task_name_str += (
                ' (' + ', '.join(modifier_list) + ')')

            task = TaskStatus(user_id, task_name_str, dict_name, 4)

    except:
        raise ResponseError('bad request')
    my_args['dict_name'] = dict_name
    my_args["task_key"] = task.key if not synchronous else None
    my_args["cache_kwargs"] = request.registry.settings["cache_kwargs"]
    my_args["published"] = publish
    my_args['sound_flag'] = sound_flag
    my_args['markup_flag'] = markup_flag
    my_args['f_type'] = f_type
    my_args['__debug_flag__'] = debug_flag

    (sync_save_dictionary if synchronous else async_save_dictionary.delay)(**my_args)


class SaveDictionary(graphene.Mutation):

    class Arguments:
        id = LingvodocID(required=True)
        mode = graphene.String(required=True)
        sound_flag = graphene.Boolean()
        markup_flag = graphene.Boolean()
        f_type = graphene.String()
        synchronous = graphene.Boolean()
        debug_flag = graphene.Boolean()

    triumph = graphene.Boolean()

    @staticmethod
    def mutate(
        root,
        info,
        id,
        mode,
        sound_flag = False,
        markup_flag = False,
        f_type = 'xlsx',
        synchronous = False,
        debug_flag = False):

        request = info.context.request

        client_id = info.context.client_id
        locale_id = info.context.locale_id

        user_id = (
            Client.get_user_by_client_id(client_id).id
                if client_id else anonymous_userid(request))

        if debug_flag and user_id != 1:

            return (

                ResponseError(
                    message = 'Only administrator can use debug mode.'))

        dictionary = (

            DBSession
                .query(dbDictionary)
                .filter_by(client_id = id[0], object_id = id[1])
                .first())

        if mode == 'published':
            publish = True
        elif mode == 'all':
            publish = None
        else:
            return ResponseError(message="mode: <all|published>")

        if mode == 'all':

            for perspective in dictionary.dictionaryperspective:

                info.context.acl_check(
                    'view',
                    'lexical_entries_and_entities',
                    perspective.id)

        save_dictionary(
            id,
            dictionary,
            request,
            user_id,
            locale_id,
            publish,
            sound_flag,
            markup_flag,
            f_type,
            synchronous,
            debug_flag)

        return DownloadDictionary(triumph=True)

class SaveAllDictionaries(graphene.Mutation):

    class Arguments:
        mode = graphene.String(required=True)

    triumph = graphene.Boolean()

    @staticmethod
    # @client_id_check()
    def mutate(root, info, **args):
        request = info.context.request
        locale_id = info.context.locale_id
        mode = args['mode']
        client = DBSession.query(Client).filter_by(id=info.context.client_id).first()
        user = DBSession.query(dbUser).filter_by(id=client.user_id).first()
        user_id = user.id
        if user_id != 1:
            raise ResponseError(message="not admin")
        # counter = 0
        dictionaries = DBSession.query(dbDictionary).filter_by(marked_for_deletion=False).all()
        if mode == 'published':
            publish = True
        elif mode == 'all':
            publish = None

        else:
            raise ResponseError(message="mode: <all|published>")

        for dictionary in dictionaries:

            save_dictionary(
                [dictionary.client_id, dictionary.object_id],
                dictionary,
                request,
                user_id,
                locale_id,
                publish)

            # if not counter % 5:
            #     time.sleep(5)
            # counter += 1

        return DownloadDictionary(triumph=True)


class MoveColumn(graphene.Mutation):
    class Arguments:
        perspective_id = LingvodocID(required=True)
        from_id = LingvodocID(required=True)
        to_id = LingvodocID(required=True)

    triumph = graphene.Boolean()

    @staticmethod
    def mutate(
        root,
        info,
        perspective_id,
        from_id,
        to_id):

        request = info.context.request

        client = DBSession.query(Client).filter_by(id=info.context.client_id).first()
        user = DBSession.query(dbUser).filter_by(id=client.user_id).first()
        user_id = user.id
        client_ids = DBSession.query(Client.id).filter(Client.user_id==user_id).all()
        # if user_id != 1:
        #     raise ResponseError(message="not admin")
        # counter = 0
        perspective = DBSession.query(dbPerspective).filter_by(client_id=perspective_id[0],
                                                                         object_id=perspective_id[1],
                                                                         marked_for_deletion=False).first()
        if not perspective:
            raise ResponseError('No such perspective')
        info.context.acl_check('edit', 'perspective',
                           (perspective.client_id, perspective.object_id))

        lexes = DBSession.query(dbLexicalEntry).join(dbEntity).join(dbPublishingEntity).filter(
            dbLexicalEntry.parent_client_id == perspective_id[0],
            dbLexicalEntry.parent_object_id == perspective_id[1],
            dbLexicalEntry.marked_for_deletion == False,
            dbEntity.client_id.in_(client_ids),
            dbEntity.field_client_id == from_id[0],
            dbEntity.field_object_id == from_id[1],
            dbEntity.marked_for_deletion == False,
            dbPublishingEntity.accepted == True).all()

        for lex in lexes:
            entities = DBSession.query(dbEntity).join(dbPublishingEntity).filter(
                dbEntity.client_id.in_(client_ids),
                dbEntity.parent_client_id == lex.client_id,
                dbEntity.parent_object_id == lex.object_id,
                dbEntity.field_client_id == from_id[0],
                dbEntity.field_object_id == from_id[1],
                dbEntity.marked_for_deletion == False,
                dbPublishingEntity.accepted == True).all()
            for entity in entities:
                existing = DBSession.query(dbEntity).join(dbPublishingEntity).filter(
                    dbEntity.parent_client_id == lex.client_id,
                    dbEntity.parent_object_id == lex.object_id,
                    dbEntity.field_client_id == to_id[0],
                    dbEntity.field_object_id == to_id[1],
                    dbEntity.marked_for_deletion == False,
                    dbPublishingEntity.accepted == True,
                    dbEntity.content == entity.content).first()
                if not existing:
                    self_id = None
                    if entity.self_client_id and entity.self_object_id:
                        self_id = [entity.self_client_id, entity.self_object_id]
                    link_id = None
                    if entity.link_client_id and entity.link_object_id:
                        link_id = [entity.link_client_id, entity.link_object_id]
                    create_entity(id=[client.id, None],
                                  parent_id=[lex.client_id, lex.object_id],
                                  field_id=to_id,
                                  self_id=self_id,
                                  additional_metadata=entity.additional_metadata,
                                  link_id=link_id,
                                  locale_id=entity.locale_id,
                                  filename=None,
                                  content=entity.content,
                                  registry=None,
                                  request=request,
                                  )
                del_object(entity, "move_column", info.context.client_id)

        return MoveColumn(triumph=True)

class AddRolesBulk(graphene.Mutation):
    """
    Adds specified roles for a specified user for all dictionaries and perspectives of a specified language,
    maybe with all its descendants.

    mutation addRolesBulk {
      add_roles_bulk(user_id: 5, language_id: [508, 36]) {
        triumph } }
    """

    class Arguments:
        user_id = graphene.Int(required=True)
        language_id = LingvodocID(required=True)

    language_count = graphene.Int()
    dictionary_count = graphene.Int()
    perspective_count = graphene.Int()

    @staticmethod
    def mutate(root, info, **args):

        user_id = args.get('user_id')
        language_id = args.get('language_id')

        request_client_id = info.context.client_id
        request_user = Client.get_user_by_client_id(request_client_id)

        if request_user.id != 1:
            raise ResponseError('Only administrator can perform bulk roles additions.')

        user = DBSession.query(dbUser).filter_by(id = user_id).first()

        if not user:
            raise ResponseError('No user with id {0}'.format(user_id))

        # Getting permission groups info.

        dictionary_group_list = [row[0] for row in DBSession.query(
            dbBaseGroup.id).filter_by(dictionary_default = True).all()]

        perspective_group_list = [row[0] for row in DBSession.query(
            dbBaseGroup.id).filter_by(perspective_default = True).all()]

        log.debug(
            'add_roles_bulk (user_id: {0}, language_id: {1}/{2}):'
            '\ndictionary_group_list: {3}\nperspective_group_list: {4}'.format(
                user_id,
                language_id[0], language_id[1],
                dictionary_group_list,
                perspective_group_list))

        def process_language(language):
            """
            Recursively adds required permissions for a specified user for all dictionaries and perspectives
            of a given language and its descendants.
            """

            dictionary_list = DBSession.query(dbDictionary).filter_by(
                parent_client_id = language.client_id,
                parent_object_id = language.object_id,
                marked_for_deletion = False).all()

            language_count = 1
            dictionary_count = len(dictionary_list)
            perspective_count = 0

            # All dictionaries of the language.

            for dictionary in dictionary_list:

                for basegroup_id in dictionary_group_list:

                    edit_role(dictionary, user_id, basegroup_id, request_client_id,
                        dictionary_default = True, action = 'add')

                perspective_list = DBSession.query(dbPerspective).filter_by(
                    parent_client_id = dictionary.client_id,
                    parent_object_id = dictionary.object_id,
                    marked_for_deletion = False).all()

                perspective_count += len(perspective_list)

                # All perspectives of each dictionary.

                for perspective in perspective_list:

                    for basegroup_id in perspective_group_list:

                        edit_role(perspective, user_id, basegroup_id, request_client_id,
                            perspective_default = True, action = 'add')

            language_list = DBSession.query(dbLanguage).filter_by(
                parent_client_id = language.client_id,
                parent_object_id = language.object_id,
                marked_for_deletion = False).all()

            # All child languages.

            for child_language in language_list:
                l_count, d_count, p_count = process_language(child_language)

                language_count += l_count
                dictionary_count += d_count
                perspective_count += p_count

            return language_count, dictionary_count, perspective_count

        # Adding required permissions.

        language = DBSession.query(dbLanguage).filter_by(
            marked_for_deletion = False,
            client_id = language_id[0],
            object_id = language_id[1]).first()

        if not language:
            raise ResponseError('Language {0}/{1} not found'.format(*language_id))

        language_count, dictionary_count, perspective_count = process_language(language)

        return AddRolesBulk(
            language_count = language_count,
            dictionary_count = dictionary_count,
            perspective_count = perspective_count)


class NewUnstructuredData(graphene.Mutation):
    """
    Creates new unstructured data entry, returns its id.
    """

    class Arguments:

        data = ObjectVal(required = True)
        metadata = ObjectVal()

    triumph = graphene.Boolean()
    id = graphene.String()

    @staticmethod
    def get_random_unstructured_data_id():
        """
        Returns reasonably short random unused base59 string unstructured data id.
        """

        # Not using 'l', 'I' and 'O' which in some cases can cause confusion.

        base59_str = '0123456789abcdefghijkmnopqrstuvwxyzABCDEFGHJKLMNPQRSTUVWXYZ'

        rng = (
            random.Random(
                int(time.time() * 1000000000)))

        id_count = DBSession.query(dbUnstructuredData).count()

        id_length = 5

        id_length_limit = (
            5 if id_count <= 0 else
            min(5, int(math.ceil(math.log(2 * id_count, 59)))))

        def random_id_generator(how_many):

            nonlocal id_length

            for i in range(how_many):

                yield (

                    ''.join(
                        rng.choice(base59_str)
                        for i in range(id_length)))

                if id_length < id_length_limit:
                    id_length += 1

        sql_str = ('''

            select id
            from (values {}) T(id)

            where id not in (
              select id from unstructured_data)

            order by length(id)
            limit 1;

            ''')

        def get_another_query_str():

            return (

                sql_str.format(

                    ', '.join(
                        '(\'{}\')'.format(random_id)
                        for random_id in random_id_generator(8))))

        result = (

            DBSession
                .execute(get_another_query_str())
                .first())

        while not result:

            result = (

                DBSession
                    .execute(get_another_query_str())
                    .first())

        return result[0]

    @staticmethod
    def mutate(root, info, **args):

        try:

            client_id = info.context.client_id
            client = DBSession.query(Client).filter_by(id = client_id).first()

            if not client:
                return ResponseError('Only registered users can create unstructured data entries.')

            data = args.get('data')
            metadata = args.get('metadata')

            log.debug(
                '\nnew_unstructured_data'
                '\nclient_id: {}'
                '\ndata:\n{}'
                '\nmetadata:\n{}'.format(
                    client_id,
                    pprint.pformat(data, width = 144),
                    pprint.pformat(metadata, width = 144)))

            id_str = NewUnstructuredData.get_random_unstructured_data_id()

            unstructured_data = (

                dbUnstructuredData(
                    id = id_str,
                    client_id = client_id,
                    data = data,
                    additional_metadata = metadata))

            DBSession.add(unstructured_data)

            return (

                NewUnstructuredData(
                    id = id_str,
                    triumph = True))

        except Exception as exception:

            traceback_string = (

                ''.join(
                    traceback.format_exception(
                        exception, exception, exception.__traceback__))[:-1])

            log.warning('new_unstructured_data: exception')
            log.warning(traceback_string)

            return (

                ResponseError(
                    'Exception:\n' + traceback_string))


class Docx2Eaf(graphene.Mutation):
    """
    Tries to convert a table-containing .docx to .eaf.

    Example:

      curl 'http://localhost:6543/graphql'
        -H 'Content-Type: multipart/form-data'
        -H 'Cookie: locale_id=2; auth_tkt=$TOKEN; client_id=$ID'
        -F operations='{
          "query":
            "mutation docx2eaf(
              $docxFile: Upload,
              $separateFlag: Boolean)
            {
              docx2eaf(
                docx_file: $docxFile,
                separate_flag: $separateFlag,
                debug_flag: true)
              {
                triumph
                eaf_url
                alignment_url
                check_txt_url
                check_docx_url
                message
              }
            }",
          "variables": {
            "docxFile": null,
            "separateFlag": false
          }}'
        -F map='{ "1": ["variables.docx_file"] }'
        -F 1=@"/root/lingvodoc-extra/Чертыкова_Беседы_14.09.2019.docx"
  
      Set $TOKEN and $ID to valid admin user authentication info.
  
      To use in a shell, join into a single line or add escaping backslashes at the end of the lines.
    """

    class Arguments:

        docx_file = Upload()
        separate_flag = graphene.Boolean()
        all_tables_flag = graphene.Boolean()
        no_header_flag = graphene.Boolean()
        no_parsing_flag = graphene.Boolean()
        debug_flag = graphene.Boolean()

    triumph = graphene.Boolean()

    eaf_url = graphene.String()
    alignment_url = graphene.String()
    check_txt_url = graphene.String()
    check_docx_url = graphene.String()

    message = graphene.String()

    @staticmethod
    def mutate(root, info, **args):

        try:

            client_id = info.context.client_id
            client = DBSession.query(Client).filter_by(id = client_id).first()

            if not client:

                return (

                    Docx2Eaf(
                        triumph = False,
                        message = 'Only registered users can convert .docx to .eaf.'))

            request = info.context.request

            if '1' not in request.POST:
                return ResponseError('.docx file is required.')

            multipart = request.POST.pop('1')

            docx_file_name = multipart.filename
            docx_file = multipart.file

            separate_flag = args.get('separate_flag', False)
            all_tables_flag = args.get('all_tables_flag', False)
            no_header_flag = args.get('no_header_flag', False)
            no_parsing_flag = args.get('no_parsing_flag', False)

            __debug_flag__ = args.get('debug_flag', False)

            if __debug_flag__ and client.user_id != 1:

                return (

                    ResponseError(
                        message = 'Only administrator can use debug mode.'))

            log.debug(
                '\n{}\n{}'.format(
                    docx_file_name,
                    type(docx_file)))

            if __debug_flag__:

                with open('docx2eaf_input.docx', 'wb') as input_file:

                    shutil.copyfileobj(docx_file, input_file)
                    docx_file.seek(0)

            url_list = []

            with tempfile.TemporaryDirectory() as tmp_dir_path:

                tmp_docx_file_path = (
                    os.path.join(tmp_dir_path, 'docx2eaf_input.docx'))

                tmp_eaf_file_path = (
                    os.path.join(tmp_dir_path, 'docx2eaf_output.eaf'))

                tmp_check_txt_file_path = (
                    os.path.join(tmp_dir_path, 'docx2eaf_check.txt'))

                tmp_check_docx_file_path = (
                    os.path.join(tmp_dir_path, 'docx2eaf_check.docx'))

                with open(tmp_docx_file_path, 'wb') as tmp_docx_file:
                    shutil.copyfileobj(docx_file, tmp_docx_file)

                docx_import.docx2eaf(
                    tmp_docx_file_path,
                    tmp_eaf_file_path,
                    separate_by_paragraphs_flag = separate_flag,
                    modify_docx_flag = True,
                    all_tables_flag = all_tables_flag,
                    no_header_flag = no_header_flag,
                    no_parsing_flag = no_parsing_flag,
                    check_file_path = tmp_check_txt_file_path,
                    check_docx_file_path = tmp_check_docx_file_path,
                    __debug_flag__ = __debug_flag__)

                # Saving local copies, if required.

                if __debug_flag__:

                    shutil.copyfile(tmp_eaf_file_path, 'docx2eaf_output.eaf')
                    shutil.copyfile(tmp_check_txt_file_path, 'docx2eaf_check.txt')

                    if not separate_flag and not all_tables_flag:
                        shutil.copyfile(tmp_check_docx_file_path, 'docx2eaf_check.docx')

                # Saving processed files.

                storage = (
                    request.registry.settings['storage'])

                storage_temporary = storage['temporary']

                host = storage_temporary['host']
                bucket = storage_temporary['bucket']

                minio_client = (

                    minio.Minio(
                        host,
                        access_key = storage_temporary['access_key'],
                        secret_key = storage_temporary['secret_key'],
                        secure = True))

                current_time = time.time()

                input_file_name = (

                    pathvalidate.sanitize_filename(
                        os.path.splitext(os.path.basename(docx_file_name))[0]))

                for file_path, suffix in (

                    (tmp_eaf_file_path, '.eaf'),
                    (tmp_docx_file_path, ' alignment.docx'),
                    (tmp_check_txt_file_path, ' check.txt'),
                    (tmp_check_docx_file_path, ' check.docx')):

                    if ((separate_flag or all_tables_flag) and
                        (suffix == ' check.docx' or suffix == ' alignment.docx')):

                        url_list.append(None)
                        continue

                    object_name = (

                        storage_temporary['prefix'] +

                        '/'.join((
                            'docx2eaf',
                            '{:.6f}'.format(current_time),
                            input_file_name + suffix)))

                    (etag, version_id) = (

                        minio_client.fput_object(
                            bucket,
                            object_name,
                            file_path))

                    url = (

                        '/'.join((
                            'https:/',
                            host,
                            bucket,
                            object_name)))

                    log.debug(
                        '\nobject_name:\n{}'
                        '\netag:\n{}'
                        '\nversion_id:\n{}'
                        '\nurl:\n{}'.format(
                            object_name,
                            etag,
                            version_id,
                            url))

                    url_list.append(url)

                log.debug(
                    '\nurl_list:\n' +
                    pprint.pformat(url_list, width = 192))

            return (

                Docx2Eaf(
                    triumph = True,
                    eaf_url = url_list[0],
                    alignment_url = url_list[1],
                    check_txt_url = url_list[2],
                    check_docx_url = url_list[3]))

        except docx_import.Docx2EafError as exception:

            return (

                Docx2Eaf(
                    triumph = False,
                    message = exception.args[0]))

        except Exception as exception:

            traceback_string = (

                ''.join(
                    traceback.format_exception(
                        exception, exception, exception.__traceback__))[:-1])

            log.warning('docx2eaf: exception')
            log.warning(traceback_string)

            return (

                ResponseError(
                    'Exception:\n' + traceback_string))


class Docx2Xlsx(graphene.Mutation):
    """
    Tries to convert a table-containing .docx to .xlsx.

    Example:

      curl 'http://localhost:6543/graphql'
        -H 'Content-Type: multipart/form-data'
        -H 'Cookie: locale_id=2; auth_tkt=$TOKEN; client_id=$ID'
        -F operations='{
          "query":
            "mutation docx2xlsx(
              $docxFile: Upload,
              $separateFlag: Boolean)
            {
              docx2xlsx(
                docx_file: $docxFile,
                separate_flag: $separateFlag,
                debug_flag: true)
              {
                triumph
                xlsx_url
                message
              }
            }",
          "variables": {
            "docxFile": null,
            "separateFlag": false
          }}'
        -F map='{ "1": ["variables.docx_file"] }'
        -F 1=@"/root/lingvodoc-extra/Чертыкова_Беседы_14.09.2019.docx"
  
      Set $TOKEN and $ID to valid admin user authentication info.
  
      To use in a shell, join into a single line or add escaping backslashes at the end of the lines.
    """

    class Arguments:

        docx_file = Upload()
        separate_flag = graphene.Boolean()
        debug_flag = graphene.Boolean()

    triumph = graphene.Boolean()

    xlsx_url = graphene.String()
    message = graphene.String()

    @staticmethod
    def mutate(root, info, **args):

        try:
            client_id = info.context.client_id
            client = DBSession.query(Client).filter_by(id = client_id).first()

            if not client:
                return (
                    Docx2Xlsx(
                        triumph = False,
                        message = 'Only registered users can convert .docx to .xlsx.'))

            request = info.context.request

            if '1' not in request.POST:
                return ResponseError('.docx file is required.')

            multipart = request.POST.pop('1')

            docx_file_name = multipart.filename
            docx_file = multipart.file

            separate_flag = args.get('separate_flag', False)
            __debug_flag__ = args.get('debug_flag', False)

            if __debug_flag__ and client.user_id != 1:
                return (
                    ResponseError(
                        message = 'Only administrator can use debug mode.'))

            log.debug(
                '\n{}\n{}'.format(
                    docx_file_name,
                    type(docx_file)))

            if __debug_flag__:
                with open('docx2xlsx_input.docx', 'wb') as input_file:
                    shutil.copyfileobj(docx_file, input_file)
                    docx_file.seek(0)

            with tempfile.TemporaryDirectory() as tmp_dir_path:

                tmp_docx_file_path = (
                    os.path.join(tmp_dir_path, 'docx2xlsx_input.docx'))

                tmp_xlsx_file_path = (
                    os.path.join(tmp_dir_path, 'docx2xlsx_output.xlsx'))

                with open(tmp_docx_file_path, 'wb') as tmp_docx_file:
                    shutil.copyfileobj(docx_file, tmp_docx_file)

                input_file_name = (
                    pathvalidate.sanitize_filename(
                        os.path.splitext(os.path.basename(docx_file_name))[0]))

                result_dict = docx_to_xlsx.get_entries(tmp_docx_file_path, __debug_flag__)
                docx_to_xlsx.write_xlsx(result_dict, tmp_xlsx_file_path, input_file_name, separate_flag)

                # Saving local copies, if required.

                if __debug_flag__:
                    shutil.copyfile(tmp_xlsx_file_path, 'docx2xlsx_output.xlsx')

                # Saving processed files.
                storage = (
                    request.registry.settings['storage'])
                storage_temporary = storage['temporary']
                host = storage_temporary['host']
                bucket = storage_temporary['bucket']

                minio_client = (
                    minio.Minio(
                        host,
                        access_key = storage_temporary['access_key'],
                        secret_key = storage_temporary['secret_key'],
                        secure = True))

                current_time = time.time()

                object_name = (
                    storage_temporary['prefix'] +
                    '/'.join((
                        'docx2xlsx',
                        '{:.6f}'.format(current_time),
                        input_file_name + '.xlsx')))

                (etag, version_id) = (
                    minio_client.fput_object(
                        bucket,
                        object_name,
                        tmp_xlsx_file_path))

                url = (
                    '/'.join((
                        'https:/',
                        host,
                        bucket,
                        object_name)))

                log.debug(
                    '\nobject_name:\n{}'
                    '\netag:\n{}'
                    '\nversion_id:\n{}'
                    '\nurl:\n{}'.format(
                        object_name,
                        etag,
                        version_id,
                        url))

            return (
                Docx2Xlsx(
                    triumph = True,
                    xlsx_url = url))

        except docx_to_xlsx.Docx2XlsxError as exception:
            return (
                Docx2Xlsx(
                    triumph = False,
                    message = exception.args[0]))

        except Exception as exception:
            traceback_string = (
                ''.join(
                    traceback.format_exception(
                        exception, exception, exception.__traceback__))[:-1])

            log.warning('docx2xlsx: exception')
            log.warning(traceback_string)

            return (
                ResponseError(
                    'Exception:\n' + traceback_string))


class CognatesSummary(graphene.Mutation):
    """
    Gets 'languages -< dictionaries -< perspectives -< etymologies' tree
    using offset and limit for languages amount in output result
    """

    class Arguments:

        only_in_toc = graphene.Boolean(required=True)
        group = graphene.String()
        title = graphene.String()
        limit = graphene.Int()
        offset = graphene.Int()
        debug_flag = graphene.Boolean()

    triumph = graphene.Boolean()
    message = graphene.String()

    @staticmethod
    def mutate(root, info, **args):

        try:
            client_id = (
                info.context.client_id)

            user = (
                Client.get_user_by_client_id(client_id))

            if not user:
                return (
                    CognatesSummary(
                        triumph=False,
                        message='Only registered users can get cognates summary'))

            only_in_toc = args.get('only_in_toc', False)
            group = args.get('group', None)
            title = args.get('title', None)
            offset = args.get('offset', 0)
            limit = args.get('limit', 10)
            debug_flag = args.get('debug_flag', False)

            if debug_flag and user.id != 1:
                return (
                    CognatesSummary(
                        triumph=False,
                        message='Only administrator can use debug mode'))

            task_status = TaskStatus(user.id, 'Cognates summary computation', '', 3)
            request = info.context.request

            list_cognates.async_get_json_tree.delay(
                task_status.key,
                request.registry.settings['sqlalchemy.url'],
                request.registry.settings['cache_kwargs'],
                request.registry.settings['storage'],
                only_in_toc,
                group,
                title,
                offset,
                limit,
                debug_flag)

            return CognatesSummary(triumph=True)

        except Exception as exception:

            traceback_string = (
                ''.join(
                    traceback.format_exception(
                        exception, exception, exception.__traceback__))[:-1])

            log.warning('cognates_summary: exception')
            log.warning(traceback_string)

            return (
                ResponseError(
                    'Exception:\n' + traceback_string))


class BidirectionalLinks(graphene.Mutation):
    """
    Ensures that links between Lexical Entries and Paradigms perspectives of each specified dictionary are
    bidirectional.

    Example:

      curl 'http://localhost:6543/graphql' \
        -H 'Content-Type: application/json' \
        -H 'Cookie: locale_id=2; auth_tkt=$TOKEN; client_id=$ID' \
        --data-raw '{ \
          "operationName": "bidirectionalLinks", \
          "variables":{"dictionaryIdList":[[4755,333300]]}, \
          "query": \
            "mutation bidirectionalLinks($dictionaryIdList: [LingvodocID]!) { \
              bidirectional_links(dictionary_id_list: $dictionaryIdList, debug_flag: true) { \
                triumph }}"}'
  
      Set $TOKEN and $ID to valid admin user authentication info.
  
      #! Remove any newline after --data-raw to call this from Ubuntu.
      #! Be careful, used token may be not actual after database restart.
    """

    class Arguments:

        dictionary_id_list = graphene.List(LingvodocID, required = True)
        debug_flag = graphene.Boolean()

    triumph = graphene.Boolean()

    @staticmethod
    def mutate(
        root,
        info,
        dictionary_id_list,
        debug_flag = False):

        try:
            client_id = info.context.client_id
            log.debug(f"client_id: {client_id}")

            client = DBSession.query(Client).filter_by(id = client_id).first()

            if not client or client.user_id != 1:
                return ResponseError('Only administrator can fix links using request.')

            backref_fid = (
                get_id_to_field_dict()['Backref'])

            # Processing each dictionary.

            for dictionary_id in dictionary_id_list:

                log.debug(
                    f'\ndictionary_id: {dictionary_id}')

                dictionary = (

                    DBSession
                        .query(dbDictionary)
                        .get((dictionary_id[1], dictionary_id[0])))

                if dictionary is None:
                    return (
                        ResponseError(
                            f'Dictionary {dictionary_id} does not exist.'))

                if dictionary.marked_for_deletion:
                    return (
                        ResponseError(
                            f'Dictionary {dictionary_id} is deleted.'))

                if debug_flag:

                    log.debug(
                        f'\ndictionary.get_translation():\n{repr(dictionary.get_translation())}')

                # Processing all linked lexical entries of all perspectives of the dictionary, intentionally
                # counting even deleted links.

                link_list = (

                    DBSession

                        .query(
                            dbPerspective.client_id,
                            dbPerspective.object_id,
                            dbLexicalEntry.client_id,
                            dbLexicalEntry.object_id,
                            dbEntity.client_id,
                            dbEntity.object_id,
                            dbEntity.link_client_id,
                            dbEntity.link_object_id,
                            dbEntity.marked_for_deletion,
                            dbPublishingEntity.published,
                            dbPublishingEntity.accepted)

                        .filter(
                            dbPerspective.parent_client_id == dictionary_id[0],
                            dbPerspective.parent_object_id == dictionary_id[1],
                            dbPerspective.marked_for_deletion == False,
                            dbLexicalEntry.parent_client_id == dbPerspective.client_id,
                            dbLexicalEntry.parent_object_id == dbPerspective.object_id,
                            dbLexicalEntry.marked_for_deletion == False,
                            dbEntity.parent_client_id == dbLexicalEntry.client_id,
                            dbEntity.parent_object_id == dbLexicalEntry.object_id,
                            dbEntity.field_client_id == backref_fid[0],
                            dbEntity.field_object_id == backref_fid[1],
                            dbPublishingEntity.client_id == dbEntity.client_id,
                            dbPublishingEntity.object_id == dbEntity.object_id)

                        .all())

                log.debug(
                    f'\nlen(link_list): {len(link_list)}')

                link_list = [

                    ((p_cid, p_oid),
                        (l_cid, l_oid),
                        (e_cid, e_oid),
                        (k_cid, k_oid),
                        (d, p, a))

                    for p_cid, p_oid, l_cid, l_oid, e_cid, e_oid, k_cid, k_oid, d, p, a in link_list]

                # Compiling info of existing links.

                link_dict = {
                    (entry_id, link_id): perspective_id
                    for perspective_id, entry_id, _, link_id, _ in link_list}

                entry_list = (

                    DBSession

                        .query(
                            dbPerspective.client_id,
                            dbPerspective.object_id,
                            dbLexicalEntry.client_id,
                            dbLexicalEntry.object_id)

                        .filter(
                            dbPerspective.parent_client_id == dictionary_id[0],
                            dbPerspective.parent_object_id == dictionary_id[1],
                            dbPerspective.marked_for_deletion == False,
                            dbLexicalEntry.parent_client_id == dbPerspective.client_id,
                            dbLexicalEntry.parent_object_id == dbPerspective.object_id,
                            dbLexicalEntry.marked_for_deletion == False)

                        .all())

                entry_dict = {
                    (l_cid, l_oid): (p_cid, p_oid)
                    for p_cid, p_oid, l_cid, l_oid in entry_list}

                log.debug(
                    f'\nlen(link_dict): {len(link_dict)}'
                    f'\nlen(entry_dict): {len(entry_dict)}')

                if debug_flag:

                    perspective_list = (

                        DBSession

                            .query(
                                dbPerspective)

                            .filter(
                                dbPerspective.parent_client_id == dictionary_id[0],
                                dbPerspective.parent_object_id == dictionary_id[1],
                                dbPerspective.marked_for_deletion == False)

                            .all())

                    translation_dict = {
                        perspective.id: perspective.get_translation()
                        for perspective in perspective_list}

                fixed = 0
                for (from_id, to_id) in link_dict:

                    if (to_id, from_id) in link_dict:
                        continue

                    # if the linked lexical entry is deleted
                    if  to_id not in entry_dict:
                        continue

                    if debug_flag:

                        log.debug(f'\nfrom perspective: {translation_dict[entry_dict[from_id]]}'
                                  f'\nto perspective: {translation_dict[entry_dict[to_id]]}')

                    create_entity(id=[entry_dict[to_id][0], None],
                                  parent_id=to_id,
                                  field_id=backref_fid,
                                  link_id=from_id)
                    fixed += 1

                print(f'\nTotal fixed links: {fixed}')

        except Exception as exception:

            traceback_string = (

                ''.join(
                    traceback.format_exception(
                        exception, exception, exception.__traceback__))[:-1])

            log.warning('bidirectional_links: exception')
            log.warning(traceback_string)

            return (

                ResponseError(
                    'Exception:\n' + traceback_string))


class ReorderColumns(graphene.Mutation):
    """
    curl 'https://lingvodoc.ispras.ru/api/graphql' \
    -H 'Content-Type: application/json' \
    -H 'Cookie: locale_id=2; auth_tkt=$TOKEN!userid_type:int; client_id=$ID' \
    --data-raw '{ "operationName": "reorder_columns", "variables": {"perspectiveId": null}, \
    "query": "mutation reorder_columns($perspectiveId: LingvodocID) \
    { reorder_columns(perspective_id: $perspectiveId, debug_flag: true) { triumph }}"}'

    #! set perspectiveId to concrete LingvodocID for one perspective's positions reordering,
    #! otherwise all perspectives with duplicated columns positions will be processed
    """

    class Arguments:

        perspective_id = LingvodocID()
        debug_flag = graphene.Boolean()

    triumph = graphene.Boolean()

    @staticmethod
    def mutate(
        root,
        info,
        perspective_id = None,
        debug_flag = False):

        try:
            client_id = info.context.client_id
            if debug_flag:
                log.debug(f"client_id: {client_id}")

            client = DBSession.query(Client).filter_by(id = client_id).first()

            if not client or client.user_id != 1:
                return ResponseError('Only administrator can reorder columns.')

            perspective_list = [perspective_id] if perspective_id else (
                DBSession
                    .query(
                        dbColumn.parent_client_id,
                        dbColumn.parent_object_id)
                    .filter_by(
                        marked_for_deletion= 'false')
                    .group_by(
                        dbColumn.parent_client_id,
                        dbColumn.parent_object_id)
                    .having(
                        func.count(dbColumn.position) -
                        func.count(func.distinct(dbColumn.position)) > 0)
                    .all()) or []

            for per_id in perspective_list:
                column_list = (
                    DBSession
                        .query(dbColumn)
                        .filter_by(
                            parent_id = per_id,
                            marked_for_deletion = 'false')
                        .order_by(
                            dbColumn.position,
                            dbColumn.client_id,
                            dbColumn.object_id,
                            dbColumn.created_at)
                        .all()) or []

                if debug_flag:
                    log.debug(f'Processing {per_id} ...')

                for i in range(1, len(column_list)):
                    if column_list[i].position <= column_list[i-1].position:
                        column_list[i].position = column_list[i-1].position + 1

                        if debug_flag:
                            log.debug(f">> Changed {i}'th")

            if debug_flag:
                log.debug(f"Total processed {len(perspective_list)} perspectives.")

            return ReorderColumns(triumph = True)

        except Exception as exception:

            traceback_string = (

                ''.join(
                    traceback.format_exception(
                        exception, exception, exception.__traceback__))[:-1])

            log.warning('columns_reordering: exception')
            log.warning(traceback_string)

            return (

                ResponseError(
                    'Exception:\n' + traceback_string))


class Tsakorpus(graphene.Mutation):
    class Arguments:

        perspective_id = LingvodocID()
        debug_flag = graphene.Boolean()

    triumph = graphene.Boolean()

    parser_language_dict = {
        'timarkh_udm': 'udmurt',
        'timarkh_erzya': 'erzya',
        'timarkh_meadow_mari': 'meadow_mari',
        'timarkh_moksha': 'moksha',
        'timarkh_komi_zyryan': 'komi_zyryan'}

    tag_category_dict = {
        '1': 'pers',
        '2': 'pers',
        '3': 'pers',
        'A': 'pos',
        'ADJ': 'pos',
        'ADJPRO': 'pos',
        'POST': 'pos',
        'PART': 'pos',
        'INTRJ': 'pos',
        'INTERJ': 'pos',
        'ADV': 'pos',
        'ADVPRO': 'pos',
        'CONJ': 'pos',
        'CNJ': 'pos',
        'N': 'pos',
        'NUM': 'pos',
        'PREP': 'pos',
        'PRO': 'pos',
        'IMIT': 'pos',
        'PARENTH': 'pos',
        'PREDIC': 'pos',
        'PRED': 'pos',
        'V': 'pos',
        'rel_n': 'nType',
        'rel_adj': 'adjType',
        'topn': 'nType',
        'famn': 'nType',
        'anim': 'nType',
        'hum': 'nType',
        'supernat': 'nType',
        'transport': 'nType',
        'body': 'nType',
        'persn': 'nType',
        'patrn': 'nType',
        'vn': 'nType',
        'nloc': 'nType',
        'obl': 'stemType',
        'dem': 'proType',
        'pers': 'proType',
        'refl': 'proType',
        'abl': 'case',
        'acc': 'case',
        'acc0': 'case',
        'gen': 'case',
        'gen2': 'case',
        'nom': 'case',
        'dat': 'case',
        'ins': 'case',
        'loc': 'case',
        'ill': 'case',
        'el': 'case',
        'egr': 'case',
        'term': 'case',
        'car': 'case',
        'app': 'case',
        'rcs': 'case',
        'prol': 'case',
        'adv': 'case',
        'dms': 'case',
        'prs': 'tense',
        'pst': 'tense',
        'pst2': 'tense',
        'fut': 'tense',
        'pl': 'number',
        'sg': 'number',
        'ptcp': 'v_form',
        'cvb': 'v_form',
        'neg': 'v_form',
        'perf': 'ptcp_form',
        'deb': 'v_form',
        'lim': 'cnv_form',
        'simult': 'cnv_form',
        'res': 'cnv_form',
        'mon': 'cnv_form',
        'dur': 'cnv_form',
        'imp': 'v_form',
        'prh': 'v_form',
        'inf': 'v_form',
        'I': 'conjtype',
        'II': 'conjtype',
        'iter': 'v_deriv',
        'act': 'ptcp_form',
        'pass': 'ptcp_form',
        'detr': 'v_deriv',
        'caus': 'v_deriv',
        'attr': 'attributivizer',
        'attr_o': 'attributivizer',
        'attr_em': 'attributivizer',
        'attr_tem': 'attributivizer',
        'intensifier': 'case_c',
        'poss_comp': 'case_c',
        'case_comp': 'n_deriv',
        '1sg': 'poss',
        '2sg': 'poss',
        '3sg': 'poss',
        '1pl': 'poss',
        '2pl': 'poss',
        '3pl': 'poss',
        'comp': 'comp',
        'ord': 'num_deriv',
        'coll': 'num_deriv',
        'distr': 'num_deriv',
        'subj': 'mood',
        'abbr': 'additional',
        'rus': 'additional',
        'oblin': 'n_deriv',
        'intr': 'v_subcat',
        'tr': 'v_subcat',
        'with_instr': 'v_subcat',
        'with_dat': 'v_subcat',
        'with_el': 'v_subcat',
        'with_ill': 'v_subcat',
        'with_gen2': 'v_subcat',
        'impers': 'v_subcat',
        'with_inf': 'v_subcat'}

    config_dict = {
        'accidental_word_fields': [],
        'all_language_search_enabled': True,
        'ambiguous_analyses': True,
        'author_metafield': '',
        'auto_switch_tiers': {},
        'citation': 'Corpus of Udmurt texts (2021)',
        'context_header_rtl': False,
        'corpus_name': 'test_corpus_2',
        'debug': True,
        'default_locale': 'en',
        'default_values': {},
        'default_view': 'standard',
        'detect_lemma_queries': False,
        'display_freq_rank': True,
        'elastic_url': 'http://127.0.0.1:9200',
        'fulltext_page_size': 100,
        'fulltext_search_enabled': True,
        'fulltext_view_enabled': False,
        'generate_dictionary': False,
        'gloss_search_enabled': True,
        'images': False,
        'input_format': 'json-gzip',
        'input_methods': [],
        'integer_meta_fields': [],
        'interface_languages': ['en', 'ru'],
        'keep_lemma_order': False,
        'keyboards': {},
        'lang_props': {
            'udmurt': {
                'gloss_shortcuts': {
                    'case': '(nom|gen2?|acc|dat|ins|loc|ill|el|egr|term|prol|car|app|rcs|adv|dms)(\\.(sg|pl))?',
                    'poss': '(p\\.[123](sg|pl)(\\.acc)?)',
                    'p.acc': '(p\\.[123](sg|pl)\\.acc)',
                    'tense': '(pst2?|prs|fut|evid)(\\.[123](sg|pl)?)?',
                    'pst_any': '(pst(\\(1sg\\)|\\.[123](sg|pl))?)',
                    'fut_any': '(fut(\\(1sg\\)|\\.[123](sg|pl))?)',
                    'deriv': '(detr|iter|caus)',
                    'pers': '[123](sg|pl)?'},
                'gr_fields_order': [
                    'nType', 'adjType', 'proType', 'stemType', 'num_deriv', 'v_form', 'attributivizer',
                    'tense', 'person', 'number', 'case', 'poss'],
                'gramm_shortcuts': {
                    'NV': '(N|V)',
                    'V_nderiv': 'V,~(caus|detr|iter)'}}},
        'languages': [],
        'line_plot_meta': ['year'],
        'max_context_expand': 5,
        'max_docs_retrieve': 9999,
        'max_hits_retrieve': 10000,
        'max_words_in_sentence': 40,
        'media': False,
        'media_length': 60,
        'media_youtube': False,
        'multiple_choice_fields': {},
        'negative_search_enabled': True,
        'query_log': False,
        'query_timeout': 60,
        'rtl_languages': [],
        'search_meta': {
            'columns': [],
            'stat_options': []},
        'search_remove_whitespaces': False,
        'sentence_meta': [],
        'start_page_url': '',
        'transliterations': [],
        'video': False,
        'viewable_meta': [],
        'wf_lowercase': True,
        'word_fields': ['trans_ru'],
        'word_search_display_gr': True,
        'word_table_fields': [],
        'year_sort_enabled': False}

    gloss_column_list = [

        [
            {'type': 'gloss', 'value': 'CASE', 'tooltip': 'Any&nbsp;case&nbsp;marker'},
            {'type': 'gloss', 'value': 'NOM', 'tooltip': 'Nominative case'},
            {'type': 'gloss', 'value': 'ACC', 'tooltip': 'Marked accusative case'},
            {'type': 'gloss', 'value': 'ACC.PL', 'tooltip': 'Accusative (plural) case'},
            {'type': 'gloss', 'value': 'GEN', 'tooltip': 'Genitive case'},
            {'type': 'gloss', 'value': 'GEN2', 'tooltip': '2nd genitive (Ablative) case'},
            {'type': 'gloss', 'value': 'DAT', 'tooltip': 'Dative case'},
            {'type': 'gloss', 'value': 'INS', 'tooltip': 'Instrumental case'},
            {'type': 'gloss', 'value': 'CAR', 'tooltip': 'Caritive case'},
            {'type': 'gloss', 'value': 'ADV', 'tooltip': 'Adverbial case'}
        ],

        [
            {'type': 'gloss', 'value': 'LOC', 'tooltip': 'Locative case'},
            {'type': 'gloss', 'value': 'ILL', 'tooltip': 'Illative case'},
            {'type': 'gloss', 'value': 'EL', 'tooltip': 'Elative case'},
            {'type': 'gloss', 'value': 'EGR', 'tooltip': 'Egressive case'},
            {'type': 'gloss', 'value': 'TERM', 'tooltip': 'Terminative case'},
            {'type': 'gloss', 'value': 'PROL', 'tooltip': 'Prolative case'},
            {'type': 'gloss', 'value': 'APP', 'tooltip': 'Approximative case'},
            {'type': 'gloss', 'value': 'RCS', 'tooltip': 'Recessive case'},
            {'type': 'separator'},
            {'type': 'gloss', 'value': 'DMS', 'tooltip': '&ldquo;Domus&rdquo; (personal-local marker)'}
        ],
        [
            {'type': 'gloss', 'value': 'PL', 'tooltip': 'Nominal plural'},
            {'type': 'gloss', 'value': 'ADJ.PL', 'tooltip': 'Adjectival plural'},
            {'type': 'separator'},
            {'type': 'gloss', 'value': 'POSS', 'tooltip': 'Any&nbsp;possessive&nbsp;marker'},
            {'type': 'gloss', 'value': 'P.1SG', 'tooltip': '1sg possessive'},
            {'type': 'gloss', 'value': 'P.1PL', 'tooltip': '1pl possessive'},
            {'type': 'gloss', 'value': 'P.2SG', 'tooltip': '2sg possessive'},
            {'type': 'gloss', 'value': 'P.2PL', 'tooltip': '2pl possessive'},
            {'type': 'gloss', 'value': 'P.3SG', 'tooltip': '3sg possessive'},
            {'type': 'gloss', 'value': 'P.3PL', 'tooltip': '3pl possessive'}
        ],

        [
            {'type': 'gloss', 'value': 'P.ACC', 'tooltip': 'Any possessive accusative'},
            {'type': 'gloss', 'value': 'P.1SG.ACC', 'tooltip': 'Accusative 1sg possessive'},
            {'type': 'gloss', 'value': 'P.1PL.ACC', 'tooltip': 'Accusative 1pl possessive'},
            {'type': 'gloss', 'value': 'P.2SG.ACC', 'tooltip': 'Accusative 2sg possessive'},
            {'type': 'gloss', 'value': 'P.2PL.ACC', 'tooltip': 'Accusative 2pl possessive'},
            {'type': 'gloss', 'value': 'P.3SG.ACC', 'tooltip': 'Accusative 3sg possessive'},
            {'type': 'gloss', 'value': 'P.3PL.ACC', 'tooltip': 'Accusative 3pl possessive'},
            {'type': 'separator'},
            {'type': 'gloss', 'value': 'ATTR', 'tooltip': 'Attributivizer'},
            {'type': 'gloss', 'value': 'NEG.ATTR', 'tooltip': 'Negative attributivizer'}
        ],

        [
            {'type': 'gloss', 'value': 'ORD', 'tooltip': 'Ordinal numeral'},
            {'type': 'gloss', 'value': 'DISTR', 'tooltip': 'Distributive numeral'},
            {'type': 'gloss', 'value': 'COLL', 'tooltip': 'Collective numeral'},
            {'type': 'gloss', 'value': 'PERIOD', 'tooltip': 'Time period (-oj)'},
            {'type': 'gloss', 'value': 'DELIM', 'tooltip': 'Delimitative (-skən)'},
            {'type': 'separator'},
            {'type': 'gloss', 'value': 'VN', 'tooltip': 'Verbal noun (-on)'},
            {'type': 'gloss', 'value': 'NLOC', 'tooltip': 'Locative noun (-nʼig)'},
            {'type': 'gloss', 'value': 'COMP', 'tooltip': '&ldquo;Comparative&rdquo;'}
        ],

        [
            {'type': 'gloss', 'value': 'DERIV', 'tooltip': 'Any verbal derivation'},
            {'type': 'gloss', 'value': 'ITER', 'tooltip': 'Iterative'},
            {'type': 'gloss', 'value': 'CAUS', 'tooltip': 'Causative'},
            {'type': 'gloss', 'value': 'DETR', 'tooltip': 'Detransitive'},
            {'type': 'separator'},
            {'type': 'gloss', 'value': 'PRS.1SG', 'tooltip': 'Present 1sg'},
            {'type': 'gloss', 'value': 'PRS.1PL', 'tooltip': 'Present 1pl'},
            {'type': 'gloss', 'value': 'PRS.2SG', 'tooltip': 'Present 2sg'},
            {'type': 'gloss', 'value': 'PRS.2PL', 'tooltip': 'Present 2pl'},
            {'type': 'gloss', 'value': 'PRS.3SG', 'tooltip': 'Present 3sg'},
            {'type': 'gloss', 'value': 'PRS.3PL', 'tooltip': 'Present 3pl'}
        ],

        [
            {'type': 'gloss', 'value': 'PST_ANY', 'tooltip': 'Any past marker'},
            {'type': 'gloss', 'value': 'PST', 'tooltip': 'Past (non-fusional)'},
            {'type': 'gloss', 'value': 'PST(1SG)', 'tooltip': 'Past 1sg (-i)'},
            {'type': 'gloss', 'value': 'PST.1SG', 'tooltip': 'Past 1sg (-j)'},
            {'type': 'gloss', 'value': 'PST.1PL', 'tooltip': 'Past 1pl'},
            {'type': 'gloss', 'value': 'PST.2SG', 'tooltip': 'Past 2sg'},
            {'type': 'gloss', 'value': 'PST.2PL', 'tooltip': 'Past 2pl'},
            {'type': 'gloss', 'value': 'PST.3SG', 'tooltip': 'Past 3sg'},
            {'type': 'gloss', 'value': 'PST.3PL', 'tooltip': 'Past 3pl'}
        ],

        [
            {'type': 'gloss', 'value': 'FUT_ANY', 'tooltip': 'Any future marker'},
            {'type': 'gloss', 'value': 'FUT', 'tooltip': 'Future (non-fusional)'},
            {'type': 'gloss', 'value': 'FUT(1SG)', 'tooltip': 'Future 1sg (-i)'},
            {'type': 'gloss', 'value': 'FUT.1SG', 'tooltip': 'Future 1sg (-j)'},
            {'type': 'gloss', 'value': 'FUT.1PL', 'tooltip': 'Future 1pl'},
            {'type': 'gloss', 'value': 'FUT.2SG', 'tooltip': 'Future 2sg'},
            {'type': 'gloss', 'value': 'FUT.2PL', 'tooltip': 'Future 2pl'},
            {'type': 'gloss', 'value': 'FUT.3SG', 'tooltip': 'Future 3sg'},
            {'type': 'gloss', 'value': 'FUT.3PL', 'tooltip': 'Future 3pl'}
        ],

        [
            {'type': 'gloss', 'value': 'PST2', 'tooltip': '2nd (evidential) past'},
            {'type': 'separator'},
            {'type': 'gloss', 'value': 'PERS', 'tooltip': 'Any person marker'},
            {'type': 'gloss', 'value': '1', 'tooltip': '1st peron (in 2nd past)'},
            {'type': 'gloss', 'value': '1SG', 'tooltip': '1st person singular'},
            {'type': 'gloss', 'value': '1PL', 'tooltip': '1st person plural'},
            {'type': 'gloss', 'value': '2SG', 'tooltip': '2nd person singular'},
            {'type': 'gloss', 'value': '2PL', 'tooltip': '2nd person plural'},
            {'type': 'gloss', 'value': '3SG', 'tooltip': '3rd person singular'},
            {'type': 'gloss', 'value': '3PL', 'tooltip': '3rd person plural'}
        ],

        [
            {'type': 'gloss', 'value': 'CVB', 'tooltip': 'General converb'},
            {'type': 'gloss', 'value': 'CVB.NEG', 'tooltip': 'General negative converb'},
            {'type': 'gloss', 'value': 'CVB.DUR', 'tooltip': 'Durative converb'},
            {'type': 'gloss', 'value': 'RES', 'tooltip': 'Resultative'},
            {'type': 'gloss', 'value': 'DEB', 'tooltip': 'Debitive'},
            {'type': 'gloss', 'value': 'CVB.LIM', 'tooltip': 'Limitative converb'},
            {'type': 'gloss', 'value': 'CVB.MON', 'tooltip': 'Converb in -mon'},
            {'type': 'gloss', 'value': 'PTCP.PST', 'tooltip': 'Past participle'},
            {'type': 'gloss', 'value': 'PTCP.ACT', 'tooltip': 'Active participle'},
            {'type': 'gloss', 'value': 'PTCP.NEG.ACT', 'tooltip': 'Negative active participle'},
            {'type': 'gloss', 'value': 'PTCP.NEG.PASS', 'tooltip': 'Negative passive participle'}
        ]
    ]

    gramm_column_list = [

        [
            {'type': 'header', 'value': 'Parts of speech'},
            {'type': 'gramm', 'value': 'N', 'category': 'pos', 'tooltip': 'Noun'},
            {'type': 'gramm', 'value': 'V', 'category': 'pos', 'tooltip': 'Verb'},
            {'type': 'gramm', 'value': 'A', 'category': 'pos', 'tooltip': 'Adjective'},
            {'type': 'gramm', 'value': 'ADV', 'category': 'pos', 'tooltip': 'Adverb'},
            {'type': 'gramm', 'value': 'PRED', 'category': 'pos', 'tooltip': 'Predicative'},
            {'type': 'gramm', 'value': 'NUM', 'category': 'pos', 'tooltip': 'Numeral'},
            {'type': 'gramm', 'value': 'PRO', 'category': 'pos', 'tooltip': 'Pronoun'},
            {'type': 'gramm', 'value': 'POST', 'category': 'pos', 'tooltip': 'Postposition'},
            {'type': 'gramm', 'value': 'CONJ', 'category': 'pos', 'tooltip': 'Conjunction'},
            {'type': 'gramm', 'value': 'PART', 'category': 'pos', 'tooltip': 'Particle'},
            {'type': 'gramm', 'value': 'INTERJ', 'category': 'pos', 'tooltip': 'Interjection'},
            {'type': 'gramm', 'value': 'IMIT', 'category': 'pos', 'tooltip': 'Ideophone'}
        ],

        [
            {'type': 'header', 'value': 'Lexical classes'},
            {'type': 'gramm', 'value': 'rel_n', 'category': 'nType', 'tooltip': 'Relational noun'},
            {'type': 'gramm', 'value': 'rel_adj', 'category': 'adjType', 'tooltip': 'Relational adjective'},
            {'type': 'gramm', 'value': 'oblin', 'category': 'adjType', 'tooltip': 'Oblinative (-ešʼ)'},
            {'type': 'gramm', 'value': 'topn', 'category': 'nType', 'tooltip': 'Place name'},
            {'type': 'gramm', 'value': 'persn', 'category': 'nType', 'tooltip': 'Personal name'},
            {'type': 'gramm', 'value': 'famn', 'category': 'nType', 'tooltip': 'Last name'},
            {'type': 'gramm', 'value': 'patrn', 'category': 'nType', 'tooltip': 'Patronymic'},
            {'type': 'gramm', 'value': 'anim', 'category': 'nType', 'tooltip': 'Animate'},
            {'type': 'gramm', 'value': 'hum', 'category': 'nType', 'tooltip': 'Human'},
            {'type': 'gramm', 'value': 'supernat', 'category': 'nType', 'tooltip': 'Supernatural'},
            {'type': 'gramm', 'value': 'body', 'category': 'nType', 'tooltip': 'Body part'},
            {'type': 'gramm', 'value': 'transport', 'category': 'nType', 'tooltip': 'Transport'}
        ],

        [
            {'type': 'header', 'value': 'Pronoun types'},
            {'type': 'gramm', 'value': 'dem', 'category': 'proType', 'tooltip': 'Demonstrative pronoun'},
            {'type': 'gramm', 'value': 'pers', 'category': 'proType', 'tooltip': 'Personal pronoun'},
            {'type': 'gramm', 'value': 'refl', 'category': 'proType', 'tooltip': 'Reflexive pronoun'},
            {'type': 'header', 'value': 'Attributivizers'},
            {'type': 'gramm', 'value': 'attr', 'category': 'attr', 'tooltip': 'Any attributivizer'},
            {'type': 'gramm', 'value': 'attr_o', 'category': 'attr', 'tooltip': '-o'},
            {'type': 'gramm', 'value': 'attr_em', 'category': 'attr', 'tooltip': '-jem'},
            {'type': 'gramm', 'value': 'attr_tem', 'category': 'attr', 'tooltip': 'Negative (-tem)'},
            {'type': 'header', 'value': 'Numerals'},
            {'type': 'gramm', 'value': 'ord', 'category': 'nType', 'tooltip': 'Ordinal'},
            {'type': 'gramm', 'value': 'period', 'category': 'nType', 'tooltip': 'Period (-skən)'},
            {'type': 'gramm', 'value': 'coll', 'category': 'nType', 'tooltip': 'Collective'},
            {'type': 'header', 'value': 'Loanwords'},
            {'type': 'gramm', 'value': 'rus', 'category': 'additional', 'tooltip': 'Russian borrowing'}
        ],

        [
            {'type': 'header', 'value': 'Core cases'},
            {'type': 'gramm', 'value': 'nom', 'category': 'case', 'tooltip': 'Nominative (or unmarked accusative)'},
            {'type': 'gramm', 'value': 'acc', 'category': 'case', 'tooltip': 'Accusative'},
            {'type': 'gramm', 'value': 'gen', 'category': 'case', 'tooltip': 'Genitive'},
            {'type': 'gramm', 'value': 'gen2', 'category': 'case', 'tooltip': '2nd genitive (ablative)'},
            {'type': 'gramm', 'value': 'dat', 'category': 'case', 'tooltip': 'Dative'},
            {'type': 'gramm', 'value': 'ins', 'category': 'case', 'tooltip': 'Instrumental'},
            {'type': 'gramm', 'value': 'car', 'category': 'case', 'tooltip': 'Caritive'},
            {'type': 'gramm', 'value': 'adv', 'category': 'case', 'tooltip': 'Adverbial'}
        ],

        [
            {'type': 'header', 'value': 'Spatial cases'},
            {'type': 'gramm', 'value': 'loc', 'category': 'case', 'tooltip': 'Locative'},
            {'type': 'gramm', 'value': 'ill', 'category': 'case', 'tooltip': 'Illative'},
            {'type': 'gramm', 'value': 'el', 'category': 'case', 'tooltip': 'Elative'},
            {'type': 'gramm', 'value': 'egr', 'category': 'case', 'tooltip': 'Egressive'},
            {'type': 'gramm', 'value': 'term', 'category': 'case', 'tooltip': 'Terminative'},
            {'type': 'gramm', 'value': 'prol', 'category': 'case', 'tooltip': 'Prolative'},
            {'type': 'gramm', 'value': 'app', 'category': 'case', 'tooltip': 'Approximative'},
            {'type': 'gramm', 'value': 'rcs', 'category': 'case', 'tooltip': 'Recessive'},
            {'type': 'separator'},
            {'type': 'gramm', 'value': 'dms', 'category': 'case',
             'tooltip': '&ldquo;Domus&rdquo; (personal-local marker)'}
        ],

        [
            {'type': 'gramm', 'value': 'obl', 'category': 'stemType', 'tooltip': 'Oblique stem'},
            {'type': 'gramm', 'value': 'case_comp', 'category': 'add', 'tooltip': 'Case compounding'},
            {'type': 'header', 'value': 'Possessiveness'},
            {'type': 'gramm', 'value': '1sg', 'category': 'poss', 'tooltip': '1sg possessive'},
            {'type': 'gramm', 'value': '1pl', 'category': 'poss', 'tooltip': '1pl possessive'},
            {'type': 'gramm', 'value': '2sg', 'category': 'poss', 'tooltip': '2sg possessive'},
            {'type': 'gramm', 'value': '2pl', 'category': 'poss', 'tooltip': '2pl possessive'},
            {'type': 'gramm', 'value': '3sg', 'category': 'poss', 'tooltip': '3sg possessive'},
            {'type': 'gramm', 'value': '3pl', 'category': 'poss', 'tooltip': '3pl possessive'},
            {'type': 'header', 'value': 'Number'},
            {'type': 'gramm', 'value': 'sg', 'category': 'number', 'tooltip': 'Singular'},
            {'type': 'gramm', 'value': 'pl', 'category': 'number', 'tooltip': 'Plural'}
        ],

        [
            {'type': 'header', 'value': 'Tense/Mood'},
            {'type': 'gramm', 'value': 'prs', 'category': 'tense', 'tooltip': 'Present tense'},
            {'type': 'gramm', 'value': 'fut', 'category': 'tense', 'tooltip': 'Future tense'},
            {'type': 'gramm', 'value': 'pst', 'category': 'tense', 'tooltip': 'Past tense'},
            {'type': 'gramm', 'value': 'pst2', 'category': 'tense', 'tooltip': 'Evidential past'},
            {'type': 'gramm', 'value': 'subj', 'category': 'tense', 'tooltip': 'Subjunctive'},
            {'type': 'header', 'value': 'Verbal derivation'},
            {'type': 'gramm', 'value': 'iter', 'category': 'deriv', 'tooltip': 'Iterative'},
            {'type': 'gramm', 'value': 'caus', 'category': 'deriv', 'tooltip': 'Causative'},
            {'type': 'gramm', 'value': 'detr', 'category': 'deriv', 'tooltip': 'Detransitive'},
            {'type': 'gramm', 'value': 'vn', 'category': 'deriv', 'tooltip': 'Verbal noun (-on)'},
            {'type': 'gramm', 'value': 'nloc', 'category': 'deriv', 'tooltip': 'Nomen loci (-onʼnʼig)'}
        ],

        [
            {'type': 'header', 'value': 'Non-finite forms'},
            {'type': 'gramm', 'value': 'inf', 'category': 'nonfin', 'tooltip': 'Infinitive'},
            {'type': 'gramm', 'value': 'ptcp', 'category': 'nonfin', 'tooltip': 'Any participle'},
            {'type': 'gramm', 'value': 'ptcp,act', 'category': 'nonfin', 'tooltip': 'Active (-išʼ/-əšʼ)'},
            {'type': 'gramm', 'value': 'ptcp,act,neg', 'category': 'nonfin', 'tooltip': 'Active negative (-tem)'},
            {'type': 'gramm', 'value': 'ptcp,pass,neg', 'category': 'nonfin', 'tooltip': 'Passive negative (-mte)'},
            {'type': 'gramm', 'value': 'ptcp,pst', 'category': 'nonfin', 'tooltip': 'Past participle (-em)'},
            {'type': 'gramm', 'value': 'res', 'category': 'nonfin', 'tooltip': 'Resultative (-mən)'},
            {'type': 'gramm', 'value': 'deb', 'category': 'nonfin', 'tooltip': 'Debitive (-ono)'}
        ],

        [
            {'type': 'header', 'value': 'Converbs'},
            {'type': 'gramm', 'value': 'cvb', 'category': 'nonfin', 'tooltip': 'Any converb'},
            {'type': 'gramm', 'value': 'cvb,dur', 'category': 'nonfin', 'tooltip': 'Durative (-onʼnʼa)'},
            {'type': 'gramm', 'value': 'cvb,lim', 'category': 'nonfin', 'tooltip': 'Limitative (-čʼožʼ)'},
            {'type': 'gramm', 'value': 'cvb,simult', 'category': 'nonfin', 'tooltip': 'Simultaneous (-ku)'},
            {'type': 'gramm', 'value': 'cvb,neg', 'category': 'nonfin', 'tooltip': 'Negative (-tek)'},
            {'type': 'header', 'value': 'Imperatives'},
            {'type': 'gramm', 'value': 'imp', 'category': 'nonfin', 'tooltip': 'Imperative'},
            {'type': 'header', 'value': 'Transitivity'},
            {'type': 'gramm', 'value': 'tr', 'category': 'vTr', 'tooltip': 'Transitive'},
            {'type': 'gramm', 'value': 'intr', 'category': 'vTr', 'tooltip': 'Intransitive'},
            {'type': 'gramm', 'value': 'impers', 'category': 'vTr', 'tooltip': 'Impersonal'}
        ]
    ]

    re_sentence_end = (

        re.compile(
            '[.։。?;？!！]*'
            '["«»‘’“”„‟『』《》]*'
            '[.։。?;？!！]+'
            '["«»‘’“”„‟『』《》]*'
            '[.։。?;？!！]*'))

    @staticmethod
    def process_parser_result(
            sentence_list,
            tag_set,
            gloss_set,
            content,
            debug_flag):
        """
        Compiles sentence data from a single parser result.
        """

        text_list = []
        text_length = 0

        token_list = []
        token_index = 0

        # Processing all extracted paragraphs with their data.

        for (
                paragraph_index,
                (paragraph_item_list, check_list, check_flag)) in (

                export_parser_result.iterate_beautiful_soup(
                    content, debug_flag)):

            if not paragraph_item_list:
                continue

            # Processing each data item of the paragraph.

            for item in paragraph_item_list:

                variant_list = []

                if isinstance(item, str):

                    # Text part which is not a parsed token.

                    item_str = item

                else:

                    # Parsed token.

                    item_str = item[0]
                    variant_list = item[1]

                token_str = (
                    re.sub(r'\s+', ' ', item_str))

                # All whitespace is transformed to single spaces, is skipped if at the start of the sentence
                # and are not treated as tokens.

                if token_str == ' ':

                    if not token_list:
                        continue

                    text_list.append(' ')
                    text_length += 1

                    continue

                # Got outselves a proper token, checking spurious whitespace.

                lead_ws, token_str, trail_ws = (
                    re.match(r'^(\s*)(.*?)(\s*)$', token_str).groups())

                punct_flag = (
                        re.search(r'[^\w\s]', token_str) is not None)

                sentence_end_flag = (
                        Tsakorpus.re_sentence_end.match(token_str) is not None)

                if not token_list:
                    lead_ws = ''

                if sentence_end_flag:
                    trail_ws = ''

                if lead_ws:
                    text_list.append(' ')
                    text_length += 1

                # Processing the token.

                token_dict = {'wf': token_str}

                token_dict['wtype'] = (
                    'punct' if punct_flag else 'word')

                token_dict['off_start'] = text_length

                text_list.append(token_str)
                text_length += len(token_str)

                token_dict['off_end'] = text_length

                if variant_list:

                    analyses_list = []

                    for variant_dict in variant_list:

                        # Checking grammar tags.

                        if 'gr' in variant_dict:

                            analysis_dict = variant_dict.copy()

                            for tag in variant_dict['gr'].split(','):

                                tag = (
                                    tag.strip())

                                if not tag:
                                    continue

                                tag_set.add(tag)

                                category = (
                                    Tsakorpus.tag_category_dict.get(tag, 'undefined'))

                                analysis_dict['gr.' + category] = tag

                            del analysis_dict['gr']

                        else:

                            analysis_dict = variant_dict

                        analyses_list.append(analysis_dict)

                        # Checking glosses.

                        if 'gloss' in variant_dict:
                            gloss_set.update(
                                variant_dict['gloss'].split('-'))

                    token_dict['ana'] = analyses_list

                if (token_list or not punct_flag) and not sentence_end_flag:
                    token_dict['sentence_index'] = token_index
                    token_index += 1

                token_list.append(token_dict)

                token_dict['next_word'] = len(token_list)

                if trail_ws:
                    text_list.append(' ')
                    text_length += 1

                if debug_flag:
                    log.debug(
                        '\ntoken_dict:\n' +
                        pprint.pformat(
                            token_dict, width=192))

                # Checking if we finished another sentence.

                if sentence_end_flag:

                    for token_dict in token_list:

                        if 'sentence_index' in token_dict:
                            token_dict['sentence_index_neg'] = (
                                    token_index - token_dict['sentence_index'])

                    sentence_dict = {
                        'text': ''.join(text_list),
                        'words': token_list,
                        'lang': 0,
                        'meta': {}}

                    sentence_list.append(
                        sentence_dict)

                    text_list = []
                    text_length = 0

                    token_list = []
                    token_index = 0

                    if debug_flag:
                        log.debug(
                            '\nsentence_dict:\n' +
                            pprint.pformat(
                                sentence_dict, width=192))

            # Checking if we have paragraph's last sentence.

            if token_list:

                for token_dict in token_list:

                    if 'sentence_index' in token_dict:
                        token_dict['sentence_index_neg'] = (
                                token_index - token_dict['sentence_index'])

                sentence_dict = {
                    'text': ''.join(text_list),
                    'words': token_list,
                    'lang': 0,
                    'meta': {}}

                sentence_list.append(
                    sentence_dict)

                text_list = []
                text_length = 0

                token_list = []
                token_index = 0

                if debug_flag:
                    log.debug(
                        '\nsentence_dict:\n' +
                        pprint.pformat(
                            sentence_dict, width=192))

    @staticmethod
    def process_perspective(
            perspective_id,
            tag_set,
            gloss_set,
            debug_flag=False):

        # Converting perspective's parser results into tsakorpus corpus format.
        #
        # Cf. create_valency_data.

        entity_list = (

            DBSession

                .query(
                dbEntity)

                .filter(
                dbLexicalEntry.parent_client_id == perspective_id[0],
                dbLexicalEntry.parent_object_id == perspective_id[1],
                dbLexicalEntry.marked_for_deletion == False,
                dbEntity.parent_client_id == dbLexicalEntry.client_id,
                dbEntity.parent_object_id == dbLexicalEntry.object_id,
                dbEntity.marked_for_deletion == False,
                dbPublishingEntity.client_id == dbEntity.client_id,
                dbPublishingEntity.object_id == dbEntity.object_id,
                dbPublishingEntity.published == True,
                dbPublishingEntity.accepted == True)

                .order_by(
                dbLexicalEntry.created_at,
                dbLexicalEntry.client_id,
                dbLexicalEntry.object_id,
                dbEntity.created_at,
                dbEntity.client_id,
                dbEntity.object_id)

                .all())

        # Processing entities.

        entry_dict = collections.defaultdict(dict)
        entry_list = []

        sentence_list = []

        language_str = None

        for entity in entity_list:

            entry_id = (
                (entity.parent_client_id, entity.parent_object_id))

            new_entry_flag = (
                    entry_id not in entry_dict)

            entry_info_dict = entry_dict[entry_id]

            if new_entry_flag:
                entry_info_dict['parser_result_list'] = []
                entry_list.append(entry_info_dict)

            if entity.field_id == (674, 5):

                entry_info_dict['comment'] = entity.content
                continue

            elif not is_subject_for_parsing(entity.content):
                continue

            parser_result_list = (

                DBSession

                    .query(
                    dbParserResult,
                    dbParser.method)

                    .filter(
                    dbParserResult.entity_client_id == entity.client_id,
                    dbParserResult.entity_object_id == entity.object_id,
                    dbParserResult.marked_for_deletion == False,
                    dbParser.client_id == dbParserResult.parser_client_id,
                    dbParser.object_id == dbParserResult.parser_object_id,
                    dbParser.method.in_(
                        Tsakorpus.parser_language_dict.keys()))

                    .order_by(
                    dbParserResult.created_at,
                    dbParserResult.client_id,
                    dbParserResult.object_id)

                    .all())

            # Processing all parser results of this entity.

            for parser_result, parser_str in parser_result_list:

                current_language_str = (
                    Tsakorpus.parser_language_dict[parser_str])

                if language_str is None:
                    language_str = current_language_str

                elif language_str != current_language_str:
                    raise NotImplementedError

                Tsakorpus.process_parser_result(
                    sentence_list,
                    tag_set,
                    gloss_set,
                    parser_result.content,
                    debug_flag)

        return sentence_list, language_str

    @staticmethod
    def save_configuration(
            corpus_name,
            tag_set,
            gloss_set,
            language_str,
            english_title,
            russian_title,
            dir_path='.'):

        # Checking categories, saving category configuration.

        extra_tag_set = (
                tag_set - Tsakorpus.tag_category_dict.keys())

        log.debug(
            f'\nextra_tag_set:\n{extra_tag_set}')

        tag_category_dict = (
            Tsakorpus.tag_category_dict.copy())

        for tag in extra_tag_set:
            tag_category_dict[tag] = 'undefined'

        categories_file_path = (
            os.path.join(dir_path, 'categories.json'))

        with open(
                categories_file_path, 'w') as categories_file:
            json.dump(
                {language_str: tag_category_dict},
                categories_file,
                ensure_ascii=False,
                indent=2)

        # Compiling and saving corpus configuration.

        config_dict = (
            Tsakorpus.config_dict.copy())

        config_dict['citation'] = english_title or russian_title
        config_dict['corpus_name'] = corpus_name

        config_dict['keyboards'] = {
            language_str: language_str}

        base_gloss_set = (

            set(
                gloss_info['value']
                for gloss_info_list in Tsakorpus.gloss_column_list
                for gloss_info in gloss_info_list
                if gloss_info['type'] == 'gloss'))

        gloss_column_list = (
            Tsakorpus.gloss_column_list.copy())

        gloss_column_list.append([
            {'type': 'gloss', 'value': gloss, 'tooltip': ''}
            for gloss in gloss_set - base_gloss_set])

        gramm_column_list = (
            Tsakorpus.gramm_column_list.copy())

        gramm_column_list.append([
            {'type': 'gramm', 'value': tag, 'category': 'undefined', 'tooltip': ''}
            for tag in extra_tag_set])

        language_dict = (
            config_dict['lang_props']['udmurt'])

        language_dict.update({
            'gloss_selection': {'columns': gloss_column_list},
            'gramm_selection': {'columns': gramm_column_list}})

        config_dict['lang_props'] = {
            language_str: language_dict}

        config_dict['languages'] = [
            language_str]

        corpus_file_path = (
            os.path.join(dir_path, 'corpus.json'))

        with open(
                corpus_file_path, 'w') as corpus_file:
            json.dump(
                config_dict,
                corpus_file,
                ensure_ascii=False,
                indent=2)

        # Saving English and Russian corpus titles.

        en_file_path = (
            os.path.join(dir_path, 'corpus-specific-en.txt'))

        with open(en_file_path, 'w') as en_file:
            en_file.write(
                f'corpus_title\t{english_title}')

        ru_file_path = (
            os.path.join(dir_path, 'corpus-specific-ru.txt'))

        with open(ru_file_path, 'w') as ru_file:
            ru_file.write(
                f'corpus_title\t{russian_title}')

    @staticmethod
    def mutate(root, info, **args):

        try:

            client_id = info.context.get('client_id')
            client = DBSession.query(Client).filter_by(id=client_id).first()

            if not client or client.user_id != 1:
                return (

                    ResponseError(
                        message='Not an administrator.'))

            perspective_id = args.get('perspective_id')
            debug_flag = args.get('debug_flag', False)

            # Processing a single perspective.

            if perspective_id:

                perspective = (
                    DBSession.query(dbPerspective).filter_by(
                        client_id=perspective_id[0], object_id=perspective_id[1]).first())

                if not perspective:
                    return (

                        ResponseError(
                            message='No perspective {}/{} in the system.'.format(*perspective_id)))

                dictionary = perspective.parent

                locale_id = (
                        info.context.get('locale_id') or ENGLISH_LOCALE)

                dictionary_name = dictionary.get_translation(locale_id)
                perspective_name = perspective.get_translation(locale_id)

                full_name = (
                        dictionary_name + ' \u203a ' + perspective_name)

                if dictionary.marked_for_deletion:
                    return (

                        ResponseError(message=
                        'Dictionary \'{}\' {}/{} of perspective \'{}\' {}/{} is deleted.'.format(
                            dictionary_name,
                            dictionary.client_id,
                            dictionary.object_id,
                            perspective_name,
                            perspective.client_id,
                            perspective.object_id)))

                if perspective.marked_for_deletion:
                    return (

                        ResponseError(message=
                        'Perspective \'{}\' {}/{} is deleted.'.format(
                            full_name,
                            perspective.client_id,
                            perspective.object_id)))

                tag_set = set()
                gloss_set = set()

                sentence_list, language_str = (

                    Tsakorpus.process_perspective(
                        perspective_id,
                        tag_set,
                        gloss_set,
                        debug_flag))

                # Saving corpus data.

                corpus_dict = {
                    'meta': {},
                    'sentences': sentence_list}

                with gzip.open('corpus.json.gz', 'wt') as corpus_file:

                    json.dump(
                        corpus_dict,
                        corpus_file,
                        ensure_ascii=False,
                        indent=2)

                Tsakorpus.save_configuration(
                    f'{perspective_id[0]}_{perspective_id[1]}',
                    tag_set,
                    gloss_set,
                    language_str,
                    dictionary.get_translation(ENGLISH_LOCALE),
                    dictionary.get_translation(RUSSIAN_LOCALE))

                current_datetime = str(datetime.datetime.utcnow())
                perspective.additional_metadata['uploaded_at'] = current_datetime

                return (

                    Tsakorpus(
                        triumph=True))

            # Not a single perspective, processing all corpora we can.

            perspective_data_list = (

                DBSession

                    .query(
                    dbDictionary,
                    dbPerspective,
                    dbParser.method)

                    .filter(
                    dbDictionary.marked_for_deletion == False,
                    dbPerspective.parent_client_id == dbDictionary.client_id,
                    dbPerspective.parent_object_id == dbDictionary.object_id,
                    dbPerspective.marked_for_deletion == False,
                    dbLexicalEntry.parent_client_id == dbPerspective.client_id,
                    dbLexicalEntry.parent_object_id == dbPerspective.object_id,
                    dbLexicalEntry.marked_for_deletion == False,
                    dbEntity.parent_client_id == dbLexicalEntry.client_id,
                    dbEntity.parent_object_id == dbLexicalEntry.object_id,
                    dbEntity.marked_for_deletion == False,
                    dbEntity.content.op('~*')('.*\.(doc|docx|odt)'),
                    dbPublishingEntity.client_id == dbEntity.client_id,
                    dbPublishingEntity.object_id == dbEntity.object_id,
                    dbPublishingEntity.published == True,
                    dbPublishingEntity.accepted == True,
                    dbParserResult.entity_client_id == dbEntity.client_id,
                    dbParserResult.entity_object_id == dbEntity.object_id,
                    dbParserResult.marked_for_deletion == False,
                    dbParser.client_id == dbParserResult.parser_client_id,
                    dbParser.object_id == dbParserResult.parser_object_id,
                    dbParser.method.in_(
                        Tsakorpus.parser_language_dict.keys()))

                    .group_by(
                    dbDictionary,
                    dbPerspective,
                    dbParser.method)

                    .all())

            data_dir_path = 'tsakorpus_data'

            tag_set_dict = collections.defaultdict(set)
            gloss_set_dict = collections.defaultdict(set)

            language_str_list = []

            for dictionary, perspective, parser_str in perspective_data_list:
                language_str = (
                    Tsakorpus.parser_language_dict[parser_str])

                # Getting and saving perspective's corpus data.

                sentence_list, _ = (

                    Tsakorpus.process_perspective(
                        perspective.id,
                        tag_set_dict[language_str],
                        gloss_set_dict[language_str],
                        debug_flag))

                language_str_list.append(language_str)

                perspective_dir_path = (

                    os.path.join(
                        data_dir_path,
                        f'perspective_{perspective.client_id}_{perspective.object_id}'))

                os.makedirs(
                    perspective_dir_path,
                    exist_ok=True)

                corpus_file_path = (

                    os.path.join(
                        perspective_dir_path, 'corpus.json.gz'))

                with gzip.open(corpus_file_path, 'wt') as corpus_file:
                    corpus_dict = {
                        'meta': {},
                        'sentences': sentence_list}

                    json.dump(
                        corpus_dict,
                        corpus_file,
                        ensure_ascii=False,
                        indent=2)

            # Saving additional corpus data.

            for dictionary, perspective, parser_str in perspective_data_list:

                language_str = (
                    Tsakorpus.parser_language_dict[parser_str])

                corpus_name = (
                    f'{perspective.client_id}_{perspective.object_id}')

                perspective_dir_path = (

                    os.path.join(
                        data_dir_path,
                        f'perspective_{corpus_name}'))

                english_title = (
                    dictionary.get_translation(ENGLISH_LOCALE))

                russian_title = (
                    dictionary.get_translation(RUSSIAN_LOCALE))

                Tsakorpus.save_configuration(
                    corpus_name,
                    tag_set_dict[language_str],
                    gloss_set_dict[language_str],
                    language_str,
                    english_title,
                    russian_title,
                    dir_path=perspective_dir_path)

                # Showing processed perspective.

                info_str_list = [f'\n{english_title}']

                if russian_title != english_title:
                    info_str_list.append(f'\n{russian_title}')

                info_str_list.append(
                    '\nhttp://lingvodoc.ispras.ru/'
                    f'dictionary/{dictionary.client_id}/{dictionary.object_id}/'
                    f'perspective/{perspective.client_id}/{perspective.object_id}/'
                    'view')

                log.debug(
                    ''.join(info_str_list))

            return (

                Tsakorpus(
                    triumph=True))

        except Exception as exception:

            traceback_string = (

                ''.join(
                    traceback.format_exception(
                        exception, exception, exception.__traceback__))[:-1])

            log.warning('tsakorpus: exception')
            log.warning(traceback_string)

            transaction.abort()

            return (

                ResponseError(
                    'Exception:\n' + traceback_string))


class MyMutations(graphene.ObjectType):
    """
    Mutation classes.
    It may be replaced by
    create_field = gql_field.CreateField.Field()
    for more beautiful imports
    """
    convert_starling = starling_converter.GqlStarling.Field()
    convert_plain_text = plain_text_converter.GqlParallelCorpora.Field()
    convert_dialeqt = ConvertDictionary.Field()
    convert_corpus = ConvertFiveTiers.Field()
    create_field = CreateField.Field()
    # update_field = UpdateField.Field()
    # delete_field = DeleteField.Field()
    copy_sound_markup_fields = CopySoundMarkupFields.Field()
    copy_single_field = CopySingleField.Field()
    create_entity = CreateEntity.Field()
    update_entity = UpdateEntity.Field()
    delete_entity = DeleteEntity.Field()
    update_entity_content = UpdateEntityContent.Field()
    bulk_update_entity_content = BulkUpdateEntityContent.Field()
    approve_all_for_user = ApproveAllForUser.Field()
    bulk_create_entity = BulkCreateEntity.Field()
    bulk_delete_entity = BulkDeleteEntity.Field()
    create_user = CreateUser.Field()
    update_user = UpdateUser.Field()
    activate_deactivate_user = ActivateDeactivateUser.Field()
    create_language = CreateLanguage.Field()
    update_language = UpdateLanguage.Field()
    update_language_atom = UpdateLanguageAtom.Field()
    move_language = MoveLanguage.Field()
    delete_language = DeleteLanguage.Field()
    create_dictionary = CreateDictionary.Field()
    update_dictionary = UpdateDictionary.Field()
    update_dictionary_status = UpdateDictionaryStatus.Field()
    update_dictionary_atom = UpdateDictionaryAtom.Field()
    add_dictionary_roles = AddDictionaryRoles.Field()
    delete_dictionary_roles = DeleteDictionaryRoles.Field()
    delete_dictionary = DeleteDictionary.Field()
    undelete_dictionary = UndeleteDictionary.Field()
    create_organization = CreateOrganization.Field()
    update_organization = UpdateOrganization.Field()
    delete_organization = DeleteOrganization.Field()
    create_translationatom = CreateTranslationAtom.Field()
    update_translationatom = UpdateTranslationAtom.Field()
    delete_translationatom = DeleteTranslationAtom.Field()
    create_translationgist = CreateTranslationGist.Field()
    delete_translationgist = DeleteTranslationGist.Field()
    undelete_translationgist = UndeleteTranslationGist.Field()
    create_lexicalentry = CreateLexicalEntry.Field()
    delete_lexicalentry = DeleteLexicalEntry.Field()
    bulk_delete_lexicalentry = BulkDeleteLexicalEntry.Field()
    bulk_undelete_lexicalentry = BulkUndeleteLexicalEntry.Field()
    bulk_create_lexicalentry = BulkCreateLexicalEntry.Field()
    join_lexical_entry_group = ConnectLexicalEntries.Field()
    leave_lexical_entry_group = DeleteGroupingTags.Field()
    create_perspective = CreateDictionaryPerspective.Field()
    update_perspective = UpdateDictionaryPerspective.Field()
    update_perspective_status = UpdatePerspectiveStatus.Field()
    update_perspective_atom = UpdatePerspectiveAtom.Field()
    add_perspective_roles = AddPerspectiveRoles.Field()
    delete_perspective_roles = DeletePerspectiveRoles.Field()
    delete_perspective = DeleteDictionaryPerspective.Field()
    undelete_perspective = UndeleteDictionaryPerspective.Field()
    create_column = CreateColumn.Field()
    update_column = UpdateColumn.Field()
    delete_column = DeleteColumn.Field()
    reorder_columns = ReorderColumns.Field()
    create_grant = CreateGrant.Field()
    update_grant = UpdateGrant.Field()
    # delete_grant = DeleteGrant.Field()
    create_userblob = CreateUserBlob.Field()
    delete_userblob = DeleteUserBlob.Field()
    create_grant_permission = CreateGrantPermission.Field()
    add_dictionary_to_grant = AddDictionaryToGrant.Field()
    administrate_org = AdministrateOrg.Field()
    participate_org = ParticipateOrg.Field()
    add_dictionary_to_organization = AddDictionaryToOrganization.Field()
    accept_userrequest = AcceptUserRequest.Field()
    #delete_userrequest = DeleteUserRequest.Field()
    download_dictionary = DownloadDictionary.Field()
    save_dictionary = SaveDictionary.Field()
    save_all_dictionaries = SaveAllDictionaries.Field()
    download_dictionaries = DownloadDictionaries.Field()
    synchronize = Synchronize.Field()
    delete_task = DeleteTask.Field()
    starling_etymology = StarlingEtymology.Field()
    phonemic_analysis = PhonemicAnalysis.Field()
    cognate_analysis = CognateAnalysis.Field()
    swadesh_analysis = SwadeshAnalysis.Field()
    morph_cognate_analysis = MorphCognateAnalysis.Field()
    phonology = Phonology.Field()
    phonological_statistical_distance = PhonologicalStatisticalDistance.Field()
    sound_and_markup = SoundAndMarkup.Field()
    merge_bulk = MergeBulk.Field()
    move_column = MoveColumn.Field()
    add_roles_bulk = AddRolesBulk.Field()
    create_basegroup = CreateBasegroup.Field()
    add_user_to_basegroup = AddUserToBasegroup.Field()
    execute_parser = ExecuteParser.Field()
    delete_parser_result = DeleteParserResult.Field()
    update_parser_result = UpdateParserResult.Field()
    xlsx_bulk_disconnect = XlsxBulkDisconnect.Field()
    new_unstructured_data = NewUnstructuredData.Field()
    docx2eaf = Docx2Eaf.Field()
    docx2xlsx = Docx2Xlsx.Field()
    valency = Valency.Field()
    valency_verb_cases = ValencyVerbCases.Field()
    create_valency_data = CreateValencyData.Field()
    save_valency_data = SaveValencyData.Field()
    set_valency_annotation = SetValencyAnnotation.Field()
    bidirectional_links = BidirectionalLinks.Field()
    cognates_summary = CognatesSummary.Field()
    complex_distance = ComplexDistance.Field()
    tsakorpus = Tsakorpus.Field()

schema = graphene.Schema(
    query=Query,
    mutation=MyMutations,
    auto_camelcase=False
)

# Special value to differentiate between when client_id, client and user info is uninitialized and when it's
# None because the request has no authenticated client info.
context_none_value = object()


class Context(dict):
    """
    Context for graphene query execution, works as dict for arbitrary key-value associations and supports
    ACL-based permission checking.
    """

    def __init__(self, context_dict):
        """
        Initializes query execution context by initializing context key-value dictionary and, hopefully,
        getting client id and request data.
        """

        dict.__init__(self, context_dict)

        self.client_id = context_dict.get('client_id')
        self.request = context_dict.get('request')
        self.locale_id = context_dict.get('locale_id')
        self.headers = context_dict.get('headers')
        self.cookies = context_dict.get('cookies')

        self.client_object = context_none_value
        self.user_id_value = context_none_value
        self.user_object = context_none_value

        self.acl_cache = {}

    def acl_check_if(
        self,
        action,
        subject,
        subject_id):
        """
        Checks if the client has permission to perform given action on a specified subject via ACL.
        """

        if type(subject_id) is list:

            subject_id = tuple(subject_id)

        acl_cache_key = (
            action, subject, subject_id)

        result = (

            self.acl_cache.get(
                acl_cache_key))

        if result is not None:

            return result

        result = (

            acl.check_direct(
                self.client_id,
                self.request,
                action,
                subject,
                subject_id))

        self.acl_cache[
            acl_cache_key] = result

        return result

    def acl_check(
        self,
        action,
        subject,
        subject_id):
        """
        Checks if the client has permission to perform given action on a specified subject via ACL, raises
        permission exception otherwise.
        """

        check = (

            self.acl_check_if(
                action,
                subject,
                subject_id))

        if not check:

            raise (
                PermissionException(
                    self.client_id, action, subject, subject_id))

    def acl_check_with_id_if(self, action, subject, args):
        """
        Checks via ACL if the client has permission to perform given action on a specified subject, with
        subject identifier extracted from query execution arguments.
        """

        return self.acl_check_if(action, subject, args.get('id'))

    @property
    def client(self):
        """
        Returns current client object, or None if there is no current client.
        """

        if (client := self.client_object) is not context_none_value:
            return client

        self.client_object = Client.get(self.client_id)

        return self.client_object

    @property
    def user_id(self):
        """
        Returns user id of the current client, or None if there is no current client.
        """

        if (user_id := self.user_id_value) is not context_none_value:

            return user_id

        if (client := self.client_object) is not context_none_value:

            self.user_id_value = client.user_id
            return self.user_id_value

        if (user := self.user_object) is not context_none_value:

            self.user_id_value = user.id
            return self.user_id_value

        self.user_id_value = Client.get_user_id(self.client_id)

        return self.user_id_value

    @property
    def user(self):
        """
        Returns user of the current client, or None if there is no current client.
        """

        if (user := self.user_object) is not context_none_value:
            return user

        self.user_object = Client.get_user_by_client_id(self.client_id)

        return self.user_object

    def effective_client_id(self, client_id):
        """
        Checks if the current client is compatible with another specified client, returns effective client
        by id.

        If the specified client is None, effective client is the current client.

        Otherwise checks compatibility, which is when
        a) current client is the admin, or
        b) clients' are of the same user,
        raise exception if the clients are not compatible.

        If the clients are compatible, effective client is the specified client.
        """

        if client_id is None:
            return self.client_id

        user_id = self.user_id

        if (user_id != 1 and
            user_id != Client.get_user_id(client_id)):

            raise (
                PermissionException(
                    'Clients of mismatched users',
                    self.client_id,
                    client_id))

        return client_id

