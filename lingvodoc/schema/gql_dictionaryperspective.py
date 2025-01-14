
# Standard library imports.

import builtins
from collections import defaultdict
import datetime
import itertools
import logging
import pprint

# External imports.

import graphene

import sqlalchemy

from sqlalchemy import (
    asc,
    and_,
    cast,
    column,
    desc,
    extract,
    func,
    literal,
    or_,
    tuple_,
    union)

from sqlalchemy.orm import aliased, joinedload

from sqlalchemy.sql.expression import (
    case,
    Grouping,
    nullsfirst,
    nullslast)

# Lingvodoc imports.

from lingvodoc.cache.caching import CACHE

from lingvodoc.models import (
    BaseGroup as dbBaseGroup,
    Client as dbClient,
    DBSession,
    Dictionary as dbDictionary,
    DictionaryPerspective as dbPerspective,
    DictionaryPerspectiveToField as dbColumn,
    ENGLISH_LOCALE,
    Entity as dbEntity,
    Field as dbField,
    Group as dbGroup,
    JSONB,
    Language as dbLanguage,
    LexicalEntry as dbLexicalEntry,
    ObjectTOC,
    ParserResult as dbParserResult,
    PublishingEntity as dbPublishingEntity,
    TranslationAtom as dbTranslationAtom,
    TranslationGist as dbTranslationGist,
    User as dbUser,
    user_to_group_association,
    ValencyEafData as dbValencyEafData,
    ValencyParserData as dbValencyParserData,
    ValencySourceData as dbValencySourceData,
    ValencySentenceData as dbValencySentenceData,
    AdverbInstanceData as dbAdverbInstanceData)

from lingvodoc.schema.gql_column import Column
from lingvodoc.schema.gql_dictionary import Dictionary
from lingvodoc.schema.gql_entity import Entity

from lingvodoc.schema.gql_holders import (
    acl_check_by_id,
    client_id_check,
    CommonFieldsComposite,
    del_object,
    fetch_object,
    LingvodocID,
    LingvodocObjectType,
    ObjectVal,
    ResponseError,
    StateHolder,
    undel_object,
    UserAndOrganizationsRoles)

from lingvodoc.schema.gql_language import Language
from lingvodoc.schema.gql_lexicalentry import LexicalEntry
from lingvodoc.schema.gql_user import User

from lingvodoc.utils import (
    ids_to_id_query,
    render_statement,
    statistics)

from lingvodoc.utils.creation import (
    create_dictionary_persp_to_field,
    create_gists_with_atoms,
    create_perspective,
    edit_role,
    update_metadata)

from lingvodoc.utils.deletion import real_delete_perspective
from lingvodoc.utils.search import translation_gist_search
from pdb import set_trace as A


# Setting up logging.
log = logging.getLogger(__name__)


def graphene_track_multiple(
    lexes, # Can be either an entry id list or an entry query.
    publish = None,
    accept = None,
    delete = False,
    filter = None,
    sort_by_field = None,
    is_ascending = True,
    is_case_sens = True,
    is_regexp = False,
    have_empty = False,
    created_entries = [],
    offset = None,
    limit = None,
    check_perspective = True,
    debug_flag = False):

    # Getting our base data query.
    #
    # Using just entry ids from CTE if the incoming info is a query cause trying to re-use the query is too
    # complex, there's aliasing considerations and errors cropping up randomly, so we're choosing simple and
    # safe.

    data_query = (

        DBSession

            .query(
                dbLexicalEntry))

    if isinstance(lexes, list):

        data_query = (

            data_query

                .filter(
                    dbLexicalEntry.id.in_(
                        ids_to_id_query(lexes))))

    else:

        lexes_cte = (
            lexes.cte())

        data_query = (

            data_query

                .filter(
                    dbLexicalEntry.client_id == lexes_cte.c.client_id,
                    dbLexicalEntry.object_id == lexes_cte.c.object_id))

    # Can't have the usual compile with literal binds cause when used in connected_words can have additional
    # query params.

    log.debug(
        '\ndata_query:\n' +
        render_statement(data_query.statement))

    # Entity filtering conditions.

    filter_list = []
    filter_outer_list = []

    if accept is not None:

        filter_list.append(
            dbPublishingEntity.accepted == accept)

        filter_outer_list.append(
            or_(
                dbPublishingEntity.accepted == accept,
                dbPublishingEntity.accepted == None))

    if publish is not None:

        filter_list.append(
            dbPublishingEntity.published == publish)

        filter_outer_list.append(
            or_(
                dbPublishingEntity.published == publish,
                dbPublishingEntity.published == None))

    if delete is not None:

        filter_list.append(
            dbEntity.marked_for_deletion == delete)

        filter_outer_list.append(
            or_(
                dbEntity.marked_for_deletion == delete,
                dbEntity.marked_for_deletion == None))

    # First, created entries.

    new_entities_result = []
    entry_total_count = 0

    if created_entries and not filter:

        new_entities_result = (

            data_query

                .outerjoin(
                    dbEntity,
                    dbEntity.parent_id == dbLexicalEntry.id)

                .outerjoin(
                    dbPublishingEntity,
                    dbPublishingEntity.id == dbEntity.id)

                .add_entity(dbEntity)
                .add_entity(dbPublishingEntity)

                .filter(
                    dbLexicalEntry.id.in_(
                        ids_to_id_query(created_entries)),

                    *filter_outer_list)

                # For created entries we place last created first, for their entities last created go last.

                .order_by(
                    desc(dbLexicalEntry.created_at),
                    desc(dbLexicalEntry.client_id),
                    desc(dbLexicalEntry.object_id),
                    dbEntity.created_at,
                    dbEntity.client_id,
                    dbEntity.object_id)

                .all())

        # Skipping already retrieved entities if we have some.

        data_query = (

            data_query.filter(
                dbLexicalEntry.id.notin_(
                    ids_to_id_query(created_entries))))

        entry_total_count += (
            len(new_entities_result))

    # If we have any filter, we can be sure that there will be no empty entries.

    if filter:
        have_empty = False

    # If required, checking for perspective deletion.

    if check_perspective:

        data_query = (

            data_query.filter(
                dbLexicalEntry.parent_id == dbPerspective.id,
                dbPerspective.marked_for_deletion == False,
                dbPerspective.parent_id == dbDictionary.id,
                dbDictionary.marked_for_deletion == False))

    # We'll either get total entry count now or after filtering.

    if not filter:

        entry_total_count += (
            data_query.count())

        log.debug(
            f'\n entry_total_count: {entry_total_count}')

    # Simple case, we just need to slice the entries.

    if not (sort_by_field or have_empty or filter):

        data_query = (

            data_query

                .order_by(
                    desc(dbLexicalEntry.created_at),
                    desc(dbLexicalEntry.client_id),
                    desc(dbLexicalEntry.object_id)))

        if offset:

            data_query = (
                data_query.offset(offset))

            offset = None

        if limit:

            data_query = (
                data_query.limit(limit))

            limit = None

        offset_limit_cte = (

            data_query

                .with_entities(
                    dbLexicalEntry.client_id,
                    dbLexicalEntry.object_id)

                .cte())

        data_query = (

            DBSession

                .query(
                    dbLexicalEntry)

                .filter(
                    dbLexicalEntry.client_id == offset_limit_cte.c.client_id,
                    dbLexicalEntry.object_id == offset_limit_cte.c.object_id))

    # If we are going to have empty entities and we need to sort, we'll need an emptiness check.

    if have_empty or sort_by_field:

        data_query = (

            data_query.add_column(
                (dbEntity.client_id != None).label('is_not_empty')))

    data_query = (

        data_query

            .join(
                dbEntity,
                dbEntity.parent_id == dbLexicalEntry.id,
                isouter = have_empty)

            .join(
                dbPublishingEntity,
                dbPublishingEntity.id == dbEntity.id,
                isouter = have_empty)

            .add_entity(dbEntity)
            .add_entity(dbPublishingEntity)

            .with_labels()

            .filter(
                *(filter_outer_list if have_empty else filter_list)))

    log.debug(
        '\ndata_query:\n' +
        render_statement(data_query.statement))

    data_cte = None

    # Filtering if required.

    if filter:

        if data_cte is None:
            data_cte = data_query.cte()

        dbTextField = aliased(dbField)
        dbTextAtom = aliased(dbTranslationAtom)

        dbUrlField = aliased(dbField)
        dbUrlAtom = aliased(dbTranslationAtom)

        # Filtering by text context as it is, by URL content by first extracting URL's last part.

        if is_regexp:

            op_str = '~' if is_case_sens else '~*'
            filter_str = filter

        else:

            op_str = '~~' if is_case_sens else '~~*'
            filter_str = f'%{filter}%'

        text_filter_condition = (
            data_cte.c.entity_content
                .op(op_str)(filter_str))

        url_filter_condition = (
            func.substring(data_cte.c.entity_content, '.*/([^/]*)$')
                .op(op_str)(filter_str))

        text_filter_query = (

            DBSession

                .query(
                    data_cte.c.lexicalentry_client_id.label('entry_cid_f'),
                    data_cte.c.lexicalentry_object_id.label('entry_oid_f'))

                .filter(
                    dbTextField.client_id == data_cte.c.entity_field_client_id,
                    dbTextField.object_id == data_cte.c.entity_field_object_id,
                    dbTextAtom.parent_client_id == dbTextField.data_type_translation_gist_client_id,
                    dbTextAtom.parent_object_id == dbTextField.data_type_translation_gist_object_id,
                    dbTextAtom.locale_id == ENGLISH_LOCALE,
                    dbTextAtom.content == 'Text',
                    text_filter_condition)

                .group_by(
                    'entry_cid_f',
                    'entry_oid_f'))

        url_filter_query = (

            DBSession

                .query(
                    data_cte.c.lexicalentry_client_id.label('entry_cid_f'),
                    data_cte.c.lexicalentry_object_id.label('entry_oid_f'))

                .filter(
                    dbUrlField.client_id == data_cte.c.entity_field_client_id,
                    dbUrlField.object_id == data_cte.c.entity_field_object_id,
                    dbUrlAtom.parent_client_id == dbUrlField.data_type_translation_gist_client_id,
                    dbUrlAtom.parent_object_id == dbUrlField.data_type_translation_gist_object_id,
                    dbUrlAtom.locale_id == ENGLISH_LOCALE,
                    or_(
                        dbUrlAtom.content == 'Image',
                        dbUrlAtom.content == 'Sound',
                        dbUrlAtom.content == 'Markup'),
                    url_filter_condition)

                .group_by(
                    'entry_cid_f',
                    'entry_oid_f'))

        filter_cte = (

            text_filter_query
                .union(url_filter_query)
                .cte())

        data_query = (

            DBSession

                .query(
                    data_cte)

                .filter(
                    data_cte.c.lexicalentry_client_id == filter_cte.c.entry_cid_f,
                    data_cte.c.lexicalentry_object_id == filter_cte.c.entry_oid_f))

        log.debug(
            '\ndata_query:\n' +
            render_statement(data_query.statement))

        # Getting total entry count after filtering.

        entry_total_count += (

            data_query

                .with_entities(
                    literal(1))

                .group_by(
                    data_cte.c.lexicalentry_client_id,
                    data_cte.c.lexicalentry_object_id)

                .count())

        log.debug(
            f'\n entry_total_count: {entry_total_count}')

        data_cte = None

    # General sorting things.

    if is_ascending or not sort_by_field:

        agg_f = func.min
        bool_f = func.bool_and
        order_f = asc
        reverse_f = desc
        null_f = nullsfirst

    else:

        agg_f = func.max
        bool_f = func.bool_or
        order_f = desc
        reverse_f = asc
        null_f = nullslast

    # We get entry sorting data if we need to sort by field.

    if sort_by_field:

        if data_cte is None:
            data_cte = data_query.cte()

        sort_cte = (

            DBSession

                .query(
                    data_cte.c.lexicalentry_client_id.label('entry_cid_s'),
                    data_cte.c.lexicalentry_object_id.label('entry_oid_s'),
                    agg_f(func.lower(data_cte.c.entity_content)).label('sort_content'),
                    func.count().label('sort_count'))

                .filter(
                    data_cte.c.entity_field_client_id == sort_by_field[0],
                    data_cte.c.entity_field_object_id == sort_by_field[1])

                .group_by(
                    'entry_cid_s',
                    'entry_oid_s')

                .cte())

    # If we need to also offset and/or limit, we'll need an additional grouping step.

    if offset or limit:

        if data_cte is None:
            data_cte = data_query.cte()

        query_list = [
            data_cte.c.lexicalentry_client_id.label('entry_cid_ol'),
            data_cte.c.lexicalentry_object_id.label('entry_oid_ol'),
            agg_f(data_cte.c.lexicalentry_created_at).label('created_at')]

        ol_order_by_list = []
        d_order_by_list = []

        if have_empty:

            query_list.append(
                bool_f(data_cte.c.is_not_empty).label('is_not_empty'))

            ol_order_by_list.append(
                'is_not_empty')

            d_order_by_list.append(
                data_cte.c.is_not_empty)

        if sort_by_field:

            query_list.extend([
                agg_f(sort_cte.c.sort_content).label('sort_content'),
                agg_f(sort_cte.c.sort_count).label('sort_count')])

            ol_order_by_list.extend([
                null_f(order_f('sort_content')),
                null_f(order_f('sort_count'))])

        ol_order_by_list.extend([
            reverse_f('created_at'),
            reverse_f('entry_cid_ol'),
            reverse_f('entry_oid_ol')])

        offset_limit_query = (

            DBSession
                .query(
                    *query_list))

        if sort_by_field:

            offset_limit_query = (

                offset_limit_query

                    .outerjoin(
                        sort_cte,
                        and_(
                            data_cte.c.lexicalentry_client_id == sort_cte.c.entry_cid_s,
                            data_cte.c.lexicalentry_object_id == sort_cte.c.entry_oid_s)))

        offset_limit_query = (

            offset_limit_query

                .group_by(
                    'entry_cid_ol',
                    'entry_oid_ol')

                .order_by(
                    *ol_order_by_list))

        if offset:

            offset_limit_query = (
                offset_limit_query.offset(offset))

        if limit:

            offset_limit_query = (
                offset_limit_query.limit(limit))

        # Checking out ordering if required.

        if debug_flag:

            log.debug(
                '\n offset_limit_query.all(): ')

            offset_limit_query.all()

        offset_limit_cte = (
            offset_limit_query.cte())

        if sort_by_field:

            d_order_by_list.extend([
                null_f(order_f(offset_limit_cte.c.sort_content)),
                null_f(order_f(offset_limit_cte.c.sort_count))])

        d_order_by_list.extend([
            reverse_f(data_cte.c.lexicalentry_created_at),
            reverse_f(data_cte.c.lexicalentry_client_id),
            reverse_f(data_cte.c.lexicalentry_object_id)])

        if sort_by_field:

            d_order_by_list.append(

                order_f(
                    case(
                        [(and_(
                            data_cte.c.entity_field_client_id == sort_by_field[0],
                            data_cte.c.entity_field_object_id == sort_by_field[1]),
                            func.lower(data_cte.c.entity_content))],
                        else_ = None)))

        d_order_by_list.extend([
            order_f(data_cte.c.entity_created_at),
            order_f(data_cte.c.entity_client_id),
            order_f(data_cte.c.entity_object_id)])

        data_query = (

            DBSession

                .query(
                    aliased(dbLexicalEntry, data_cte),
                    aliased(dbEntity, data_cte),
                    aliased(dbPublishingEntity, data_cte))

                .filter(
                    data_cte.c.lexicalentry_client_id == offset_limit_cte.c.entry_cid_ol,
                    data_cte.c.lexicalentry_object_id == offset_limit_cte.c.entry_oid_ol)

                .order_by(
                    *d_order_by_list))

        data_cte = None

    # Nah, just sorting and we're done.

    else:

        if data_cte is None:
            data_cte = data_query.cte()

        data_query = (

            DBSession

                .query(
                    aliased(dbLexicalEntry, data_cte),
                    aliased(dbEntity, data_cte),
                    aliased(dbPublishingEntity, data_cte)))

        d_order_by_list = []

        if have_empty:

            d_order_by_list.append(
                data_cte.c.is_not_empty)

        if sort_by_field:

            data_query = (

                data_query

                    .outerjoin(
                        sort_cte,
                        and_(
                            data_cte.c.lexicalentry_client_id == sort_cte.c.entry_cid_s,
                            data_cte.c.lexicalentry_object_id == sort_cte.c.entry_oid_s)))

            d_order_by_list.extend([
                null_f(order_f(sort_cte.c.sort_content)),
                null_f(order_f(sort_cte.c.sort_count))])

        d_order_by_list.extend([
            reverse_f(data_cte.c.lexicalentry_created_at),
            reverse_f(data_cte.c.lexicalentry_client_id),
            reverse_f(data_cte.c.lexicalentry_object_id)])

        if sort_by_field:

            d_order_by_list.append(

                order_f(
                    case(
                        (and_(
                            data_cte.c.entity_field_client_id == sort_by_field[0],
                            data_cte.c.entity_field_object_id == sort_by_field[1]),
                            func.lower(data_cte.c.entity_content)),
                        else_ = None)))

        d_order_by_list.extend([
            order_f(data_cte.c.entity_created_at),
            order_f(data_cte.c.entity_client_id),
            order_f(data_cte.c.entity_object_id)])

        data_query = (

            data_query
                .order_by(
                    *d_order_by_list))

        data_cte = None

    log.debug(
        '\ndata_query:\n' +
        render_statement(data_query.statement))

    return (
        new_entities_result,
        data_query,
        entry_total_count)


def entries_with_entities(
    lexes,
    mode = None,
    is_edit_mode = True,
    created_entries = [],
    limit = None,
    offset = None,
    **query_args):

    if mode == 'debug':
        return [LexicalEntry(lex) for lex in lexes]

    if mode == 'not_accepted':
        query_args['accept'] = False
        query_args['delete'] = False

    new_entities, old_entities, total_count = (

        graphene_track_multiple(
            lexes,
            have_empty = is_edit_mode,
            created_entries = created_entries,
            offset = offset,
            limit = limit,
            **query_args))

    result_list = []

    for entry_id, lep_iter in (
        itertools.groupby(
            itertools.chain(new_entities, old_entities),
            lambda lep: lep[0].id)):

        lep_list = list(lep_iter)
        db_entry = lep_list[0][0]

        gql_entity_list = [

            Entity(
                db_entity,
                publishingentity = db_p_entity)

            for _, db_entity, db_p_entity in lep_list
            if db_entity]

        result_list.append(

            LexicalEntry(
                db_entry,
                gql_Entities = gql_entity_list))

    return result_list, total_count


class PerspectivePage(graphene.ObjectType):

    lexical_entries = graphene.List(LexicalEntry)
    entries_total = graphene.Int()


class DictionaryPerspective(LingvodocObjectType):
    """
     #created_at                       | timestamp without time zone | NOT NULL
     #object_id                        | bigint                      | NOT NULL
     #client_id                        | bigint                      | NOT NULL
     #parent_object_id                 | bigint                      |
     #parent_client_id                 | bigint                      |
     #translation_gist_client_id       | bigint                      | NOT NULL
     #translation_gist_object_id       | bigint                      | NOT NULL
     #state_translation_gist_client_id | bigint                      | NOT NULL
     #state_translation_gist_object_id | bigint                      | NOT NULL
     #marked_for_deletion              | boolean                     | NOT NULL
     #is_template                      | boolean                     | NOT NULL
     #import_source                    | text                        |
     #import_hash                      | text                        |
     #additional_metadata              | jsonb                       |
     + .translation
     + status
     + tree

    query myQuery {
      perspective(id: [78, 4]) {
        id
        statistic(starting_time: 0, ending_time: 1506812557)
        entities(mode: "all") {
          id
          parent_id
          published
          accepted
        }
        lexical_entries(ids: [[78, 6], [78, 8]]) {
          id
        }
            columns{
                id
                field_id
            }
      }
    }

    """
    data_type = graphene.String()

    import_source = graphene.String()
    import_hash = graphene.String()

    tree = graphene.List(CommonFieldsComposite, )  # TODO: check it
    columns = graphene.List(Column)

    lexical_entries = graphene.List(
        LexicalEntry,
        ids = graphene.List(LingvodocID),
        mode = graphene.String())

    perspective_page = graphene.Field(
        PerspectivePage,
        ids = graphene.List(LingvodocID),
        mode = graphene.String(),
        filter = graphene.String(),
        is_regexp = graphene.Boolean(),
        is_case_sens = graphene.Boolean(),
        is_edit_mode = graphene.Boolean(),
        is_ascending = graphene.Boolean(),
        sort_by_field = LingvodocID(),
        offset = graphene.Int(),
        limit = graphene.Int(),
        created_entries = graphene.List(LingvodocID),
        debug_flag = graphene.Boolean())

    authors = graphene.List('lingvodoc.schema.gql_user.User')
    roles = graphene.Field(UserAndOrganizationsRoles)
    role_check = graphene.Boolean(subject = graphene.String(required = True), action = graphene.String(required = True))

    statistic = (

        graphene.Field(
            ObjectVal,
            starting_time = graphene.Int(),
            ending_time = graphene.Int(),
            disambiguation_flag = graphene.Boolean()))

    is_template = graphene.Boolean()
    counter = graphene.Int(mode=graphene.String())
    last_modified_at = graphene.Float()

    is_hidden_for_client = graphene.Boolean()
    has_valency_data = graphene.Boolean()
    has_adverb_data = graphene.Boolean()
    new_valency_data_count = graphene.Int()
    new_adverb_data_count = graphene.Int()

    dbType = dbPerspective

    entries_total = 0

    class Meta:
        interfaces = (CommonFieldsComposite, StateHolder)

    def check_is_hidden_for_client(self, info):
        """
        Checks if the perspective is hidden for the current client.

        Perspective is hidden for the current client if either it or its dictionary status is 'Hidden' and
        it is not in the 'Available dictionaries' list for the client, see 'def resolve_dictionaries()' in
        query.py switching based on 'mode'.
        """

        try:
            return self.is_hidden_for_client_flag

        except AttributeError:
            pass

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
                    dbTranslationAtom.locale_id == 2)

                .first())

        # Checking if either the perspective or its dictionary has 'Hidden' status.

        is_hidden = (
            self.dbObject.state_translation_gist_client_id == hidden_id[0] and
            self.dbObject.state_translation_gist_object_id == hidden_id[1])

        if not is_hidden:

            is_hidden = (

                DBSession

                    .query(
                        and_(
                            dbDictionary.state_translation_gist_client_id == hidden_id[0],
                            dbDictionary.state_translation_gist_object_id == hidden_id[1]))

                    .filter(
                        dbDictionary.client_id == self.dbObject.parent_client_id,
                        dbDictionary.object_id == self.dbObject.parent_object_id)

                    .scalar())

        if not is_hidden:

            self.is_hidden_for_client_flag = False
            return False

        # Perspective is hidden, checking if it's hidden for the client.

        client_id = info.context.request.authenticated_userid

        if not client_id:

            self.is_hidden_for_client_flag = True
            return True

        user = dbClient.get_user_by_client_id(client_id)

        if user.id == 1:

            self.is_hidden_for_client_flag = False
            return False

        # Not an admin, we check if the perspective's dictionary is available for the client, see 'available
        # dictionaries' branch in resolve_dictionaries() in query.py.

        exists_query = (

            DBSession

                .query(
                    literal(1))

                .filter(
                    user_to_group_association.c.user_id == user.id,
                    dbGroup.id == user_to_group_association.c.group_id,
                    dbBaseGroup.id == dbGroup.base_group_id,

                    or_(
                        and_(
                            dbGroup.subject_override,
                            or_(
                                dbBaseGroup.dictionary_default,
                                dbBaseGroup.perspective_default)),
                        and_(
                            dbGroup.subject_client_id == self.dbObject.client_id,
                            dbGroup.subject_object_id == self.dbObject.object_id),
                        and_(
                            dbGroup.subject_client_id == self.dbObject.parent_client_id,
                            dbGroup.subject_object_id == self.dbObject.parent_object_id,
                            dbBaseGroup.dictionary_default)))

                .exists())

        is_available = (

            DBSession
                .query(exists_query)
                .scalar())

        self.is_hidden_for_client_flag = not is_available
        return self.is_hidden_for_client_flag

    # @fetch_object()
    # def resolve_additional_metadata(self, args, context, info):
    #     return self.dbObject.additional_metadata

    # @fetch_object('translation')
    # def resolve_translation(self, args, context, info):
    #     return self.dbObject.get_translation(context.get('locale_id'))

    @fetch_object('is_template')
    def resolve_is_template(self, info):
        return self.dbObject.is_template

    @fetch_object('tree') # tested
    def resolve_tree(self, info):

        dictionary_db = self.dbObject.parent
        dictionary = Dictionary(id = dictionary_db.id)

        return [self] + dictionary.resolve_tree(info)

    @fetch_object('columns') # tested
    def resolve_columns(self, info):
        columns = DBSession.query(dbColumn).filter_by(parent=self.dbObject, marked_for_deletion=False).order_by(dbColumn.position).all()
        result = list()
        for dbfield in columns:
            gr_field_obj = Column(id=[dbfield.client_id, dbfield.object_id])
            gr_field_obj.dbObject = dbfield
            result.append(gr_field_obj)
        return result

    @fetch_object()
    def resolve_counter(self, info, mode):

        lexes = (

            DBSession

                .query(
                    dbLexicalEntry)

                .filter(
                    dbLexicalEntry.parent == self.dbObject)

                .join(
                    dbEntity,
                    dbEntity.parent_id == dbLexicalEntry.id)

                .join(
                    dbPublishingEntity,
                    dbPublishingEntity.id == dbEntity.id))

        if mode == 'all':
            # info.context.acl_check('view', 'lexical_entries_and_entities',
            #                        (self.dbObject.client_id, self.dbObject.object_id))
            counter_query = lexes.filter(dbPublishingEntity.accepted == True, dbLexicalEntry.marked_for_deletion == False,
                                 dbEntity.marked_for_deletion == False)
        elif mode == 'published':
            counter_query = lexes.filter(dbPublishingEntity.published == True, dbLexicalEntry.marked_for_deletion == False,
                                 dbEntity.marked_for_deletion == False)
        elif mode == 'unpublished':
            counter_query = lexes.filter(dbPublishingEntity.published == False, dbLexicalEntry.marked_for_deletion == False,
                                 dbEntity.marked_for_deletion == False)
        elif mode == 'not_accepted':
            counter_query = lexes.filter(dbPublishingEntity.accepted == False, dbLexicalEntry.marked_for_deletion == False,
                                 dbEntity.marked_for_deletion == False)
        else:
            raise ResponseError(message="mode: <all|published|unpublished|not_accepted>")
        counter = counter_query.group_by(dbLexicalEntry.id).count()
        return counter

    @fetch_object('last_modified_at')
    def resolve_last_modified_at(self, info):
        """
        Perspective's last modification time, defined as latest time of creation or deletion of the
        perspective and all its lexical entries and entities.
        """

        # select
        #   max((value ->> 'deleted_at') :: float)
        #
        #   from
        #     ObjectTOC,
        #     jsonb_each(additional_metadata)
        #
        #   where
        #     client_id = <client_id> and
        #     object_id = <object_id>;

        deleted_at_query = (

            DBSession

            .query(
                func.max(cast(
                    column('value').op('->>')('deleted_at'),
                    sqlalchemy.Float)))

            .select_from(
                ObjectTOC,
                func.jsonb_each(ObjectTOC.additional_metadata))

            .filter(
                ObjectTOC.client_id == self.dbObject.client_id,
                ObjectTOC.object_id == self.dbObject.object_id,
                ObjectTOC.additional_metadata != JSONB.NULL))

        # Query for last modification time of the perspective's lexical entries and entities.

        sql_str = ('''

            select

              max(
                greatest(

                  extract(epoch from L.created_at),

                  (select
                    max((value ->> 'deleted_at') :: float)

                    from
                      jsonb_each(OL.additional_metadata)),

                  (select

                    max(
                      greatest(

                        extract(epoch from E.created_at),

                        (select
                          max((value ->> 'deleted_at') :: float)

                          from
                            jsonb_each(OE.additional_metadata))))

                    from
                      public.entity E,
                      ObjectTOC OE

                    where
                      E.parent_client_id = L.client_id and
                      E.parent_object_id = L.object_id and
                      OE.client_id = E.client_id and
                      OE.object_id = E.object_id and
                      OE.additional_metadata != 'null' :: jsonb)))

            from
              lexicalentry L,
              ObjectTOC OL

            where
              L.parent_client_id = :client_id and
              L.parent_object_id = :object_id and
              OL.client_id = L.client_id and
              OL.object_id = L.object_id and
              OL.additional_metadata != 'null' :: jsonb

            ''')

        # Complete query for the perspective, excluding created_at which we already have.

        DBSession.execute(
            'set extra_float_digits to 3;')

        result = (

            DBSession

            .query(
                  func.greatest(
                      deleted_at_query.label('deleted_at'),
                      Grouping(sqlalchemy.text(sql_str))))

            .params({
                'client_id': self.dbObject.client_id,
                'object_id': self.dbObject.object_id})

            .scalar())

        if result is not None:

            return max(
                self.dbObject.created_at,
                result)

        else:

            return self.dbObject.created_at

    @fetch_object()
    def resolve_is_hidden_for_client(self, info):
        """
        If the perspective is hidden for the current client.
        """

        return self.check_is_hidden_for_client(info)

    def resolve_has_valency_data(self, info):
        """
        If the perspective has valency annotation data.
        """

        exists_query = (

            DBSession

                .query(
                    literal(1))

                .filter(
                    dbValencySourceData.perspective_client_id == self.id[0],
                    dbValencySourceData.perspective_object_id == self.id[1])

                .exists())

        return (

            DBSession
                .query(exists_query)
                .scalar())

    def resolve_has_adverb_data(self, info):
        """
        If the perspective has adverb annotation data.
        """

        exists_query = (
            DBSession

                .query(
                    literal(1))

                .filter(
                    dbValencySourceData.perspective_client_id == self.id[0],
                    dbValencySourceData.perspective_object_id == self.id[1],
                    dbValencyParserData.id == dbValencySourceData.id,
                    dbValencyParserData.hash_adverb != '')

                .exists())

        return (
            DBSession
                .query(exists_query)
                .scalar())

    def resolve_new_valency_data_count(self, info):
        """
        How many unprocessed valency sources perspective has.
        """

        debug_flag = False

        total_hash_union = (

            union(

                DBSession

                    .query(

                        func.encode(
                            func.digest(
                                dbParserResult.content, 'sha256'),
                            'hex')

                            .label('hash'))

                    .filter(
                        dbLexicalEntry.parent_client_id == self.id[0],
                        dbLexicalEntry.parent_object_id == self.id[1],
                        dbLexicalEntry.marked_for_deletion == False,
                        dbEntity.parent_client_id == dbLexicalEntry.client_id,
                        dbEntity.parent_object_id == dbLexicalEntry.object_id,
                        dbEntity.marked_for_deletion == False,
                        dbPublishingEntity.client_id == dbEntity.client_id,
                        dbPublishingEntity.object_id == dbEntity.object_id,
                        dbPublishingEntity.published == True,
                        dbPublishingEntity.accepted == True,
                        dbParserResult.entity_client_id == dbEntity.client_id,
                        dbParserResult.entity_object_id == dbEntity.object_id,
                        dbParserResult.marked_for_deletion == False),

                DBSession

                    .query(

                        cast(
                            dbEntity.additional_metadata['hash'],
                            sqlalchemy.UnicodeText)

                            .label('hash'))

                    .filter(
                        dbLexicalEntry.parent_client_id == self.id[0],
                        dbLexicalEntry.parent_object_id == self.id[1],
                        dbLexicalEntry.marked_for_deletion == False,
                        dbEntity.parent_client_id == dbLexicalEntry.client_id,
                        dbEntity.parent_object_id == dbLexicalEntry.object_id,
                        dbEntity.marked_for_deletion == False,
                        dbEntity.content.ilike('%.eaf'),
                        dbEntity.additional_metadata.contains({'data_type': 'elan markup'}),
                        dbPublishingEntity.client_id == dbEntity.client_id,
                        dbPublishingEntity.object_id == dbEntity.object_id,
                        dbPublishingEntity.published == True,
                        dbPublishingEntity.accepted == True))

                .alias())

        total_hash_subquery = (

            DBSession
                .query(total_hash_union)
                .subquery())

        total_hash_count = (

            DBSession
                .query(total_hash_union)
                .count())

        if debug_flag:
            log.debug(
                f'total_hash_count: {total_hash_count}')

        has_hash_union = (

            union(

                DBSession

                    .query(
                        dbValencyParserData.hash)

                    .filter(
                        dbValencySourceData.perspective_client_id == self.id[0],
                        dbValencySourceData.perspective_object_id == self.id[1],
                        dbValencyParserData.id == dbValencySourceData.id),

                DBSession

                    .query(
                        dbValencyEafData.hash)

                    .filter(
                        dbValencySourceData.perspective_client_id == self.id[0],
                        dbValencySourceData.perspective_object_id == self.id[1],
                        dbValencyEafData.id == dbValencySourceData.id))

                .alias())

        has_hash_count = (

            DBSession
                .query(has_hash_union)
                .count())

        if debug_flag:
            log.debug(
                f'has_hash_count: {has_hash_count}')

        new_hash_count = (

            DBSession

                .query(
                    total_hash_subquery.c.hash)

                .filter(
                    total_hash_subquery.c.hash.notin_(
                        has_hash_union))

                .count())

        if debug_flag:

            log.debug(
                f'new_hash_count: {new_hash_count}')

        return new_hash_count + (has_hash_count > total_hash_count)

    def resolve_new_adverb_data_count(self, info):
        """
        How many unprocessed adverb sources perspective has.
        """

        debug_flag = False

        ready_hash_subquery = (
            DBSession

                .query(

                    func.encode(
                        func.digest(
                            dbParserResult.content, 'sha256'),
                        'hex')

                        .label('hash'))

                .filter(
                    dbLexicalEntry.parent_client_id == self.id[0],
                    dbLexicalEntry.parent_object_id == self.id[1],
                    dbLexicalEntry.marked_for_deletion == False,
                    dbEntity.parent_client_id == dbLexicalEntry.client_id,
                    dbEntity.parent_object_id == dbLexicalEntry.object_id,
                    dbEntity.marked_for_deletion == False,
                    dbPublishingEntity.client_id == dbEntity.client_id,
                    dbPublishingEntity.object_id == dbEntity.object_id,
                    dbPublishingEntity.published == True,
                    dbPublishingEntity.accepted == True,
                    dbParserResult.entity_client_id == dbEntity.client_id,
                    dbParserResult.entity_object_id == dbEntity.object_id,
                    dbParserResult.marked_for_deletion == False)

                .subquery())

        ready_hash_count = (
            DBSession
                .query(ready_hash_subquery)
                .count())

        has_hash_subquery = (
            DBSession

                .query(
                    dbValencyParserData.hash_adverb)

                .filter(
                    dbValencySourceData.perspective_client_id == self.id[0],
                    dbValencySourceData.perspective_object_id == self.id[1],
                    dbValencyParserData.id == dbValencySourceData.id,
                    dbValencyParserData.hash_adverb != '')

                .subquery())

        has_hash_count = (
            DBSession
                .query(has_hash_subquery)
                .count())

        if debug_flag:
            log.debug(
                f'ready_hash_count: {ready_hash_count}\n'
                f'has_hash_count: {has_hash_count}')

        new_hash_count = (
            DBSession

                .query(
                    ready_hash_subquery.c.hash)

                .filter(
                    ready_hash_subquery.c.hash.notin_(
                        has_hash_subquery))

                .count())

        if debug_flag:

            log.debug(
                f'new_hash_count: {new_hash_count}')

        # Actually here we answer if database has sources with old hash_adverbs,
        # with wrong (maybe deleted) related parser results or duplicate sources
        return new_hash_count + (has_hash_count > ready_hash_count)

    @fetch_object()
    def resolve_lexical_entries(self, info, ids=None,
                                mode=None, authors=None, clients=None,
                                start_date=None, end_date=None, position=1,
                                **query_args):

        if self.check_is_hidden_for_client(info):
            return []

        if mode == 'all':
            publish = None
            accept = True
            delete = False
            info.context.acl_check('view', 'lexical_entries_and_entities',
                                   (self.dbObject.client_id, self.dbObject.object_id))
        elif mode == 'published':
            publish = True
            accept = True
            delete = False
        elif mode == 'unpublished':
            publish = False
            accept = True
            delete = False
        elif mode == 'not_accepted':
            publish = None
            accept = False
            delete = False
        elif mode == 'deleted':
            publish = None
            accept = None
            delete = True
            info.context.acl_check('view', 'lexical_entries_and_entities',
                                   (self.dbObject.client_id, self.dbObject.object_id))
        elif mode == 'all_with_deleted':
            publish = None
            accept = None
            delete = None
            info.context.acl_check('view', 'lexical_entries_and_entities',
                                   (self.dbObject.client_id, self.dbObject.object_id))
        elif mode == 'debug':
            publish = None
            accept = True
            delete = False
            info.context.acl_check('view', 'lexical_entries_and_entities',
                                   (self.dbObject.client_id, self.dbObject.object_id))
        else:
            raise ResponseError(message="mode: <all|published|unpublished|not_accepted|deleted|all_with_deleted>")

        lexes = (

            DBSession

                .query(
                    dbLexicalEntry.client_id,
                    dbLexicalEntry.object_id)

                .filter(
                    dbLexicalEntry.parent == self.dbObject))

        if ids is not None:
            id_info = list(ids)
            if len(ids) > 2:
                id_info = ids_to_id_query(id_info)
            lexes = lexes.filter(tuple_(dbLexicalEntry.client_id, dbLexicalEntry.object_id).in_(id_info))
        if authors or start_date or end_date:
            lexes = lexes.join(dbLexicalEntry.entity).join(dbEntity.publishingentity)

        if delete is not None:
            if authors or start_date or end_date:
                lexes = lexes.filter(or_(dbLexicalEntry.marked_for_deletion == delete, dbEntity.marked_for_deletion == delete))
            else:
                lexes = lexes.filter(dbLexicalEntry.marked_for_deletion == delete)
        if authors:
            lexes = lexes.join(dbClient, dbEntity.client_id == dbClient.id).join(dbClient.user).filter(dbUser.id.in_(authors))
        if start_date:
            lexes = lexes.filter(dbEntity.created_at >= start_date)
        if end_date:
            lexes = lexes.filter(dbEntity.created_at <= end_date)

        db_la_gist = translation_gist_search('Limited access')
        limited_client_id, limited_object_id = db_la_gist.client_id, db_la_gist.object_id

        if (self.dbObject.state_translation_gist_client_id == limited_client_id and
                self.dbObject.state_translation_gist_object_id == limited_object_id and
                mode != 'not_accepted'):

            if not info.context.acl_check_if('view', 'lexical_entries_and_entities',
                                             (self.dbObject.client_id, self.dbObject.object_id)):

                lexes = lexes.limit(20)

        lexical_entries, self.entries_total = (
            entries_with_entities(lexes, mode, accept=accept, delete=delete, publish=publish,
                                  check_perspective = False, **query_args))

        # If we were asked for specific lexical entries, we try to return them in creation order.

        if ids is not None:
            lexical_entries.sort(key = lambda e: (e.dbObject.created_at, e.dbObject.object_id))

        return lexical_entries

    def resolve_perspective_page(
            self,
            info,
            **query_args):

        return PerspectivePage(
            lexical_entries = self.resolve_lexical_entries(info, **query_args),
            entries_total = self.entries_total)

    @fetch_object()
    def resolve_authors(self, info):
        client_id, object_id = self.dbObject.client_id, self.dbObject.object_id

        parent = DBSession.query(dbPerspective).filter_by(client_id=client_id, object_id=object_id).first()
        if parent and not parent.marked_for_deletion:
            authors = DBSession.query(dbUser).join(dbUser.clients).join(dbEntity, dbEntity.client_id == dbClient.id) \
                .join(dbEntity.parent).join(dbEntity.publishingentity) \
                .filter(dbLexicalEntry.parent_client_id == parent.client_id,# TODO: filter by accepted==True
                        dbLexicalEntry.parent_object_id == parent.object_id,
                        dbLexicalEntry.marked_for_deletion == False,
                        dbEntity.marked_for_deletion == False)

            authors_list = [User(id=author.id,
                                 name=author.name,
                                 intl_name=author.intl_name,
                                 login=author.login) for author in authors]
            return authors_list
        raise ResponseError(message="Error: no such perspective in the system.")

    @fetch_object(ACLSubject='perspective_role', ACLKey='id')
    def resolve_roles(self, info):
        client_id, object_id = self.dbObject.client_id, self.dbObject.object_id
        perspective = DBSession.query(dbPerspective).filter_by(client_id=client_id, object_id=object_id).first()
        if not perspective or perspective.marked_for_deletion:
            raise ResponseError(message="Perspective with such ID doesn`t exists in the system")


        bases = DBSession.query(dbBaseGroup).filter_by(perspective_default=True)
        roles_users = defaultdict(list)
        roles_organizations = defaultdict(list)
        for base in bases:
            group = DBSession.query(dbGroup).filter_by(base_group_id=base.id,
                                                     subject_object_id=object_id,
                                                     subject_client_id=client_id).first()
            if not group:
                continue
            for user in group.users:
                roles_users[user.id].append(base.id)
            for org in group.organizations:
                roles_organizations[org.id].append(base.id)
        roles_users = [{"user_id": x, "roles_ids": roles_users[x]} for x in roles_users]
        roles_organizations = [{"user_id": x, "roles_ids": roles_organizations[x]} for x in roles_organizations]
        return UserAndOrganizationsRoles(roles_users=roles_users, roles_organizations=roles_organizations)

    @fetch_object()
    def resolve_role_check(self, info, subject = '', action = ''):

        # Checking for specified permission for the current user for the perspective.

        return (
            info.context.acl_check_if(
                action, subject, (self.dbObject.client_id, self.dbObject.object_id)))

    @fetch_object()
    def resolve_statistic(
        self,
        info,
        starting_time = None,
        ending_time = None,
        disambiguation_flag = False):

        return (

            statistics.new_format(
                statistics.stat_perspective(
                    self.id,
                    starting_time,
                    ending_time,
                    disambiguation_flag,
                    locale_id = info.context.locale_id)))


class CreateDictionaryPerspective(graphene.Mutation):
    """
    example:
    mutation  {
            create_perspective( parent_id:[66,4], translation_gist_id: [714, 3],is_template: true
             additional_metadata: {hash:"1234567"}, import_source: "source", import_hash: "hash") {
                triumph

                perspective{
                    is_template
                    id
                }
            }
    }
    (this example works)
    returns:
    {
        "data": {
            "create_perspective": {
                "triumph": true,
                "perspective": {
                    "id": [
                        1197,
                        320
                    ]
                }
            }
        }
    }
    with atoms:
    mutation {
      create_perspective(parent_id: [1198, 16], translation_atoms: [{locale_id: 2, content: "123"}], additional_metadata: {hash: "1234567"}, import_source: "source", import_hash: "hash") {
        triumph
        perspective {
          id
          translation
        }
      }
    }

    """

    class Arguments:
        id = LingvodocID()
        parent_id = LingvodocID(required=True)
        translation_gist_id = LingvodocID()
        translation_atoms = graphene.List(ObjectVal)
        additional_metadata = ObjectVal()
        import_source = graphene.String()
        import_hash = graphene.String()
        is_template = graphene.Boolean()
        fields = graphene.List(ObjectVal)

    perspective = graphene.Field(DictionaryPerspective)
    triumph = graphene.Boolean()


    @staticmethod
    @client_id_check()
    @acl_check_by_id('create', 'perspective', id_key = "parent_id")
    def mutate(root, info, **args):
        id = args.get("id")
        client_id = id[0] if id else info.context["client_id"]
        object_id = id[1] if id else None
        id = [client_id, object_id]
        parent_id = args.get('parent_id')
        translation_gist_id = args.get('translation_gist_id')
        translation_atoms = args.get("translation_atoms")

        translation_gist_id = create_gists_with_atoms(translation_atoms,
                                                      translation_gist_id,
                                                      [client_id,object_id],
                                                      gist_type="Perspective")
        import_source = args.get('import_source')
        import_hash = args.get('import_hash')
        additional_metadata = args.get('additional_metadata')
        is_template = args.get("is_template")

        field_info_list = args.get('fields')

        dbperspective = create_perspective(id=id,
                                parent_id=parent_id,
                                translation_gist_id=translation_gist_id,
                                additional_metadata=additional_metadata,
                                import_source=import_source,
                                import_hash=import_hash,
                                is_template=is_template
                                )

        perspective_id = (
            (dbperspective.client_id, dbperspective.object_id))

        perspective = DictionaryPerspective(id = perspective_id)
        perspective.dbObject = dbperspective

        # Creating fields, if required.

        if field_info_list:

            log.debug(
                '\nfield_info_list:\n' +
                pprint.pformat(
                    field_info_list, width = 192))

            counter = 0
            fake_id_dict = {}

            for field_info in field_info_list:

                counter += 1

                self_id = field_info['self_id']

                if self_id is not None:

                    if self_id not in fake_id_dict:
                        raise ResponseError(f'Unknown fake id \'{self_id}\'.')

                    self_id = fake_id_dict[self_id]

                persp_to_field = (

                    create_dictionary_persp_to_field(
                        id = (client_id, None),
                        parent_id = perspective_id,
                        field_id = field_info['field_id'],
                        self_id = self_id,
                        link_id = field_info['link_id'],
                        position = counter))

                if 'id' in field_info:

                    fake_id_dict[field_info['id']] = (
                        (persp_to_field.client_id, persp_to_field.object_id))

        return CreateDictionaryPerspective(perspective=perspective, triumph=True)


class UpdateDictionaryPerspective(graphene.Mutation):
    """
    example:
      mutation  {
            update_perspective(id:[949,2491], parent_id:[449,2491], translation_gist_id: [714, 3]) {
                triumph
                perspective{
                    id
                }
            }
    }

    (this example works)
    returns:

    {
      "update_perspective": {
        "triumph": true,
        "perspective": {
          "id": [
            949,
            2491
          ],
        }
      }
    }
    """
    class Arguments:
        id = LingvodocID(required=True)  #List(graphene.Int) # lingvidicID
        translation_gist_id = LingvodocID()
        parent_id = LingvodocID()
        additional_metadata = ObjectVal()

    perspective = graphene.Field(DictionaryPerspective)
    triumph = graphene.Boolean()

    @staticmethod
    @acl_check_by_id('edit', 'perspective')
    def mutate(root, info, **args):
        id = args.get("id")
        client_id = id[0]
        object_id = id[1]
        parent_id = args.get('parent_id')
        additional_metadata = args.get('additional_metadata')
        # dbperspective = DBSession.query(dbPerspective).filter_by(client_id=client_id, object_id=object_id).first()
        dbperspective = CACHE.get(objects =
            {
                dbPerspective : ((client_id, object_id), )
            },
        DBSession=DBSession)
        if not dbperspective or dbperspective.marked_for_deletion:
            raise ResponseError(message="Error: No such perspective in the system")

        # dictionaryperspective_parent_object_id_fkey  (parent_object_id, parent_client_id)=(2491, 449)  in dictionary
        translation_gist_id = args.get("translation_gist_id")
        translation_gist_client_id = translation_gist_id[0] if translation_gist_id else None
        translation_gist_object_id = translation_gist_id[1] if translation_gist_id else None
        if translation_gist_client_id:
            dbperspective.translation_gist_client_id = translation_gist_client_id
        if translation_gist_object_id:
            dbperspective.translation_gist_object_id = translation_gist_object_id  # TODO: refactor like dictionaries
        if parent_id:
            parent_client_id, parent_object_id = parent_id
            # dbparent_dictionary = DBSession.query(dbDictionary).filter_by(client_id=parent_client_id,
            #                                                               object_id=parent_object_id).first()
            dbparent_dictionary = CACHE.get(objects=
                {
                    dbDictionary : (parent_id, )
                },
            DBSession=DBSession)
            if not dbparent_dictionary:
                raise ResponseError(message="Error: No such dictionary in the system")
            dbperspective.parent_client_id = parent_client_id
            dbperspective.parent_object_id = parent_object_id

        update_metadata(dbperspective, additional_metadata)

        CACHE.set(objects = [dbperspective,], DBSession=DBSession)
        perspective = DictionaryPerspective(id=[dbperspective.client_id, dbperspective.object_id])
        perspective.dbObject = dbperspective
        return UpdateDictionaryPerspective(perspective=perspective, triumph=True)

class UpdatePerspectiveStatus(graphene.Mutation):
    """
    mutation  {
    update_perspective_status(id:[66, 5], state_translation_gist_id: [1, 192]) {
        triumph
        perspective{
            id
        }
    }
    }

    """
    class Arguments:
        id = LingvodocID(required=True)
        state_translation_gist_id = LingvodocID(required=True)

    perspective = graphene.Field(DictionaryPerspective)
    triumph = graphene.Boolean()

    @staticmethod
    @acl_check_by_id("edit", "perspective_status")
    def mutate(root, info, **args):
        client_id, object_id = args.get('id')
        state_translation_gist_client_id, state_translation_gist_object_id = args.get('state_translation_gist_id')
        # dbperspective = DBSession.query(dbPerspective).filter_by(client_id=client_id, object_id=object_id).first()
        dbperspective = CACHE.get(objects =
            {
                dbPerspective : ((client_id, object_id), )
            },
        DBSession=DBSession)
        if dbperspective and not dbperspective.marked_for_deletion:
            dbperspective.state_translation_gist_client_id = state_translation_gist_client_id
            dbperspective.state_translation_gist_object_id = state_translation_gist_object_id
            atom = DBSession.query(dbTranslationAtom).filter_by(parent_client_id=state_translation_gist_client_id,
                                                              parent_object_id=state_translation_gist_object_id,
                                                              locale_id=info.context.get('locale_id')).first()
            perspective = DictionaryPerspective(id=[dbperspective.client_id, dbperspective.object_id],
                                                status=atom.content)
            perspective.dbObject = dbperspective
            CACHE.set(objects = [dbperspective,], DBSession=DBSession)
            return UpdatePerspectiveStatus(perspective=perspective, triumph=True)

class AddPerspectiveRoles(graphene.Mutation):
    """
    mutation myQuery {
        add_perspective_roles(id: [1279,7], user_id:2 , roles_users:[8,12,13,15,20,21,22,23,24,26,16,34]){
                    triumph

                }
    }
    """
    class Arguments:
        id = LingvodocID(required=True)
        user_id = graphene.Int(required=True)
        roles_users = graphene.List(graphene.Int)
        roles_organizations = graphene.List(graphene.Int)

    perspective = graphene.Field(DictionaryPerspective)
    triumph = graphene.Boolean()

    @staticmethod
    @acl_check_by_id("create", "perspective_role")
    def mutate(root, info, **args):
        perspective_client_id, perspective_object_id = args.get('id')
        user_id = args.get("user_id")
        roles_users = args.get('roles_users')
        roles_organizations = args.get('roles_organizations')
        # dbperspective = DBSession.query(dbPerspective).filter_by(client_id=perspective_client_id, object_id=perspective_object_id).first()
        dbperspective = CACHE.get(objects =
            {
                dbPerspective : (args.get('id'), )
            },
        DBSession=DBSession)
        client_id = info.context.get('client_id')
        if not dbperspective or dbperspective.marked_for_deletion:
            raise ResponseError(message="No such perspective in the system")
        if roles_users:
            for role_id in roles_users:
                edit_role(dbperspective, user_id, role_id, client_id, perspective_default=True)
        if roles_organizations:
            for role_id in roles_organizations:
                edit_role(dbperspective, user_id, role_id, client_id, perspective_default=True, organization=True)
        perspective = Dictionary(id=[dbperspective.client_id, dbperspective.object_id])
        perspective.dbObject = dbperspective
        CACHE.set(objects = [dbperspective,], DBSession=DBSession)
        return AddPerspectiveRoles(perspective=perspective, triumph=True)


class DeletePerspectiveRoles(graphene.Mutation):
    class Arguments:
        id = LingvodocID(required=True)
        user_id = graphene.Int(required=True)
        roles_users = graphene.List(graphene.Int)
        roles_organizations = graphene.List(graphene.Int)

    perspective = graphene.Field(Dictionary)
    triumph = graphene.Boolean()

    @staticmethod
    @acl_check_by_id("delete", "perspective_role")
    def mutate(root, info, **args):
        perspective_client_id, perspective_object_id = args.get('id')
        user_id = args.get("user_id")
        roles_users = args.get('roles_users')
        roles_organizations = args.get('roles_organizations')
        # dbperspective = DBSession.query(dbPerspective).filter_by(client_id=perspective_client_id,
        #                                                          object_id=perspective_object_id).first()
        dbperspective = CACHE.get(objects =
            {
                dbPerspective : (args.get('id'), )
            },
        DBSession=DBSession)
        client_id = info.context.get('client_id')
        if not dbperspective or dbperspective.marked_for_deletion:
            raise ResponseError(message="No such perspective in the system")
        if roles_users:
            for role_id in roles_users:
                edit_role(dbperspective, user_id, role_id, client_id, perspective_default=True, action="delete")

        if roles_organizations:
            for role_id in roles_organizations:
                edit_role(dbperspective, user_id, role_id, client_id, perspective_default=True, organization=True,
                          action="delete")
        perspective = DictionaryPerspective(id=[dbperspective.client_id, dbperspective.object_id])
        perspective.dbObject = dbperspective
        CACHE.set(objects = [dbperspective,], DBSession=DBSession)
        return DeletePerspectiveRoles(perspective=perspective, triumph=True)


class UpdatePerspectiveAtom(graphene.Mutation):
    """
        example:
    mutation up{
        update_perspective_atom(id: [2138, 6], locale_id: 2, content: "test6"){
            triumph
        }

    }

        now returns:

    {
        "data": {
            "update_perspective_atom": {
                "triumph": true
            }
        }
    }
    """

    class Arguments:
        id = LingvodocID(required=True)
        content = graphene.String()
        locale_id = graphene.Int()
        atom_id = LingvodocID()

    triumph = graphene.Boolean()
    locale_id = graphene.Int()
    perspective = graphene.Field(DictionaryPerspective)

    @staticmethod
    @acl_check_by_id('edit', 'perspective')
    def mutate(root, info, **args):
        content = args.get('content')
        client_id, object_id = args.get('id')
        # dbperspective = DBSession.query(dbPerspective).filter_by(client_id=client_id, object_id=object_id).first()
        dbperspective = CACHE.get(objects =
            {
                dbPerspective : ((client_id, object_id), )
            },
        DBSession=DBSession)
        if not dbperspective:
            raise ResponseError(message="No such perspective in the system")
        locale_id = args.get("locale_id")

        if 'atom_id' in args:

            atom_id = args['atom_id']

            dbtranslationatom = (

                DBSession
                    .query(dbTranslationAtom)
                    .filter_by(
                        client_id = atom_id[0],
                        object_id = atom_id[1])
                    .first())

        else:

            dbtranslationatom = (

                DBSession
                    .query(dbTranslationAtom)
                    .filter_by(
                        parent_client_id=dbperspective.translation_gist_client_id,
                        parent_object_id=dbperspective.translation_gist_object_id,
                        locale_id=locale_id)
                    .first())

        if dbtranslationatom:
            if dbtranslationatom.locale_id == locale_id:
                key = "translation:%s:%s:%s" % (
                    str(dbtranslationatom.parent_client_id),
                    str(dbtranslationatom.parent_object_id),
                    str(dbtranslationatom.locale_id))
                CACHE.rem(key)
                key = "translations:%s:%s" % (
                    str(dbtranslationatom.parent_client_id),
                    str(dbtranslationatom.parent_object_id))
                CACHE.rem(key)
                if content:
                    dbtranslationatom.content = content
            else:
                if args.get('atom_id'):
                    atom_client_id, atom_object_id = args.get('atom_id')
                else:
                    raise ResponseError(message="atom field is empty")
                args_atom = DBSession.query(dbTranslationAtom).filter_by(client_id=atom_client_id,
                                                                         object_id=atom_object_id).first()
                if not args_atom:
                    raise ResponseError(message="No such dictionary in the system")
                dbtranslationatom.locale_id = locale_id
        else:
            dbtranslationatom = dbTranslationAtom(client_id=client_id,
                                                object_id=None,
                                                parent_client_id=dbperspective.translation_gist_client_id,
                                                parent_object_id=dbperspective.translation_gist_object_id,
                                                locale_id=locale_id,
                                                content=content)
            DBSession.add(dbtranslationatom)
            DBSession.flush()

        perspective = DictionaryPerspective(id=[dbPerspective.client_id, dbPerspective.object_id])
        perspective.dbObject = dbPerspective
        return UpdatePerspectiveAtom(perspective=perspective, triumph=True)


class DeleteDictionaryPerspective(graphene.Mutation):
    """
    example:
      mutation  {
            delete_perspective(id:[949,2491], parent_id:[449,2491]) {
                triumph
                perspective{
                    id
                }
            }
    }

    (this example works)
    returns:

    {
      "delete_perspective": {
        "triumph": true,
        "perspective": {
          "id": [
            949,
            2491
          ],
        }
      }
    }
    """
    class Arguments:
        id = LingvodocID(required=True)

    perspective = graphene.Field(DictionaryPerspective)
    triumph = graphene.Boolean()

    @staticmethod
    @acl_check_by_id('delete', 'perspective')
    def mutate(root, info, **args):
        id = args.get("id")
        client_id, object_id = id
        dbperspective = DBSession.query(dbPerspective).filter_by(client_id=client_id, object_id=object_id).first()
        if not dbperspective or dbperspective.marked_for_deletion:
            raise ResponseError(message="No such perspective in the system")
        settings = info.context["request"].registry.settings
        if 'desktop' in settings:
            real_delete_perspective(dbperspective, settings)
        else:
            del_object(dbperspective, "delete_perspective", info.context.get('client_id'))
        perspective = DictionaryPerspective(id=[dbperspective.client_id, dbperspective.object_id])
        perspective.dbObject = dbperspective
        return DeleteDictionaryPerspective(perspective=perspective, triumph=True)


class UndeleteDictionaryPerspective(graphene.Mutation):

    class Arguments:
        id = LingvodocID(required=True)

    perspective = graphene.Field(DictionaryPerspective)
    triumph = graphene.Boolean()

    @staticmethod
    @acl_check_by_id('delete', 'perspective')
    def mutate(root, info, **args):
        id = args.get("id")
        client_id, object_id = id
        dbperspective = DBSession.query(dbPerspective).filter_by(client_id=client_id, object_id=object_id).first()
        if not dbperspective:
            raise ResponseError(message="No such perspective in the system")
        if not dbperspective.marked_for_deletion:
            raise ResponseError(message="Perspective is not deleted")
        undel_object(dbperspective, "undelete_perspective", info.context.get('client_id'))
        perspective = DictionaryPerspective(id=[dbperspective.client_id, dbperspective.object_id])
        perspective.dbObject = dbperspective
        return UndeleteDictionaryPerspective(perspective=perspective, triumph=True)
