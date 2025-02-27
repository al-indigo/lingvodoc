
# Standard library imports.
import collections
import datetime
import io
import logging
import os
import shutil
import time
import traceback
import urllib.parse
import xlsxwriter

import graphene.types

from sqlalchemy.orm.attributes import flag_modified

from lingvodoc.models import (
    DBSession,
    DictionaryPerspective as dbDictionaryPerspective,
    Entity as dbEntity,
    User as dbUser,
    MarkupGroup as dbMarkupGroup,
    Client as dbClient,
    LexicalEntry as dbLexicalEntry,
    get_client_counter)

from lingvodoc.utils import sanitize_worksheet_name

from lingvodoc.schema.gql_holders import (
    LingvodocID,
    ResponseError,
    ObjectVal)

from sqlalchemy import (tuple_)

from pdb import set_trace as A

# Setting up logging.
log = logging.getLogger(__name__)


class MarkupGroup(graphene.ObjectType):
    client_id = graphene.Int()
    object_id = graphene.Int()
    perspective_client_id = graphene.Int()
    perspective_object_id = graphene.Int()
    type = graphene.String()
    author_id = graphene.Int()
    author_name = graphene.String()
    created_at = graphene.Float()

    def resolve_client_id(self, info):
        return self.client_id

    def resolve_object_id(self, info):
        return self.object_id

    def resolve_perspective_client_id(self, info):
        return self.perspective_client_id

    def resolve_perspective_object_id(self, info):
        return self.perspective_object_id

    def resolve_type(self, info):
        return self.type

    def resolve_author_id(self, info):
        return (
            DBSession.query(dbClient.user_id).filter_by(id=self.client_id).scalar())

    def resolve_author_name(self, info):
        return (
            DBSession
                .query(dbUser.name)
                .filter(
                    dbUser.id == dbClient.user_id,
                    dbClient.id == self.client_id)
                .scalar())

    def resolve_created_at(self, info):
        return self.created_at


class Markup(graphene.ObjectType):
    field_translation = graphene.String()
    field_position = graphene.Int()
    offset = graphene.Int()
    text = graphene.String()
    id = graphene.String()
    group_ids = graphene.List(LingvodocID)
    markup_groups = graphene.List(
        MarkupGroup,
        group_type = graphene.String(),
        author = graphene.Int())

    def resolve_field_translation(self, info):
        return self.field_translation

    def resolve_field_position(self, info):
        return self.field_position

    def resolve_offset(self, info):
        return self.offset

    def resolve_text(self, info):
        return self.text

    def resolve_id(self, info):
        return self.id

    def resolve_group_ids(self, info):
        return self.group_ids

    def resolve_markup_groups(self, info, group_type=None, author=None):

        custom_filters = list()
        if group_type:
            custom_filters.extend([
                dbMarkupGroup.type == group_type])
        if author:
            custom_filters.extend([
                dbMarkupGroup.client_id == dbClient.id,
                dbClient.user_id == author])

        markup_groups = (
            DBSession
                .query(dbMarkupGroup)
                .filter(
                    tuple_(
                        dbMarkupGroup.client_id,
                        dbMarkupGroup.object_id
                    ).in_(self.group_ids),
                    dbMarkupGroup.marked_for_deletion == False,
                    *custom_filters)
                .order_by(
                    dbMarkupGroup.created_at)
                .all()) if len(self.group_ids) else []

        return markup_groups


def list2dict(markup_list):
    markup_dict = collections.defaultdict(list)

    for (cid, oid, offset) in (markup_list or []):
        markup_dict[(cid, oid)].append(offset)

    return markup_dict


def list2tuple(group_list):
    return [tuple(g) for g in (group_list or [])]

'''
class UpdateEntityMarkup(graphene.Mutation):
    """
    curl 'https://lingvodoc.ispras.ru/api/graphql' \
    -H 'Content-Type: application/json' \
    -H 'Cookie: locale_id=2; auth_tkt=$TOKEN!userid_type:int; client_id=$ID' \
    --data-raw '{ "operationName": "update_entity_markup", "variables": {"id": [123, 321], \
    "result": [[4,6]], "groups_to_delete": [7,8,9]}, "query": "mutation \
    updateEntityMarkupMutation($id: LingvodocID, $result: [[LingvodocID]]!, $groups_to_delete: [LingvodocID])" \
    { update_entity_markup(result: $result, groups_to_delete: $groups_to_delete) { triumph }}"}'
    """

    class Arguments:

        id = LingvodocID(required = True)
        result = graphene.List(graphene.List(LingvodocID), required = True)
        groups_to_delete = graphene.List(LingvodocID)
        debug_flag = graphene.Boolean()

    triumph = graphene.Boolean()

    @staticmethod
    def mutate(root, info, **args):

        try:
            client_id = info.context.client_id

            entity_id = args.get('id')
            result = args.get('result')
            force = args.get('force', False)
            group_ids = args.get('groups_to_delete', [])
            debug_flag = args.get('debug_flag', False)

            if debug_flag:
                log.debug(f"{entity_id=}\n{result=}\n{group_ids=}")

            client = DBSession.query(dbClient).filter_by(id = client_id).first()

            if force and (not client or client.user_id != 1):
                return ResponseError('Only administrator can use force mode.')

            # Update entity's markups in additional_metadata
            entity_to_update = (
                DBSession

                    .query(dbEntity)
                    .filter(dbEntity.id == entity_id)
                    .one())

            if type(entity_to_update.additional_metadata) is dict:
                entity_to_update.additional_metadata['markups'] = result
            else:
                entity_to_update.additional_metadata = {'markups': result}

            flag_modified(entity_to_update, 'additional_metadata')

            # Delete groups if any is in deleted markups
            groups_to_delete = (
                DBSession

                    .query(dbMarkupGroup)
                    .filter(tuple_(dbMarkupGroup.client_id, dbMarkupGroup.object_id).in_(group_ids))
                    .all()

            ) if len(group_ids) else []

            for group_obj in groups_to_delete:
                group_obj.marked_for_deletion = True
                flag_modified(group_obj, 'marked_for_deletion')

            return UpdateEntityMarkup(triumph = True)

        except Exception as exception:

            traceback_string = (

                ''.join(
                    traceback.format_exception(
                        exception, exception, exception.__traceback__))[:-1])

            log.warning('update_entity_markup: exception')
            log.warning(traceback_string)

            return (

                ResponseError(
                    'Exception:\n' + traceback_string))
'''


class CreateMarkupGroup(graphene.Mutation):

    class Arguments:

        group_type = graphene.String(required=True)
        markups = graphene.List(graphene.List(graphene.Int, required=True))
        perspective_id = LingvodocID(required=True)
        debug_flag = graphene.Boolean()

    entry_ids = graphene.List(LingvodocID)
    triumph = graphene.Boolean()

    @staticmethod
    def mutate(root, info, **args):

        try:
            client_id = info.context.client_id

            group_type = args.get('group_type')
            markups = list2dict(args.get('markups'))
            perspective_id = args.get('perspective_id')
            debug_flag = args.get('debug_flag', False)

            if debug_flag:
                log.debug(f"{group_type=}\n{markups=}")

            client = DBSession.query(dbClient).filter_by(id=client_id).first()

            if not client:
                return ResponseError('Only authorized users can create markup groups.')

            group_object_id = get_client_counter(client_id)

            group_dict = {
                'type': group_type,
                'client_id': client_id,
                'object_id': group_object_id,
                'perspective_client_id': perspective_id[0],
                'perspective_object_id': perspective_id[1],
                'created_at': datetime.datetime.now(datetime.timezone.utc).timestamp(),
                'marked_for_deletion': False
            }

            DBSession.execute(
                dbMarkupGroup.__table__
                    .insert()
                    .values([group_dict]))

            entity_objs = (
                DBSession
                    .query(dbEntity)
                    .filter(tuple_(dbEntity.client_id, dbEntity.object_id).in_(markups))
                    .all())

            entry_ids = set()

            for ent in entity_objs:

                markup_objs = ent.additional_metadata.get('markups')

                if not markup_objs:
                    raise NotImplementedError

                for mrk in markup_objs:
                    if not len(mrk) or len(mrk[0]) != 2:
                        continue

                    offset, _ = mrk[0]
                    if offset in markups[(ent.client_id, ent.object_id)]:
                        mrk.append([client_id, group_object_id])  # this really works?!
                        entry_ids.add(ent.parent_id)
                        flag_modified(ent, 'additional_metadata')
                        break
                else:
                    # If no break
                    raise NotImplementedError

            return CreateMarkupGroup(triumph=True, entry_ids=list(entry_ids))

        except Exception as exception:

            traceback_string = (

                ''.join(
                    traceback.format_exception(
                        exception, exception, exception.__traceback__))[:-1])

            log.warning('create_markup_group: exception')
            log.warning(traceback_string)

            return (

                ResponseError(
                    'Exception:\n' + traceback_string))


class DeleteMarkupGroup(graphene.Mutation):

    class Arguments:

        group_ids = graphene.List(graphene.List(graphene.Int, required=True))
        markups = graphene.List(graphene.List(graphene.Int, required=True))
        perspective_id = LingvodocID()
        debug_flag = graphene.Boolean()

    entry_ids = graphene.List(LingvodocID)
    triumph = graphene.Boolean()

    @staticmethod
    def mutate(root, info, **args):

        try:
            client_id = info.context.client_id

            group_ids = args.get('group_ids')
            markups = list2dict(args.get('markups'))
            perspective_id = args.get('perspective_id')
            debug_flag = args.get('debug_flag', False)

            if debug_flag:
                log.debug(f"{group_ids=}\n{markups=}")

            client = DBSession.query(dbClient).filter_by(id=client_id).first()

            if not client:
                return ResponseError('Only authorized users can delete markup groups.')

            group_objs = (
                DBSession

                    .query(dbMarkupGroup)
                    .filter(tuple_(dbMarkupGroup.client_id, dbMarkupGroup.object_id).in_(group_ids))
                    .all())

            for grp in group_objs:
                grp.marked_for_deletion = True
                flag_modified(grp, 'marked_for_deletion')

            entity_objs = []

            if markups:

                entity_objs = (
                    DBSession
                        .query(dbEntity)
                        .filter(tuple_(dbEntity.client_id, dbEntity.object_id).in_(markups))
                        .all())

            elif perspective_id:

                entity_objs = (
                    DBSession
                        .query(dbEntity)
                        .filter(
                            dbEntity.parent_id == dbLexicalEntry.id,
                            dbEntity.marked_for_deletion == False,
                            dbLexicalEntry.parent_id == perspective_id,
                            dbLexicalEntry.marked_for_deletion == False)
                        .all())

            entry_ids = set()

            for ent in entity_objs:

                if type(metadata := ent.additional_metadata) is dict:
                    markup_objs = metadata.get('markups', [])
                else:
                    continue

                for i, mrk in enumerate(markup_objs):
                    if not len(mrk):
                        continue

                    indexes = mrk.pop(0)

                    if set(list2tuple(group_ids)) & set(list2tuple(mrk)):
                        ent.additional_metadata['markups'][i] = (
                            [indexes] + [id for id in mrk if id not in group_ids])
                        entry_ids.add(ent.parent_id)
                        flag_modified(ent, 'additional_metadata')

            return DeleteMarkupGroup(triumph=True, entry_ids=list(entry_ids))

        except Exception as exception:

            traceback_string = (

                ''.join(
                    traceback.format_exception(
                        exception, exception, exception.__traceback__))[:-1])

            log.warning('delete_markup_group: exception')
            log.warning(traceback_string)

            return (

                ResponseError(
                    'Exception:\n' + traceback_string))


class SaveMarkupGroups(graphene.Mutation):

    class Arguments:

        perspective_id = LingvodocID(required=True)
        field_list = graphene.List(ObjectVal, required=True)
        group_list = graphene.List(ObjectVal, required=True)
        debug_flag = graphene.Boolean()

    xlsx_url = graphene.String()
    message = graphene.String()
    triumph = graphene.Boolean()

    # Writing xlsx with any number of source fields
    @staticmethod
    def write_xlsx(field_list, group_list):

        workbook_stream = (
            io.BytesIO())

        workbook = (
            xlsxwriter.Workbook(
                workbook_stream, {'in_memory': True}))

        worksheet = (
            workbook.add_worksheet(
                sanitize_worksheet_name('Markup groups')))

        bold = workbook.add_format({'bold': True})
        align = workbook.add_format()
        align.set_align('vcenter')
        align.set_text_wrap()

        worksheet.set_column(0, len(field_list) - 3, 50)
        worksheet.set_column(len(field_list) - 2, len(field_list) - 1, 25)

        for column, item in enumerate(field_list):
            worksheet.write(0, column, item, bold)

        for row, group in enumerate(group_list, 1):
            text = group.get('text', "")
            type = group.get('type', "n/a")
            author = group.get('author', "n/a")

            max_len = max(map(lambda t: len(t), text))
            max_lines = max(map(lambda t: len(t.splitlines()), text))
            h1 = (max_len // 50 + 1) * 17
            h2 = max_lines * 17
            worksheet.set_row(row, max(h1, h2))
            for column, item in enumerate(text + [type, author]):
                worksheet.write(row, column, item, align)

        workbook.close()
        return workbook_stream

    @staticmethod
    def save_xlsx(info, perspective_id, xlsx_stream, section_name='markup_groups'):

        storage = info.context.request.registry.settings['storage']
        locale_id = info.context.locale_id
        time_str = '{0:.6f}'.format(time.time())

        storage_dir = (

            os.path.join(
                storage['path'],
                section_name,
                time_str))

        os.makedirs(storage_dir, exist_ok=True)

        db_perspective = (
            DBSession
                .query(dbDictionaryPerspective)
                .filter_by(id=perspective_id)
                .one())

        xlsx_filename = f'{db_perspective.parent.get_translation(locale_id)} (markup groups).xlsx'

        xlsx_path = os.path.join(
            storage_dir, xlsx_filename)

        with open(xlsx_path, 'wb') as xlsx_file:
            xlsx_stream.seek(0)
            shutil.copyfileobj(xlsx_stream, xlsx_file)

        return ''.join([
            storage['prefix'],
            storage['static_route'],
            section_name, '/',
            time_str, '/',
            urllib.parse.quote(xlsx_filename)])

    @staticmethod
    def mutate(root, info, **args):

        try:
            client_id = info.context.client_id

            perspective_id = args.get('perspective_id')
            field_list = args.get('field_list')
            group_list = args.get('group_list')
            debug_flag = args.get('debug_flag', False)

            if debug_flag:
                log.debug(f"{perspective_id=}\n{field_list=}\n{group_list=}")

            client = DBSession.query(dbClient).filter_by(id=client_id).first()

            if not client:
                return ResponseError('Only authorized users can save markup groups.')

            xlsx_stream = SaveMarkupGroups.write_xlsx(field_list, group_list)
            xlsx_url = SaveMarkupGroups.save_xlsx(info, perspective_id, xlsx_stream)
            message = ""

            return SaveMarkupGroups(triumph=True, xlsx_url=xlsx_url, message=message)

        except Exception as exception:

            traceback_string = (

                ''.join(
                    traceback.format_exception(
                        exception, exception, exception.__traceback__))[:-1])

            log.warning('save_markup_groups: exception')
            log.warning(traceback_string)

            return (

                ResponseError(
                    'Exception:\n' + traceback_string))
