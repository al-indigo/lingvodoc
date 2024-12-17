
# Standard library imports.

import errno
import hashlib
import logging
import os
import shutil

# Library imports.

import dateutil.parser
import graphene

from pathvalidate import sanitize_filename

from sqlalchemy import (
    and_,
    BigInteger,
    cast,
    func,
    literal,
    or_,
    tuple_)

# Project imports.

from lingvodoc.models import (
    BaseGroup as dbBaseGroup,
    Client,
    DBSession,
    Dictionary as dbDictionary,
    DictionaryPerspective as dbPerspective,
    ENGLISH_LOCALE,
    Entity as dbEntity,
    Field as dbField,
    Group as dbGroup,
    Language as dbLanguage,
    LexicalEntry as dbLexicalEntry,
    PublishingEntity as dbPublishingEntity,
    TranslationAtom as dbTranslationAtom,
    User as dbUser,
    user_to_group_association)

from lingvodoc.schema.gql_holders import (
    Accepted,
    AdditionalMetadata,
    CompositeIdHolder,
    Content,
    CreatedAt,
    del_object,
    DeletedAt,
    fetch_object,
    FieldHolder,
    LingvodocID,
    LingvodocObjectType,
    LocaleId,
    MarkedForDeletion,
    ObjectVal,
    ParentLink,
    Published,
    Relationship,
    ResponseError,
    SelfHolder,
    Upload)

from lingvodoc.utils.creation import create_entity
from lingvodoc.utils.deletion import real_delete_entity
from lingvodoc.utils.elan_functions import eaf_wordlist
from lingvodoc.utils.lexgraph_marker import marker_between_arith as marker_between
from lingvodoc.utils.verification import check_client_id, check_lingvodoc_id
from pdb import set_trace as A


# Setting up logging.
log = logging.getLogger(__name__)


def object_file_path(obj, base_path, folder_name, filename, create_dir=False):
    filename = sanitize_filename(filename)
    storage_dir = os.path.join(base_path, obj.__tablename__, folder_name, str(obj.client_id), str(obj.object_id))
    if create_dir:
        os.makedirs(storage_dir, exist_ok=True)
    storage_path = os.path.join(storage_dir, filename)
    return storage_path, filename


def create_object(content, obj, data_type, filename, folder_name, storage, json_input=True):
    storage_path, filename = object_file_path(obj, storage["path"], folder_name, filename, True)
    directory = os.path.dirname(storage_path)  # TODO: find out, why object_file_path were not creating dir
    try:
        os.makedirs(directory)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    with open(str(storage_path), 'wb+') as f:
        if json_input:
            f.write(content)
        else:
            shutil.copyfileobj(content, f)

    real_location = storage_path
    url = "".join((storage["prefix"],
                   storage["static_route"],
                   obj.__tablename__,
                   '/',
                   folder_name,
                   '/',
                   str(obj.client_id), '/',
                   str(obj.object_id), '/',
                   filename))
    return real_location, url


def is_subject_for_parsing(content):

    if not content:
        return False

    rfind_result = content.rfind('.')

    return (
        rfind_result >= 0 and
        content[rfind_result + 1:] in ('doc', 'docx', 'odt'))


# Read
class Entity(LingvodocObjectType):
    """
        query myQuery {
      entity(id: [66, 298] ){
				id
				created_at
			}

		}
    """
    # TODO: Accepted, entity_type
    content = graphene.String()
    data_type = graphene.String()
    dbType = dbEntity
    publishingentity = None
    is_subject_for_parsing = graphene.Boolean()

    class Meta:
        interfaces = (CompositeIdHolder,
                      AdditionalMetadata,
                      CreatedAt,
                      DeletedAt,
                      MarkedForDeletion,
                      Relationship,
                      SelfHolder,
                      FieldHolder,
                      ParentLink,
                      Content,
                      # TranslationHolder,
                      LocaleId,
                      Published,
                      Accepted
                      )

    def __init__(
        self,
        *args,
        publishingentity = None,
        **kwargs):

        if publishingentity is not None:
            self.publishingentity = publishingentity

        super().__init__(*args, **kwargs)

    @fetch_object('is_subject_for_parsing')
    def resolve_is_subject_for_parsing(self, info):
        return is_subject_for_parsing(self.dbObject.content)

    @fetch_object('data_type')
    def resolve_data_type(self, info):
        return self.dbObject.field.data_type


# Create
class CreateEntity(graphene.Mutation):

    class Arguments:
        """
        input values from request. Look at "LD methods" exel table
        """

        parent_id = LingvodocID(required=True)
        field_id = LingvodocID(required=True)

        client_id = graphene.Int()
        self_id = LingvodocID()
        link_id = LingvodocID()
        link_perspective_id = LingvodocID()

        locale_id = graphene.Int()
        content = graphene.String()
        filename = graphene.String()
        file_content = Upload()
        lexgraph_after = graphene.String()
        additional_metadata = ObjectVal()

    # Result object

    entity = graphene.Field(Entity)

    """
    example:
    curl -i -X POST  -H "Cookie: auth_tkt="
    -H "Content-Type: multipart/form-data" -F "blob=@белка.wav" -F 'query=mutation {
            create_entity(parent_id: [66, 69],  field_id:  [66,12] ) {entity{id, parent_id} triumph}}' http://localhost:6543/graphql

    or
    mutation  {
    create_entity(parent_id: [66, 69], field_id: [66, 6], content: "test") {
        entity {
            created_at,
	content
        }

    triumph
    }
    }
    """
    # Used for convenience

    triumph = graphene.Boolean()

    @staticmethod
    def mutate(
        root,
        info,

        parent_id,
        field_id,

        client_id = None,
        self_id = None,
        link_id = None,
        link_perspective_id = None,

        locale_id = ENGLISH_LOCALE,
        content = None,
        filename = None,
        lexgraph_after = None,
        additional_metadata = None,

        **args):

        context = info.context

        # When creating object, we check, if a client id was supplied, if it's compatible.

        client_id = (

            context.effective_client_id(
                client_id))

        lexical_entry = dbLexicalEntry.get(parent_id)

        if not lexical_entry:

            return (
                ResponseError("No such lexical entry in the system"))

        perspective_id = (
            lexical_entry.parent_id)

        info.context.acl_check(
            'create',
            'lexical_entries_and_entities',
            perspective_id)

        data_type = (

            DBSession

                .query(
                    func.lower(dbTranslationAtom.content))

                .filter(
                    dbTranslationAtom.locale_id == ENGLISH_LOCALE,
                    dbTranslationAtom.parent_id == dbField.data_type_translation_gist_id,
                    dbField.id == field_id)

                .scalar())

        if not data_type:

            return (
                ResponseError("No such field in the system"))

        if self_id:

            if not dbEntity.exists(self_id):
                return ResponseError("No such upper level in the system")

        db_entity = (

            dbEntity(
                client_id = client_id,
                parent_id = parent_id,
                self_id = self_id,
                field_id = field_id,
                locale_id = locale_id,
                additional_metadata = additional_metadata))

        # Acception permission check.
        # Admin is assumed to have all permissions.

        user_id = (
            context.user_id)

        if user_id == 1:

            create_flag = True

        elif user_id is not None:

            # Not just calling acl.check_direct because ours is much more specific case.

            user = context.user

            if (create_flag := user.is_active):

                group_query = (

                    DBSession

                        .query(literal(1))

                        .filter(
                            dbBaseGroup.subject == 'lexical_entries_and_entities',
                            dbBaseGroup.action == 'create',
                            dbGroup.base_group_id == dbBaseGroup.id,
                            or_(dbGroup.subject_override,
                                dbGroup.subject_id == perspective_id),
                            user_to_group_association.c.user_id == user_id,
                            user_to_group_association.c.group_id == dbGroup.id))

                create_flag = (

                    DBSession
                        .query(group_query.exists())
                        .scalar())

                #
                # NOTE: tests show that exists is faster than the current limit-count-based way from acl.py,
                # which is like this:
                #
                # create_flag = (
                #
                #     DBSession
                #
                #         .query(user_to_group_association)
                #
                #         .filter(
                #             dbBaseGroup.subject == 'lexical_entries_and_entities',
                #             dbBaseGroup.action == 'create',
                #             dbGroup.base_group_id == dbBaseGroup.id,
                #             or_(dbGroup.subject_override,
                #                 dbGroup.subject_id == perspective_id),
                #             user_to_group_association.c.user_id == user_id,
                #             user_to_group_association.c.group_id == dbGroup.id)
                #
                #         .limit(1)
                #         .count())
                #

        else:

            create_flag = False

        if create_flag:

            db_entity.publishingentity.accepted = True

        # If the entity is being created by the admin, we automatically publish it.

        if user_id == 1:

            db_entity.publishingentity.published = True

        real_location = None
        url = None
        if data_type == 'image' or data_type == 'sound' or 'markup' in data_type:
            blob = info.context.request.POST.pop("1")
            filename=blob.filename
            content = blob.file.read()
            #filename=
            real_location, url = create_object(content, db_entity, data_type, filename, "graphql_files", info.context.request.registry.settings["storage"])
            db_entity.content = url
            old_meta = db_entity.additional_metadata
            need_hash = True
            if old_meta:
                if old_meta.get('hash'):
                    need_hash = False
            if need_hash:
                hash = hashlib.sha224(content).hexdigest()
                hash_dict = {'hash': hash}
                if old_meta:
                    old_meta.update(hash_dict)
                else:
                    old_meta = hash_dict
                db_entity.additional_metadata = old_meta
            if 'markup' in data_type:
                name = filename.split('.')
                ext = name[len(name) - 1]
                if ext.lower() == 'textgrid':
                    data_type = 'praat markup'

                elif ext.lower() == 'eaf':
                    data_type = 'elan markup'

            db_entity.additional_metadata['data_type'] = data_type

            if 'elan' in data_type:
                bag_of_words = list(eaf_wordlist(content))
                db_entity.additional_metadata['bag_of_words'] = bag_of_words
        elif data_type == 'link':
            if link_id:
                db_entity.link_id = link_id
            else:
                return ResponseError(
                    "The field is of link type. You should provide client_id and object id in the content")
        elif data_type == 'directed link':
            if link_id:
                db_entity.link_id = link_id
            else:
                return ResponseError(
                    "The field is of link type. You should provide client_id and object id in the content")

            if link_perspective_id:
                if not dbPerspective.exists(link_perspective_id):
                    return ResponseError("link_perspective not found")
                db_entity.additional_metadata['link_perspective_id'] = link_perspective_id

            else:
                return ResponseError(
                    "The field is of link type. You should provide link_perspective_id id in the content")

        elif lexgraph_after is not None:
            db_entity.content = marker_between(marker_after=lexgraph_after)


        else:
            db_entity.content = content

        DBSession.add(db_entity)
        DBSession.flush()

        return (

            CreateEntity(
                entity = Entity(db_entity),
                triumph = True))

    # Update
    """
    example #1:
    mutation  {
        update_entity(id: [ 742, 5494], additional_metadata: {hash:"1234567"} ) {
            entity {
                created_at,
                additional_metadata{
                hash
                }
            }

        status
        }
    }
    example #2:
    mutation  {
        update_entity(id: [ 742, 5494], additional_metadata: {hash:"12345"} ){status}
    }
    resolve:
    {
        "update_entity": {
            "status": true
        }
    }
    """
    # Delete
    """
    query:
    mutation  {
        delete_entity(id: [879, 8]) {
        entity{id, content, created_at}
        status
        }
    }
    response:
    {
        "delete_entity": {
            "entity": {
                "id": [
                    879,
                    8
                ],
                "content": "123",
                "created_at": "2017-06-27T09:49:24"
            },
            "status": true
        }
    }
    or
    {
        "errors": [
            "No such entity in the system"
        ]
    }
    """


class UpdateEntity(graphene.Mutation):
    """
    mutation Mu{
	update_entity_content(id:[1995,2017], published: true){
		entity{
			created_at
		}
	}
    }
    """
    class Arguments:
        id = LingvodocID(required=True)
        published = graphene.Boolean()
        accepted = graphene.Boolean()
        new_parent_id = LingvodocID()

    entity = graphene.Field(Entity)
    triumph = graphene.Boolean()

    @staticmethod
    def mutate(root, info, **args):
        client_id, object_id = args.get('id')
        dbpublishingentity = DBSession.query(dbPublishingEntity).filter_by(client_id=client_id,
                                                                           object_id=object_id).first()
        if not dbpublishingentity:
            raise ResponseError(message="No such entity in the system")
        # lexical_entry = dbpublishingentity.parent.parent
        lexical_entry = DBSession.query(dbLexicalEntry).join(dbLexicalEntry.entity).join(
            dbEntity.publishingentity).filter(dbPublishingEntity.client_id == client_id,
                                              dbPublishingEntity.object_id == object_id).one()
        if not lexical_entry:
            raise ResponseError(message="No such lexical_entry in the system")
        published = args.get('published')
        accepted = args.get('accepted')
        if published is not None and not dbpublishingentity.published:
            info.context.acl_check('create', 'approve_entities',
                                   (lexical_entry.parent_client_id, lexical_entry.parent_object_id))

        if published is not None and not published and dbpublishingentity.published:
            info.context.acl_check('delete', 'approve_entities',
                                   (lexical_entry.parent_client_id, lexical_entry.parent_object_id))

        if accepted is not None and not dbpublishingentity.accepted:
            info.context.acl_check('create', 'lexical_entries_and_entities',
                                   (lexical_entry.parent_client_id, lexical_entry.parent_object_id))

        if accepted is not None and not accepted and dbpublishingentity.accepted:
            raise ResponseError(message="Not allowed action")

        if published is not None:
            dbpublishingentity.published = published
        if accepted is not None:
            dbpublishingentity.accepted = accepted

        dbentity = dbEntity.get((client_id, object_id))

        new_parent_id = args.get('new_parent_id')

        if new_parent_id is not None:

            info.context.acl_check('delete', 'lexical_entries_and_entities',
                                   (lexical_entry.parent_client_id, lexical_entry.parent_object_id))
            info.context.acl_check('create', 'lexical_entries_and_entities',
                                   (lexical_entry.parent_client_id, lexical_entry.parent_object_id))

            dbentity.parent_id = new_parent_id

        return UpdateEntity(entity=Entity(dbentity), triumph=True)


class ApproveAllForUser(graphene.Mutation):
    """
    mutation Mu{
	update_entity_content(id:[1995,2017], published: true){
		entity{
			created_at
		}
	}
    }
    """

    class Arguments:

        user_id = graphene.Int()

        published = graphene.Boolean()
        accepted = graphene.Boolean()

        field_ids = graphene.List(LingvodocID)

        skip_deleted_entities = graphene.Boolean()

        perspective_id = LingvodocID()

        language_id = LingvodocID()
        language_recursive = graphene.Boolean()
        language_all = graphene.Boolean()

        time_from = graphene.String()

    #entity = graphene.Field(Entity)
    triumph = graphene.Boolean()
    update_count = graphene.Int()

    @staticmethod
    def mutate(
        root,
        info,
        user_id = None,
        published = False,
        accepted = False,
        field_ids = None,
        skip_deleted_entities = True,
        perspective_id = None,
        language_id = None,
        language_recursive = False,
        language_all = False,
        time_from = None):

        # We have a starting time specification, trying to parse it.

        if time_from is not None:
            time_from = dateutil.parser.parse(time_from)

        log.debug(
            '\napprove_all_for_user'
            '\n  user_id: {}'
            '\n  published: {}'
            '\n  accepted: {}'
            '\n  field_ids: {}'
            '\n  skip_deleted_entities: {}'
            '\n  perspective_id: {}'
            '\n  language_id: {}'
            '\n  language_recursive: {}'
            '\n  language_all: {}'
            '\n  time_from: {}'.format(
            user_id,
            published,
            accepted,
            field_ids,
            skip_deleted_entities,
            perspective_id,
            language_id,
            language_recursive,
            language_all,
            time_from))

        request_client_id = info.context.request.authenticated_userid
        request_user = Client.get_user_by_client_id(request_client_id)

        if (not perspective_id and
            not language_id and
            not language_all):

            return ResponseError(
                'Please specify either a perspective, a language or the flag for all languages.')

        if ((language_id or language_all) and
            request_user.id != 1):

            return ResponseError(
                'Only administrator can perform bulk approve for languages.')

        if (user_id is None and
            request_user.id != 1):

            return ResponseError(
                'Only administrator can perform bulk approve for all users.')

        # Entity selection condition.

        if published and accepted:

            entity_select_condition = or_(
                dbPublishingEntity.published == False,
                dbPublishingEntity.accepted == False)

        elif published:
            entity_select_condition = dbPublishingEntity.published == False

        elif accepted:
            entity_select_condition = dbPublishingEntity.accepted == False

        else:
            return ResponseError('Neither publish nor accept action is specified.')

        # Filtering by user id, if required.

        if user_id is not None:

            client_id_query = (
                DBSession.query(Client.id).filter_by(user_id = user_id))

            entity_select_condition = (

                and_(
                    entity_select_condition,

                    func.coalesce(
                        cast(dbEntity.additional_metadata[
                            ('merge', 'original_client_id')].astext, BigInteger),
                        dbEntity.client_id)

                        .in_(client_id_query)))

        # If have any specified fields, checking their info and updating selection condition.

        list_of_fields = list()

        if field_ids:

            for field_id in field_ids:
                field = DBSession.query(dbField).filter_by(client_id=field_id[0], object_id=field_id[1]).first()
                if not field:
                    return ResponseError("field not found")
                list_of_fields.append((field.client_id, field.object_id))

        field_id_list = tuple(list_of_fields)

        if field_ids is not None:

            entity_select_condition = and_(
                entity_select_condition,
                tuple_(dbEntity.field_client_id, dbEntity.field_object_id).in_(field_id_list))

        # Checking if we should or should not update status of deleted entities.

        if skip_deleted_entities:

            entity_select_condition = and_(
                entity_select_condition,
                dbEntity.marked_for_deletion == False)

        # Checking if we should filter entities based on creation datetime.

        if time_from:

            entity_select_condition = and_(
                entity_select_condition,
                dbEntity.created_at >= time_from)

        # Bulk approve for a single perspective.

        if perspective_id:

            given_perspective = DBSession.query(dbPerspective).filter_by(marked_for_deletion=False,
                                                                                   client_id = perspective_id[0],
                                                                                   object_id = perspective_id[1]).first()
            if not given_perspective:
                return ResponseError("Perspective not found")

            # Currently bulk approve can only give published/accepted status and can't remove it, so we
            # don't check for 'delete' permissions.
            #
            # We also do not check permissions if the request is from the administrator.

            if published is not None and request_user.id != 1:

                info.context.acl_check('create', 'approve_entities',
                                       (perspective_id[0], perspective_id[1]))

            if accepted is not None and request_user.id != 1:

                info.context.acl_check('create', 'lexical_entries_and_entities',
                                       (perspective_id[0], perspective_id[1]))

            # Performing bulk approve.

            update_dict = {}

            if published:
                update_dict['published'] = True

            if accepted:
                update_dict['accepted'] = True

            update_count = (

                DBSession

                    .query(
                        dbPublishingEntity)

                    .filter(
                        dbLexicalEntry.parent_client_id == perspective_id[0],
                        dbLexicalEntry.parent_object_id == perspective_id[1],
                        dbLexicalEntry.marked_for_deletion == False,
                        dbEntity.parent_client_id == dbLexicalEntry.client_id,
                        dbEntity.parent_object_id == dbLexicalEntry.object_id,
                        dbPublishingEntity.client_id == dbEntity.client_id,
                        dbPublishingEntity.object_id == dbEntity.object_id,
                        entity_select_condition)

                    .update(
                        values = update_dict,
                        synchronize_session = False))

            log.debug(
                'approve_all_for_user (perspective {0}/{1}): updated {2} entit{3}'.format(
                perspective_id[0], perspective_id[1],
                update_count,
                'y' if update_count == 1 else 'ies'))

        # Bulk approve for a language, possibly recursive, or all undeleted languages with transitively
        # undeleted parents.

        if language_id or language_all:

            if language_id:

                # A single specified language.

                marked_for_deletion = (

                    DBSession

                        .query(
                            dbLanguage.marked_for_deletion)

                        .filter_by(
                            client_id = language_id[0],
                            object_id = language_id[1])

                        .scalar())

                if marked_for_deletion is None:
                    raise ResponseError('Language {0}/{1} not found'.format(*language_id))

                elif marked_for_deletion is True:
                    raise ResponseError('Language {0}/{1} is deleted'.format(*language_id))

                language_id_list = [
                    (language_id[0], language_id[1])]

                language_str = (

                    'language {0}/{1}{2}'.format(
                        language_id[0],
                        language_id[1],
                        ', recursive' if language_recursive else ''))

            else:

                # All undeleted languages with transitively undeleted parents.

                language_id_query = (

                    DBSession

                        .query(
                            dbLanguage.client_id,
                            dbLanguage.object_id)

                        .filter_by(
                            marked_for_deletion = False,
                            parent_client_id = None,
                            parent_object_id = None))

                language_id_list = [

                    (client_id, object_id)
                    for client_id, object_id in language_id_query.all()]

                language_str = 'all languages'

            # Getting all child languages, if required.

            if language_recursive or language_all:

                parent_id_list = language_id_list

                while parent_id_list:

                    language_list = DBSession.query(dbLanguage).filter(
                        dbLanguage.marked_for_deletion == False,
                        tuple_(dbLanguage.parent_client_id, dbLanguage.parent_object_id).in_(
                            parent_id_list)).all()

                    parent_id_list = [
                        (language.client_id, language.object_id)
                        for language in language_list]

                    language_id_list.extend(parent_id_list)

            log.debug(
                '\napprove_all_for_user ({0}):'
                '\nlanguage_id_list ({1}): {2}'.format(
                language_str,
                len(language_id_list),
                language_id_list))

            # Performing bulk approve of entities for all non-deleted lexical entries from all non-deleted
            # perspectives of all non-deleted dictionaries of specified languages.

            update_dict = {}

            if published:
                update_dict['published'] = True

            if accepted:
                update_dict['accepted'] = True

            update_query = (

                dbPublishingEntity.__table__
                    .update()

                    .where(and_(
                        tuple_(dbDictionary.parent_client_id, dbDictionary.parent_object_id).in_(
                            language_id_list),
                        dbDictionary.marked_for_deletion == False,
                        dbPerspective.parent_client_id == dbDictionary.client_id,
                        dbPerspective.parent_object_id == dbDictionary.object_id,
                        dbPerspective.marked_for_deletion == False,
                        dbLexicalEntry.parent_client_id == dbPerspective.client_id,
                        dbLexicalEntry.parent_object_id == dbPerspective.object_id,
                        dbLexicalEntry.marked_for_deletion == False,
                        dbEntity.parent_client_id == dbLexicalEntry.client_id,
                        dbEntity.parent_object_id == dbLexicalEntry.object_id,
                        dbPublishingEntity.client_id == dbEntity.client_id,
                        dbPublishingEntity.object_id == dbEntity.object_id,
                        entity_select_condition))

                    .values(**update_dict))

            update_result = update_query.execute()
            update_count = update_result.rowcount

            log.debug(
                'approve_all_for_user ({0}): updated {1} entit{2}'.format(
                language_str,
                update_count,
                'y' if update_count == 1 else 'ies'))

        DBSession.flush()
        return ApproveAllForUser(triumph = True, update_count = update_count)


class DeleteEntity(graphene.Mutation):

    class Arguments:
        id = LingvodocID(required=True)

    triumph = graphene.Boolean()
    entity = graphene.Field(Entity)

    @staticmethod
    def mutate(root, info, id):

        db_entity = (

            dbEntity.get(
                id, deleted = False))

        if not db_entity:
            return ResponseError(f'No entity {id} in the system.')

        info.context.acl_check(
            'delete',
            'lexical_entries_and_entities',
            db_entity.parent.parent_id)

        settings = (
            info.context.request.registry.settings)

        if 'desktop' in settings:

            real_delete_entity(
                db_entity,
                settings)

        else:

            del_object(
                db_entity,
                'delete_entity',
                info.context.client_id)

        return (

            DeleteEntity(
                entity = Entity(db_entity),
                triumph = True))


class BulkCreateEntity(graphene.Mutation):
    """
    mutation {
            bulk_create_entity(entities: [{id: [1199, 4], parent_id: [66, 69],  field_id:  [66, 6]}]) {
                   triumph
        }
    }
    """

    class Arguments:
        entities = graphene.List(ObjectVal)
    entities = graphene.List(Entity)
    triumph = graphene.Boolean()

    @staticmethod
    def mutate(root, info, entities = None):

        # No assurances that this even works, will need to be rewritten if we would need this.

        raise NotImplementedError

        client = Client.get(info.context.client_id)
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.",
                           info.context.client_id)
        request = info.context.request

        dbentities_list = []

        for entity_obj in entities:
            ids = entity_obj.get("id")  # TODO: id check

            if ids is None:
                ids = (info.context.client_id, None)
            else:
                if not check_lingvodoc_id(ids):
                    raise KeyError("Wrong id")
                if not check_client_id(info.context.client_id, ids[0]):
                    raise KeyError("Invalid client id (not registered on server). Try to logout and then login.",
                                   ids[0])
            parent_id = entity_obj.get("parent_id")

            if not parent_id:
                return ResponseError(message="Bad lexical_entry object")
            if not check_lingvodoc_id(parent_id):
                raise KeyError("Wrong parent_id")

            lexical_entry = dbLexicalEntry.get(parent_id)

            info.context.acl_check(
                'create',
                'lexical_entries_and_entities',
                lexical_entry.parent_id)

            additional_metadata = None
            if 'additional_metadata' in entity_obj:
                additional_metadata = entity_obj["additional_metadata"]
            field_id = entity_obj.get('field_id')
            if not check_lingvodoc_id(field_id):
                raise ResponseError('no field_id provided')
            self_id = entity_obj.get("self_id")
            if self_id:
                if not check_lingvodoc_id(self_id):
                    raise KeyError("Wrong self_id")
            link_id = entity_obj.get("link_id")
            if link_id:
                if not check_lingvodoc_id(link_id):
                    raise KeyError("Wrong link_id")
            locale_id = entity_obj.get("locale_id", 2)
            filename = entity_obj.get("filename")
            content = entity_obj.get("content")
            registry = entity_obj.get("registry")

            dbentity = create_entity(ids, parent_id, additional_metadata, field_id, self_id, link_id, locale_id,
                                     filename, content, registry, request, False)

            dbentities_list.append(dbentity)

        DBSession.bulk_save_objects(dbentities_list)
        DBSession.flush()

        entity_list = (
            list(map(Entity, dbentities_list)))

        return (

            BulkCreateEntity(
                entities = entity_list,
                triumph = True))


class BulkDeleteEntity(graphene.Mutation):    
    """
    Deletes entities based on various criteria, entities of a user, of a client, of a perspective /
    perspectives, created in a specified time interval.

    Example:

      curl 'http://localhost:6543/graphql'
        -H 'content-type: application/json'
        -H 'Cookie: locale_id=2; auth_tkt=$TOKEN; client_id=$ID'
        --data-raw '{
          "operationName": "bulkDeleteEntity",
          "variables": {
            "userId": 349,
            "perspectiveIdList": [[5070, 3105], [5523, 2]]},
          "query":
            "mutation bulkDeleteEntity(
              $userId: Int,
              $perspectiveIdList: [LingvodocID])
            {
              bulk_delete_entity(
                user_id: $userId,
                field_id: [66, 25],
                perspective_id_list: $perspectiveIdList)
              {
                triumph
                delete_count
              }
            }"}'
  
      Set $TOKEN and $ID to valid admin user authentication info.
  
      To use in a shell, join into a single line or add escaping backslashes at the end of the lines.
    """

    class Arguments:

        user_id = graphene.Int()

        field_id = LingvodocID()
        field_id_list = graphene.List(LingvodocID)

        perspective_id = LingvodocID()
        perspective_id_list = graphene.List(LingvodocID)

        time_from = graphene.String()
        time_to = graphene.String()

    triumph = graphene.Boolean()
    delete_count = graphene.Int()

    @staticmethod
    def mutate(
        root,
        info,
        user_id = None,
        field_id = None,
        field_id_list = None,
        perspective_id = None,
        perspective_id_list = None,
        time_from = None,
        time_to = None):

        # Conceivable filterting conditions we won't be implementing at the time.

        if (field_id_list is not None or
            perspective_id is not None or
            time_from is not None or
            time_to is not None):

            raise NotImplementedError

        user = info.context.user

        if not user or user.id != 1:

            return ResponseError(
                'Only administrator can perform bulk entity deletion.')

        # Entity selection conditions.

        condition_list = [
            dbEntity.marked_for_deletion == False]

        if field_id is not None:

            condition_list.extend([
                dbEntity.field_client_id == field_id[0],
                dbEntity.field_object_id == field_id[1]])

        # Filtering by user id, if required.

        if user_id is not None:

            client_id_query = (
                DBSession.query(Client.id).filter_by(user_id = user_id))

            condition_list.append(

                func.coalesce(
                    cast(dbEntity.additional_metadata[
                        ('merge', 'original_client_id')].astext, BigInteger),
                    dbEntity.client_id)

                    .in_(client_id_query))

        # Filtering by perspectives, if required.

        if perspective_id_list is not None:

            if not perspective_id_list:

                return ResponseError(
                    'If specified, perspective id list should be non-empty.')

            condition_list.extend([

                dbLexicalEntry.client_id == dbEntity.parent_client_id,
                dbLexicalEntry.object_id == dbEntity.parent_object_id,

                tuple_(
                    dbLexicalEntry.parent_client_id,
                    dbLexicalEntry.parent_object_id)

                    .in_(
                        perspective_id_list)])

        entity_list = (

            DBSession
                .query(dbEntity)
                .filter(*condition_list)
                .all())

        log.debug(f'\n{len(entity_list)} entities')

        # Deleting entities.
        #
        # Currently unoptimized, should think about bulk object deletion operations.

        settings = info.context.request.registry.settings

        if 'desktop' in settings:

            for entity in entity_list:

                real_delete_entity(
                    entity,
                    settings)

        else:

            for entity in entity_list:

                del_object(
                    entity,
                    'bulk_delete_entity',
                    info.context.client_id)

        return (

            BulkDeleteEntity(
                triumph = True,
                delete_count = len(entity_list)))


class UpdateEntityContent(graphene.Mutation):
    """


		mutation My {
	update_entity_content(id:[1907,10], content: "cat"){
		entity{
			created_at
		}
	}
}
    """

    class Arguments:
        """
        input values from request. Look at "LD methods" exel table
        """
        id = LingvodocID()
        content = graphene.String()
        markups = graphene.List(ObjectVal)
        lexgraph_before = graphene.String()
        lexgraph_after = graphene.String()

    entity = graphene.Field(Entity)
    triumph = graphene.Boolean()

    @staticmethod
    def mutate(root, info, **args):
        # delete
        old_client_id, object_id = args.get('id')
        client_id = DBSession.query(Client).filter_by(id=info.context.client_id).first().id
        content = args.get("content")
        markups = args.get("markups")
        lexgraph_before = args.get("lexgraph_before", '')
        lexgraph_after = args.get("lexgraph_after", '')
        if content is None and (lexgraph_before or lexgraph_after):
            content = marker_between(lexgraph_before, lexgraph_after)

        dbentity_old = dbEntity.get((old_client_id, object_id), deleted = False)
        if not dbentity_old:
            raise ResponseError(message="No such entity in the system")
        if dbentity_old.field.data_type != "Text" and dbentity_old.field.data_type != "Ordering":
            raise ResponseError(message="Can't edit non-text entities")
        lexical_entry = dbentity_old.parent

        info.context.acl_check(
            'delete',
            'lexical_entries_and_entities',
            lexical_entry.parent_id)

        settings = info.context.request.registry.settings
        if 'desktop' in settings:
            real_delete_entity(dbentity_old, settings)
        else:
            del_object(dbentity_old, "update_entity_content", info.context.client_id)
        # create
        client = DBSession.query(Client).filter_by(id=client_id).first()
        user = DBSession.query(dbUser).filter_by(id=client.user_id).first()
        if not user:
            return ResponseError("This client id is orphaned. Try to logout and then login once more.")

        parent = dbLexicalEntry.get(dbentity_old.parent_id)
        if not parent:
            return ResponseError("No such lexical entry in the system")

        info.context.acl_check('create', 'lexical_entries_and_entities',
                               (parent.parent_client_id, parent.parent_object_id))

        if markups:
            if type(dbentity_old.additional_metadata) is dict:
                dbentity_old.additional_metadata['markups'] = markups
            else:
                dbentity_old.additional_metadata = {'markups': markups}

        dbentity = dbEntity(client_id=client_id,
                        object_id=None,
                        field_client_id=dbentity_old.field_client_id,
                        field_object_id=dbentity_old.field_object_id,
                        locale_id=dbentity_old.locale_id,
                        additional_metadata=dbentity_old.additional_metadata,
                        parent=dbentity_old.parent)

        dbentity.publishingentity.accepted = dbentity_old.publishingentity.accepted
        dbentity.content = content

        DBSession.add(dbentity)
        DBSession.flush()

        return UpdateEntityContent(entity = Entity(dbentity), triumph = True)


class BulkUpdateEntityContent(graphene.Mutation):
    """
            mutation My {
        update_entity_content(ids:[[1907,10]], contents: ["cat"]){
            entity{
                created_at
            }
        }
    }
    """
    class Arguments:
        """
        input values from request. Look at "LD methods" exel table
        """
        ids = graphene.List(LingvodocID)
        contents = graphene.List(graphene.String)

    entities = graphene.List(Entity)
    triumph = graphene.Boolean()

    @staticmethod
    def mutate(root, info, **args):

        # delete
        old_ids = args.get('ids')
        contents = args.get('contents')

        if len(old_ids) != len(contents):
            return ResponseError("Length of contents list is not equal to length of entities list")

        client = Client.get(info.context.client_id)
        client_id = client.id

        user = dbUser.get(client.user_id)
        if not user:
            return ResponseError("This client id is orphaned. Try to logout and then login once more.")

        dbentities_old = []

        for old_id in old_ids:
            dbentity_old = dbEntity.get(old_id, deleted = False)

            if not dbentity_old:
                return ResponseError("No entity " + format(old_id) + " in the system")
            if dbentity_old.field.data_type != "Text":
                return ResponseError("Can't edit non-text entity" + format(old_id))

            lexical_entry = dbentity_old.parent

            info.context.acl_check(
                'delete',
                'lexical_entries_and_entities',
                lexical_entry.parent_id)

            settings = info.context.request.registry.settings
            if 'desktop' in settings:
                real_delete_entity(dbentity_old, settings)
            else:
                del_object(dbentity_old, "update_entity_content", info.context.client_id)
            dbentities_old.append(dbentity_old)

        # Creating new entities.

        dbentities_new = []

        i = -1
        for content in contents:
            i += 1

            parent = dbLexicalEntry.get(dbentities_old[i].parent_id)
            if not parent:
                return ResponseError("No such lexical entry in the system")

            info.context.acl_check(
                'create',
                'lexical_entries_and_entities',
                parent.parent_id)

            dbentity = dbEntity(client_id=client_id,
                                object_id=None,
                                field_client_id=dbentities_old[i].field_client_id,
                                field_object_id=dbentities_old[i].field_object_id,
                                locale_id=dbentities_old[i].locale_id,
                                additional_metadata=dbentities_old[i].additional_metadata,
                                parent=dbentities_old[i].parent)

            dbentity.publishingentity.accepted = dbentities_old[i].publishingentity.accepted
            dbentity.content = content
            # if args.get('is_translatable', None): # TODO: fix it
            #     field.is_translatable = bool(args['is_translatable'])
            dbentities_new.append(dbentity)

        DBSession.bulk_save_objects(dbentities_new)
        DBSession.flush()

        entities = list(map(Entity, dbentities_new))

        return BulkUpdateEntityContent(entities=entities, triumph=True)

