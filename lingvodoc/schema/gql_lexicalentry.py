
# Standard library imports.

import time
import string
import random
import logging
from uuid import uuid4

# External imports.

import graphene

from pyramid.security import authenticated_userid

from sqlalchemy import tuple_

# Lingvodoc imports.

from lingvodoc.cache.caching import CACHE

from lingvodoc.models import (
    BaseGroup as dbBaseGroup,
    Client,
    DBSession,
    DictionaryPerspective as dbPerspective,
    Entity as dbEntity,
    Field as dbField,
    Group as dbGroup,
    LexicalEntry as dbLexicalEntry,
    PublishingEntity as dbPublishingEntity,
    User as dbUser)

from lingvodoc.schema.gql_entity import Entity

from lingvodoc.schema.gql_holders import (
    acl_check_by_id,
    AdditionalMetadata,
    client_id_check,
    CompositeIdHolder,
    CreatedAt,
    del_object,
    DeletedAt,
    fetch_object,
    LingvodocID,
    LingvodocObjectType,
    MarkedForDeletion,
    MovedTo,
    ObjectVal,
    Relationship,
    ResponseError,
    undel_object)

from lingvodoc.views.v2.delete import real_delete_lexical_entry

from lingvodoc.utils import ids_to_id_query
from lingvodoc.utils.creation import create_lexicalentry
from lingvodoc.utils.deletion import real_delete_entity
from lingvodoc.utils.search import find_all_tags, find_lexical_entries_by_tags
from pdb import set_trace as A


# Setting up logging.
log = logging.getLogger(__name__)


class LexicalEntry(LingvodocObjectType):
    """
     #created_at          | timestamp without time zone | NOT NULL
     #object_id           | bigint                      | NOT NULL
     #client_id           | bigint                      | NOT NULL
     #parent_object_id    | bigint                      |
     #parent_client_id    | bigint                      |
     #marked_for_deletion | boolean                     | NOT NULL
     #moved_to            | text                        |
     #additional_metadata | jsonb                       |
    """
    entities = graphene.List(Entity, mode=graphene.String())
    dbType = dbLexicalEntry
    gql_Entities = None

    class Meta:
        interfaces = (
            CompositeIdHolder,
            AdditionalMetadata,
            CreatedAt,
            DeletedAt,
            MarkedForDeletion,
            Relationship,
            MovedTo)

    def __init__(
        self,
        *args,
        gql_Entities = None,
        **kwargs):

        if gql_Entities is not None:
            self.gql_Entities = gql_Entities

        super().__init__(*args, **kwargs)

    @fetch_object('entities')
    # @acl_check_by_id('view', 'lexical_entries_and_entities')
    def resolve_entities(self, info, mode='all'):
        if self.gql_Entities is not None:
            return self.gql_Entities
        if mode == 'all':
            publish = None
            accept = True
        elif mode == 'published':
            publish = True
            accept = True
        elif mode == 'unpublished':
            publish = False
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
            raise ResponseError(message="mode: <all|published|unpublished|not_accepted>")

        result = list()
        entities = DBSession.query(dbEntity, dbPublishingEntity).\
            filter(dbEntity.parent_client_id == self.dbObject.client_id,
                   dbEntity.parent_object_id == self.dbObject.object_id,
                   dbEntity.client_id == dbPublishingEntity.client_id,
                   dbEntity.object_id == dbPublishingEntity.object_id)
        if publish is not None:
            entities = entities.filter(dbPublishingEntity.published == publish)
        if accept is not None:
            entities = entities.filter(dbPublishingEntity.accepted == accept)
        entities = entities.filter(dbEntity.marked_for_deletion == False).yield_per(100)

        def graphene_entity(cur_entity, cur_publishing):
            ent = Entity(id = (cur_entity.client_id, cur_entity.object_id))
            ent.dbObject = cur_entity
            ent.publishingentity = cur_publishing
            return ent

        result = [graphene_entity(entity[0], entity[1]) for entity in entities]
        # for db_entity in self.dbObject.entity:
        #     publ = db_entity.publishingentity
        #     if publish is not None and publ.published != publish:
        #         continue
        #     if accept is not None and publ.accepted != accept:
        #         continue
        #     if db_entity.marked_for_deletion:
        #         continue
        #     ent = Entity(id = [db_entity.client_id, db_entity.object_id])
        #     ent.dbObject = db_entity
        #     ent.publishingentity = publ
        #     result.append(ent)


        return result






class CreateLexicalEntry(graphene.Mutation):
    """
    example:
    mutation {
        create_lexicalentry(id: [949,21], perspective_id: [71,5]) {
            field {
                id
            }
            triumph
        }
    }

    (this example works)
    returns:

    {
      "create_lexicalentry": {
        "field": {
          "id": [
            949,
            21
          ]
        },
        "triumph": true
      }
    }
    """

    class Arguments:
        id = LingvodocID()
        perspective_id = LingvodocID(required=True)

    lexicalentry = graphene.Field(LexicalEntry)
    triumph = graphene.Boolean()

    @staticmethod
    @client_id_check()
    def mutate(root, info, **args):
        perspective_id = args.get('perspective_id')
        id = args.get('id')
        client_id = id[0] if id else info.context["client_id"]
        object_id = id[1] if id else None
        id = [client_id, object_id]
        info.context.acl_check('create', 'lexical_entries_and_entities', perspective_id)
        dblexentry = create_lexicalentry(id, perspective_id, True)
        """
        perspective_client_id = perspective_id[0]
        perspective_object_id = perspective_id[1]

        object_id = None
        client_id_from_args = None
        if len(id) == 1:
            client_id_from_args = id[0]
        elif len(id) == 2:
            client_id_from_args = id[0]
            object_id = id[1]

        client_id = info.context["client_id"]
        client = DBSession.query(Client).filter_by(id=client_id).first()

        user = DBSession.query(dbUser).filter_by(id=client.user_id).first()
        if not user:
            raise ResponseError(message="This client id is orphaned. Try to logout and then login once more.")

        perspective = DBSession.query(dbDictionaryPerspective). \
            filter_by(client_id=perspective_client_id, object_id=perspective_object_id).first()
        if not perspective:
            raise ResponseError(message="No such perspective in the system")

        if client_id_from_args:
            if check_client_id(authenticated=client.id, client_id=client_id_from_args):
                client_id = client_id_from_args
            else:
                raise ResponseError(message="Error: client_id from another user")

        dblexentry = dbLexicalEntry(object_id=object_id, client_id=client_id,
                               parent_object_id=perspective_object_id, parent=perspective)
        DBSession.add(dblexentry)
        DBSession.flush()
        """
        lexicalentry = LexicalEntry(id=[dblexentry.client_id, dblexentry.object_id])
        lexicalentry.dbObject = dblexentry
        return CreateLexicalEntry(lexicalentry=lexicalentry, triumph=True)

class DeleteLexicalEntry(graphene.Mutation):
    """
    example:
    mutation {
        delete_lexicalentry(id: [949,21]) {
            lexicalentry {
                id
            }
            triumph
        }
    }
    now returns:
      {
      "delete_lexicalentry": {
        "lexicalentry": {
          "id": [
            949,
            21
          ]
        },
        "triumph": true
      }
    }
    """

    class Arguments:
        id = LingvodocID(required=True)

    triumph = graphene.Boolean()

    @staticmethod
    def mutate(root, info, **args):
        lex_id = args.get('id')
        client_id, object_id = lex_id
        # dblexicalentry = DBSession.query(dbLexicalEntry).filter_by(client_id=client_id, object_id=object_id).first()
        dblexicalentry = CACHE.get(objects =
            {
                dbLexicalEntry : (lex_id, )
            },
        DBSession=DBSession)
        if not dblexicalentry or dblexicalentry.marked_for_deletion:
            raise ResponseError(message="Error: No such entry in the system")
        info.context.acl_check('delete', 'lexical_entries_and_entities',
                                   (dblexicalentry.parent_client_id, dblexicalentry.parent_object_id))
        settings = info.context.request.registry.settings
        if 'desktop' in settings:
            real_delete_lexical_entry(dblexicalentry, settings)
        else:
            del_object(dblexicalentry, "delete_lexicalentry", info.context.client_id)
        return DeleteLexicalEntry(triumph=True)


class BulkDeleteLexicalEntry(graphene.Mutation):

    class Arguments:
        ids = graphene.List(LingvodocID)
        perspective_id = LingvodocID()
        empty = graphene.Boolean()

    deleted_entries = graphene.List(LexicalEntry)
    triumph = graphene.Boolean()

    @staticmethod
    def mutate(
        root,
        info,
        ids = None,
        perspective_id = None,
        empty = None):

        entry_check_flag = True
        acl_check_flag = True

        # Deleting entries specified by ids.

        if ids is not None:

            entry_query = (

                DBSession

                    .query(
                        dbLexicalEntry)

                    .filter(
                        dbLexicalEntry.id.in_(
                            ids_to_id_query(ids))))

        # Deleting entries in a perspective.

        elif perspective_id is not None:

            if not dbPerspective.exists(perspective_id):

                return ResponseError('Perspective {perspective_id} does not exist.')

            info.context.acl_check(
                'delete', 'lexical_entries_and_entities', perspective_id)

            entry_check_flag = False
            acl_check_flag = False

            entry_query = (

                DBSession

                    .query(
                        dbLexicalEntry)

                    .filter(
                        dbLexicalEntry.parent_id == perspective_id,
                        dbLexicalEntry.marked_for_deletion == False))

            # If we need only empty entries, we check by entities.

            if empty:

                entry_query = (

                    entry_query
                        .outerjoin(dbEntity)
                        .filter(dbEntity.client_id == None))

        else:

            return ResponseError('Please specify what entries to delete.')

        task_id = str(uuid4())

        settings = info.context.request.registry.settings
        client_id = info.context.client_id

        deleted_entries = []

        for index, dblexicalentry in enumerate(entry_query.yield_per(4096)):

            if (entry_check_flag and (
                not dblexicalentry or dblexicalentry.marked_for_deletion)):

                raise ResponseError(message="Error: No such entry in the system")

            if acl_check_flag:

                info.context.acl_check(
                    'delete', 'lexical_entries_and_entities', dblexicalentry.parent_id)

            if 'desktop' in settings:

                real_delete_lexical_entry(
                    dblexicalentry, settings)

            else:

                del_object(
                    dblexicalentry,
                    "bulk_delete_lexicalentry",
                    client_id,
                    task_id = task_id,
                    counter = index + 1)

            deleted_entries.append(
                LexicalEntry(dblexicalentry))

        return BulkDeleteLexicalEntry(deleted_entries=deleted_entries, triumph=True)


class BulkUndeleteLexicalEntry(graphene.Mutation):

    class Arguments:
        ids = graphene.List(LingvodocID, required=True)

    triumph = graphene.Boolean()

    @staticmethod
    def mutate(root, info, **args):

        ids = args.get('ids')
        task_id = str(uuid4())

        lexical_entries = (

            DBSession

                .query(dbLexicalEntry)

                .filter(
                    tuple_(dbLexicalEntry.client_id, dbLexicalEntry.object_id).in_(ids))

                .all())

        settings = info.context.request.registry.settings
        client_id = info.context.client_id

        for dblexicalentry in lexical_entries:
            # client_id, object_id = lex_id
            # dblexicalentry = DBSession.query(dbLexicalEntry).filter_by(client_id=client_id, object_id=object_id).first()

            if not dblexicalentry:
                raise ResponseError(message = "Error: No such entry in the system")
            if not dblexicalentry.marked_for_deletion:
                raise ResponseError(message = f"Entry {dblexicalentry.id} is not deleted")

            info.context.acl_check(
                'delete', 'lexical_entries_and_entities', dblexicalentry.parent_id)

            undel_object(
                dblexicalentry,
                "bulk_undelete_lexicalentry",
                client_id,
                task_id = task_id,
                counter = len(ids))

        return BulkUndeleteLexicalEntry(triumph=True)


def create_n_entries_in_persp(n, pid, client):
    lexentries_list = list()
    client = client
    for i in range(0, n):
        id = [client.id, None]
        perspective_id = pid
        dblexentry = create_lexicalentry(id, perspective_id, True)
        lexentries_list.append(dblexentry)
    # DBSession.bulk_save_objects(lexentries_list)
    # DBSession.flush()
    CACHE.set(objects = lexentries_list, DBSession=DBSession)
    result = list()
    for lexentry in lexentries_list:
        result.append(LexicalEntry(id=[lexentry.client_id, lexentry.object_id]))
        result[-1].dbObject = lexentry
    return result


class BulkCreateLexicalEntry(graphene.Mutation):
    class Arguments:
        lexicalentries = graphene.List(ObjectVal)

    triumph = graphene.Boolean()

    @staticmethod
    def mutate(root, info, **args):
        lexicalentries = args.get('lexicalentries')
        lexentries_list = list()
        client = DBSession.query(Client).filter_by(id=info.context["client_id"]).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.",
                           info.context["client_id"])
        for lexentry in lexicalentries:
            id = lexentry["id"]

            perspective_id = lexentry["perspective_id"]

            dblexentry = create_lexicalentry(id, perspective_id, False)
            lexentries_list.append(dblexentry)

        # DBSession.bulk_save_objects(lexentries_list)
        # DBSession.flush()
        CACHE.set(objects = lexentries_list, DBSession=DBSession)
        return BulkCreateLexicalEntry(triumph=True)


class ConnectLexicalEntries(graphene.Mutation):
    class Arguments:
        connections = graphene.List(LingvodocID, required = True)
        field_id = LingvodocID(required=True)
        tag = graphene.String()

    triumph = graphene.Boolean()

    @staticmethod
    @client_id_check()
    def mutate(root, info, **args):
        client = DBSession.query(Client).filter_by(id=info.context["client_id"]).first()
        user = DBSession.query(dbUser).filter_by(id=client.user_id).first()
        tags = list()
        tag = args.get('id')
        if tag is not None:
            tags.append(tag)
        # if 'tag' in req:
        #     tags.append(req['tag'])
        field_id = args['field_id']
        field = DBSession.query(dbField).\
            filter_by(client_id=field_id[0], object_id=field_id[1]).first()

        if not field:
            raise ResponseError('No such field in the system')

        if field.data_type != 'Grouping Tag':
            raise ResponseError("wrong field data type")
        connections = args['connections']
        for par in connections:
            # parent = DBSession.query(dbLexicalEntry).\
            #     filter_by(client_id=par[0], object_id=par[1]).first()
            parent = CACHE.get(objects =
                {
                    dbLexicalEntry : (par, )
                },
            DBSession=DBSession)
            if not parent:
                raise ResponseError("No such lexical entry in the system")
            par_tags = find_all_tags(parent, field_id[0], field_id[1], False, False)
            for tag in par_tags:
                if tag not in tags:
                    tags.append(tag)
        if not tags:
            n = 10  # better read from settings
            rnd = random.SystemRandom()
            choice_str = string.digits + string.ascii_letters
            tag = (
                time.asctime(time.gmtime()) +
                ''.join(rnd.choice(choice_str) for c in range(n)))
            tags.append(tag)
        lexical_entries = find_lexical_entries_by_tags(tags, field_id[0], field_id[1], False, False)
        for par in connections:
            # parent = DBSession.query(dbLexicalEntry).\
            #     filter_by(client_id=par[0], object_id=par[1]).first()
            parent = CACHE.get(objects =
                {
                    dbLexicalEntry : (par, )
                },
            DBSession=DBSession)
            if parent not in lexical_entries:
                lexical_entries.append(parent)

        # Override create permission check, depends only on the user.
        # Admin is assumed to have all permissions.

        create_override_flag = (user.id == 1)

        if not create_override_flag:

            group = DBSession.query(dbGroup).join(dbBaseGroup).filter(
                dbBaseGroup.subject == 'lexical_entries_and_entities',
                dbGroup.subject_override == True,
                dbBaseGroup.action == 'create').one()

            create_override_flag = (
                user.is_active and user in group.users)

        create_flag_dict = {}

        for lex in lexical_entries:

            create_flag = create_override_flag

            # Create permission check, depends on the perspective of the lexical entry.

            if not create_flag:

                perspective_id = (
                    lex.parent_client_id, lex.parent_object_id)

                if perspective_id in create_flag_dict:
                    create_flag = create_flag_dict[perspective_id]

                else:

                    group = DBSession.query(dbGroup).join(dbBaseGroup).filter(
                        dbBaseGroup.subject == 'lexical_entries_and_entities',
                        dbGroup.subject_client_id == perspective_id[0],
                        dbGroup.subject_object_id == perspective_id[1],
                        dbBaseGroup.action == 'create').one()

                    create_flag = (
                        user.is_active and user in group.users)

                    create_flag_dict[perspective_id] = create_flag

            # Ensuring that the lexical entry has all link tags.

            for tag in tags:

                tag_entity = DBSession.query(dbEntity) \
                    .join(dbEntity.field) \
                    .join(dbEntity.publishingentity) \
                    .filter(dbEntity.parent == lex,
                            dbField.client_id == field_id[0],
                            dbField.object_id == field_id[1],
                            dbEntity.content == tag,
                            dbEntity.marked_for_deletion == False).first()

                if not tag_entity:

                    tag_entity = dbEntity(
                        client_id = client.id,
                        field = field,
                        content = tag,
                        parent = lex)

                    if create_flag:
                        tag_entity.publishingentity.accepted = True

                # If we are the admin, we automatically publish link entities.

                if user.id == 1:
                    tag_entity.publishingentity.published = True

        return ConnectLexicalEntries(triumph=True)


class DeleteGroupingTags(graphene.Mutation):
    class Arguments:
        id = LingvodocID(required=True)
        field_id = LingvodocID(required=True)

    triumph = graphene.Boolean()

    @staticmethod
    def mutate(root, info, **args):
        """
        mutation DeleteTag{
            delete_grouping_tags(field_id: [66,25]
        id:[1523, 9499]
             ) {
                    triumph
                }
            }
        """
        settings = info.context.request.registry.settings
        request = info.context.request
        variables = {'auth': authenticated_userid(request)}
        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        user = DBSession.query(dbUser).filter_by(id=client.user_id).first()

        client_id, object_id = args.get("id")
        field_client_id, field_object_id = args.get("field_id")
        field = DBSession.query(dbField).filter_by(client_id=field_client_id,
                                                 object_id=field_object_id).first()
        if not field:
            return ResponseError("No such field in the system")
        elif field.data_type != 'Grouping Tag':
            return ResponseError("Wrong type of field")

        perspective_id = (

            DBSession

                .query(
                    dbLexicalEntry.parent_client_id,
                    dbLexicalEntry.parent_object_id)

                .filter(
                    dbLexicalEntry.client_id == client_id,
                    dbLexicalEntry.object_id == object_id,
                    dbLexicalEntry.marked_for_deletion == False)

                .first())

        if not perspective_id:
            return ResponseError('No such lexical entry in the system.')

        # Checking permissions.

        info.context.acl_check(
            'delete',
            'lexical_entries_and_entities',
            perspective_id)

        entities = DBSession.query(dbEntity).filter_by(field_client_id=field_client_id,
                                                     field_object_id=field_object_id,
                                                     parent_client_id=client_id,
                                                     parent_object_id=object_id, marked_for_deletion=False).all()
        if entities:
            for dbentity in entities:
                if 'desktop' in settings:
                    real_delete_entity(dbentity, settings)
                else:
                    del_object(dbentity, "delete_grouping_tags", info.context.client_id)
                # entity.marked_for_deletion = True
                # objecttoc = DBSession.query(dbObjectTOC).filter_by(client_id=entity.client_id,
                #                                                  object_id=entity.object_id).one()
                # objecttoc.marked_for_deletion = True
            return DeleteGroupingTags(triumph=True)
        return DeleteGroupingTags(triumph=False)
