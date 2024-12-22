import json
import datetime
import graphene
import logging
import re
import time

from graphql.language.ast import ObjectValue, ListValue, IntValue
from graphql.language import ast

from graphene.types import Scalar
from graphene.types.json import JSONString as JSONtype
from graphene.types.generic import GenericScalar

from sqlalchemy import or_, tuple_

from lingvodoc.models import (
    ObjectTOC,
    DBSession,
    Client as dbClient,
    LexicalEntry,
    DictionaryPerspectiveToField,
    TranslationGist as dbTranslationGist,
    TranslationAtom as dbTranslationAtom,
    UnstructuredData as dbUnstructuredData,
    User as dbUser
)
from lingvodoc.utils.verification import check_client_id
from lingvodoc.cache.caching import CACHE
from pdb import set_trace as A


# Setting up logging.
log = logging.getLogger(__name__)


# Object types
# TODO: delete comment lines

class ResponseError(Exception):
    """
    usage:
    (in resolve_***  functions)
    if not perm_check(client_id, "field"):
        return ResponseError(message = "Permission Denied")
    """
    def __init__(self, message, code=None, params=None, self_object=None):
        super().__init__(message)
        self.message = str(message)
        self.code = code
        self.params = params
        if self_object:
            self_object.ErrorHappened = True
        #DBSession.rollback()


class PermissionException(ResponseError):
    """
    Exception used to signal absense of permission.
    """

    def __init__(self, client_id, action, subject, subject_id):
        """
        Saves permission absense information.
        """

        super().__init__(
            'Client {0} doesn\'t have \'{1}\' permissions for \'{2}\'{3}.'.format(
                client_id,
                action,
                subject,
                '' if subject_id is None else ' {0}'.format(subject_id)),
            params=(client_id, action, subject, subject_id))

        self.client_id = client_id
        self.action = action
        self.subject = subject
        self.subject_id = subject_id


def acl_check_by_id(action, subject, id_key='id'):
    """
    Decorator enabling ACL-based permission checks, compatible with any functions/methods with signature
    'def f(arg1, args, context, *args)', e.g. resolve methods.
    Example:
        @acl_check_by_id('view', 'dictionary_role')
        def resolve_dictionary_role(self, args, context, info):
            ...
    With different identifier argument key, that is, assuming that subject identifier is  args.get(
    'different_id_key'):
     @client_id_check()
     @acl_check_by_id('create', 'perspective', id_key = "parent_id")
            ...
    """

    def decorator(resolve_f):
        def wrapper(self,info, **kwargs):

            if info.context.acl_check_if(action, subject, kwargs.get(id_key)):
                return resolve_f(self, info, **kwargs )

            else:
                raise PermissionException(info.context.client_id, action, subject, kwargs.get(id_key))

        return wrapper

    return decorator


def client_id_check():
    """
    client_id checks
    """
    def get_client_id_from_args(wrapper_args):
        if "id" in wrapper_args:
            if wrapper_args["id"]:
                return wrapper_args["id"][0]

    def decorator(resolve_f):
        def wrapper(self, *args, **kwargs):
            client_id = get_client_id_from_args(kwargs)
            info = args[0]
            authenticated = info.context.client_id
            if client_id:
                if not check_client_id(authenticated, client_id):
                    raise KeyError("Clients of mismatched users.", authenticated, client_id)
            else:
                client = DBSession.query(dbClient).filter_by(id=authenticated).first()
                if not client:
                    raise KeyError("Invalid client id (not registered on server). Try to logout and then login.",
                                   authenticated)
            return resolve_f(self, args[0], **kwargs)

        return wrapper

    return decorator


class LingvodocObjectType(graphene.ObjectType):

    dbObject = None
    ErrorHappened = None

    def __init__(
        self,
        dbObject = None,
        **kwargs):

        if dbObject is not None:

            self.dbObject = dbObject

            if hasattr(dbObject, 'id'):
                kwargs['id'] = dbObject.id

        super().__init__(**kwargs)


class LingvodocID(Scalar):
    """
    ObjectVal is GraphQL scalar value must be Object
    This class was obtained from from "GenericScalar" and restrict users to input string or float data
    """

    @staticmethod
    def identity(value):
        return value

    serialize = identity
    parse_value = identity

    @staticmethod
    def parse_literal(ast):
        if isinstance(ast, ListValue):
            if len(ast.values) != 2:
                return None
            result = list()
            for intvalue in ast.values:
                if isinstance(intvalue, IntValue):
                    result.append(int(intvalue.value))
                else:
                    return None
            return result
        else:
            return None


class Upload(graphene.types.Scalar):
    @staticmethod
    def serialize(value):
        return value

    @staticmethod
    def parse_literal(node):
        return node

    @staticmethod
    def parse_value(value):
        return value

# class LevelID(Scalar):
#     """
#     made specifically for returning language tree
#     """
#
#     @staticmethod
#     def identity(value):
#         return value
#
#     serialize = identity
#     parse_value = identity
#
#     @staticmethod
#     def parse_literal(ast):
#         if isinstance(ast, ListValue):
#             if len(ast.values) != 3:
#                 return None
#             result = list()
#             for intvalue in ast.values:
#                 if isinstance(intvalue, IntValue):
#                     result.append(int(intvalue.value))
#                 else:
#                     return None
#             return result
#         else:
#             return None


class ObjectVal(Scalar):
    """
    ObjectVal is GraphQL scalar value must be Object
    This class was obtained from from "GenericScalar" and restrict users to input string or float data
    """

    @staticmethod
    def identity(value):
        return value

    serialize = identity
    parse_value = identity

    @staticmethod
    def parse_literal(ast):
        if isinstance(ast, ObjectValue):
            return {field.name.value: GenericScalar.parse_literal(field.value) for field in ast.fields}
        else:
            return None


class JSONString(JSONtype):
    """
    graphene.types.json.JSONString replacement which has no JSON encoder and decoder
    """
    @staticmethod
    def serialize(dt):
        return dt

    @staticmethod
    def parse_literal(node):
        if isinstance(node, ast.StringValue):
            return json.loads(node.value)

    @staticmethod
    def parse_value(value):
        # json.loads(value)
        return value


class DateTime(Scalar):  # TODO: change format
    """
    graphene.types.json.JSONString replacement having different time format
    """
    @staticmethod
    def serialize(dt):
        # need to add assert
        if type(dt) is datetime.datetime:
            return dt.isoformat()
        else:
            dt = datetime.datetime.utcfromtimestamp(dt)  # can return wrong time
        return dt.isoformat()

    @staticmethod
    def parse_literal(node):
        if isinstance(node, ast.StringValue):
            return datetime.datetime.strptime(
                node.value, "%Y-%m-%dT%H:%M:%S.%f")

    @staticmethod
    def parse_value(value):
        return datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%f")

# Functions


def delete_message(
    function_name,
    deleted_by,
    task_id=None,
    counter=1,
    reason="Manually deleted",
    deleted_at=None,
    subject=None,
    undelete=False,
    __additional_info__=None):
    """
    This function generates a message for additional metadata to track reason of garbage collection.
    :param function_name: name of function that delete object
    :param deleted_by: client_id of user that delete object
    :param subject: object that initially became a reason for deletion (it can be some of parents objects of self)
    :param reason: how object was deleted
    :param counter: number of deleted objects
    :param task_id: uuid of operation
    :param deleted_at: time
    :param subject: like parent of deleted object. For ex. for atom it`s gist; for gist it`s perspective
    :return:
    """

    if not deleted_at:
        deleted_at = datetime.datetime.now(datetime.timezone.utc).timestamp()

    if reason == "Manually deleted" and undelete:
        reason = "Manually undeleted"

    message = {function_name:
                {"undeleted_at" if undelete else "deleted_at": deleted_at,
                 "reason": reason,
                 "undeleted_by" if undelete else "deleted_by": deleted_by,
                 "task_id": task_id,
                 "counter": counter,
                 "subject": subject},
                }

    if __additional_info__ is not None:
        message['__additional_info__'] = __additional_info__

    return message


def undelete_message(*args, **kwargs):
    return delete_message(*args, undelete = True, **kwargs)


def delete_gist_with_atoms(
    deleted_by, gist, task_id, subject = None, reason = "Manually deleted", **kwargs):

    key = "translations:%s:%s" % (
        str(gist.client_id),
        str(gist.object_id))
    CACHE.rem(key)

    atoms = DBSession.query(dbTranslationAtom).filter_by(parent=gist,
                                                         marked_for_deletion=False).all()

    atom_id_list = []

    for dbtranslationatom in atoms:
        key = "translation:%s:%s:%s" % (
            str(dbtranslationatom.parent_client_id),
            str(dbtranslationatom.parent_object_id),
            str(dbtranslationatom.locale_id))
        CACHE.rem(key)
        atom_id_list.append(
            [dbtranslationatom.client_id, dbtranslationatom.object_id])
        dbtranslationatom.mark_deleted(
            delete_message("del_object",
                           deleted_by,
                           task_id,
                           counter=len(atoms),
                           subject=(gist.client_id, gist.object_id),
                           reason="Automatically deleted after gist removal"))

    additional_info = kwargs.copy()
    additional_info['atom_id_list'] = sorted(atom_id_list)

    gist.mark_deleted(delete_message("del_object",
                                     deleted_by,
                                     task_id,
                                     reason=reason,
                                     subject=subject,
                                     __additional_info__=additional_info))


def undelete_gist_with_atoms(
    deleted_by, gist, task_id, subject = None, reason = "Manually undeleted", **kwargs):

    objecttoc_obj = (

        DBSession.query(ObjectTOC).filter_by(
            client_id = gist.client_id,
            object_id = gist.object_id).first())

    # Restoring translation atoms, if required.

    additional_info = objecttoc_obj.additional_metadata.get('__additional_info__')

    if additional_info:

        atom_id_list = additional_info.get('atom_id_list')

        if atom_id_list:

            atom_list = (

                DBSession.query(dbTranslationAtom)

                    .filter(
                        tuple_(dbTranslationAtom.client_id, dbTranslationAtom.object_id)
                            .in_(atom_id_list))

                    .all())

            for atom in atom_list:

                atom.mark_undeleted(
                    undelete_message(
                        "undel_object",
                        deleted_by,
                        task_id,
                        counter=len(atom_list),
                        subject=(gist.client_id, gist.object_id),
                        reason="Automatically undeleted after gist restore"))

    gist.mark_undeleted(
        undelete_message(
            "undel_object",
            deleted_by,
            task_id,
            reason = reason,
            subject = subject,
            __additional_info__ = kwargs or None))


def del_object(
    tmp_object,
    function_name,
    deleted_by,
    task_id=None,
    counter=1,
    update_cache = True,
    **kwargs):
    """
    This function can delete perspective/dictionary/any object with child gist and translationatoms.
    """

    if tmp_object.marked_for_deletion:
        return

    # if object is translationgist, delete translationgist with atoms
    if function_name == "delete_translationgist":
        gist = tmp_object
        delete_gist_with_atoms(deleted_by, gist, task_id, **kwargs)
        return

    # if object is Dictionary/Language/Field/etc. delete its translationgist
    elif hasattr(tmp_object, "translation_gist_object_id"):
        gist = DBSession.query(dbTranslationGist).filter_by(client_id=tmp_object.translation_gist_client_id,
                                                            object_id=tmp_object.translation_gist_object_id,
                                                            marked_for_deletion=False).first()
        if gist and gist.type != "Perspective":

            delete_gist_with_atoms(
                deleted_by,
                gist,
                task_id,
                reason = 'Automatically deleted after object removal',
                subject = (tmp_object.client_id, tmp_object.object_id))

    # delete object
    message = (

        delete_message(
            function_name,
            deleted_by,
            task_id,
            counter,
            __additional_info__ = kwargs or None))

    tmp_object.mark_deleted(message)
    if update_cache and CACHE.get(objects = { tmp_object.__class__ : ((tmp_object.client_id, tmp_object.object_id), ) }, DBSession=DBSession):
        CACHE.set(objects = [tmp_object, ], DBSession=DBSession)

def undel_object(
    tmp_object,
    function_name,
    deleted_by,
    task_id=None,
    counter=1,
    update_cache = True,
    **kwargs):
    """
    Reverse for del_object(), should be modified accordingly whenever del_object() is modified.
    """

    if not tmp_object.marked_for_deletion:
        return
    # if object is translationgist, undelete translationgist with atoms
    if function_name == "undelete_translationgist":
        gist = tmp_object
        undelete_gist_with_atoms(deleted_by, gist, task_id, **kwargs)
        return
    # if object is Dictionary/Language/Field/etc. undelete its translationgist
    elif hasattr(tmp_object, "translation_gist_object_id"):
        gist = DBSession.query(dbTranslationGist).filter_by(client_id=tmp_object.translation_gist_client_id,
                                                            object_id=tmp_object.translation_gist_object_id,
                                                            marked_for_deletion=True).first()
        if gist and gist.type != "Perspective":

            undelete_gist_with_atoms(
                deleted_by,
                gist,
                task_id,
                reason = 'Automatically undeleted after object restore',
                subject = (tmp_object.client_id, tmp_object.object_id))

    # delete object
    message = (

        undelete_message(
            function_name,
            deleted_by,
            task_id,
            counter,
            __additional_info__ = kwargs or None))

    tmp_object.mark_undeleted(message)
    if update_cache and CACHE.get(objects = { tmp_object.__class__ : ((tmp_object.client_id, tmp_object.object_id), ) }, DBSession=DBSession):
        CACHE.set(objects = [tmp_object, ], DBSession=DBSession)


#
# Special unique placeholder value signifying that the field has None value. We need that because None
# itself is used to mark field as having no value.
#
gql_none_value = object()


def fetch_object(attrib_name=None, ACLSubject=None, ACLKey=None):
    """
    This magic decorator, which the resolve_* functions have, sets the dbObject atribute
    after execution of a request to a DB or force the function to return attribute by attrib_name.
    class must have dbType and dbObject attributes
    example:
    dbType = dbLanguage

    @fetch_object("marked_for_deletion")
    def resolve_marked_for_deletion(self, args, context, info):
        ...
    """
    def dec(func):
        def wrapper(*args, **kwargs):
            cls = args[0]
            context = args[1].context

            if ACLSubject and ACLKey == 'id':
                context.acl_check('view', ACLSubject, cls.id)
            if cls.ErrorHappened:
                return None

            if (attrib_name is not None and
                ACLSubject is None):

                try:

                    value = getattr(cls, attrib_name)

                    if value is not None:
                        return (None if value is gql_none_value else value)

                except AttributeError:
                    pass

            if not cls.dbObject:
                if isinstance(cls.id, (int, str)):
                    # example: (id: 1),  (id: 'ihGLq')
                    cls.dbObject = DBSession.query(cls.dbType).filter_by(id=cls.id).first()
                    if cls.dbObject is None:
                        #cls.ErrorHappened = True
                        raise ResponseError(message="%s was not found" % cls.__class__, self_object=cls)
                elif isinstance(cls.id, (list, tuple)):
                    # example: (id: [2,3])
                    cls.dbObject = DBSession.query(cls.dbType).filter_by(client_id=cls.id[0],
                                                                         object_id=cls.id[1]).first()
                    # cls.dbObject = CACHE.get(objects = {cls.dbType : (cls.id, )})
                    if cls.dbObject is None:
                        #cls.ErrorHappened = True
                        raise ResponseError(message="%s was not found" % cls.__class__, self_object=cls)
            if ACLSubject and '_' in ACLKey:
                context.acl_check('view', ACLSubject,
                                  [getattr(cls.dbObject, ACLKey.replace('_', '_client_')),
                                   getattr(cls.dbObject, ACLKey.replace('_', '_object_'))])
            return func(*args, **kwargs)

        return wrapper

    return dec


# Common interfaces

class IdHolder(graphene.Interface):
    id = graphene.Int()

    @fetch_object("id")
    def resolve_id(self, info):
        return self.dbObject.id


class CompositeIdHolder(graphene.Interface):

    id = LingvodocID()
    created_by = graphene.Field('lingvodoc.schema.gql_user.User')

    # client_id = graphene.Int()
    # object_id = graphene.Int()

    @fetch_object("id")
    def resolve_id(self, info):
        return (self.dbObject.client_id, self.dbObject.object_id)

    # @fetch_object("client_id")
    # def resolve_client_id(self, info):
    #     return self.dbObject.client_id
    #
    # @fetch_object("object_id")
    # def resolve_object_id(self, info):
    #     return self.dbObject.object_id

    @fetch_object('created_by')
    def resolve_created_by(self, info):
        """
        Returns user whose client created the object.
        """

        from .gql_user import User

        dbuser = (

            DBSession
                .query(dbUser)
                .filter(
                    dbClient.id == self.dbObject.client_id,
                    dbUser.id == dbClient.user_id)
                .first())

        user = User(id = dbuser.id)
        user.dbObject = dbuser

        return user

class CreatedAt(graphene.Interface):
    created_at = graphene.Float() #DateTime()

    @staticmethod
    def from_timestamp(value):

        if isinstance(value, (int, float)):
            return value

        return value.replace(tzinfo = datetime.timezone.utc).timestamp()

    @fetch_object("created_at")
    def resolve_created_at(self, info):
        return CreatedAt.from_timestamp(self.dbObject.created_at)


class DeletedAt(graphene.Interface):
    deleted_at = graphene.Float()

    @fetch_object("deleted_at")
    def resolve_deleted_at(self, info):
        return self.dbObject.deleted_at


class Relationship(graphene.Interface):
    parent_id = LingvodocID()

    @fetch_object("parent_id")
    def resolve_parent_id(self, info):
        if self.dbObject.parent_client_id is None or self.dbObject.parent_object_id is None:
            return None
        return (self.dbObject.parent_client_id, self.dbObject.parent_object_id)

class SelfHolder(graphene.Interface):
    self_id = LingvodocID()

    @fetch_object("self_id")
    def resolve_self_id(self, info):
        if self.dbObject.self_client_id and self.dbObject.self_object_id:
            return (self.dbObject.self_client_id, self.dbObject.self_object_id)
        return None

class FieldHolder(graphene.Interface):
    field_id = LingvodocID()

    @fetch_object("field_id")
    def resolve_field_id(self, info):
        return (self.dbObject.field_client_id, self.dbObject.field_object_id)

class ParentLink(graphene.Interface):
    link_id = LingvodocID()

    @fetch_object("link_id")
    def resolve_link_id(self, info):
        if self.dbObject.link_client_id and self.dbObject.link_object_id:
            return (self.dbObject.link_client_id, self.dbObject.link_object_id)
        return None

class MarkedForDeletion(graphene.Interface):
    marked_for_deletion = graphene.Boolean()

    @fetch_object("marked_for_deletion")
    def resolve_marked_for_deletion(self, info):
        return self.dbObject.marked_for_deletion


class Position(graphene.Interface):
    position = graphene.Int()

    @fetch_object("position")
    def resolve_position(self, info):
        return self.dbObject.position


class TranslationGistHolder(graphene.Interface):
    translation_gist_id = LingvodocID()

    @fetch_object("translation_gist_id")
    def resolve_translation_gist_id(self, info):
        return (self.dbObject.translation_gist_client_id,
                self.dbObject.translation_gist_object_id)


class UserId(graphene.Interface):
    user_id = graphene.Int()

    @fetch_object("user_id")
    def resolve_user_id(self, info):
        return self.dbObject.user_id


class StateHolder(graphene.Interface):

    state_translation_gist_id = LingvodocID()

    status = graphene.Field(graphene.String, locale_id = graphene.Int())
    status_translations = ObjectVal()

    @fetch_object('state_translation_gist_id')
    def resolve_state_translation_gist_id(self, info):
        return (self.dbObject.state_translation_gist_client_id,
                self.dbObject.state_translation_gist_object_id)

    @fetch_object('status')
    def resolve_status(self, info, locale_id = None):

        if locale_id is None:
            locale_id = int(info.context.get('locale_id'))

        atom = DBSession.query(dbTranslationAtom.content).filter_by(
            parent_client_id=self.dbObject.state_translation_gist_client_id,
            parent_object_id=self.dbObject.state_translation_gist_object_id,
            marked_for_deletion=False,
            locale_id=locale_id).first()
        if atom:
            return atom[0]
        else:
            return None

    @fetch_object('status_translations')
    def resolve_status_translations(self, info):

        if self.dbObject:
            return self.dbObject.get_state_translations()


class TableName(graphene.Interface):
    table_name = graphene.String()

    @fetch_object("table_name")
    def resolve_table_name(self, info):
        return self.dbObject.table_name


class Name(graphene.Interface):
    name = graphene.String()

    @fetch_object("name")
    def resolve_name(self, info):
        return self.dbObject.name

class RealStoragePath(graphene.Interface):
    real_storage_path = graphene.String()

    @fetch_object("real_storage_path")
    def resolve_real_storage_path(self, info):
        return self.dbObject.real_storage_path

class LocaleId(graphene.Interface):
    locale_id = graphene.Int()

    @fetch_object("locale_id")
    def resolve_locale_id(self, info):
        return self.dbObject.locale_id


class Content(graphene.Interface):
    content = graphene.String()

    @fetch_object("content")
    def resolve_content(self, info):
        return self.dbObject.content


class TypeHolder(graphene.Interface):
    type = graphene.String()  # rename (?)

    @fetch_object("type")
    def resolve_type(self, info):
        return self.dbObject.type


class TranslationHolder(graphene.Interface):

    translation = graphene.String(locale_id=graphene.Int())
    translations = ObjectVal()

    @fetch_object("translation")
    def resolve_translation(self, info, locale_id = None):

        return (
            self.dbObject.get_translation( # TODO: fix it
                locale_id if locale_id is not None else info.context.get('locale_id')))

    @fetch_object("translations")
    def resolve_translations(self, info):

        return (
            self.dbObject.get_translations())


# rare interfaces

# Organization interface


class About(graphene.Interface):

    about = graphene.String()
    about_translations = ObjectVal()

    @fetch_object("about")
    def resolve_about(self, info, locale_id = None):

        if self.dbObject:
            return str(self.dbObject.get_about_translation( # TODO: fix it
                locale_id if locale_id is not None else info.context.get('locale_id')))

    @fetch_object("about_translations")
    def resolve_about_translations(self, info):

        if self.dbObject:
            return self.dbObject.get_about_translations()


# PublishedEntity interface


class Published(graphene.Interface):
    published = graphene.Boolean()

    @fetch_object("published")
    def resolve_published(self, info):
        if not self.publishingentity:
            self.publishingentity = self.dbObject.publishingentity
        return self.publishingentity.published


class Accepted(graphene.Interface):
    accepted = graphene.Boolean()

    @fetch_object("accepted")
    def resolve_accepted(self, info):
        if not self.publishingentity:
            self.publishingentity = self.dbObject.publishingentity
        return self.dbObject.publishingentity.accepted

# userBlobs interface


class DataType(graphene.Interface):  # TODO: check all data_type fields
    data_type = graphene.String()

    @fetch_object("data_type")
    def resolve_data_type(self, info):
        return self.dbObject.data_type

class FakeIds(graphene.Interface):
    fake_id = graphene.String()
    self_fake_id = graphene.String()


# LexicalEntry interface


class MovedTo(graphene.Interface):
    moved_to = graphene.String()

    @fetch_object("moved_to")
    def resolve_moved_to(self, info):
        return self.dbObject.moved_to

# Field interface


class DataTypeTranslationGistId(graphene.Interface):
    data_type_translation_gist_id = LingvodocID()


    @fetch_object("data_type_translation_gist_id")
    def resolve_data_type_translation_gist_id(self, info):
        return (self.dbObject.data_type_translation_gist_client_id,
                self.dbObject.data_type_translation_gist_object_id)
    #

class IsTranslatable(graphene.Interface):
    is_translatable = graphene.Boolean()

    @fetch_object("is_translatable")
    def resolve_is_translatable(self, info):
        return self.dbObject.is_translatable


class MergeMetadata(graphene.ObjectType):
    min_created_at = graphene.Float()
    original_client_id = graphene.Int()
    merge_tree = graphene.List(LingvodocID)


class Metadata(graphene.ObjectType):
    """
    graphene object that have all metadata attributes
    if new attributes of metadata are added, then this class has to be updated
    """
    hash = graphene.String()
    origin_id = LingvodocID()
    blobs = graphene.List(LingvodocID)
    merged_by = LingvodocID()
    data_type = graphene.String()
    blob_description = graphene.String()
    merge = graphene.Field(MergeMetadata)  # TODO: MergeMetadata
    original_filename = graphene.String()
    location = ObjectVal()
    client_id = graphene.Int()
    row_id = graphene.Int()
    merged_to = LingvodocID()
    is_protected = graphene.Boolean()
    previous_objects = graphene.List(LingvodocID)
    younger_siblings = graphene.List(LingvodocID)
    starling_fields = graphene.List(graphene.String)
    participant = graphene.List(LingvodocID)
    link_perspective_id = LingvodocID()
    tag_list = graphene.List(graphene.String)
    authors = graphene.List(graphene.String)
    #hasAudio = graphene.Boolean()
    years = graphene.List(graphene.String)
    kind = graphene.String()
    speakersAmount = graphene.String()
    humanSettlement = graphene.List(graphene.String)
    transcription_rules = graphene.String()
    admins = graphene.List(graphene.Int)
    #authors_process = graphene.List(graphene.String)
    informant = graphene.String()
    #discourse_type = graphene.List(graphene.String)
    #speech_type = graphene.List(graphene.String)
    #speech_genre = graphene.List(graphene.String)
    #text_topic = graphene.List(graphene.String)
    interrogator = graphene.List(graphene.String)
    processing = graphene.List(graphene.String)
    typeOfDiscourse = graphene.String()
    typeOfSpeech = graphene.String()
    speechGenre = graphene.String()
    theThemeOfTheText = graphene.String()

    license = graphene.String()

    titleOfTheWork = graphene.String()
    genre = graphene.String()
    timeOfWriting = graphene.String()
    quantitativeCharacteristic = graphene.String()
    bibliographicDataOfTheSource = graphene.String()
    translator = graphene.String()
    bibliographicDataOfTheTranslation = graphene.String()

    toc_mark = graphene.Boolean()
    attached_users = graphene.List(graphene.Int)

    parallel = graphene.Boolean()

    uploaded_at = graphene.String()

    markups = graphene.List(ObjectVal)

metadata_key_set = {

    key
    for key in Metadata.__dict__
    if not key.startswith('_')}

metadata_list_key_dict = {
    'blobs': [],
    'previous_objects': [],
    'younger_siblings': [],
    'starling_fields': [],
    'participant': [],
    'tag_list': [],
    'authors': [],
    'years': [],
    'humanSettlement': [],
    'admins': [],
    'interrogator': [],
    'processing': []}

metadata_list_key_list = [
    'blobs',
    'previous_objects',
    'younger_siblings',
    'starling_fields',
    'participant',
    'tag_list',
    'authors',
    'years',
    'humanSettlement',
    'admins',
    'interrogator',
    'processing']


class AdditionalMetadata(graphene.Interface):
    """
    Interface allowing to work with metadata as with the dictionary

    example:
    additional_metadata{
      hash
     }
    """

    additional_metadata = graphene.Field(Metadata)

    @staticmethod
    def from_object(metadata_dict):

        if not metadata_dict:
            return Metadata(**metadata_list_key_dict)

        metadata_dict = {
            key: value
            for key, value in metadata_dict.items()
            if key in metadata_key_set}

        participant = (
            metadata_dict.get('participant'))

        if participant:

            metadata_dict['participant'] = [
                [x['client_id'], x['object_id']] for x in participant]

        blobs = (
            metadata_dict.get('blobs'))

        if blobs:

            metadata_dict['blobs'] = [
                [x['client_id'], x['object_id']] for x in blobs]

        # New 'authors' metadata is a list of author strings, while some old dictionaries (in particular,
        # ones converted from Dialeqt files via old convertion code) has 'authors' metadata which is a
        # single string with comma-separated author names.
        #
        # So if 'authors' is a string, we assume that it's comma-separated and split it.

        authors = metadata_dict.get('authors')

        if isinstance(authors, str):

            metadata_dict['authors'] = (
                re.split(r'\s*,', authors))

        for key in metadata_list_key_list:

            if key not in metadata_dict:
                metadata_dict[key] = []

        return Metadata(**metadata_dict)

    @fetch_object("additional_metadata")
    def resolve_additional_metadata(self, info):

        return (
            AdditionalMetadata.from_object(
                self.dbObject.additional_metadata))


class CommonFieldsComposite(
    MarkedForDeletion,
    AdditionalMetadata,
    CreatedAt,
    DeletedAt,
    CompositeIdHolder,
    Relationship,
    TranslationGistHolder,
    TranslationHolder):
    """
    used in Dictionary, DictionaryPerspective and Language classes as Interfaces because function
    tree = graphene.List(CommonFieldsComposite, ) does not support listing
    """
    pass#fieldType = graphene.String()  # depricated? -> data_typeString()


class UserAndOrganizationsRoles(graphene.ObjectType):
    roles_users = graphene.List(ObjectVal)
    roles_organizations = graphene.List(ObjectVal)

    def resolve_roles_users(self, info):
        return self.roles_users

    def resolve_roles_organizations(self, info):
        return self.roles_organizations


class UnstructuredData(LingvodocObjectType):

    dbType = dbUnstructuredData

    id = graphene.String()
    client_id = graphene.Int()
    data = ObjectVal()
    additional_metadata = ObjectVal()

    class Meta:
        interfaces = (CreatedAt,)

    @fetch_object('id')
    def resolve_id(self, info):
        return self.dbObject.id

    @fetch_object('client_id')
    def resolve_client_id(self, info):
        return self.dbObject.client_id

    @fetch_object('data')
    def resolve_data(self, info):
        return self.dbObject.data

    @fetch_object('additional_metadata')
    def resolve_additional_metadata(self, info):
        return self.dbObject.additional_metadata


def get_published_translation_gist_id_query(session = DBSession):

    return (

        session

            .query(
                dbTranslationGist.client_id,
                dbTranslationGist.object_id)

            .filter(
                dbTranslationGist.marked_for_deletion == False,
                dbTranslationGist.type == 'Service',
                dbTranslationAtom.parent_client_id == dbTranslationGist.client_id,
                dbTranslationAtom.parent_object_id == dbTranslationGist.object_id,
                dbTranslationAtom.locale_id == 2,
                dbTranslationAtom.marked_for_deletion == False,

                or_(
                    dbTranslationAtom.content == 'Published',
                    dbTranslationAtom.content == 'Limited access')))


def get_published_translation_gist_id_subquery_query(session = DBSession):

    return (

        session.query(
            get_published_translation_gist_id_query(session)
                .subquery()))


def get_published_translation_gist_id_cte(query = None, session = DBSession):

    if query is None:
        query = get_published_translation_gist_id_query(session)

    return query.cte()


def get_published_translation_gist_id_cte_query(cte = None, session = DBSession):

    if cte is None:
        cte = get_published_translation_gist_id_cte(session = session)

    return session.query(cte)

