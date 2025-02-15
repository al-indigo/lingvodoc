import graphene

from lingvodoc.schema.gql_holders import (
    LingvodocObjectType,
    CompositeIdHolder,
    AdditionalMetadata,
    CreatedAt,
    MarkedForDeletion,
    Relationship,
    MovedTo,
    fetch_object,
    client_id_check,
    del_object,
    acl_check_by_id,
    ResponseError,
    LingvodocID,
    ObjectVal
)

from lingvodoc.models import (
    Entity as dbEntity,
    Field as dbField,
    PublishingEntity as dbPublishingEntity,
    LexicalEntry as dbLexicalEntry,
    Client,
    DBSession,
    DictionaryPerspective as dbDictionaryPerspective,
    Group as dbGroup,
    LexicalEntry as dbLexicalEntry,
    User as dbUser,
    ObjectTOC as dbObjectTOC,
    BaseGroup as dbBaseGroup,
    Dictionary as dbDictionary,
    TranslationGist as dbTranslationGist,
    Client as dbClient
)
from pyramid.security import authenticated_userid
import logging
from lingvodoc.cache.caching import TaskStatus
from pyramid.httpexceptions import (
    HTTPOk,
    HTTPBadRequest,
    HTTPUnauthorized
)
from lingvodoc.views.v2.desktop_sync.core import async_download_dictionary
import os
import json
import requests
from pyramid.request import Request
from pyramid.response import Response
from lingvodoc.utils.search import recursive_sort
from pdb import set_trace as A

from lingvodoc.cache.caching import CACHE

log = logging.getLogger(__name__)


def make_request(path, cookies, req_type='get', json_data=None):
    session = requests.Session()
    session.headers.update({'Connection': 'Keep-Alive'})
    adapter = requests.adapters.HTTPAdapter(pool_connections=1, pool_maxsize=1, max_retries=10)
    # with open('authentication_data.json', 'r') as f:
    #     cookies = json.loads(f.read())
    session.mount('http://', adapter)
    # log.error(path)
    if req_type == 'get':
        status = session.get(path, cookies=cookies)
    elif req_type == 'post':
        status = session.post(path, json=json_data, cookies=cookies)
    else:
        return None
    return status

# single_id_tables = [Client, dbUser]
#
# composite_id_tables = [dbLanguage, dbDictionary]
#
# dict_from_obj = lambda r: {c.name: getattr(r, c.name) for c in r.__table__.columns}
# str_from_ids = lambda x: "_".join([str(x.client_id), str(x.object_id)])
#
#
#
#
# def server_basic_sync(existing):
#     existing_sets = dict()
#
#     for tablename in existing:
#         existing_sets[tablename] = set(existing[tablename])
#
#     server_answer = dict()
#     server_answer['update'] = dict()
#     server_answer['insert'] = dict()
#
#     for table in single_id_tables:
#         server_current = DBSession.query(table).all()
#         server_answer['update'][tablename] = [dict_from_obj(obj) for obj in server_current if obj.id in existing_sets[tablename]]
#         server_answer['insert'][tablename] = [dict_from_obj(obj) for obj in server_current if obj.id  not in existing_sets[tablename]]
#
#     for table in composite_id_tables[1:]:
#         server_current = DBSession.query(table).all()
#         server_answer['update'][tablename] = [dict_from_obj(obj) for obj in server_current if str_from_ids(obj.client_id, obj.client_object_id) in existing_sets[tablename]]
#         server_answer['insert'][tablename] = [dict_from_obj(obj) for obj in server_current if not str_from_ids(obj.client_id, obj.client_object_id) in existing_sets[tablename]]
#
#
#     langs = DBSession.query(dbLanguage).filter_by(marked_for_deletion=False).order_by(dbLanguage.parent_client_id,
#                                                                                     dbLanguage.parent_object_id,
#                                                                                     dbLanguage.additional_metadata[
#                                                                                         'younger_siblings']).all()
#     visited = set()
#     stack = set()
#     current_languages = list()
#     recursive_sort(langs, visited, stack, current_languages)
#     # sort_language_tree()
#     server_answer['language'] = [dict_from_obj(obj) for obj in current_languages]
#
#     return server_answer
#
#
# def bulk_upsert(items):
#     # todo: find out, how to do it right
#     inserts = [DBSession.insert(dbLanguage).values(item).on_conflict_do_update(
#         index_elements=[dbLanguage.client_id, dbLanguage.object_id], set_=item) for item in items]
#     for insert in inserts:
#         dbLanguage.execute(insert)
#
#
# def desktop_basic_sync():
#     existing = dict()
#     for table in single_id_tables:
#         existing[table.__tablename__] = DBSession.query(table.id).all()
#
#     for table in composite_id_tables:
#         existing[table.__tablename__] = str_from_ids(DBSession.query(table.client_id, table.client_object_id).all())
#
#
#     server_answer = send(existing)
#     # updated objects may link to new, so all inserts should be done first
#     for tablename in [t.__tablename__ for t in single_id_tables]:
#         DBSession.bulk_insert_mappings(server_answer['insert'][tablename])
#         DBSession.bulk_insert_mappings(server_answer['update'][tablename])
#
#     bulk_upsert(server_answer['language'])
#
#     for tablename in [t.__tablename__ for t in composite_id_tables]:
#         DBSession.bulk_insert_mappings(server_answer['insert'][tablename])
#         DBSession.bulk_insert_mappings(server_answer['update'][tablename])


def download_dictionary(dict_id, request, user_id, locale_id):
    my_args = dict()
    my_args["client_id"] = dict_id[0]
    my_args["object_id"] = dict_id[1]
    my_args["central_server"] = request.registry.settings["desktop"]['central_server']
    my_args["storage"] = request.registry.settings["storage"]
    my_args['sqlalchemy_url'] = request.registry.settings["sqlalchemy.url"]
    my_args["cookies"] = json.loads(request.cookies.get('server_cookies'))
    try:
        # dictionary_obj = DBSession.query(dbDictionary).filter_by(client_id=dict_id[0],
        #                                                        object_id=dict_id[1]).first()
        dictionary_obj = CACHE.get(objects =
            {
                dbDictionary : (args["dictionary_id"], )
            },
        DBSession=DBSession)
        if not dictionary_obj:
            dict_json = make_request(my_args["central_server"] + 'dictionary/%s/%s' % (
                dict_id[0],
                dict_id[1]), my_args["cookies"])
            if dict_json.status_code != 200 or not dict_json.json():
                task = TaskStatus(user_id, "Dictionary sync with server", "dictionary name placeholder", 5)
            else:
                dict_json = dict_json.json()
                gist = DBSession.query(dbTranslationGist). \
                    filter_by(client_id=dict_json['translation_gist_client_id'],
                              object_id=dict_json['translation_gist_object_id']).first()
                if gist:
                    task = TaskStatus(user_id, "Dictionary sync with server", gist.get_translation(locale_id), 5)
                else:
                    task = TaskStatus(user_id, "Dictionary sync with server", "dictionary name placeholder", 5)

        else:
            gist = DBSession.query(dbTranslationGist). \
                filter_by(client_id=dictionary_obj.translation_gist_client_id,
                          object_id=dictionary_obj.translation_gist_object_id).first()
            task = TaskStatus(user_id, "Dictionary sync with server", gist.get_translation(locale_id), 5)
    except:
        raise ResponseError('bad request')
    my_args["task_key"] = task.key
    my_args["cache_kwargs"] = request.registry.settings["cache_kwargs"]
    res = async_download_dictionary.delay(**my_args)
    # async_convert_dictionary_new(user_id, req['blob_client_id'], req['blob_object_id'], req["language_client_id"], req["language_object_id"], req["gist_client_id"], req["gist_object_id"], request.registry.settings["sqlalchemy.url"], request.registry.settings["storage"])
    log.debug("Conversion started")


class DownloadDictionary(graphene.Mutation):
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
        id = LingvodocID(required=True)

    triumph = graphene.Boolean()

    @staticmethod
    @client_id_check()
    def mutate(root, info, **args):
        request = info.context.request
        path = request.route_url('basic_sync')
        subreq = Request.blank(path)
        subreq.method = 'POST'
        subreq.headers = request.headers
        resp = request.invoke_subrequest(subreq)
        if resp.status_code != 200:
            raise ResponseError('network error')
        request = info.context.request
        locale_id = int(request.cookies.get('locale_id') or 2)
        dict_id = args['id']
        variables = {'auth': authenticated_userid(request)}
        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        user = DBSession.query(dbUser).filter_by(id=client.user_id).first()
        user_id = user.id
        download_dictionary(dict_id, request, user_id, locale_id)

        return DownloadDictionary(triumph=True)

class DownloadDictionaries(graphene.Mutation):
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
        ids = graphene.List(LingvodocID, required=True)

    triumph = graphene.Boolean()

    @staticmethod
    @client_id_check()
    def mutate(root, info, **args):
        request = info.context.request

        path = request.route_url('basic_sync')
        subreq = Request.blank(path)
        subreq.method = 'POST'
        subreq.headers = request.headers
        resp = request.invoke_subrequest(subreq)
        if resp.status_code != 200:
            raise ResponseError('network error')
        locale_id = int(request.cookies.get('locale_id') or 2)
        ids = args['ids']
        variables = {'auth': authenticated_userid(request)}
        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        user = DBSession.query(dbUser).filter_by(id=client.user_id).first()
        user_id = user.id

        for dict_id in ids:
            download_dictionary(dict_id, request, user_id, locale_id)

        return DownloadDictionary(triumph=True)



class Synchronize(graphene.Mutation):


    triumph = graphene.Boolean()

    @staticmethod
    @client_id_check()
    def mutate(root, info, **args):
        print('locking client')
        log.error('locking client')

        request = info.context.request
        DBSession.execute("LOCK TABLE client IN EXCLUSIVE MODE;")
        variables = {'auth': authenticated_userid(request)}
        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        if not client:
            raise ResponseError('try to login again')

        client_id = request.authenticated_userid
        user_id = Client.get_user_by_client_id(client_id).id

        is_admin = False
        if user_id == 1:
            is_admin = True

        task = TaskStatus(user_id, "Synchronisation with server", '', 16)
        task.set(1, 10, "Started", "")
        path = request.route_url('check_version')
        subreq = Request.blank(path)
        subreq.method = 'GET'
        subreq.headers = request.headers
        resp = request.invoke_subrequest(subreq)
        if resp.status_code != 200:
            raise ResponseError('network error 1')

        path = request.route_url('basic_sync')
        subreq = Request.blank(path)
        subreq.method = 'POST'
        subreq.headers = request.headers
        resp = request.invoke_subrequest(subreq)
        if resp.status_code != 200:
            raise ResponseError('network error 2')
        task.set(2, 15, "Basic synchronisation completed", "")

        path = request.route_url('diff_desk')
        subreq = Request.blank(path)
        subreq.method = 'POST'
        subreq.headers = request.headers
        subreq.json = {'task_key': task.key}
        log.error('before diff_desk')
        resp = request.invoke_subrequest(subreq)
        log.error('after diff_desk')
        if resp.status_code != 200:
            raise ResponseError('network error 3')
        task.set(15, 95, "All data uploaded to server", "")

        log.error('before downloading dictionaries')
        locale_id = int(request.cookies.get('locale_id') or 2)
        for dict_obj in DBSession.query(dbDictionary).all():
            download_dictionary([dict_obj.client_id, dict_obj.object_id], request, user_id, locale_id)
        log.error('after downloading dictionaries')

        path = request.route_url('basic_sync')
        subreq = Request.blank(path)
        subreq.method = 'POST'
        subreq.headers = request.headers
        resp = request.invoke_subrequest(subreq)
        if resp.status_code != 200:
            raise ResponseError('network error 2')
        task.set(16, 100, "Synchronisation complete (New data still can be downloading from server, look a other tasks)", "")
        return Synchronize(triumph=True)


class StopMutation(graphene.Mutation):

    class Arguments:
        stamp = graphene.String(required=True)

    triumph = graphene.Boolean()

    @staticmethod
    def mutate(root, info, stamp):

        client_id = info.context.client_id
        client = DBSession.query(dbClient).filter_by(id=client_id).first()

        if not client:
            return ResponseError('Only authorized users can stop running mutations.')

        storage = info.context.request.registry.settings['storage']
        stamp_path = os.path.join(storage['path'], 'lingvodoc_stamps')
        stamp_file = os.path.join(stamp_path, stamp)
        os.makedirs(stamp_path, exist_ok=True)

        # Touch stamp file
        open(stamp_file, 'a').close()

        print("!!! Stamp-to-stop")

        return StopMutation(triumph=True)
