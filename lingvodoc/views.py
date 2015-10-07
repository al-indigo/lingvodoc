from pyramid.response import Response
from pyramid.view import view_config

from sqlalchemy.exc import DBAPIError

from .scripts.lingvodoc_converter import convert_one

from .models import (
    DBSession,
    Dictionary,
    User,
    Client,
    Email,
    Passhash,
    Group,
    BaseGroup,
    Language,
    Locale,
    UITranslationString,
    UserEntitiesTranslationString,
    Dictionary,
    DictionaryPerspective,
    DictionaryPerspectiveField,
    LexicalEntry,
    LevelOneEntity,
    LevelTwoEntity,
    GroupingEntity,
    PublishLevelOneEntity,
    PublishGroupingEntity,
    PublishLevelTwoEntity,
    Base,
    Organization,
    UserBlobs,
    About
    )

from .merge_perspectives import (
    mergeDicts
    )

from sqlalchemy.orm import sessionmaker
from pyramid.security import (
    Everyone,
    Allow,
    Deny
    )

from sqlalchemy.exc import IntegrityError
from sqlalchemy import (
    func,
    or_
)
from sqlalchemy.orm import joinedload, subqueryload, noload, join, joinedload_all

from pyramid.httpexceptions import HTTPForbidden
from pyramid.httpexceptions import HTTPFound
from pyramid.httpexceptions import HTTPNotFound, HTTPOk, HTTPBadRequest, HTTPConflict, HTTPInternalServerError
from pyramid.httpexceptions import HTTPUnauthorized
from pyramid.security import authenticated_userid
from pyramid.security import forget
from pyramid.security import remember
from pyramid.view import forbidden_view_config

# from pyramid.chameleon_zpt import render_template_to_response
from pyramid.renderers import render_to_response
from pyramid.response import FileResponse

import os
import shutil
import datetime
import base64
import string
import time

import keystoneclient.v3 as keystoneclient
import swiftclient.client as swiftclient
import random
import sqlalchemy
from sqlalchemy import create_engine

from sqlalchemy.inspection import inspect
from pyramid.request import Request
# import redis
import json

import logging
log = logging.getLogger(__name__)


class CommonException(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def find_by_translation_string(locale_id, translation_string):
        trstr = DBSession.query(UserEntitiesTranslationString).\
            filter_by(locale_id=locale_id, translation_string=translation_string).first()
        if not trstr:
            trstr = DBSession.query(UserEntitiesTranslationString).\
                filter_by(locale_id=1, translation_string=translation_string).first()
        if not trstr:
            return translation_string
        return trstr.translation


def add_translation_to_translation_string(locale_id, translation, translation_string, client_id):
        client = DBSession.query(Client).filter_by(id=client_id).first()
        uets = DBSession.query(UserEntitiesTranslationString).filter_by(locale_id=locale_id,
                                                                        translation_string=translation_string).first()
        if not uets:
            uets = UserEntitiesTranslationString(object_id=DBSession.query(UserEntitiesTranslationString).filter_by(client_id=client.id).count()+1,
                                                 client_id=client.id,
                                                 locale_id=locale_id,
                                                 translation_string=translation_string,
                                                 translation=translation)
            DBSession.add(uets)
            DBSession.flush()
        else:
            uets.translation = translation


def find_locale_id(request):
    try:
        return int(request.cookies['locale_id'])
    except:
        return 1

@view_config(route_name='basic_search', renderer='json', request_method='GET')
def basic_search(request):
    searchstring = request.params.get('leveloneentity')
    results_cursor = DBSession.query(LevelOneEntity).filter(LevelOneEntity.content.like('%'+searchstring+'%')).all()
    results = []
    for item in results_cursor:
        result = dict()
        result['client_id'] = item.parent_client_id
        result['object_id'] = item.parent_object_id
        result['origin_perspective_client_id'] = item.parent.parent.client_id
        result['origin_perspective_object_id'] = item.parent.parent.object_id
        result['origin_perspective_name'] = item.parent.parent.name
        result['origin_dictionary_client_id'] = item.parent.parent.parent.client_id
        result['origin_dictionary_object_id'] = item.parent.parent.parent.object_id
        result['origin_dictionary_name'] = item.parent.parent.parent.name
        results.append(result)
    return results

from multiprocessing import Process

#TODO: make it normal, it's just a test
@view_config(route_name='convert_dictionary', renderer='json', request_method='POST')
def convert_dictionary(request):
    req = request.json_body

    client_id = req['blob_client_id']
    object_id = req['blob_object_id']
    parent_client_id = req['parent_client_id']
    parent_object_id = req['parent_object_id']
    client = DBSession.query(Client).filter_by(id=authenticated_userid(request)).first()
    user = client.user

    blob = DBSession.query(UserBlobs).filter_by(client_id=client_id, object_id=object_id).first()

    p = Process(target=convert_one, args=(blob.real_storage_path,
                                          user.login,
                                          user.password.hash,
                                          parent_client_id,
                                          parent_object_id))
    p.start()
    request.response.status = HTTPOk.code
    return {"status": "Your dictionary is being converted."
                      " Wait 5-15 minutes and you will see new dictionary in your dashboard."}


@view_config(route_name='language', renderer='json', request_method='GET')
def view_language(request):
    response = dict()
    client_id = request.matchdict.get('client_id')
    object_id = request.matchdict.get('object_id')
    language = DBSession.query(Language).filter_by(client_id=client_id, object_id=object_id).first()
    if language:
        if not language.marked_for_deletion:
            response['parent_client_id'] = language.parent_client_id
            response['parent_object_id'] = language.parent_object_id
            response['client_id'] = language.client_id
            response['object_id'] = language.object_id
            response['translation_string'] = find_by_translation_string(locale_id=find_locale_id(request),
                                                                        translation_string=language.translation_string)
            if language.locale:
                response['locale_exist'] = True
            else:
                response['locale_exist'] = False

            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such language in the system")}


def language_info(lang, request):
    result = dict()
    result['client_id'] = lang.client_id
    result['object_id'] = lang.object_id
    result['translation_string'] = find_by_translation_string(locale_id=find_locale_id(request),
                                                              translation_string=lang.translation_string)
    if lang.locale:
        result['locale_exist'] = True
    else:
        result['locale_exist'] = False

    if lang.language:
        contains = []
        for childlang in lang.language:
            contains += [language_info(childlang, request)]
        result['contains'] = contains

    return result


@view_config(route_name='get_languages', renderer='json', request_method='GET')
def view_languages_list(request):
    response = dict()
    langs = []
    languages = DBSession.query(Language).filter_by(parent = None).all()
    if languages:
        for lang in languages:

            langs += [language_info(lang, request)]

    response['languages'] = langs

    request.response.status = HTTPOk.code
    return response


@view_config(route_name='language', renderer='json', request_method='PUT', permission='edit')
def edit_language(request):
    try:
        response = dict()
        client_id = request.matchdict.get('client_id')
        object_id = request.matchdict.get('object_id')
        client = DBSession.query(Client).filter_by(id=request.authenticated_userid).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.")
        language = DBSession.query(Language).filter_by(client_id=client_id, object_id=object_id).first()
        if language:
            if not language.marked_for_deletion:
                req = request.json_body
                if 'parent_client_id' in req:
                    language.parent_client_id = req['parent_client_id']
                if 'parent_object_id' in req:
                    language.parent_object_id = req['parent_object_id']
                # TODO: fix translation stuff
                if 'translation' in req:
                    add_translation_to_translation_string(locale_id=find_locale_id(request),
                                                          translation_string=language.translation_string,
                                                          translation=req['translation'], client_id=client.id)
                request.response.status = HTTPOk.code
                return response
        request.response.status = HTTPNotFound.code
        return {'error': str("No such language in the system")}
    except KeyError as e:
        request.response.status = HTTPBadRequest.code
        return {'error': str(e)}


@view_config(route_name='language', renderer='json', request_method='DELETE', permission='delete')
def delete_language(request):
    response = dict()
    client_id = request.matchdict.get('client_id')
    object_id = request.matchdict.get('object_id')
    language = DBSession.query(Language).filter_by(client_id=client_id, object_id=object_id).first()
    if language:
        if not language.marked_for_deletion:
            language.marked_for_deletion = True
            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such language in the system")}


@view_config(route_name='create_language', renderer='json', request_method='POST', permission='create')
def create_language(request):
    try:
        variables = {'auth': request.authenticated_userid}

        req = request.json_body
        try:
            parent_client_id = req['parent_client_id']
            parent_object_id = req['parent_object_id']
        except:
            parent_client_id = None
            parent_object_id = None
        # TODO: fix translation stuff
        translation_string = req['translation_string']
        translation = req['translation']
        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.",
                           variables['auth'])
        user = DBSession.query(User).filter_by(id=client.user_id).first()
        if not user:
            raise CommonException("This client id is orphaned. Try to logout and then login once more.")

        parent = None
        if parent_client_id and parent_object_id:
            parent = DBSession.query(Language).filter_by(client_id=parent_client_id, object_id=parent_object_id).first()
        add_translation_to_translation_string(locale_id=find_locale_id(request), translation = translation,
                                              translation_string = translation_string, client_id = client.id)
        language = Language(object_id=DBSession.query(Language).filter_by(client_id=client.id).count()+1, client_id=variables['auth'],
                            translation_string = translation_string)
        DBSession.add(language)
        if parent:
            language.parent = parent
        DBSession.flush()
        basegroups = []
        basegroups += [DBSession.query(BaseGroup).filter_by(translation_string="Can edit languages").first()]
        basegroups += [DBSession.query(BaseGroup).filter_by(translation_string="Can delete languages").first()]
        groups = []
        for base in basegroups:
            group = Group(subject_client_id=language.client_id, subject_object_id=language.object_id, parent=base)
            groups += [group]
        for group in groups:
            user.groups.append(group)
        request.response.status = HTTPOk.code
        return {'object_id': language.object_id,
                'client_id': language.client_id}
    except KeyError as e:
        request.response.status = HTTPBadRequest.code
        return {'error': str(e)}

    except IntegrityError as e:
        request.response.status = HTTPInternalServerError.code
        return {'error': str(e)}

    except CommonException as e:
        request.response.status = HTTPConflict.code
        return {'error': str(e)}


@view_config(route_name='dictionary', renderer='json', request_method='GET')  # Authors -- names of users, who can edit?
def view_dictionary(request):
    response = dict()
    client_id = request.matchdict.get('client_id')
    object_id = request.matchdict.get('object_id')
    dictionary = DBSession.query(Dictionary).filter_by(client_id=client_id, object_id=object_id).first()
    if dictionary:
        if not dictionary.marked_for_deletion:
            response['parent_client_id'] = dictionary.parent_client_id
            response['parent_object_id'] = dictionary.parent_object_id
            response['client_id'] = dictionary.client_id
            response['object_id'] = dictionary.object_id
            response['name'] = find_by_translation_string(locale_id=find_locale_id(request),
                                                          translation_string=dictionary.name)
            response['status'] = dictionary.state
            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such dictionary in the system")}


@view_config(route_name='dictionary', renderer='json', request_method='PUT', permission='edit')
def edit_dictionary(request):
    try:
        response = dict()
        client_id = request.matchdict.get('client_id')
        object_id = request.matchdict.get('object_id')
        client = DBSession.query(Client).filter_by(id=request.authenticated_userid).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.")
        dictionary = DBSession.query(Dictionary).filter_by(client_id=client_id, object_id=object_id).first()
        if dictionary:
            if not dictionary.marked_for_deletion:
                req = request.json_body
                if 'parent_client_id' in req:
                    dictionary.parent_client_id = req['parent_client_id']
                if 'parent_object_id' in req:
                    dictionary.parent_object_id = req['parent_object_id']
                # TODO: fix translation stuff
                if 'name' in req:
                    add_translation_to_translation_string(locale_id=find_locale_id(request),
                                                          translation_string=dictionary.name,
                                                          translation=req['name'], client_id=client.id)
                request.response.status = HTTPOk.code
                return response

        request.response.status = HTTPNotFound.code
        return {'error': str("No such dictionary in the system")}
    except KeyError as e:
        request.response.status = HTTPBadRequest.code
        return {'error': str(e)}


@view_config(route_name='dictionary', renderer='json', request_method='DELETE', permission='delete')
def delete_dictionary(request):
    response = dict()
    client_id = request.matchdict.get('client_id')
    object_id = request.matchdict.get('object_id')
    dictionary = DBSession.query(Dictionary).filter_by(client_id=client_id, object_id=object_id).first()
    if dictionary:
        if not dictionary.marked_for_deletion:
            dictionary.marked_for_deletion = True
            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such dictionary in the system")}


@view_config(route_name='create_dictionary', renderer='json', request_method='POST', permission='create')
def create_dictionary(request):
    try:

        variables = {'auth': request.authenticated_userid}
        req = request.json_body
        parent_client_id = req['parent_client_id']
        parent_object_id = req['parent_object_id']
        name = req['name']
        translation = req['translation']
        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.")
        user = DBSession.query(User).filter_by(id=client.user_id).first()
        if not user:
            raise CommonException("This client id is orphaned. Try to logout and then login once more.")
        parent = DBSession.query(Language).filter_by(client_id=parent_client_id, object_id=parent_object_id).first()
        add_translation_to_translation_string(locale_id=find_locale_id(request), translation=translation,
                                              translation_string=name, client_id=client.id)

        dictionary = Dictionary(object_id=DBSession.query(Dictionary).filter_by(client_id=client.id).count() + 1,
                                client_id=variables['auth'],
                                name=name,
                                state='WiP',
                                parent=parent)
        DBSession.add(dictionary)
        DBSession.flush()
        for base in DBSession.query(BaseGroup).filter_by(dictionary_default=True):
            new_group = Group(parent=base,
                              subject_object_id=dictionary.object_id, subject_client_id=dictionary.client_id)
            new_group.users.append(user)
            DBSession.add(new_group)
            DBSession.flush()
        request.response.status = HTTPOk.code
        return {'object_id': dictionary.object_id,
                'client_id': dictionary.client_id}
    except KeyError as e:
        request.response.status = HTTPBadRequest.code
        return {'error': str(e)}

    except IntegrityError as e:
        request.response.status = HTTPInternalServerError.code
        return {'error': str(e)}

    except CommonException as e:
        request.response.status = HTTPConflict.code
        return {'error': str(e)}


@view_config(route_name='dictionary_status', renderer='json', request_method='GET')
def view_dictionary_status(request):
    response = dict()
    client_id = request.matchdict.get('client_id')
    object_id = request.matchdict.get('object_id')
    dictionary = DBSession.query(Dictionary).filter_by(client_id=client_id, object_id=object_id).first()
    if dictionary:
        if not dictionary.marked_for_deletion:
            response['status'] = dictionary.state
            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such dictionary in the system")}


@view_config(route_name='dictionary_status', renderer='json', request_method='PUT', permission='edit')
def edit_dictionary_status(request):
    response = dict()
    client_id = request.matchdict.get('client_id')
    object_id = request.matchdict.get('object_id')
    dictionary = DBSession.query(Dictionary).filter_by(client_id=client_id, object_id=object_id).first()
    if dictionary:
        if not dictionary.marked_for_deletion:
            req = request.json_body
            status = req['status']
            dictionary.state = status
            DBSession.add(dictionary)
            request.response.status = HTTPOk.code
            response['status'] = status
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such dictionary in the system")}


@view_config(route_name='perspective', renderer='json', request_method='GET')
def view_perspective(request):
    response = dict()
    client_id = request.matchdict.get('perspective_client_id')
    object_id = request.matchdict.get('perspective_id')
    parent_client_id = request.matchdict.get('dictionary_client_id')
    parent_object_id = request.matchdict.get('dictionary_object_id')
    parent = DBSession.query(Dictionary).filter_by(client_id=parent_client_id, object_id=parent_object_id).first()
    if not parent:
        request.response.status = HTTPNotFound.code
        return {'error': str("No such dictionary in the system")}

    perspective = DBSession.query(DictionaryPerspective).filter_by(client_id=client_id, object_id=object_id).first()
    if perspective:
        if not perspective.marked_for_deletion:
            if perspective.parent != parent:
                request.response.status = HTTPNotFound.code
                return {'error': str("No such pair of dictionary/perspective in the system")}
            response['parent_client_id'] = perspective.parent_client_id
            response['parent_object_id'] = perspective.parent_object_id
            response['client_id'] = perspective.client_id
            response['object_id'] = perspective.object_id
            response['name'] = find_by_translation_string(locale_id=find_locale_id(request),
                                                          translation_string=perspective.name)
            response['status'] = perspective.state
            response['marked_for_deletion'] = perspective.marked_for_deletion
            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such perspective in the system"), 'persp': str(perspective)}


@view_config(route_name='perspective', renderer='json', request_method='PUT', permission='edit')
def edit_perspective(request):
    try:
        response = dict()
        client_id = request.matchdict.get('perspective_client_id')
        object_id = request.matchdict.get('perspective_id')
        client = DBSession.query(Client).filter_by(id=request.authenticated_userid).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.")
        parent_client_id = request.matchdict.get('dictionary_client_id')
        parent_object_id = request.matchdict.get('dictionary_object_id')
        parent = DBSession.query(Dictionary).filter_by(client_id=parent_client_id, object_id=parent_object_id).first()
        if not parent:
            request.response.status = HTTPNotFound.code
            return {'error': str("No such dictionary in the system")}

        perspective = DBSession.query(DictionaryPerspective).filter_by(client_id=client_id, object_id=object_id).first()
        if perspective:
            if not perspective.marked_for_deletion:
                if perspective.parent != parent:
                    request.response.status = HTTPNotFound.code
                    return {'error': str("No such pair of dictionary/perspective in the system")}
                req = request.json_body
                # TODO: fix translation stuff
                if 'name_translation' in req:
                    add_translation_to_translation_string(locale_id=find_locale_id(request),
                                                          translation_string=perspective.name,
                                                          translation=req['name_translation'], client_id = client.id)
                if 'parent_client_id' in req:
                    perspective.parent_client_id = req['parent_client_id']
                if 'parent_object_id' in req:
                    perspective.parent_object_id = req['parent_object_id']
                request.response.status = HTTPOk.code
                return response
        else:
            request.response.status = HTTPNotFound.code
            return {'error': str("No such perspective in the system")}
    except KeyError as e:
        request.response.status = HTTPBadRequest.code
        return {'error': str(e)}


@view_config(route_name='perspective', renderer='json', request_method='DELETE', permission='delete')
def delete_perspective(request):
    response = dict()
    client_id = request.matchdict.get('perspective_client_id')
    object_id = request.matchdict.get('perspective_id')
    parent_client_id = request.matchdict.get('dictionary_client_id')
    parent_object_id = request.matchdict.get('dictionary_object_id')
    parent = DBSession.query(Dictionary).filter_by(client_id=parent_client_id, object_id=parent_object_id).first()
    if not parent:
        request.response.status = HTTPNotFound.code
        return {'error': str("No such dictionary in the system")}

    perspective = DBSession.query(DictionaryPerspective).filter_by(client_id=client_id, object_id=object_id).first()
    if perspective:
        if not perspective.marked_for_deletion:
            if perspective.parent != parent:
                request.response.status = HTTPNotFound.code
                return {'error': str("No such pair of dictionary/perspective in the system")}
            perspective.marked_for_deletion = True
            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such perspective in the system")}


@view_config(route_name='perspectives', renderer='json', request_method='GET')
def view_perspectives(request):
    response = dict()
    parent_client_id = request.matchdict.get('dictionary_client_id')
    parent_object_id = request.matchdict.get('dictionary_object_id')
    parent = DBSession.query(Dictionary).filter_by(client_id=parent_client_id, object_id=parent_object_id).first()
    if not parent:
        request.response.status = HTTPNotFound.code
        return {'error': str("No such dictionary in the system")}
    perspectives = []
    for perspective in parent.dictionaryperspective:
        path = request.route_url('perspective',
                                 dictionary_client_id=parent_client_id,
                                 dictionary_object_id=parent_object_id,
                                 perspective_client_id=perspective.client_id,
                                 perspective_id=perspective.object_id)
        subreq = Request.blank(path)
        # subreq = request.copy()
        subreq.method = 'GET'
        subreq.headers = request.headers
        resp = request.invoke_subrequest(subreq)
        perspectives += [resp.json]
    response['perspectives'] = perspectives
    request.response.status = HTTPOk.code
    return response


@view_config(route_name = 'create_perspective', renderer = 'json', request_method = 'POST', permission='create')
def create_perspective(request):
    try:
        variables = {'auth': authenticated_userid(request)}
        parent_client_id = request.matchdict.get('dictionary_client_id')
        parent_object_id = request.matchdict.get('dictionary_object_id')
        req = request.json_body
        name = req['name']
        translation = req['translation']
        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.")
        user = DBSession.query(User).filter_by(id=client.user_id).first()
        if not user:
            raise CommonException("This client id is orphaned. Try to logout and then login once more.")

        parent = DBSession.query(Dictionary).filter_by(client_id=parent_client_id, object_id=parent_object_id).first()
        if not parent:
            request.response.status = HTTPNotFound.code
            return {'error': str("No such dictionary in the system")}
        add_translation_to_translation_string(locale_id=find_locale_id(request), translation_string=name,
                                              translation=translation, client_id=client.id)
        perspective = DictionaryPerspective(object_id=DBSession.query(DictionaryPerspective).filter_by(client_id=client.id).count() + 1,
                                            client_id=variables['auth'],
                                            name=name,
                                            state='WiP',
                                            parent = parent)
        DBSession.add(perspective)
        DBSession.flush()
        owner_client = DBSession.query(Client).filter_by(id=parent.client_id).first()
        owner = owner_client.user
        for base in DBSession.query(BaseGroup).filter_by(perspective_default=True):
            new_group = Group(parent=base,
                              subject_object_id=perspective.object_id, subject_client_id=perspective.client_id)
            new_group.users.append(user)
            new_group.users.append(owner)
            DBSession.add(new_group)
            DBSession.flush()
        request.response.status = HTTPOk.code
        return {'object_id': perspective.object_id,
                'client_id': perspective.client_id}
    except KeyError as e:
        request.response.status = HTTPBadRequest.code
        return {'error': str(e)}

    except IntegrityError as e:
        request.response.status = HTTPInternalServerError.code
        return {'error': str(e)}

    except CommonException as e:
        request.response.status = HTTPConflict.code
        return {'error': str(e)}


@view_config(route_name = 'perspective_status', renderer = 'json', request_method = 'GET')
def view_perspective_status(request):
    response = dict()
    client_id = request.matchdict.get('perspective_client_id')
    object_id = request.matchdict.get('perspective_id')
    parent_client_id = request.matchdict.get('dictionary_client_id')
    parent_object_id = request.matchdict.get('dictionary_object_id')
    parent = DBSession.query(Dictionary).filter_by(client_id=parent_client_id, object_id=parent_object_id).first()
    if not parent:
        request.response.status = HTTPNotFound.code
        return {'error': str("No such dictionary in the system")}

    perspective = DBSession.query(DictionaryPerspective).filter_by(client_id=client_id, object_id=object_id).first()
    if perspective:
        if not perspective.marked_for_deletion:
            if perspective.parent != parent:
                request.response.status = HTTPNotFound.code
                return {'error': str("No such pair of dictionary/perspective in the system")}
            response['status'] = perspective.state
            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such perspective in the system")}


@view_config(route_name = 'perspective_status', renderer = 'json', request_method = 'PUT', permission='edit')
def edit_perspective_status(request):
    response = dict()
    client_id = request.matchdict.get('perspective_client_id')
    object_id = request.matchdict.get('perspective_id')
    parent_client_id = request.matchdict.get('dictionary_client_id')
    parent_object_id = request.matchdict.get('dictionary_object_id')
    parent = DBSession.query(Dictionary).filter_by(client_id=parent_client_id, object_id=parent_object_id).first()
    if not parent:
        request.response.status = HTTPNotFound.code
        return {'error': str("No such dictionary in the system")}

    perspective = DBSession.query(DictionaryPerspective).filter_by(client_id=client_id, object_id=object_id).first()
    if perspective:
        if not perspective.marked_for_deletion:
            if perspective.parent != parent:
                request.response.status = HTTPNotFound.code
                return {'error': str("No such pair of dictionary/perspective in the system")}
            req = request.json_body
            perspective.state = req['status']
            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such perspective in the system")}


@view_config(route_name = 'dictionary_roles', renderer = 'json', request_method = 'GET', permission='view')
def view_dictionary_roles(request):
    response = dict()
    client_id = request.matchdict.get('client_id')
    object_id = request.matchdict.get('object_id')
    dictionary = DBSession.query(Dictionary).filter_by(client_id=client_id, object_id=object_id).first()
    if dictionary:
        if not dictionary.marked_for_deletion:
            bases = DBSession.query(BaseGroup).filter_by(dictionary_default=True)
            roles_users = dict()
            roles_organizations = dict()
            for base in bases:
                group = DBSession.query(Group).filter_by(base_group_id=base.id,
                                                             subject_object_id=object_id,
                                                             subject_client_id=client_id).first()
                perm = base.translation_string
                users = []
                for user in group.users:
                    users += [user.id]
                organizations = []
                for org in group.organizations:
                    organizations += [org.id]
                roles_users[perm] = users
                roles_organizations[perm] = organizations
            response['roles_users'] = roles_users
            response['roles_organizations'] = roles_organizations

            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such dictionary in the system")}


@view_config(route_name = 'dictionary_roles', renderer = 'json', request_method = 'POST', permission='create')
def edit_dictionary_roles(request):
    response = dict()
    log.debug("VERY IMPORTANT 1")
    client_id = request.matchdict.get('client_id')
    object_id = request.matchdict.get('object_id')
    req = request.json_body
    user_id = None
    log.debug("VERY IMPORTANT",req)
    if 'user_id' in req:
        user_id = req['user_id']
    organization_id = None
    if 'organization_id' in req:
        organization_id = req['organization_id']

    role_names = req['role_names']
    dictionary = DBSession.query(Dictionary).filter_by(client_id=client_id, object_id=object_id).first()
    if dictionary:
        if not dictionary.marked_for_deletion:
            if user_id:
                user = DBSession.query(User).filter_by(id=user_id).first()
                if user:
                    for role_name in role_names:
                        base = DBSession.query(BaseGroup).filter_by(translation_string=role_name, dictionary_default=True).first()
                        if not base:
                            request.response.status = HTTPNotFound.code
                            return {'error': str("No such role in the system")}

                        group = DBSession.query(Group).filter_by(base_group_id=base.id,
                                                                 subject_object_id=object_id,
                                                                 subject_client_id=client_id).first()
                        client = DBSession.query(Client).filter_by(id=request.authenticated_userid).first()
                        userlogged = DBSession.query(User).filter_by(id=client.user_id).first()
                        if userlogged in group.users:
                            group.users.append(user)
                    request.response.status = HTTPOk.code
                    return response
                else:
                    request.response.status = HTTPNotFound.code
                    return {'error': str("No such user in the system")}
            if organization_id:
                org = DBSession.query(Organization).filter_by(id=organization_id).first()
                if org:
                    for role_name in role_names:
                        base = DBSession.query(BaseGroup).filter_by(translation_string=role_name, dictionary_default=True).first()
                        if not base:
                            request.response.status = HTTPNotFound.code
                            return {'error': str("No such role in the system")}

                        group = DBSession.query(Group).filter_by(base_group_id=base.id,
                                                                 subject_object_id=object_id,
                                                                 subject_client_id=client_id).first()
                        client = DBSession.query(Client).filter_by(id=request.authenticated_userid).first()
                        userlogged = DBSession.query(User).filter_by(id=client.user_id).first()
                        if userlogged in group.users:
                            group.org.append(org)
                    request.response.status = HTTPOk.code
                    return response
                else:
                    request.response.status = HTTPNotFound.code
                    return {'error': str("No such organization in the system")}
    request.response.status = HTTPNotFound.code
    return {'error': str("No such dictionary in the system")}


@view_config(route_name = 'dictionary_roles', renderer = 'json', request_method = 'DELETE', permission='delete')
def delete_dictionary_roles(request):
    response = dict()
    client_id = request.matchdict.get('client_id')
    object_id = request.matchdict.get('object_id')
    req = request.json_body
    user_id = req['user_id']
    role_names = req['role_names']
    dictionary = DBSession.query(Dictionary).filter_by(client_id=client_id, object_id=object_id).first()
    if dictionary:
        if not dictionary.marked_for_deletion:
            user = DBSession.query(User).filter_by(id=user_id).first()
            if user:
                for role_name in role_names:
                    base = DBSession.query(BaseGroup).filter_by(translation_string=role_name, dictionary_default=True).first()
                    if not base:
                        request.response.status = HTTPNotFound.code
                        return {'error': str("No such role in the system")}

                    group = DBSession.query(Group).filter_by(base_group_id=base.id,
                                                             subject_object_id=object_id,
                                                             subject_client_id=client_id).first()
                    if user in group.users:
                        group.users.remove(user)
                request.response.status = HTTPOk.code
                return response
            else:
                request.response.status = HTTPNotFound.code
                return {'error': str("No such user in the system")}
    request.response.status = HTTPNotFound.code
    return {'error': str("No such dictionary in the system")}


@view_config(route_name = 'perspective_roles', renderer = 'json', request_method = 'GET', permission='view')
def view_perspective_roles(request):
    response = dict()
    client_id = request.matchdict.get('perspective_client_id')
    object_id = request.matchdict.get('perspective_id')
    parent_client_id = request.matchdict.get('client_id')
    parent_object_id = request.matchdict.get('object_id')
    parent = DBSession.query(Dictionary).filter_by(client_id=parent_client_id, object_id=parent_object_id).first()
    if not parent:
        request.response.status = HTTPNotFound.code
        return {'error': str("No such dictionary in the system")}

    perspective = DBSession.query(DictionaryPerspective).filter_by(client_id=client_id, object_id=object_id).first()
    if perspective:
        if not perspective.marked_for_deletion:
            bases = DBSession.query(BaseGroup).filter_by(perspective_default=True)
            roles_users = dict()
            roles_organizations = dict()
            for base in bases:
                group = DBSession.query(Group).filter_by(base_group_id=base.id,
                                                             subject_object_id=object_id,
                                                             subject_client_id=client_id).first()
                perm = base.translation_string
                users = []
                for user in group.users:
                    users += [user.id]
                organizations = []
                for org in group.organizations:
                    organizations += [org.id]
                roles_users[perm] = users
                roles_organizations[perm] = organizations
            response['roles_users'] = roles_users
            response['roles_organizations'] = roles_organizations

            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such perspective in the system")}


@view_config(route_name = 'perspective_roles', renderer = 'json', request_method = 'POST', permission='create')
def edit_perspective_roles(request):
    response = dict()
    client_id = request.matchdict.get('perspective_client_id')
    object_id = request.matchdict.get('perspective_id')
    parent_client_id = request.matchdict.get('client_id')
    parent_object_id = request.matchdict.get('object_id')

    parent = DBSession.query(Dictionary).filter_by(client_id=parent_client_id, object_id=parent_object_id).first()
    if not parent:
        request.response.status = HTTPNotFound.code
        return {'error': str("No such dictionary in the system")}
    perspective = DBSession.query(DictionaryPerspective).filter_by(client_id=client_id, object_id=object_id).first()
    if perspective:
        if not perspective.marked_for_deletion:
            if perspective.parent != parent:
                request.response.status = HTTPNotFound.code
                return {'error': str("No such pair of dictionary/perspective in the system")}
            req = request.json_body
            user_id = req['user_id']
            role_names = req['role_names']
            user = DBSession.query(User).filter_by(id=user_id).first()
            if user:
                for role_name in role_names:
                    base = DBSession.query(BaseGroup).filter_by(translation_string=role_name, perspective_default=True).first()
                    if not base:
                        request.response.status = HTTPNotFound.code
                        return {'error': str("No such role in the system")}

                    group = DBSession.query(Group).filter_by(base_group_id=base.id,
                                                             subject_object_id=object_id,
                                                             subject_client_id=client_id).first()
                    client = DBSession.query(Client).filter_by(id=request.authenticated_userid).first()
                    userlogged = DBSession.query(User).filter_by(id=client.user_id).first()
                    if userlogged in group.users:
                        group.users.append(user)
                request.response.status = HTTPOk.code
                return response
            else:
                request.response.status = HTTPNotFound.code
                return {'error': str("No such user in the system")}
    request.response.status = HTTPNotFound.code
    return {'error': str("No such perspective in the system")}


@view_config(route_name = 'perspective_roles', renderer = 'json', request_method = 'DELETE', permission='delete')
def delete_perspective_roles(request):
    response = dict()
    client_id = request.matchdict.get('perspective_client_id')
    object_id = request.matchdict.get('perspective_id')
    req = request.json_body
    user_id = req['user_id']
    role_names = req['role_names']
    parent_client_id = request.matchdict.get('client_id')
    parent_object_id = request.matchdict.get('object_id')
    parent = DBSession.query(Dictionary).filter_by(client_id=parent_client_id, object_id=parent_object_id).first()
    if not parent:
        request.response.status = HTTPNotFound.code
        return {'error': str("No such dictionary in the system")}

    perspective = DBSession.query(DictionaryPerspective).filter_by(client_id=client_id, object_id=object_id).first()
    if perspective:
        if not perspective.marked_for_deletion:
            if perspective.parent != parent:
                request.response.status = HTTPNotFound.code
                return {'error': str("No such pair of dictionary/perspective in the system")}
            user = DBSession.query(User).filter_by(id=user_id).first()
            if user:
                for role_name in role_names:
                    base = DBSession.query(BaseGroup).filter_by(translation_string=role_name, perspective_default=True).first()
                    if not base:
                        request.response.status = HTTPNotFound.code
                        return {'error': str("No such role in the system")}

                    group = DBSession.query(Group).filter_by(base_group_id=base.id,
                                                             subject_object_id=object_id,
                                                             subject_client_id=client_id).first()
                    if user in group.users:
                        group.users.remove(user)
                request.response.status = HTTPOk.code
                return response
            else:
                request.response.status = HTTPNotFound.code
                return {'error': str("No such user in the system")}
    request.response.status = HTTPNotFound.code
    return {'error': str("No such perspective in the system")}


@view_config(route_name='signin', renderer='json', request_method='POST')
def signin(request):
    next = request.params.get('next') or request.route_url('home')
    req = request.json_body
    login = req['login']
    password = req['password']
    # login = request.POST.get('login', '')
    # password = request.POST.get('password', '')

    user = DBSession.query(User).filter_by(login=login).first()
    if user and user.check_password(password):
        client = Client(user_id=user.id)
        user.clients.append(client)
        DBSession.add(client)
        DBSession.flush()
        headers = remember(request, principal=client.id)
        response = Response()
        response.headers = headers
        locale_id = user.default_locale_id
        if not locale_id:
            locale_id = 1
        response.set_cookie(key='locale_id', value=str(locale_id))
        return HTTPFound(location=next, headers=response.headers)
    return HTTPUnauthorized(location=request.route_url('login'))


def all_languages(lang):
    langs = [{'object_id': lang.object_id, 'client_id': lang.client_id}]
    for la in lang.language:
        langs += all_languages(la)
    return langs


def check_for_client(obj, clients):
    if obj.client_id in clients:
        return True
    for entry in dir(obj):
        if entry in inspect(type(obj)).relationships:
            i = inspect(obj.__class__).relationships[entry]
            if i.direction.name == "ONETOMANY":
                x = getattr(obj, str(entry))
                answer = False
                for xx in x:
                    answer = answer or check_for_client(xx, clients)
                    if answer:
                        break
                return answer
    return False


@view_config(route_name = 'dictionaries', renderer = 'json', request_method='POST')
def dictionaries_list(request):
    req = request.json_body
    response = dict()
    user_created = None
    if 'user_created' in req:
        user_created = req['user_created']
    published = None
    if 'published' in req:
        published = req['published']
    user_participated = None
    if 'user_participated' in req:
        user_participated = req['user_participated']
    organization_participated = None
    if 'organization_participated' in req:
        organization_participated = req['organization_participated']
    languages = None
    if 'languages' in req:
        languages = req['languages']
    dicts = DBSession.query(Dictionary)
    if published:
        if published:
            dicts = dicts.filter_by(state='Published').join(DictionaryPerspective)\
                .filter(DictionaryPerspective.state == 'Published')
        # else:
        #     dicts = dicts.filter_by(state!='Published')
    if user_created:
        clients = DBSession.query(Client).filter(Client.user_id.in_(user_created)).all()
        cli = [o.id for o in clients]
        response['clients'] = cli
        dicts = dicts.filter(Dictionary.client_id.in_(cli))
    if languages:
        langs = []
        for lan in languages:
            lang = DBSession.query(Language).filter_by(object_id=lan['object_id'], client_id=lan['client_id']).first()
            langs += all_languages(lang)
        if langs:
            prevdicts = dicts.filter_by(parent_client_id=langs[0]['client_id'], parent_object_id=langs[0]['object_id'])
            langs.remove(langs[0])
            for lan in langs:
                prevdicts = prevdicts.subquery().select()
                prevdicts = dicts.filter_by(parent_client_id=lan['client_id'], parent_object_id=lan['object_id']).union_all(prevdicts)

            dicts = prevdicts
        else:
            dicts = DBSession.query(Dictionary).filter(sqlalchemy.sql.false())

    # add geo coordinates
    if organization_participated:
        organization = DBSession.query(Organization).filter(Organization.id.in_(organization_participated)).first()
        users = organization.users
        users_id = [o.id for o in users]

        clients = DBSession.query(Client).filter(Client.user_id.in_(users_id)).all()
        cli = [o.id for o in clients]

        dictstemp = []
        for dicti in dicts:
            if check_for_client(dicti, cli):
                dictstemp += [{'client_id': dicti.client_id, 'object_id': dicti.object_id}]
        if dictstemp:
            prevdicts = dicts.filter_by(parent_client_id=dictstemp[0]['client_id'], parent_object_id=dictstemp[0]['object_id'])
            dictstemp.remove(dictstemp[0])
            for dicti in dictstemp:
                prevdicts = prevdicts.subquery().select()
                prevdicts = dicts.filter_by(client_id=dicti['client_id'], object_id=dicti['object_id']).union_all(prevdicts)

            dicts = prevdicts
        else:
            dicts = DBSession.query(Dictionary).filter(sqlalchemy.sql.false())

    if user_participated:
        clients = DBSession.query(Client).filter(Client.user_id.in_(user_participated)).all()
        cli = [o.id for o in clients]

        dictstemp = []
        for dicti in dicts:
            if check_for_client(dicti, cli):
                dictstemp += [{'client_id': dicti.client_id, 'object_id': dicti.object_id}]
        if dictstemp:
            prevdicts = dicts.filter_by(parent_client_id=dictstemp[0]['client_id'], parent_object_id=dictstemp[0]['object_id'])
            dictstemp.remove(dictstemp[0])
            for dicti in dictstemp:
                prevdicts = prevdicts.subquery().select()
                prevdicts = dicts.filter_by(client_id=dicti['client_id'], object_id=dicti['object_id']).union_all(prevdicts)

            dicts = prevdicts
        else:
            dicts = DBSession.query(Dictionary).filter(sqlalchemy.sql.false())

    dictionaries = [{'object_id':o.object_id,'client_id':o.client_id, 'name':o.name, 'status':o.state,'parent_client_id':o.parent_client_id,'parent_object_id':o.parent_object_id} for o in dicts]

    response['dictionaries'] = dictionaries
    request.response.status = HTTPOk.code

    return response\

@view_config(route_name='all_perspectives', renderer = 'json', request_method='GET')
def perspectives_list(request):
    response = dict()
    is_template = None
    try:
        is_template = request.params.get('istemplate')
    except:
        pass
    state = None
    try:
        state = request.params.get('state')
    except:
        pass
    persps = DBSession.query(DictionaryPerspective)
    if is_template:
        persps = persps.filter(DictionaryPerspective).filter_by(is_template=is_template)
    if state:
        persps = persps.filter(DictionaryPerspective).filter_by(state=state)
    perspectives = []
    for perspective in persps:
        path = request.route_url('perspective',
                                 dictionary_client_id=perspective.parent_client_id,
                                 dictionary_object_id=perspective.parent_object_id,
                                 perspective_client_id=perspective.client_id,
                                 perspective_id=perspective.object_id)
        subreq = Request.blank(path)
        # subreq = request.copy()
        subreq.method = 'GET'
        subreq.headers = request.headers
        resp = request.invoke_subrequest(subreq)
        perspectives += [resp.json]
    response['perspectives'] = perspectives
    request.response.status = HTTPOk.code

    return response


@view_config(route_name='users', renderer='json', request_method='GET')
def users_list(request):
    response = dict()
    search = None
    try:
        search = request.params.get('search')
    except:
        pass
    users_temp = DBSession.query(User).join(User.email)
    users = []
    if search:
        name = search + '%'
        users_temp = users_temp.filter(or_(
            User.name.startswith(name),
            User.login.startswith(name),
            User.intl_name.startswith(name),
            Email.email.startswith(name)
        ))
    for user in users_temp:
        users += [{'id': user.id, 'name': user.name, 'login': user.login, 'intl_name': user.intl_name}]

    response['users'] = users
    request.response.status = HTTPOk.code

    return response


@view_config(route_name='perspective_fields', renderer='json', request_method='GET')
def view_perspective_fields(request):
    response = dict()
    client_id = request.matchdict.get('perspective_client_id')
    object_id = request.matchdict.get('perspective_id')
    perspective = DBSession.query(DictionaryPerspective).filter_by(client_id=client_id, object_id=object_id).first()
    client = DBSession.query(Client).filter_by(id=request.authenticated_userid).first()
    if not client:
        raise KeyError("Invalid client id (not registered on server). Try to logout and then login.")
    user = DBSession.query(User).filter_by(id=client.user_id).first()
    if not user:
        raise CommonException("This client id is orphaned. Try to logout and then login once more.")
    locale_id = find_locale_id(request)
    if perspective:
        fields = []
        for field in perspective.dictionaryperspectivefield:

            data = dict()
            if not field.marked_for_deletion:
                if field.level == 'leveloneentity' or field.level == 'groupingentity':
                    data['entity_type'] = find_by_translation_string(locale_id=locale_id,
                                                                     translation_string=field.entity_type)

                    data['data_type'] = find_by_translation_string(locale_id=find_locale_id(request),
                                                                   translation_string=field.data_type)
                    data['position'] = field.position
                    data['status'] = field.state
                    data['level'] = field.level
                    contains = []
                    if field.dictionaryperspectivefield:
                        for field2 in field.dictionaryperspectivefield:
                            if not field2.marked_for_deletion:
                                data2 = dict()
                                data2['entity_type'] = find_by_translation_string(locale_id=locale_id,
                                                                                  translation_string=field2.entity_type)

                                data2['data_type'] = find_by_translation_string(locale_id=locale_id,
                                                                                translation_string=field2.data_type)
                                data2['status'] = field2.state
                                data2['position'] = field2.position
                                data2['level'] = field2.level
                                data2['client_id'] = field2.client_id
                                data2['object_id'] = field2.object_id
                                contains += [data2]
                    data['contains'] = contains
                    if field.group:
                        #group = DBSession.query(UserEntitiesTranslationString).\
                        #    filter_by(translation_string=field.group, locale_id=locale_id).first()
                        data['group'] = find_by_translation_string(locale_id=locale_id,
                                                                   translation_string=field.group)
                    data['client_id'] = field.client_id
                    data['object_id'] = field.object_id
                    fields += [data]
        response['fields'] = fields
        request.response.status = HTTPOk.code
        return response
    else:
        request.response.status = HTTPNotFound.code
        return {'error': str("No such perspective in the system")}


@view_config(route_name='perspective_fields', renderer = 'json', request_method='DELETE', permission='edit')
def delete_perspective_fields(request):
    response = dict()
    client_id = request.matchdict.get('perspective_client_id')
    object_id = request.matchdict.get('perspective_id')

    perspective = DBSession.query(DictionaryPerspective).filter_by(client_id=client_id, object_id=object_id).first()
    if perspective:
        req = request.json_body
        cli_id = req['field_client_id']
        obj_id = req['field_object_id']
        field = DBSession.query(DictionaryPerspectiveField).filter_by(client_id=cli_id, object_id=obj_id).first()
        if field:
            field.marked_for_deletion = True
        else:
            request.response.status = HTTPNotFound.code
            return {'error': str("No such field in the system")}
        request.response.status = HTTPOk.code
        return response
    else:
        request.response.status = HTTPNotFound.code
        return {'error': str("No such perspective in the system")}


@view_config(route_name='perspective_fields', renderer='json', request_method='POST', permission='edit')
def create_perspective_fields(request):
    try:
        variables = {'auth': authenticated_userid(request)}
        parent_client_id = request.matchdict.get('perspective_client_id')
        parent_object_id = request.matchdict.get('perspective_id')
        dictionary_client_id = request.matchdict.get('dictionary_client_id')
        dictionary_object_id = request.matchdict.get('dictionary_object_id')
        fields = request.json_body['fields']
        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.")
        user = DBSession.query(User).filter_by(id=client.user_id).first()
        if not user:
            raise CommonException("This client id is orphaned. Try to logout and then login once more.")

        perspective = DBSession.query(DictionaryPerspective).filter_by(client_id=parent_client_id,
                                                                       object_id=parent_object_id).first()
        if not perspective:
            request.response.status = HTTPNotFound.code
            return {'error': str("No such dictionary in the system")}
        for field in perspective.dictionaryperspectivefield:
            field.marked_for_deletion = True
        locale_id = find_locale_id(request)

        for entry in fields:
            field = DictionaryPerspectiveField(object_id=DBSession.query(DictionaryPerspectiveField).filter_by(client_id=client.id).count() + 1,
                                               client_id=variables['auth'],
                                               entity_type=entry['entity_type'],
                                               data_type=entry['data_type'],
                                               parent=perspective,
                                               state=entry['status'])
            if 'group' in entry:
                field.group = entry['group']
                add_translation_to_translation_string(locale_id=locale_id,
                                                      translation_string=entry['group'],
                                                      translation=entry['group'], client_id=client.id)
            field.level = entry['level']
            field.position = entry['position']
            if 'contains' in entry:
                for ent in entry['contains']:
                    # TODO: Fix translation stuff
                    field2 = DictionaryPerspectiveField(object_id=DBSession.query(DictionaryPerspectiveField).filter_by(client_id=client.id).count() + 1,
                                                        client_id=variables['auth'],
                                                        entity_type=ent['entity_type'],
                                                        data_type=ent['data_type'],
                                                        level='leveltwoentity',
                                                        parent=perspective,
                                                        parent_entity=field,
                                                        state=entry['status'])
                    field2.position = ent['position']
                    DBSession.add(field2)
                    DBSession.flush()
                    add_translation_to_translation_string(locale_id=locale_id,
                                                          translation_string=ent['entity_type'],
                                                          translation=ent['entity_type'],
                                                          client_id=client.id)
                    add_translation_to_translation_string(locale_id=locale_id,
                                                          translation_string=ent['data_type'],
                                                          translation=ent['data_type'], client_id=client.id)
                    if 'group' in ent:
                        field.group = entry['group']  # is there need for group on second level?
                        add_translation_to_translation_string(locale_id=locale_id,
                                                              translation_string=ent['group'],
                                                              translation=ent['group'], client_id=client.id)
            DBSession.add(field)
            DBSession.flush()
            add_translation_to_translation_string(locale_id=locale_id,
                                                  translation_string=entry['entity_type'],
                                                  translation=entry['entity_type'], client_id=client.id)
            add_translation_to_translation_string(locale_id=locale_id,
                                                  translation_string=entry['data_type'],
                                                  translation=entry['data_type'], client_id=client.id)
        request.response.status = HTTPOk.code
        return {}
    except KeyError as e:
        request.response.status = HTTPBadRequest.code
        return {'error': str(e)}

    except IntegrityError as e:
        request.response.status = HTTPInternalServerError.code
        return {'error': str(e)}

    except CommonException as e:
        request.response.status = HTTPConflict.code
        return {'error': str(e)}


def object_file_path(obj, settings, data_type, filename, create_dir=False):
    base_path = settings['storage']['path']
    storage_dir = os.path.join(base_path, obj.__tablename__, data_type, str(obj.client_id), str(obj.object_id))
    if create_dir:
        os.makedirs(storage_dir, exist_ok=True)
    storage_path = os.path.join(storage_dir, filename)

    return storage_path


def openstack_upload(settings, file, file_name, content_type,  container_name):
    storage = settings['storage']
    authurl = storage['authurl']
    user = storage['user']
    key = storage['key']
    auth_version = storage['auth_version']
    tenant_name = storage['tenant_name']
    conn = swiftclient.Connection(authurl=authurl, user=user, key=key,  auth_version=auth_version,
                                  tenant_name=tenant_name)
    #storageurl = conn.get_auth()[0]
    conn.put_container(container_name)
    obje = conn.put_object(container_name, file_name,
                    contents = file,
                    content_type = content_type)
    #obje = conn.get_object(container_name, file_name)
    return str(obje)


# Json_input point to the method of file getting: if it's embedded in json, we need to decode it. If
# it's uploaded via multipart form, it's just saved as-is.
def create_object(request, content, obj, data_type, filename, json_input=True):
    # here will be object storage write as an option. Fallback (default) is filesystem write
    settings = request.registry.settings
    storage = settings['storage']
    if storage['type'] == 'openstack':
        if json_input:
            content = base64.urlsafe_b64decode(content)
        # TODO: openstack objects correct naming
        filename = str(obj.data_type) + '/' + str(obj.client_id) + '_' + str(obj.object_id)
        real_location = openstack_upload(settings, content, filename, obj.data_type, 'test')
    else:
        filename = filename or 'noname.noext'
        storage_path = object_file_path(obj, settings, data_type, filename, True)

        with open(storage_path, 'wb+') as f:
            if json_input:
                f.write(base64.urlsafe_b64decode(content))
            else:
                shutil.copyfileobj(content, f)

        real_location = storage_path

    url = "".join((settings['storage']['prefix'],
                  settings['storage']['static_route'],
                  obj.__tablename__,
                  '/',
                  data_type,
                  '/',
                  str(obj.client_id), '/',
                  str(obj.object_id), '/',
                  filename))
    return real_location, url

@view_config(route_name='upload_user_blob', renderer='json', request_method='POST')
def upload_user_blob(request):
    variables = {'auth': authenticated_userid(request)}
    response = dict()
    #print(request.POST)
    filename = request.POST['blob'].filename
    input_file = request.POST['blob'].file

    class Object(object):
        pass
    blob = Object()
    blob.client_id = variables['auth']
    client = DBSession.query(Client).filter_by(id=variables['auth']).first()
    blob.object_id = DBSession.query(UserBlobs).filter_by(client_id=client.id).count()+1
    blob.data_type = request.POST['data_type']

    blob.filename = filename

    current_user = DBSession.query(User).filter_by(id=client.user_id).first()

    blob_object = UserBlobs(object_id=blob.object_id,
                            client_id=blob.client_id,
                            name=filename,
                            data_type=blob.data_type,
                            user_id=current_user.id)

    current_user.userblobs.append(blob_object)
    DBSession.flush()
    blob_object.real_storage_path, blob_object.content = create_object(request, input_file, blob_object, blob.data_type,
                                                                       blob.filename, json_input=False)
    DBSession.add(blob_object)
    DBSession.add(current_user)
    request.response.status = HTTPOk.code
    response = {"client_id": blob.client_id, "object_id": blob.object_id, "content": blob_object.content}
    return response


# seems to be redundant
# @view_config(route_name='get_user_blob', request_method='GET')
# def get_user_blob(request):
#     client_id = request.matchdict.get('client_id')
#     object_id = request.matchdict.get('object_id')
#     blob = DBSession.query(UserBlobs).filter_by(client_id=client_id, object_id=object_id).first()
#     if blob:
#         FileResponse(blob.real_storage_path)
#     else:
#         raise HTTPNotFound

@view_config(route_name='list_user_blobs', renderer='json', request_method='GET')
def list_user_blobs(request):
    variables = {'auth': authenticated_userid(request)}
#    user_client_ids = [cl_id.id for cl_id in DBSession.query(Client).filter_by(id=variables['auth']).all()]
#    user_blobs = DBSession.query(UserBlobs).filter_by(client_id.in_(user_client_ids)).all()
    client = DBSession.query(Client).filter_by(id=variables['auth']).first()
    user_blobs = DBSession.query(UserBlobs).filter_by(user_id=client.user_id).all()
    request.response.status = HTTPOk.code
    response = [{'name': blob.name, 'content': blob.content, 'data_type': blob.data_type,
                 'client_id': blob.client_id, 'object_id': blob.object_id} for blob in user_blobs]
    return response


@view_config(route_name='create_level_one_entity', renderer='json', request_method='POST', permission='create')
def create_l1_entity(request):
    try:
        variables = {'auth': authenticated_userid(request)}
        response = dict()
        parent_client_id = request.matchdict.get('lexical_entry_client_id')
        parent_object_id = request.matchdict.get('lexical_entry_object_id')
        req = request.json_body
        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.")
        user = DBSession.query(User).filter_by(id=client.user_id).first()
        if not user:
            raise CommonException("This client id is orphaned. Try to logout and then login once more.")

        parent = DBSession.query(LexicalEntry).filter_by(client_id=parent_client_id, object_id=parent_object_id).first()
        if not parent:
            request.response.status = HTTPNotFound.code
            return {'error': str("No such lexical entry in the system")}
        additional_metadata=req.get('additional_metadata')
        if additional_metadata:
            additional_metadata = json.dumps(additional_metadata)
        entity = LevelOneEntity(client_id=client.id, object_id=DBSession.query(LevelOneEntity).filter_by(client_id=client.id).count() + 1, entity_type=req['entity_type'],
                                locale_id=req['locale_id'], additional_metadata=additional_metadata,
                                parent=parent)
        DBSession.add(entity)
        DBSession.flush()
        data_type = req.get('data_type')
        filename = req.get('filename')
        real_location = None
        url = None
        if data_type == 'image' or data_type == 'sound' or data_type == 'markup':
            real_location, url = create_object(request, req['content'], entity, data_type, filename)

        if url and real_location:
            entity.content = url
        else:
            entity.content = req['content']
        DBSession.add(entity)
        request.response.status = HTTPOk.code
        response['client_id'] = entity.client_id
        response['object_id'] = entity.object_id
        return response
#    except KeyError as e:
#        request.response.status = HTTPBadRequest.code
#        return {'error': str(e)}

    except IntegrityError as e:
        request.response.status = HTTPInternalServerError.code
        return {'error': str(e)}

    except CommonException as e:
        request.response.status = HTTPConflict.code
        return {'error': str(e)}

@view_config(route_name='create_entities_bulk', renderer='json', request_method='POST', permission='create')
def create_entities_bulk(request):
    try:
        variables = {'auth': authenticated_userid(request)}
        response = dict()
        req = request.json_body

        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.")
        user = DBSession.query(User).filter_by(id=client.user_id).first()
        if not user:
            raise CommonException("This client id is orphaned. Try to logout and then login once more.")

        inserted_items = []
        for item in req:
            if item['level'] == 'leveloneentity':
                parent = DBSession.query(LexicalEntry).filter_by(client_id=item['parent_client_id'], object_id=item['parent_object_id']).first()
                entity = LevelOneEntity(client_id=client.id,
                                        object_id=DBSession.query(LevelOneEntity).filter_by(client_id=client.id).count() + 1,
                                        entity_type=item['entity_type'],
                                        locale_id=item['locale_id'],
                                        additional_metadata=item.get('additional_metadata'),
                                        parent=parent)
            elif item['level'] == 'groupingentity':
                parent = DBSession.query(LexicalEntry).filter_by(client_id=item['parent_client_id'], object_id=item['parent_object_id']).first()
                entity = GroupingEntity(client_id=client.id,
                                        object_id=DBSession.query(GroupingEntity).filter_by(client_id=client.id).count() + 1,
                                        entity_type=item['entity_type'],
                                        locale_id=item['locale_id'],
                                        additional_metadata=item.get('additional_metadata'),
                                        parent=parent)
            elif item['level'] == 'leveltwoentity':
                parent = DBSession.query(LevelOneEntity).filter_by(client_id=item['parent_client_id'], object_id=item['parent_object_id']).first()
                entity = LevelTwoEntity(client_id=client.id,
                                        object_id=DBSession.query(LevelTwoEntity).filter_by(client_id=client.id).count() + 1,
                                        entity_type=item['entity_type'],
                                        locale_id=item['locale_id'],
                                        additional_metadata=item.get('additional_metadata'),
                                        parent=parent)
            DBSession.add(entity)
            DBSession.flush()
            data_type = item.get('data_type')
            filename = item.get('filename')
            real_location = None
            url = None
            if data_type == 'sound' or data_type == 'markup':
                real_location, url = create_object(request, item['content'], entity, data_type, filename)

            if url and real_location:
                entity.content = url
            else:
                entity.content = item['content']
            DBSession.add(entity)
            inserted_items.append({"client_id": entity.client_id, "object_id": entity.object_id})
        request.response.status = HTTPOk.code
        return inserted_items
    #    except KeyError as e:
    #        request.response.status = HTTPBadRequest.code
    #        return {'error': str(e)}

    except IntegrityError as e:
        request.response.status = HTTPInternalServerError.code
        return {'error': str(e)}

    except CommonException as e:
        request.response.status = HTTPConflict.code
        return {'error': str(e)}


@view_config(route_name='get_level_one_entity_indict', renderer='json', request_method='GET', permission='view')
@view_config(route_name='get_level_one_entity', renderer='json', request_method='GET', permission='view')
def view_l1_entity(request):
    response = dict()
    client_id = request.matchdict.get('client_id')
    object_id = request.matchdict.get('object_id')
    entity = DBSession.query(LevelOneEntity).filter_by(client_id=client_id, object_id=object_id).first()
    if entity:
        if not entity.marked_for_deletion:
            # TODO: fix urls to relative urls in content
            response = entity.track(False)
            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such entity in the system")}


@view_config(route_name='get_level_one_entity_indict', renderer='json', request_method='DELETE', permission='delete')
@view_config(route_name='get_level_one_entity', renderer='json', request_method='DELETE', permission='delete')
def delete_l1_entity(request):
    response = dict()
    client_id = request.matchdict.get('client_id')
    object_id = request.matchdict.get('object_id')

    entity = DBSession.query(LevelOneEntity).filter_by(client_id=client_id, object_id=object_id).first()
    if entity:
        if not entity.marked_for_deletion:

            entity.marked_for_deletion = True
            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such entity in the system")}

@view_config(route_name='get_level_two_entity_indict', renderer='json', request_method='GET', permission='view')
@view_config(route_name='get_level_two_entity', renderer='json', request_method='GET', permission='view')
def view_l2_entity(request):
    response = dict()
    client_id = request.matchdict.get('client_id')
    object_id = request.matchdict.get('object_id')
    entity = DBSession.query(LevelTwoEntity).filter_by(client_id=client_id, object_id=object_id).first()
    if entity:
        if not entity.marked_for_deletion:

            response['entity_type'] = find_by_translation_string(locale_id=find_locale_id(request),
                                                                 translation_string=entity.entity_type)
            response['parent_client_id'] = entity.parent_client_id
            response['parent_object_id'] = entity.parent_object_id
            response['content'] = find_by_translation_string(locale_id=find_locale_id(request),
                                                             translation_string=entity.content)
            response['locale_id'] = entity.locale_id
            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such entity in the system")}


@view_config(route_name='get_level_two_entity_indict', renderer='json', request_method='DELETE', permission='delete')
@view_config(route_name='get_level_two_entity', renderer='json', request_method='DELETE', permission='delete')
def delete_l2_entity(request):
    response = dict()
    client_id = request.matchdict.get('client_id')
    object_id = request.matchdict.get('object_id')

    entity = DBSession.query(LevelTwoEntity).filter_by(client_id=client_id, object_id=object_id).first()
    if entity:
        if not entity.marked_for_deletion:

            entity.marked_for_deletion = True
            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such entity in the system")}


@view_config(route_name='create_level_two_entity', renderer='json', request_method='POST', permission='create')
def create_l2_entity(request):
    try:
        variables = {'auth': authenticated_userid(request)}
        response = dict()
        parent_client_id = request.matchdict.get('level_one_client_id')
        parent_object_id = request.matchdict.get('level_one_object_id')
        req = request.json_body
        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.")
        user = DBSession.query(User).filter_by(id=client.user_id).first()
        if not user:
            raise CommonException("This client id is orphaned. Try to logout and then login once more.")

        parent = DBSession.query(LevelOneEntity).filter_by(client_id=parent_client_id, object_id=parent_object_id).first()
        if not parent:
            request.response.status = HTTPNotFound.code
            return {'error': str("No such level one entity in the system")}
        entity = LevelTwoEntity(client_id=client.id, object_id=DBSession.query(LevelTwoEntity).filter_by(client_id=client.id).count() + 1, entity_type=req['entity_type'],
                                locale_id=req['locale_id'], additional_metadata=req.get('additional_metadata'),
                                parent=parent)
        DBSession.add(entity)
        DBSession.flush()
        data_type = req.get('data_type')
        filename = req.get('filename')
        real_location = None
        url = None
        if data_type == 'image' or data_type == 'sound' or data_type == 'markup':
            real_location, url = create_object(request, req['content'], entity, data_type, filename)

        if url and real_location:
            entity.content = url
        else:
            entity.content = req['content']
        DBSession.add(entity)
        request.response.status = HTTPOk.code
        response['client_id'] = entity.client_id
        response['object_id'] = entity.object_id
        return response
    except KeyError as e:
        request.response.status = HTTPBadRequest.code
        return {'error': str(e)}

    except IntegrityError as e:
        request.response.status = HTTPInternalServerError.code
        return {'error': str(e)}

    except CommonException as e:
        request.response.status = HTTPConflict.code
        return {'error': str(e)}


@view_config(route_name='get_group_entity', renderer='json', request_method='GET', permission='view')
def view_group_entity(request):
    response = dict()
    client_id = request.matchdict.get('client_id')
    object_id = request.matchdict.get('object_id')

    entities = DBSession.query(GroupingEntity).filter_by(parent_client_id=client_id, parent_object_id=object_id).all()
    ents = []
    if entities:
        for entity in entities:
            if not entity.marked_for_deletion:
                ent = dict()
                ent['entity_type'] = find_by_translation_string(locale_id=find_locale_id(request),
                                                                     translation_string=entity.entity_type)
                ent['tag'] = entity.content
                entities2 = DBSession.query(GroupingEntity).filter_by(content=entity.content)
                objs = []
                for entry in entities2:
                    obj = {'client_id': entry.parent_client_id, 'object_id': entry.parent_object_id}
                    objs += [obj]
                ent['connections'] = objs
                ents += [ent]
        response['entities'] = ents
        request.response.status = HTTPOk.code
        return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No entities in the system")}


@view_config(route_name='get_connected_words', renderer='json', request_method='GET')
@view_config(route_name='get_connected_words_indict', renderer='json', request_method='GET')
def view_connected_words(request):
    response = dict()
    client_id = request.matchdict.get('client_id')
    object_id = request.matchdict.get('object_id')
    lexical_entry = DBSession.query(LexicalEntry).filter_by(client_id=client_id, object_id=object_id).first()
    if lexical_entry:
        if not lexical_entry.marked_for_deletion:
            words = []
            path = request.route_url('get_group_entity',
                                     client_id=lexical_entry.client_id,
                                     object_id=lexical_entry.object_id)
            subreq = Request.blank(path)
            subreq.method = 'GET'
            subreq.headers = request.headers
            respon = request.invoke_subrequest(subreq)
            if 'error' not in respon.json:
                connections = respon.json['entities'][0]['connections']
                for lex in connections:
                    path = request.route_url('lexical_entry',
                                             client_id=lex['client_id'],
                                             object_id=lex['object_id'])
                    subreq = Request.blank(path)
                    subreq.method = 'GET'
                    subreq.headers = request.headers
                    resp = request.invoke_subrequest(subreq)
                    words += [resp.json]
            else:
                path = request.route_url('lexical_entry',
                                         client_id=lexical_entry.client_id,
                                         object_id=lexical_entry.object_id)
                subreq = Request.blank(path)
                subreq.method = 'GET'
                subreq.headers = request.headers
                resp = request.invoke_subrequest(subreq)
                words += [resp.json]

            response['words'] = words
            return response

    request.response.status = HTTPNotFound.code
    return {'error': str("No such lexical entry in the system")}


@view_config(route_name='get_group_entity', renderer='json', request_method='DELETE', permission='delete')
def delete_group_entity(request):
    response = dict()
    client_id = request.matchdict.get('client_id')
    object_id = request.matchdict.get('object_id')
    entities = DBSession.query(GroupingEntity).filter_by(parent_client_id=client_id, parent_object_id=object_id).all()
    if entities:
        for entity in entities:
            if not entity.marked_for_deletion:
                entity.marked_for_deletion = True
        request.response.status = HTTPOk.code
        return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such entity in the system")}


@view_config(route_name='add_group_indict', renderer='json', request_method='POST')  # TODO: check for permission
@view_config(route_name='add_group_entity', renderer='json', request_method='POST')  # TODO: check for permission
def create_group_entity(request):
    try:
        variables = {'auth': authenticated_userid(request)}
        response = dict()
        req = request.json_body
        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.")
        user = DBSession.query(User).filter_by(id=client.user_id).first()
        if not user:
            raise CommonException("This client id is orphaned. Try to logout and then login once more.")

        tags = []
        if 'tag' in req:
            tags += [req['tag']]

        for par in req['connections']:
            parent = DBSession.query(LexicalEntry).\
                filter_by(client_id=par['client_id'], object_id=par['object_id']).first()
            if not parent:
                request.response.status = HTTPNotFound.code
                return {'error': str("No such lexical entry in the system")}
            par_tags = DBSession.query(GroupingEntity).\
                filter_by(entity_type=req['entity_type'], parent=parent).all()
            tags += [o.content for o in par_tags]
        if not tags:
            n = 10  # better read from settings
            tag = time.ctime() + ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits)
                                         for c in range(n))
            if not tag in tags:
                tags += [tag]
        parents = req['connections']
        # for tag in tags:
        #     pars = DBSession.query(GroupingEntity).\
        #         filter_by(content = tag).all()
        #     for par in pars:
        #         pa = {'client_id':par.parent_client_id, 'object_id':par.parent_object_id}
        #         if not pa in parents:
        #             parents += [pa]
        for par in parents:
            parent = DBSession.query(LexicalEntry).\
                filter_by(client_id=par['client_id'], object_id=par['object_id']).first()
            for tag in tags:
                ent = DBSession.query(GroupingEntity).\
                    filter_by(entity_type=req['entity_type'], content=tag, parent=parent).first()
                if not ent:
                    entity = GroupingEntity(client_id=client.id, object_id=DBSession.query(GroupingEntity).filter_by(client_id=client.id).count() + 1,
                                            entity_type=req['entity_type'], content=tag, parent=parent)
                    DBSession.add(entity)
                    DBSession.flush()
        log.debug('TAGS: %s', tags)
        request.response.status = HTTPOk.code
        return {}
    except KeyError as e:
        request.response.status = HTTPBadRequest.code
        return {'error': str(e)}

    except IntegrityError as e:
        request.response.status = HTTPInternalServerError.code
        return {'error': str(e)}

    except CommonException as e:
        request.response.status = HTTPConflict.code
        return {'error': str(e)}


@view_config(route_name='create_lexical_entry', renderer='json', request_method='POST', permission='create')
def create_lexical_entry(request):
    try:
        dictionary_client_id = request.matchdict.get('dictionary_client_id')
        dictionary_object_id = request.matchdict.get('dictionary_object_id')
        perspective_client_id = request.matchdict.get('perspective_client_id')
        perspective_id = request.matchdict.get('perspective_id')

        variables = {'auth': request.authenticated_userid}
        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.",
                           variables['auth'])
        user = DBSession.query(User).filter_by(id=client.user_id).first()
        if not user:
            raise CommonException("This client id is orphaned. Try to logout and then login once more.")
        perspective = DBSession.query(DictionaryPerspective).\
            filter_by(client_id=perspective_client_id, object_id = perspective_id).first()
        if not perspective:
            request.response.status = HTTPNotFound.code
            return {'error': str("No such perspective in the system")}

        lexentr = LexicalEntry(object_id=DBSession.query(LexicalEntry).filter_by(client_id=client.id).count() + 1, client_id=variables['auth'],
                               parent_object_id=perspective_id, parent=perspective)
        DBSession.add(lexentr)
        DBSession.flush()

        request.response.status = HTTPOk.code
        return {'object_id': lexentr.object_id,
                'client_id': lexentr.client_id}
    except KeyError as e:
        request.response.status = HTTPBadRequest.code
        return {'error': str(e)}

    except IntegrityError as e:
        request.response.status = HTTPInternalServerError.code
        return {'error': str(e)}

    except CommonException as e:
        request.response.status = HTTPConflict.code
        return {'error': str(e)}



# @view_config(route_name='lexical_entry_in_perspective', renderer='json', request_method='DELETE', permission='delete')#, permission='view')
# @view_config(route_name='lexical_entry', renderer='json', request_method='DELETE', permission='delete')
# def delete_lexical_entry(request):
#     client_id = request.matchdict.get('client_id')
#     object_id = request.matchdict.get('object_id')
#
#     entry = DBSession.query(LexicalEntry).filter_by(client_id=client_id, object_id=object_id).first()
#     if entry:
#         if not entry.marked_for_deletion:
#             entry.marked_for_deletion = True
#             request.response.status = HTTPOk.code
#             return {}
#     request.response.status = HTTPNotFound.code
#     return {'error': str("No such lexical entry in the system")}


@view_config(route_name='create_lexical_entry_bulk', renderer='json', request_method='POST', permission='create')
def create_lexical_entry_bulk(request):
    try:
        dictionary_client_id = request.matchdict.get('dictionary_client_id')
        dictionary_object_id = request.matchdict.get('dictionary_object_id')
        perspective_client_id = request.matchdict.get('perspective_client_id')
        perspective_id = request.matchdict.get('perspective_id')

        count = request.json_body.get('count')

        variables = {'auth': request.authenticated_userid}

        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.",
                           variables['auth'])
        user = DBSession.query(User).filter_by(id=client.user_id).first()
        if not user:
            raise CommonException("This client id is orphaned. Try to logout and then login once more.")
        perspective = DBSession.query(DictionaryPerspective). \
            filter_by(client_id=perspective_client_id, object_id = perspective_id).first()
        if not perspective:
            request.response.status = HTTPNotFound.code
            return {'error': str("No such perspective in the system")}

        lexes_list = []
        for i in range(0, count):
            lexentr = LexicalEntry(object_id=DBSession.query(LexicalEntry).filter_by(client_id=client.id).count() + 1, client_id=variables['auth'],
                                   parent_object_id=perspective_id, parent=perspective)
            DBSession.add(lexentr)
            lexes_list.append(lexentr)
        DBSession.flush()
        lexes_ids_list = []
        for lexentr in lexes_list:
            lexes_ids_list.append({'client_id': lexentr.client_id, 'object_id': lexentr.object_id})

        request.response.status = HTTPOk.code
        return lexes_ids_list
    except KeyError as e:
        request.response.status = HTTPBadRequest.code
        return {'error': str(e)}

    except IntegrityError as e:
        request.response.status = HTTPInternalServerError.code
        return {'error': str(e)}

    except CommonException as e:
        request.response.status = HTTPConflict.code
        return {'error': str(e)}


@view_config(route_name='lexical_entries_all', renderer='json', request_method='GET', permission='view')
def lexical_entries_all(request):
    response = dict()
    client_id = request.matchdict.get('perspective_client_id')
    object_id = request.matchdict.get('perspective_id')

    sort_criterion = request.params.get('sort_by') or 'Translation'

    log.debug('CRITERION: %s' % request.GET.get('sort_by'))
    start_from = request.params.get('start_from') or 0

    log.debug('START FROM: %s' % request.GET.get('start_from'))
    count = request.params.get('count') or 200

    parent = DBSession.query(DictionaryPerspective).filter_by(client_id=client_id, object_id=object_id).first()
    if parent:
        if not parent.marked_for_deletion:

            # lexical_entries_criterion = DBSession.query(LexicalEntry)\
            #     .filter_by(parent_client_id=parent.client_id, parent_object_id=parent.object_id)\
            #     .join(LevelOneEntity)\
            #     .filter_by(entity_type=sort_criterion)\
            #     .group_by(LexicalEntry).subquery().select()
            # lexical_entries_not_criterion = DBSession.query(LexicalEntry)\
            #     .filter_by(parent_client_id=parent.client_id, parent_object_id=parent.object_id)\
            #     .except_(lexical_entries_criterion)
            # lexical_entries_criterion2 = DBSession.query(LexicalEntry,
            #                                             func.min(LevelOneEntity.content).label("content"))\
            #     .filter_by(parent_client_id=parent.client_id, parent_object_id=parent.object_id)\
            #     .join(LevelOneEntity)\
            #     .filter_by(entity_type=sort_criterion)\
            #     .group_by(LexicalEntry)
            # lexical_entries_not_criterion = DBSession.query(LexicalEntry,
            #                                                 sqlalchemy.null().label('content'))\
            #     .correlate(lexical_entries_not_criterion)
            # lexical_entries = lexical_entries_criterion2.union(lexical_entries_not_criterion)\
            #     .filter_by(parent_client_id=parent.client_id, parent_object_id=parent.object_id).order_by('content')\
            #     .offset(start_from) \
            #     .limit(count).all()
            #
            # result = []
            # for entry in lexical_entries:
            #     result.append(entry[0].track(False))
            # response['lexical_entries'] = result

            lexical_entries = DBSession.query(LexicalEntry)\
                .filter_by(parent_client_id=parent.client_id, parent_object_id=parent.object_id)\
                .offset(start_from) \
                .limit(count).all()

            resultold = []
            for entry in lexical_entries:
                resultold.append(entry.track(False))
            response['lexical_entries'] = resultold

            request.response.status = HTTPOk.code
            return response
    else:
        request.response.status = HTTPNotFound.code
        return {'error': str("No such perspective in the system")}


@view_config(route_name='lexical_entries_all_count', renderer='json', request_method='GET', permission='view')
def lexical_entries_all_count(request):
    response = dict()
    client_id = request.matchdict.get('perspective_client_id')
    object_id = request.matchdict.get('perspective_id')

    sort_criterion = request.params.get('sort_by') or 'Translation'
    start_from = request.params.get('start_from') or 0
    count = request.params.get('count') or 200

    parent = DBSession.query(DictionaryPerspective).filter_by(client_id=client_id, object_id=object_id).first()
    if parent:
        if not parent.marked_for_deletion:
            lexical_entries_count = DBSession.query(func.count(LexicalEntry.object_id))\
                .filter_by(parent_client_id=parent.client_id, parent_object_id=parent.object_id).scalar()
            return {"count": lexical_entries_count}
    else:
        request.response.status = HTTPNotFound.code
        return {'error': str("No such perspective in the system")}


@view_config(route_name='lexical_entries_published', renderer='json', request_method='GET', permission='view')
def lexical_entries_published(request):
    response = dict()
    client_id = request.matchdict.get('perspective_client_id')
    object_id = request.matchdict.get('perspective_id')

    sort_criterion = request.params.get('sort_by') or 'Translation'
    start_from = request.params.get('start_from') or 0
    count = request.params.get('count') or 200

    parent = DBSession.query(DictionaryPerspective).filter_by(client_id=client_id, object_id=object_id).first()
    if parent:
        if not parent.marked_for_deletion:

            # lexical_entries_criterion = DBSession.query(LexicalEntry)\
            #     .filter_by(parent_client_id=parent.client_id, parent_object_id=parent.object_id)\
            #     .join(LevelOneEntity)\
            #     .filter_by(entity_type=sort_criterion)\
            #     .group_by(LexicalEntry).subquery().select()
            # lexical_entries_not_criterion = DBSession.query(LexicalEntry)\
            #     .filter_by(parent_client_id=parent.client_id, parent_object_id=parent.object_id)\
            #     .except_(lexical_entries_criterion)
            # lexical_entries_criterion2 = DBSession.query(LexicalEntry,
            #                                             func.min(LevelOneEntity.content).label("content"))\
            #     .filter_by(parent_client_id=parent.client_id, parent_object_id=parent.object_id)\
            #     .join(LevelOneEntity)\
            #     .filter_by(entity_type=sort_criterion)\
            #     .group_by(LexicalEntry)
            # lexical_entries_not_criterion = DBSession.query(LexicalEntry,
            #                                                 sqlalchemy.null().label('content'))\
            #     .correlate(lexical_entries_not_criterion)
            # lexical_entries = lexical_entries_criterion2.union(lexical_entries_not_criterion)\
            #     .filter(or_(LexicalEntry.publishleveloneentity != None,
            #                 LexicalEntry.publishleveltwoentity != None,
            #                 LexicalEntry.publishgroupingentity != None))\
            #     .filter_by(parent_client_id=parent.client_id, parent_object_id=parent.object_id)\
            #     .order_by('content')\
            #     .offset(start_from) \
            #     .limit(count).all()
            #
            # result = []
            # for entry in lexical_entries:
            #     result.append(entry[0].track(True))
            # response['lexical_entries'] = result
            lexical_entries = DBSession.query(LexicalEntry)\
                .filter_by(parent_client_id=parent.client_id, parent_object_id=parent.object_id)\
                .outerjoin(PublishGroupingEntity)\
                .outerjoin(PublishLevelOneEntity)\
                .outerjoin(PublishLevelTwoEntity)\
                .filter(or_(PublishGroupingEntity.marked_for_deletion==False,
                            PublishLevelOneEntity.marked_for_deletion==False,
                            PublishLevelTwoEntity.marked_for_deletion==False,
                            ))\
                .group_by(LexicalEntry).all()
            # lexical_entries = DBSession.query(LexicalEntry)\
            #     .filter_by(parent_client_id=parent.client_id, parent_object_id=parent.object_id)\
            #     .filter(or_(LexicalEntry.publishleveloneentity != None,
            #                 LexicalEntry.publishleveltwoentity != None,
            #                 LexicalEntry.publishgroupingentity != None))\
            #     .offset(start_from) \
            #     .limit(count).all()

            resultold = []
            for entry in lexical_entries:
                resultold.append(entry.track(True))
            response['lexical_entries'] = resultold

            request.response.status = HTTPOk.code
            return response
    else:
        request.response.status = HTTPNotFound.code
        return {'error': str("No such perspective in the system")}


@view_config(route_name='lexical_entries_published_count', renderer='json', request_method='GET', permission='view')
def lexical_entries_published_count(request):
    response = dict()
    client_id = request.matchdict.get('perspective_client_id')
    object_id = request.matchdict.get('perspective_id')

    sort_criterion = request.params.get('sort_by') or 'Translation'
    start_from = request.params.get('start_from') or 0
    count = request.params.get('count') or 200

    parent = DBSession.query(DictionaryPerspective).filter_by(client_id=client_id, object_id=object_id).first()
    if parent:
        if not parent.marked_for_deletion:
            lexical_entries_count = DBSession.query(func.count(LexicalEntry.object_id))\
                .filter_by(parent_client_id=parent.client_id, parent_object_id=parent.object_id)\
                .outerjoin(PublishGroupingEntity)\
                .outerjoin(PublishLevelOneEntity)\
                .outerjoin(PublishLevelTwoEntity)\
                .filter(or_(PublishGroupingEntity.marked_for_deletion==False,
                            PublishLevelOneEntity.marked_for_deletion==False,
                            PublishLevelTwoEntity.marked_for_deletion==False,
                            ))\
                .group_by(LexicalEntry).scalar()
            return {"count": lexical_entries_count}
    else:
        request.response.status = HTTPNotFound.code
        return {'error': str("No such perspective in the system")}


# TODO: fix ACL
@view_config(route_name='lexical_entry_in_perspective', renderer='json', request_method='GET', permission='view')#, permission='view')
@view_config(route_name='lexical_entry', renderer='json', request_method='GET', permission='view')
def view_lexical_entry(request):
    response = dict()
    client_id = request.matchdict.get('client_id')
    object_id = request.matchdict.get('object_id')

    entry = DBSession.query(LexicalEntry).filter_by(client_id=client_id, object_id=object_id).first()
    if entry:
        if not entry.marked_for_deletion:
            if entry.moved_to:
                url = request.route_url('lexical_entry',
                                        client_id=entry.moved_to.split("/")[0],
                                        object_id=entry.moved_to.split("/")[1])
                subreq = Request.blank(url)
                subreq.method = 'GET'
                subreq.headers = request.headers
                return request.invoke_subrequest(subreq)
            else:
                response['lexical_entry'] = entry.track(False)

            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such lexical entry in the system")}


@view_config(route_name='get_user_info', renderer='json', request_method='GET')
def get_user_info(request):
    response = dict()
    client_id = None
    try:
        client_id = request.params.get('client_id')
    except:
        pass
    user_id=None
    try:
        user_id = request.params.get('user_id')
    except:
        pass
    user = None
    if client_id:
        client = DBSession.query(Client).filter_by(id=client_id).first()
        if not client:

            request.response.status = HTTPNotFound.code
            return {'error': str("No such client in the system")}
        user = DBSession.query(User).filter_by(id=client.user_id).first()
        if not user:

            request.response.status = HTTPNotFound.code
            return {'error': str("No such user in the system")}
    else:
        user = DBSession.query(User).filter_by(id=user_id).first()
        if not user:

            request.response.status = HTTPNotFound.code
            return {'error': str("No such user in the system")}
    response['id']=user.id
    response['login']=user.login
    response['name']=user.name
    response['intl_name']=user.intl_name
    response['default_locale_id']=user.default_locale_id
    response['birthday']=str(user.birthday)
    response['signup_date']=str(user.signup_date)
    response['is_active']=str(user.is_active)
    email = None
    if user.email:
        for em in user.email:
            email = em.email
            break
    response['email'] = email
    about = None
    if user.about:
        for ab in user.about:
            about = ab.content
            break
    response['about'] = about
    organizations = []
    for organization in user.organizations:
        organizations += [{'organization_id':organization.id}]
    response['organizations'] = organizations
    request.response.status = HTTPOk.code
    return response


@view_config(route_name='get_user_info', renderer='json', request_method='PUT')
def edit_user_info(request):
    response = dict()
    client_id = None
    try:
        client_id = request.params.get('client_id')
    except:
        pass
    user_id=None
    try:
        user_id = request.params.get('user_id')
    except:
        pass
    user = None
    if client_id:
        client = DBSession.query(Client).filter_by(id=client_id).first()
        if not client:

            request.response.status = HTTPNotFound.code
            return {'error': str("No such client in the system")}
        user = DBSession.query(User).filter_by(id=client.user_id).first()
        if not user:

            request.response.status = HTTPNotFound.code
            return {'error': str("No such user in the system")}
    else:
        user = DBSession.query(User).filter_by(id=user_id).first()
        if not user:

            request.response.status = HTTPNotFound.code
            return {'error': str("No such user in the system")}
    req = request.json_body
    user.name=req['name']
    user.default_locale_id = req['default_locale_id']
    user.birthday = datetime.date(response['birthday'])
    if user.email:
        for em in user.email:
            em.email = req['email']
    else:
        new_email = Email(user=user, email=req['email'])
        DBSession.add(new_email)
        DBSession.flush()
    if user.about:
        for ab in user.about:
            ab.content = req['about']
    else:
        new_about = About(user=user, email=req['about'])
        DBSession.add(new_about)
        DBSession.flush()

    # response['is_active']=str(user.is_active)
    request.response.status = HTTPOk.code
    return response


@view_config(route_name='approve_all', renderer='json', request_method='PATCH', permission='create')
def approve_all(request):
    response = dict()
    client_id = request.matchdict.get('perspective_client_id')
    object_id = request.matchdict.get('perspective_id')

    parent = DBSession.query(DictionaryPerspective).filter_by(client_id=client_id, object_id=object_id).first()

    if parent:
        if not parent.marked_for_deletion:
            dictionary_client_id = parent.parent_client_id
            dictionary_object_id = parent.parent_object_id
            lexes = DBSession.query(LexicalEntry).filter_by(parent=parent).all()
            for lex in lexes:
                levones = DBSession.query(LevelOneEntity).filter_by(parent=lex).all()
                for levone in levones:
                    url = request.route_url('approve_entity',
                                            dictionary_client_id=dictionary_client_id,
                                            dictionary_object_id=dictionary_object_id,
                                            perspective_client_id=client_id,
                                            perspective_id=object_id)
                    subreq = Request.blank(url)
                    jsn = dict()
                    entities = [{'type': 'leveloneentity',
                                         'client_id': levone.client_id,
                                         'object_id': levone.object_id}]
                    jsn['entities']= entities
                    subreq.json = jsn
                    subreq.method = 'PATCH'
                    subreq.headers = request.headers
                    try:
                        request.invoke_subrequest(subreq)
                    except:
                        log.debug('JSON', jsn)
                        return{'error': str(jsn)}
                    for levtwo in levone.leveltwoentity:
                        url = request.route_url('approve_entity',
                                                dictionary_client_id=dictionary_client_id,
                                                dictionary_object_id=dictionary_object_id,
                                                perspective_client_id=client_id,
                                                perspective_id=object_id)
                        subreq = Request.blank(url)
                        jsn = dict()
                        entities = [{'type': 'leveltwoentity',
                                             'client_id':levtwo.client_id,
                                             'object_id':levtwo.object_id}]
                        jsn['entities']= entities
                        subreq.json = jsn
                        subreq.method = 'PATCH'
                        subreq.headers = request.headers
                        try:
                            request.invoke_subrequest(subreq)
                        except:
                            log.debug('JSON', jsn)
                groupents = DBSession.query(GroupingEntity).filter_by(parent=lex).all()
                for groupent in groupents:
                    url = request.route_url('approve_entity',
                                            dictionary_client_id=dictionary_client_id,
                                            dictionary_object_id=dictionary_object_id,
                                            perspective_client_id=client_id,
                                            perspective_id=object_id)
                    subreq = Request.blank(url)
                    jsn = dict()
                    entities = [{'type': 'groupingentity',
                                         'client_id': groupent.client_id,
                                         'object_id': groupent.object_id}]
                    jsn['entities'] = entities
                    subreq.json = jsn
                    subreq.method = 'PATCH'
                    subreq.headers = request.headers
                    try:
                        request.invoke_subrequest(subreq)
                    except:
                        log.debug('JSON', jsn)

            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such perspective in the system")}


@view_config(route_name='approve_entity', renderer='json', request_method='PATCH', permission='create')
def approve_entity(request):
    try:

        req = request.json_body
        variables = {'auth': request.authenticated_userid}
        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.",
                           variables['auth'])
        user = DBSession.query(User).filter_by(id=client.user_id).first()
        if not user:
            raise CommonException("This client id is orphaned. Try to logout and then login once more.")
        log.debug('REQUEST BODY:', req)
        for entry in req['entities']:
            if entry['type'] == 'leveloneentity':
                entity = DBSession.query(LevelOneEntity).\
                    filter_by(client_id=entry['client_id'], object_id=entry['object_id']).first()
                if entity:
                    if not entity.publishleveloneentity:
                        publishent = PublishLevelOneEntity(client_id=client.id, object_id=DBSession.query(PublishLevelOneEntity).filter_by(client_id=client.id).count() + 1,
                                                           entity=entity, parent=entity.parent)
                        DBSession.add(publishent)
                        DBSession.flush()
                    else:
                        for ent in entity.publishleveloneentity:
                            if ent.marked_for_deletion:
                                ent.marked_for_deletion = False
                                DBSession.flush()
            elif entry['type'] == 'leveltwoentity':
                entity = DBSession.query(LevelTwoEntity).\
                    filter_by(client_id=entry['client_id'], object_id=entry['object_id']).first()
                if entity:
                    if not entity.publishleveltwoentity:
                        publishent = PublishLevelTwoEntity(client_id=client.id, object_id=DBSession.query(PublishLevelTwoEntity).filter_by(client_id=client.id).count() + 1,
                                                           entity=entity, parent=entity.parent.parent)
                        DBSession.add(publishent)
                        DBSession.flush()
                    else:
                        for ent in entity.publishleveltwoentity:
                            if ent.marked_for_deletion:
                                ent.marked_for_deletion = False
                                DBSession.flush()
            elif entry['type'] == 'groupingentity':
                entity = DBSession.query(GroupingEntity).\
                    filter_by(client_id=entry['client_id'], object_id=entry['object_id']).first()
                if entity:
                    if not entity.publishgroupingentity:
                        publishent = PublishGroupingEntity(client_id=client.id, object_id=DBSession.query(PublishGroupingEntity).filter_by(client_id=client.id).count() + 1,
                                                           entity=entity, parent=entity.parent)
                        DBSession.add(publishent)
                        DBSession.flush()
                    else:
                        for ent in entity.publishgroupingentity:
                            if ent.marked_for_deletion:
                                ent.marked_for_deletion = False
                                DBSession.flush()
            else:
                raise CommonException("Unacceptable type")

        request.response.status = HTTPOk.code
        return {}
    except KeyError as e:
        request.response.status = HTTPBadRequest.code
        return {'error': str(e)}

    except IntegrityError as e:
        request.response.status = HTTPInternalServerError.code
        return {'error': str(e)}

    except CommonException as e:
        request.response.status = HTTPConflict.code
        return {'error': str(e)}


@view_config(route_name='approve_entity', renderer='json', request_method='DELETE', permission='delete')
def disapprove_entity(request):
    try:
        req = request.json_body
        variables = {'auth': request.authenticated_userid}
        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.",
                           variables['auth'])
        user = DBSession.query(User).filter_by(id=client.user_id).first()
        if not user:
            raise CommonException("This client id is orphaned. Try to logout and then login once more.")
        for entry in req['entities']:
            if entry['type'] == 'leveloneentity':
                entity = DBSession.query(LevelOneEntity).\
                    filter_by(client_id=entry['client_id'], object_id=entry['object_id']).first()
                if entity:
                    for ent in entity.publishleveloneentity:
                        ent.marked_for_deletion = True
                        DBSession.flush()
                else:
                    log.debug("WARNING: NO ENTITY")
            elif entry['type'] == 'leveltwoentity':
                entity = DBSession.query(LevelTwoEntity).\
                    filter_by(client_id=entry['client_id'], object_id=entry['object_id']).first()
                if entity:
                    for ent in entity.publishleveltwoentity:
                        ent.marked_for_deletion = True
                        DBSession.flush()
            elif entry['type'] == 'groupingentity':
                entity = DBSession.query(GroupingEntity).\
                    filter_by(client_id=entry['client_id'], object_id=entry['object_id']).first()
                if entity:
                    for ent in entity.publishgroupingentity:
                        ent.marked_for_deletion = True
                        DBSession.flush()
            else:
                raise CommonException("Unacceptable type")

        request.response.status = HTTPOk.code
        return {}
    except KeyError as e:
        request.response.status = HTTPBadRequest.code
        return {'error': str(e)}

    except IntegrityError as e:
        request.response.status = HTTPInternalServerError.code
        return {'error': str(e)}

    except CommonException as e:
        request.response.status = HTTPConflict.code
        return {'error': str(e)}


@view_config(route_name='get_translations', renderer='json', request_method='GET')
def get_translations(request):
    req = request.json_body
    response = []
    for entry in req:
        response += [find_by_translation_string(find_locale_id(request), entry)]
    request.response.status = HTTPOk.code
    return response


@view_config(route_name='merge_dictionaries', renderer='json', request_method='POST')  # TODO: check for permission
def merge_dictionaries(request):
    try:
        req = request.json_body
        variables = {'auth': request.authenticated_userid}
        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.",
                           variables['auth'])
        user = DBSession.query(User).filter_by(id=client.user_id).first()
        if not user:
            raise CommonException("This client id is orphaned. Try to logout and then login once more.")
        parent_object_id = req['language_object_id']
        parent_client_id = req['language_client_id']
        name = req['name']
        translation = req['translation']

        dictionaries = req['dictionaries']
        if len(dictionaries) != 2:
            raise KeyError("Wrong number of dictionaries to merge.",
                           len(dictionaries))
        new_dicts = []
        for dicti in dictionaries:
            diction = DBSession.query(Dictionary).filter_by(client_id=dicti['client_id'], object_id=dicti['object_id']).first()
            if not diction:
                raise KeyError("Dictionary do not exist in the system")
            if parent_client_id != diction.parent_client_id or parent_object_id != diction.parent_object_id:
                raise KeyError("Both dictionaries should have same language.")
            new_dicts += [diction]
        dictionaries = new_dicts
        base = DBSession.query(BaseGroup).filter_by(subject='merge', action='create').first()
        override = DBSession.query(Group).filter_by(base_group_id=base.id, subject_override = True).first()
        if user not in override.users:
            grps = []
            for dict in dictionaries:
                gr = DBSession.query(Group).filter_by(base_group_id=base.id,
                                                      subject_client_id=dict.client_id,
                                                      subject_object_id=dict.object_id).first()
                grps += [gr]
        for gr in grps:
            if user not in gr.users:
                raise KeyError("Not enough permission to do that")

        subreq = Request.blank('/dictionary')
        subreq.method = 'POST'
        subreq.json = {'parent_object_id': parent_object_id, 'parent_client_id': parent_client_id,
                            'name': name, 'translation': translation}
        subreq.headers = request.headers
        response = request.invoke_subrequest(subreq)
        client_id = response.json['client_id']
        object_id = response.json['object_id']
        new_dict = DBSession.query(Dictionary).filter_by(client_id=client_id, object_id=object_id).first()
        perspectives = []
        for dicti in dictionaries:
            for entry in dicti.dictionaryperspective:
                perspectives += [entry]
            for entry in perspectives:
                if entry in dicti.dictionaryperspective:
                    dicti.dictionaryperspective.remove(entry)
                new_dict.dictionaryperspective.append(entry)
            cli_id = dicti.client_id
            obj_id = dicti.object_id
            bases = DBSession.query(BaseGroup).filter_by(dictionary_default=True)
            groups = []
            for base in bases:

                group = DBSession.query(Group).filter_by(base_group_id=base.id,
                                                         subject_object_id=obj_id,
                                                         subject_client_id=cli_id).first()
                groups += [group]

            for group in groups:
                existing = DBSession.query(Group).join(BaseGroup).filter(BaseGroup.dictionary_default==True,
                                                         Group.subject_object_id==obj_id,
                                                         Group.subject_client_id==cli_id).first()
                if existing:
                    users = []
                    for user in group.users:
                        users += [user]
                    for user in users:
                        if user in group.users:
                            group.users.remove(user)
                        if not user in existing.users:
                            existing.users.append(user)
                else:
                    new_group = Group(base_group_id=group.base_group_id,
                                      subject_object_id=cli_id,
                                      subject_client_id=obj_id)
                    DBSession.add(new_group)
                    users = []
                    for user in group.users:
                        users += [user]
                    for user in users:
                        group.remove(user)
                        new_group.users.append(user)
                group.marked_for_deletion = True
            dicti.marked_for_deletion = True
        request.response.status = HTTPOk.code
        return {'object_id': object_id,
                'client_id': client_id}
    except KeyError as e:
        request.response.status = HTTPBadRequest.code
        return {'error': str(e)}

    except IntegrityError as e:
        request.response.status = HTTPInternalServerError.code
        return {'error': str(e)}

    except CommonException as e:
        request.response.status = HTTPConflict.code
        return {'error': str(e)}


@view_config(route_name='merge_perspectives', renderer='json', request_method='POST')  # TODO: check for permission
def merge_perspectives(request):
    try:
        req = request.json_body
        variables = {'auth': request.authenticated_userid}
        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.",
                           variables['auth'])
        user = DBSession.query(User).filter_by(id=client.user_id).first()
        if not user:
            raise CommonException("This client id is orphaned. Try to logout and then login once more.")
        dictionary_client_id = req['dictionary_client_id']
        dictionary_object_id = req['dictionary_object_id']
        name = req['name']
        translation = req['translation']

        perspectives = req['perspectives']
        if len(perspectives) != 2:
            raise KeyError("Wrong number of perspectives to merge.",
                           len(perspectives))
        new_persps = []
        for persp in perspectives:
            perspe = DBSession.query(DictionaryPerspective).filter_by(client_id=persp['client_id'],
                                                                       object_id=persp['object_id']).first()
            if not perspe:
                raise KeyError("Perspective do not exist in the system")
            if dictionary_client_id != perspe.parent_client_id or dictionary_object_id != perspe.parent_object_id:
                raise KeyError("Both perspective should from same dictionary.")
            new_persps += [perspe]
        perspectives = new_persps
        base = DBSession.query(BaseGroup).filter_by(subject='merge', action='create').first()
        override = DBSession.query(Group).filter_by(base_group_id=base.id, subject_override = True).first()
        if user not in override.users:
            group = DBSession.query(Group).filter_by(base_group_id=base.id,
                                                  subject_client_id=dictionary_client_id,
                                                  subject_object_id=dictionary_object_id).first()
            if user not in group.users:
                raise KeyError("Not enough permission to do that")

        subreq = Request.blank('/dictionary/%s/%s/perspective' % (dictionary_client_id, dictionary_object_id))
        subreq.method = 'POST'
        subreq.json = {'name': name, 'translation': translation}
        subreq.headers = request.headers
        response = request.invoke_subrequest(subreq)
        client_id = response.json['client_id']
        object_id = response.json['object_id']
        new_persp = DBSession.query(DictionaryPerspective).filter_by(client_id=client_id, object_id=object_id).first()
        perspectives = []
        # TODO: make fields for new perspective
        for persp in perspectives:
            # TODO: make movement of lexical entries in new perspective.
            #  Must be complex, because fields of perspective should change
            obj_id = persp.object_id
            cli_id = persp.client_id
            bases = DBSession.query(BaseGroup).filter_by(dictionary_default=True)
            groups = []
            for base in bases:

                group = DBSession.query(Group).filter_by(base_group_id=base.id,
                                                         subject_object_id=obj_id,
                                                         subject_client_id=cli_id).first()
                groups += [group]

            for group in groups:
                existing = DBSession.query(Group).join(BaseGroup).filter(BaseGroup.perspective_default==True,
                                                         Group.subject_object_id==obj_id,
                                                         Group.subject_client_id==cli_id).first()
                if existing:
                    users = []
                    for user in group.users:
                        users += [user]
                    for user in users:
                        if user in group.users:
                            group.users.remove(user)
                        if not user in existing.users:
                            existing.users.append(user)
                else:
                    new_group = Group(base_group_id=group.base_group_id,
                                      subject_object_id=cli_id,
                                      subject_client_id=obj_id)
                    DBSession.add(new_group)
                    users = []
                    for user in group.users:
                        users += [user]
                    for user in users:
                        group.remove(user)
                        new_group.users.append(user)
                group.marked_for_deletion = True
            persp.marked_for_deletion = True
        request.response.status = HTTPOk.code
        return {'object_id': object_id,
                'client_id': client_id}
    except KeyError as e:
        request.response.status = HTTPBadRequest.code
        return {'error': str(e)}

    except IntegrityError as e:
        request.response.status = HTTPInternalServerError.code
        return {'error': str(e)}

    except CommonException as e:
        request.response.status = HTTPConflict.code
        return {'error': str(e)}


@view_config(route_name='move_lexical_entry', renderer='json', request_method='PATCH')  # TODO: check for permission
def move_lexical_entry(request):
    req = request.json_body
    # variables = {'auth': request.authenticated_userid}
    object_id = req['object_id']
    client_id = req['client_id']
    cli_id = req['client_id']
    obj_id = req['object_id']
    entry = DBSession.query(LexicalEntry).filter_by(client_id=client_id, object_id=object_id).first()
    parent = DBSession.query(LexicalEntry).filter_by(client_id=cli_id, object_id=obj_id).first()
    if entry and parent:
        if not entry.marked_for_deletion and parent.marked_for_deletion:
            if entry.moved_to is None and parent.moved_to is None:
                for entity in entry.leveloneentity:
                    entity.parent = parent
                    for publent in entity.publishleveloneentity:
                        publent.marked_for_deletion = True
                        publent.parent = parent
                    DBSession.flush()

                    for ent in entity.leveltwoentity:
                        ent.parent = parent
                        for publent in ent.publishleveltwoentity:
                            publent.marked_for_deletion = True
                            publent.parent = parent
                        DBSession.flush()
                for entity in entry.groupingentity:
                    entity.parent = parent
                    for publent in entity.publishgroupingentity:
                        publent.marked_for_deletion = True
                        publent.parent = parent
                    DBSession.flush()
                entry.moved_to = str(cli_id) + '/' + str(obj_id)
    request.response.status = HTTPNotFound.code
    return {'error': str("No such lexical entry in the system")}


@view_config(route_name='organization', renderer='json', request_method='GET')
def view_organization(request):
    response = dict()
    organization_id = request.matchdict.get('organization_id')
    organization = DBSession.query(Organization).filter_by(id=organization_id).first()
    if organization:
        if not organization.marked_for_deletion:
            response['name'] = organization.name
            response['about'] = organization.parent_object_id
            users = []
            for user in organization.users:
                users += [user.id]
            response['users'] = users
            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such organization in the system")}


@view_config(route_name='organization', renderer='json', request_method='PUT', permission='edit')
def edit_organization(request):
    try:
        response = dict()
        organization_id = request.matchdict.get('organization_id')
        organization = DBSession.query(Organization).filter_by(id=organization_id).first()
        if organization:
            if not organization.marked_for_deletion:
                req = request.json_body
                if 'add_users' in req:
                    for user in req['add_users']:
                        if user not in organization.users:
                            organization.user.append(user)
                            bases = DBSession.query(BaseGroup).filter_by(subject='organization')
                            for base in bases:
                                group = DBSession.query(Group).filter_by(base_group_id=base.id,
                                                                         subject_object_id=organization.id).first()
                                group.users.append(user)
                if 'delete_users' in req:
                    for user in req['delete_users']:
                        if user not in organization.users:
                            organization.user.remove(user)
                            bases = DBSession.query(BaseGroup).filter_by(subject='organization')
                            for base in bases:
                                group = DBSession.query(Group).filter_by(base_group_id=base.id,
                                                                         subject_object_id=organization.id).first()
                                group.users.remove(user)
                if 'name' in req:
                    organization.name = req['name']
                if 'about' in req:
                    organization.name = req['about']
                request.response.status = HTTPOk.code
                return response

        request.response.status = HTTPNotFound.code
        return {'error': str("No such organization in the system")}
    except KeyError as e:
        request.response.status = HTTPBadRequest.code
        return {'error': str(e)}


@view_config(route_name='organization', renderer='json', request_method='DELETE', permission='delete')
def delete_organization(request):
    response = dict()
    organization_id = request.matchdict.get('organization_id')
    organization = DBSession.query(Organization).filter_by(id=organization_id).first()
    if organization:
        if not organization.marked_for_deletion:
            organization.marked_for_deletion = True
            request.response.status = HTTPOk.code
            return response
    request.response.status = HTTPNotFound.code
    return {'error': str("No such organization in the system")}


@view_config(route_name='create_organization', renderer='json', request_method='POST', permission='create')
def create_organization(request):
    try:

        variables = {'auth': request.authenticated_userid}
        req = request.json_body
        name = req['name']
        about = req['about']
        client = DBSession.query(Client).filter_by(id=variables['auth']).first()
        if not client:
            raise KeyError("Invalid client id (not registered on server). Try to logout and then login.")
        user = DBSession.query(User).filter_by(id=client.user_id).first()
        if not user:
            raise CommonException("This client id is orphaned. Try to logout and then login once more.")

        organization = Organization(name=name,
                                    about=about)
        DBSession.add(organization)
        DBSession.flush()
        bases = DBSession.query(BaseGroup).filter_by(subject='organization')
        for base in bases:
            group = Group(parent=base, subject_object_id=organization.id)
            group.users.append(user)
        request.response.status = HTTPOk.code
        return {'organization_id': organization.id}
    except KeyError as e:
        request.response.status = HTTPBadRequest.code
        return {'error': str(e)}

    except IntegrityError as e:
        request.response.status = HTTPInternalServerError.code
        return {'error': str(e)}

    except CommonException as e:
        request.response.status = HTTPConflict.code
        return {'error': str(e)}

conn_err_msg = """\
Pyramid is having a problem using your SQL database.  The problem
might be caused by one of the following things:

1.  You may need to run the "initialize_lingvodoc_db" script
    to initialize your database tables.  Check your virtual
    environment's "bin" directory for this script and try to run it.

2.  Your database server may not be running.  Check that the
    database server referred to by the "sqlalchemy.url" setting in
    your "development.ini" file is running.

After you fix the problem, please restart the Pyramid application to
try it again.
"""


@view_config(route_name='testing', renderer='json')
def testing(request):
    try:
        req = request.json_body
        return req
    except:
        return request



@view_config(route_name='login', renderer='templates/login.pt', request_method='GET')
def login_get(request):
    variables = {'auth': authenticated_userid(request)}
    return render_to_response('templates/login.pt', variables, request=request)


@view_config(route_name='login', request_method='POST')
def login_post(request):
    next = request.params.get('next') or request.route_url('home')
    login = request.POST.get('login', '')
    password = request.POST.get('password', '')
    print(login)
    user = DBSession.query(User).filter_by(login=login).first()
    if user and user.check_password(password):
        client = Client(user_id=user.id)
        user.clients.append(client)
        DBSession.add(client)
        DBSession.flush()
        headers = remember(request, principal=client.id)
        response = Response()
        response.headers = headers
        locale_id = user.default_locale_id
        if not locale_id:
            locale_id = 1
        response.set_cookie(key='locale_id', value=str(locale_id))
        headers = remember(request, principal=client.id)
        return HTTPFound(location=next, headers=response.headers)
    return HTTPUnauthorized(location=request.route_url('login'))

@view_config(route_name='cheatlogin', request_method='POST')
def login_cheat(request):
    next = request.params.get('next') or request.route_url('dashboard')
    login = request.POST.get('login', '')
    passwordhash = request.POST.get('passwordhash', '')
    print(login)
    user = DBSession.query(User).filter_by(login=login).first()
    if user and user.password.hash == passwordhash:
        client = Client(user_id=user.id)
        user.clients.append(client)
        DBSession.add(client)
        DBSession.flush()
        headers = remember(request, principal=client.id)
        response = Response()
        response.headers = headers
        locale_id = user.default_locale_id
        if not locale_id:
            locale_id = 1
        response.set_cookie(key='locale_id', value=str(locale_id))
        headers = remember(request, principal=client.id)
        return HTTPFound(location=next, headers=response.headers)
    return HTTPUnauthorized(location=request.route_url('login'))

@view_config(route_name='logout', renderer='json')
def logout_any(request):
    next = request.params.get('next') or request.route_url('login')
    headers = forget(request)
    return HTTPFound(location=next, headers=headers)


@view_config(route_name='signup', renderer='templates/signup.pt', request_method='GET')
def signup_get(request):
    variables = {'auth': authenticated_userid(request)}
    return render_to_response('templates/signup.pt', variables, request=request)

@view_config(route_name='signup', renderer='json', request_method='POST')
def signup_post(request):
    try:
        login = request.POST.getone('login')
        name = request.POST.getone('name')
        email = request.POST.getone('email')
        password = request.POST.getone('password')

        day = request.POST.get('day', "1")
        month = request.POST.get('month', "1")
        year = request.POST.get('year', "1970")
        birthday = datetime.datetime.strptime(day + month + year, "%d%m%Y").date()

        if DBSession.query(User).filter_by(login=login).first():
            raise CommonException("The user with this login is already registered")
        if DBSession.query(Email).filter_by(email=email).first():
            raise CommonException("The user with this email is already registered")
        new_user = User(login=login, name=name, signup_date=datetime.datetime.utcnow(), intl_name=login, birthday=birthday, is_active=True)
        pwd = Passhash(password=password)
        email = Email(email=email)
        new_user.password = pwd
        new_user.email.append(email)
        DBSession.add(new_user)
        basegroups = []
        basegroups += [DBSession.query(BaseGroup).filter_by(translation_string="Can create dictionaries").first()]
        basegroups += [DBSession.query(BaseGroup).filter_by(translation_string="Can create languages").first()]
        basegroups += [DBSession.query(BaseGroup).filter_by(translation_string="Can create organizations").first()]
        basegroups += [DBSession.query(BaseGroup).filter_by(translation_string="Can create translation strings").first()]
        groups = []
        for base in basegroups:
            groups += [DBSession.query(Group).filter_by(subject_override=True, base_group_id=base.id).first()]
        for group in groups:
            new_user.groups.append(group)
        DBSession.flush()
        return login_post(request)

    except KeyError as e:
        request.response.status = HTTPBadRequest.code
        return {'status': request.response.status, 'error': str(e)}

    except CommonException as e:
        request.response.status = HTTPConflict.code
        return {'status': request.response.status, 'error': str(e)}

    except ValueError as e:
        request.response.status = HTTPConflict.code
        return {'status': request.response.status, 'error': str(e)}


def get_user_by_client_id(client_id):
    user = None
    client = DBSession.query(Client).filter_by(id=client_id).first()
    if client is not None:
        user = DBSession.query(User).filter_by(id=client.user_id).first()
    return user

@view_config(route_name='home', renderer='templates/home.pt', request_method='GET')
def home_get(request):
    client_id = authenticated_userid(request)
    user = get_user_by_client_id(client_id)
    variables = {'client_id': client_id, 'user': user}
    return render_to_response('templates/home.pt', variables, request=request)

@view_config(route_name='dashboard', renderer='templates/dashboard.pt', request_method='GET')
def dashboard_get(request):
    client_id = authenticated_userid(request)
    user = get_user_by_client_id(client_id)
    if user is None:
        response = Response()
        return HTTPFound(location=request.route_url('login'), headers=response.headers)
    variables = {'client_id': client_id, 'user': user}
    return render_to_response('templates/dashboard.pt', variables, request=request)

@view_config(route_name='languages', renderer='templates/languages.pt', request_method='GET')
def languages_get(request):
    client_id = authenticated_userid(request)
    user = get_user_by_client_id(client_id)
    if user is None:
        response = Response()
        return HTTPFound(location=request.route_url('login'), headers=response.headers)
    variables = {'client_id': client_id, 'user': user}
    return render_to_response('templates/languages.pt', variables, request=request)

@view_config(route_name='new_dictionary', renderer='templates/create_dictionary.pt', request_method='GET')
def new_dictionary_get(request):
    client_id = authenticated_userid(request)
    user = get_user_by_client_id(client_id)
    if user is None:
        response = Response()
        return HTTPFound(location=request.route_url('login'), headers=response.headers)
    variables = {'client_id': client_id, 'user': user}
    return render_to_response('templates/create_dictionary.pt', variables, request=request)

@view_config(route_name='edit_dictionary', renderer='templates/edit_dictionary.pt', request_method='GET')
def edit_dictionary_get(request):
    client_id = authenticated_userid(request)
    user = get_user_by_client_id(client_id)
    if user is None:
        response = Response()
        return HTTPFound(location=request.route_url('login'), headers=response.headers)

    dictionary_client_id = request.matchdict.get('dictionary_client_id')
    dictionary_object_id = request.matchdict.get('dictionary_object_id')
    perspective_client_id = request.matchdict.get('perspective_client_id')
    perspective_id = request.matchdict.get('perspective_id')

    variables = {'client_id': client_id, 'user': user, 'dictionary_client_id': dictionary_client_id,
                 'dictionary_object_id': dictionary_object_id, 'perspective_client_id': perspective_client_id,
                 'perspective_id': perspective_id}

    return render_to_response('templates/edit_dictionary.pt', variables, request=request)

@view_config(route_name='view_dictionary', renderer='templates/view_dictionary.pt', request_method='GET')
def view_dictionary_get(request):
    client_id = authenticated_userid(request)
    user = get_user_by_client_id(client_id)
    if user is None:
        response = Response()
        return HTTPFound(location=request.route_url('login'), headers=response.headers)

    dictionary_client_id = request.matchdict.get('dictionary_client_id')
    dictionary_object_id = request.matchdict.get('dictionary_object_id')
    perspective_client_id = request.matchdict.get('perspective_client_id')
    perspective_id = request.matchdict.get('perspective_id')

    variables = {'client_id': client_id, 'user': user, 'dictionary_client_id': dictionary_client_id,
                 'dictionary_object_id': dictionary_object_id, 'perspective_client_id': perspective_client_id,
                 'perspective_id': perspective_id}

    return render_to_response('templates/view_dictionary.pt', variables, request=request)


@view_config(route_name='publish_dictionary', renderer='templates/publish_dictionary.pt', request_method='GET')
def publish_dictionary_get(request):
    client_id = authenticated_userid(request)
    user = get_user_by_client_id(client_id)
    if user is None:
        response = Response()
        return HTTPFound(location=request.route_url('login'), headers=response.headers)

    dictionary_client_id = request.matchdict.get('dictionary_client_id')
    dictionary_object_id = request.matchdict.get('dictionary_object_id')
    perspective_client_id = request.matchdict.get('perspective_client_id')
    perspective_id = request.matchdict.get('perspective_id')

    variables = {'client_id': client_id, 'user': user, 'dictionary_client_id': dictionary_client_id,
                 'dictionary_object_id': dictionary_object_id, 'perspective_client_id': perspective_client_id,
                 'perspective_id': perspective_id}

    return render_to_response('templates/publish_dictionary.pt', variables, request=request)


@view_config(route_name='blob_upload', renderer='templates/user_upload.pt', request_method='GET')
def blob_upload_get(request):
    client_id = authenticated_userid(request)
    user = get_user_by_client_id(client_id)
    if user is None:
        response = Response()
        return HTTPFound(location=request.route_url('login'), headers=response.headers)

    dictionary_client_id = request.matchdict.get('dictionary_client_id')
    dictionary_object_id = request.matchdict.get('dictionary_object_id')
    perspective_client_id = request.matchdict.get('perspective_client_id')
    perspective_id = request.matchdict.get('perspective_id')

    variables = {'client_id': client_id, 'user': user, 'dictionary_client_id': dictionary_client_id,
                 'dictionary_object_id': dictionary_object_id, 'perspective_client_id': perspective_client_id,
                 'perspective_id': perspective_id}

    return render_to_response('templates/user_upload.pt', variables, request=request)


@view_config(route_name='merge_suggestions', renderer='json', request_method='POST')
def merge_suggestions(request):
    subreq = Request.blank('/dictionary/' + request.matchdict.get('dictionary_client_id_1') + '/' + 
    request.matchdict.get('dictionary_object_id_1') + '/perspective/' +
    request.matchdict.get('perspective_client_id_1') + '/' +
    request.matchdict.get('perspective_object_id_1') + '/all')
    subreq.method = 'GET'
    response_1 = request.invoke_subrequest(subreq).json
    subreq = Request.blank('/dictionary/' + request.matchdict.get('dictionary_client_id_2') + '/' + 
    request.matchdict.get('dictionary_object_id_2') + '/perspective/' +
    request.matchdict.get('perspective_client_id_2') + '/' +
    request.matchdict.get('perspective_object_id_2') + '/all')
    subreq.method = 'GET'
    response_2 = request.invoke_subrequest(subreq).json
    #entity_type_primary = 'Word'
    #entity_type_secondary = 'Transcription'
    #threshold = 0.2
    #levenstein = 2
    entity_type_primary = request.matchdict.get('entity_type_primary')
    entity_type_secondary = request.matchdict.get('entity_type_secondary')
    threshold = request.matchdict.get('threshold')
    levenstein = request.matchdict.get('levenstein')
    def parse_response(elem):
        words = filter(lambda x: x['entity_type'] == entity_type_primary and not x['marked_for_deletion'], elem['contains'])
        words = map(lambda x: x['content'], words)
        trans = filter(lambda x: x['entity_type'] == entity_type_secondary and not x['marked_for_deletion'], elem['contains'])
        trans = map(lambda x: x['content'], trans)
        tuples_res = [(i_word, i_trans, (elem['client_id'], elem['object_id'])) for i_word in words for i_trans in trans]
        return tuples_res
    tuples_1 = [parse_response(i) for i in response_1['lexical_entries']]
    tuples_1 = [item for sublist in tuples_1 for item in sublist]
    tuples_2 = [parse_response(i) for i in response_2['lexical_entries']]
    tuples_2 = [item for sublist in tuples_2 for item in sublist]
    def get_dict(elem):
        return {'suggestion': [
            {'lexical_entry_client_id': elem[0][0], 'lexical_entry_object_id': elem[0][1]},
            {'lexical_entry_client_id': elem[1][0], 'lexical_entry_object_id': elem[1][1]}
        ], 'confidence': elem[2]}
    results = [get_dict(i) for i in mergeDicts(tuples_1, tuples_2, float(threshold), int(levenstein))]
    return json.dumps(results)

@view_config(route_name='profile', renderer='templates/profile.pt', request_method='GET')
def profile_get(request):
    client_id = authenticated_userid(request)
    user = get_user_by_client_id(client_id)
    if user is None:
        response = Response()
        return HTTPFound(location=request.route_url('login'), headers=response.headers)
    variables = {'client_id': client_id, 'user': user }
    return render_to_response('templates/profile.pt', variables, request=request)

@view_config(route_name='organizations', renderer='templates/organizations.pt', request_method='GET')
def organizations_get(request):
    client_id = authenticated_userid(request)
    user = get_user_by_client_id(client_id)
    if user is None:
        response = Response()
        return HTTPFound(location=request.route_url('login'), headers=response.headers)
    variables = {'client_id': client_id, 'user': user }
    return render_to_response('templates/organizations.pt', variables, request=request)