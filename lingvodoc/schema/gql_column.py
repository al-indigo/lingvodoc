import graphene
from lingvodoc.schema.gql_holders import (
    LingvodocObjectType,
    CompositeIdHolder,
    CreatedAt,
    Relationship,
    SelfHolder,
    FieldHolder,
    ParentLink,
    MarkedForDeletion,
    Position,
    client_id_check,
    del_object,
    ResponseError,
    LingvodocID,
    fetch_object,
)
from lingvodoc.schema.gql_field import Field
from lingvodoc.models import (
    DBSession,
    DictionaryPerspectiveToField as dbDictionaryPerspectiveToField,
    Field as dbField,
)

from lingvodoc.utils.creation import create_dictionary_persp_to_field

class Column(LingvodocObjectType):
    """
     #created_at          | timestamp without time zone | NOT NULL
     #object_id           | bigint                      | NOT NULL
     #client_id           | bigint                      | NOT NULL
     #parent_object_id    | bigint                      |
     #parent_client_id    | bigint                      |
     #self_client_id      | bigint                      |
     #self_object_id      | bigint                      |
     #field_client_id     | bigint                      | NOT NULL
     #field_object_id     | bigint                      | NOT NULL
     #link_client_id      | bigint                      |
     #link_object_id      | bigint                      |
     #marked_for_deletion | boolean                     | NOT NULL
     #position            | integer                     | NOT NULL
    """
    dbType = dbDictionaryPerspectiveToField

    field = graphene.Field(Field)

    class Meta:
        interfaces = (CreatedAt,
                      CompositeIdHolder,
                      Relationship,
                      SelfHolder,
                      FieldHolder,
                      ParentLink,
                      MarkedForDeletion,
                      Position)

    @fetch_object('field_id')
    def resolve_field(self, info):

        field = (

            DBSession

                .query(dbField)

                .filter_by(
                    client_id = self.dbObject.field_client_id,
                    object_id = self.dbObject.field_object_id)

                .first())

        return Field(id = [field.client_id, field.object_id])


class CreateColumn(graphene.Mutation):
    """
    example:
    mutation  {
        create_column(parent_id: [1204,19664], field_id: [66, 6],
  position: 1) {
            triumph
            column{
                id
                position
            }
        }
    }

    (this example works)
    returns:

    {
      "create_column": {
        "triumph": true,
        "column": {
          "id": [
            949,
            2493
          ],
          "position": 1
        }
      }
    }
    """

    class Arguments:
        id = LingvodocID()
        parent_id = LingvodocID(required=True)
        field_id = LingvodocID(required=True)
        self_id = LingvodocID()
        link_id = LingvodocID()
        position = graphene.Int(required=True)

    column = graphene.Field(Column)
    triumph = graphene.Boolean()

    @staticmethod
    @client_id_check()
    def mutate(root, info, **args):
        id = args.get("id")
        client_id = id[0] if id else info.context["client_id"]
        object_id = id[1] if id else None
        id = [client_id, object_id]
        parent_id = args.get('parent_id')
        info.context.acl_check('edit', 'perspective', parent_id)
        field_id = args.get('field_id')
        self_id = args.get('self_id')
        link_id = args.get('link_id')
        position = args.get('position')
        field_object = create_dictionary_persp_to_field(id=id,
                                              parent_id=parent_id,
                                              field_id=field_id,
                                              self_id=self_id,
                                              link_id=link_id,
                                              position=position)
        DBSession.add(field_object)
        DBSession.flush()
        column = Column(id=[field_object.client_id, field_object.object_id])
        column.dbObject = field_object
        return CreateColumn(column=column, triumph=True)


class UpdateColumn(graphene.Mutation):
    """
    example:
      mutation  {
        update_column(id: [949, 2493], position: 5) {
            triumph
            perspective_to_field{
                id
                position
            }
        }
    }

    (this example works)
    returns:

    {
      "update_column": {
        "triumph": true,
        "column": {
          "id": [
            949,
            2493
          ],
          "position": 5
        }
      }
    }
    """

    class Arguments:
        id = LingvodocID(required=True)
        parent_id = LingvodocID()
        field_id = LingvodocID()
        self_id = LingvodocID()
        link_id = LingvodocID()
        position = graphene.Int()

    column = graphene.Field(Column)
    triumph = graphene.Boolean()

    @staticmethod
    def mutate(root, info, **args):
        id = args.get("id")
        client_id, object_id = id
        field_object = DBSession.query(dbDictionaryPerspectiveToField).filter_by(client_id=client_id,
                                                                                 object_id=object_id).first()
        if not field_object or field_object.marked_for_deletion:
            raise ResponseError(message="Error: No such field object in the system")

        info.context.acl_check('edit', 'perspective',
                                   (field_object.parent_client_id, field_object.parent_object_id))
        field_id = args.get('field_id')
        self_id = args.get('self_id')
        link_id = args.get('link_id')
        position = args.get('position')
        if field_id:
            field_object.field_client_id, field_object.field_object_id = field_id

        # Attaching or de-attaching as a nested field.

        if self_id:

            field_object.self_client_id, field_object.self_object_id = (
                self_id if self_id[0] > 0 else (None, None))

        if link_id:
            field_object.link_client_id, field_object.link_object_id = link_id
        if position:
            field_object.position = position
        column = Column(id=[field_object.client_id, field_object.object_id])
        column.dbObject = field_object
        return UpdateColumn(column=column, triumph=True)



class DeleteColumn(graphene.Mutation):
    """
    example:
      mutation  {
       delete_column(id: [949, 2493]) {
            triumph
            column{
                id
            }
        }
    }

    (this example works)
    returns:

    {
      "delete_column": {
        "triumph": true,
        "column": {
          "id": [
            949,
            2493
          ]
        }
      }
    }
    """
    class Arguments:
        id = LingvodocID(required=True)

    column = graphene.Field(Column)
    triumph = graphene.Boolean()

    @staticmethod
    def mutate(root, info, **args):
        id = args.get('id')
        client_id, object_id = id
        column_object = DBSession.query(dbDictionaryPerspectiveToField).filter_by(client_id=client_id,
                                                                                 object_id=object_id).first()
        perspective_ids = (column_object.parent_client_id, column_object.parent_object_id)
        info.context.acl_check('edit', 'perspective', perspective_ids)
        if not column_object or column_object.marked_for_deletion:
            raise ResponseError(message="No such column object in the system")
        del_object(column_object, "delete_column", info.context.get('client_id'))
        column = Column(id=id)
        column.dbObject = column_object
        return DeleteColumn(column=column, triumph=True)

