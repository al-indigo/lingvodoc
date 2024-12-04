"""markup_group_table

Revision ID: 882f3864b825
Revises: 50847f29994c
Create Date: 2024-12-04 18:14:28.470514

"""

# revision identifiers, used by Alembic.
revision = '882f3864b825'
down_revision = '50847f29994c'
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa
from lingvodoc.models import SLBigInteger


def upgrade():
    op.create_table('markupgroup',
                    sa.Column('object_id', SLBigInteger(), nullable=False),
                    sa.Column('client_id', SLBigInteger(), nullable=False),
                    sa.Column('perspective_object_id', SLBigInteger(), nullable=False),
                    sa.Column('perspective_client_id', SLBigInteger(), nullable=False),
                    sa.Column('type', sa.UnicodeText(), nullable=False),
                    sa.Column('marked_for_deletion', sa.Boolean(), nullable=False),
                    sa.Column('created_at', sa.TIMESTAMP(), nullable=False),
                    sa.PrimaryKeyConstraint('object_id', 'client_id'))

def downgrade():
    op.drop_table('markupgroup')
