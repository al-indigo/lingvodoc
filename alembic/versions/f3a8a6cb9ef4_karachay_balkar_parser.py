"""karachay_balkar_parser

Revision ID: f3a8a6cb9ef4
Revises: 882f3864b825
Create Date: 2025-01-13 17:07:07.383095

"""

# revision identifiers, used by Alembic.
revision = 'f3a8a6cb9ef4'
down_revision = '882f3864b825'
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa


def upgrade():
    op.execute('''
    INSERT INTO public.parser(additional_metadata, created_at, object_id, client_id, name, parameters, method)
    VALUES(null, '2025-01-13 14:14:14.579664', 17, 1, 'Парсер карачаево-балкарского языка Apertium', '[]',
           'apertium_krc');
    ''')

def downgrade():
    op.execute('''
    DELETE FROM parser WHERE method = 'apertium_krc';
    ''')
