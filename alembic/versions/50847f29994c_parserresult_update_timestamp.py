"""parserresult update timestamp

Revision ID: 50847f29994c
Revises: 83fac9948381
Create Date: 2024-11-19 16:43:55.507274

"""

# revision identifiers, used by Alembic.
revision = '50847f29994c'
down_revision = '83fac9948381'
branch_labels = None
depends_on = None

from alembic import op
import sqlalchemy as sa

def upgrade():
    op.execute('''
        -- Adding 'updated_at' column
        ALTER TABLE parserresult 
          ADD COLUMN updated_at TIMESTAMP NOT NULL DEFAULT TIMEZONE('UTC', NOW());
    
        -- Function which will update 'updated_at' column
        CREATE OR REPLACE FUNCTION trigger_set_timestamp()
        RETURNS TRIGGER AS $$
        BEGIN
          NEW.updated_at = TIMEZONE('UTC', NOW());
          RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    
        -- Trigger which will call the function before an update is performed on 'updated_at'
        CREATE TRIGGER set_timestamp
          BEFORE UPDATE ON parserresult
          FOR EACH ROW
          EXECUTE PROCEDURE trigger_set_timestamp();
    ''')

def downgrade():
    op.execute('''
        -- Removing trigger and function
        DROP TRIGGER set_timestamp ON parserresult;
        DROP FUNCTION trigger_set_timestamp;
        
        -- Removing 'updated_at'
        ALTER TABLE parserresult
          DROP COLUMN updated_at;
    ''')
