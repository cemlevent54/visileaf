"""is_starred

Revision ID: 708bf79769db
Revises: ea82047169a7
Create Date: 2025-12-05 22:56:47.682654

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '708bf79769db'
down_revision = 'ea82047169a7'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add is_starred column with default value False for existing rows
    op.add_column('images', sa.Column('is_starred', sa.Boolean(), nullable=False, server_default='false'))


def downgrade() -> None:
    op.drop_column('images', 'is_starred')
