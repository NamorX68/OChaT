"""Add health check metrics to Model.

Revision ID: 208dc0c232e2
Revises: f608934696fd
Create Date: 2026-08-28 14:48:52.044819

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '208dc0c232e2'
down_revision: str | None = 'f608934696fd'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('model', sa.Column('last_check_latency_ms', sa.Float(), nullable=True))
    op.add_column('model', sa.Column('last_check_tokens_per_second', sa.Float(), nullable=True))
    op.add_column('model', sa.Column('last_check_error', sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('model', 'last_check_error')
    op.drop_column('model', 'last_check_tokens_per_second')
    op.drop_column('model', 'last_check_latency_ms')
