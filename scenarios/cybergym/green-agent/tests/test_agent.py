
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from a2a.types import Message, Task, TaskStatus, TaskState, TextPart, Part, Role
from a2a.server.tasks import TaskUpdater

